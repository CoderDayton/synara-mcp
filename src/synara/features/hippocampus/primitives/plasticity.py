"""Per-edge plasticity layer over the successor representation.

Sits on top of ``SuccessorRepresentation`` and tracks E-LTP / L-LTP /
habit / LTD dynamics on directed edges ``(i, j)``. Existing SR rows
``M[i, j]`` continue to be the discounted-closure ranking signal; this
layer adds a separate durable weight ``W[i, j]`` and a transient bonus
``b[i, j]`` that together encode the reinforcement history.

State per edge ``(i, j)``
-------------------------
- ``weight``       L-LTP-consolidated durable strength (real number ``>= 0``).
- ``bonus``        Transient E-LTP amplification (decays with
                   ``e_ltp_decay_seconds`` real wall-clock).
- ``bonus_set_at`` Wall-clock of the most recent reinforcement that
                   updated the bonus.
- ``hits_in_window``  Reinforcements observed inside the current E-LTP
                      window (zeroed on transition to durable weight).
- ``total_hits``   Lifetime reinforcement count (monotone non-decreasing).
- ``ever_habit``   Latched once ``total_hits >= habit_threshold_hits``;
                   never cleared. Drives slower LTD and faster relearn.
- ``last_touch``   Wall-clock of most recent reinforcement.

Math
----
Reinforcement of ``(i, j)`` with score ``s in [0, 1]`` at time ``t``:

    elapsed = max(0, t - bonus_set_at)
    bonus  *= exp(-elapsed / tau_E)                    # E-LTP decay
    n      = (n + 1) if elapsed <= tau_E else 1        # window count
    gain   = s * (sigma if ever_habit else 1)          # savings boost
    bonus += 0.5 * gain
    if n >= theta_L:                                    # L-LTP transition
        weight += bonus
        bonus, n = 0.0, 0
    total_hits += 1
    if total_hits >= theta_H: ever_habit = True

LTD over real-elapsed ``dt`` seconds (compressed by ``time_compression``):

    idle_days = dt * C / 86400
    mu        = mu_habit if ever_habit else 1
    weight   *= (1 - lambda * mu) ** idle_days

Edges with ``weight < prune_floor`` and not ``ever_habit`` are removed.
``ever_habit`` edges persist even at zero weight so the savings flag
survives long disuse (Ebbinghaus savings).
"""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass

from .successor import SuccessorRepresentation


@dataclass(slots=True)
class _Edge:
    weight: float = 0.0
    bonus: float = 0.0
    bonus_set_at: float = -math.inf
    hits_in_window: int = 0
    total_hits: int = 0
    ever_habit: bool = False
    last_touch: float = -math.inf


class PlasticityGraph:
    """E-LTP / L-LTP / habit / LTD layer over a ``SuccessorRepresentation``.

    Composition (not inheritance): SR keeps its TD(0) closure semantics;
    this class owns the durable-weight + transient-bonus state. Both can
    be queried jointly via :meth:`boost`.
    """

    def __init__(
        self,
        sr: SuccessorRepresentation | None,
        *,
        e_ltp_decay_seconds: float,
        l_ltp_threshold_hits: int,
        habit_threshold_hits: int,
        habit_ltd_multiplier: float,
        habit_savings_factor: float,
        ltd_decay_per_idle_day: float,
        time_compression: float,
        prune_floor: float = 1e-3,
    ) -> None:
        if e_ltp_decay_seconds <= 0:
            raise ValueError("e_ltp_decay_seconds must be positive")
        if l_ltp_threshold_hits <= 0:
            raise ValueError("l_ltp_threshold_hits must be positive")
        if habit_threshold_hits <= 0:
            raise ValueError("habit_threshold_hits must be positive")
        if not 0.0 <= habit_ltd_multiplier <= 1.0:
            raise ValueError("habit_ltd_multiplier must be in [0, 1]")
        if habit_savings_factor < 1.0:
            raise ValueError("habit_savings_factor must be >= 1")
        if not 0.0 <= ltd_decay_per_idle_day <= 1.0:
            raise ValueError("ltd_decay_per_idle_day must be in [0, 1]")
        if time_compression <= 0:
            raise ValueError("time_compression must be positive")
        self._sr = sr
        self.e_ltp_decay_seconds = float(e_ltp_decay_seconds)
        self.l_ltp_threshold_hits = int(l_ltp_threshold_hits)
        self.habit_threshold_hits = int(habit_threshold_hits)
        self.habit_ltd_multiplier = float(habit_ltd_multiplier)
        self.habit_savings_factor = float(habit_savings_factor)
        self.ltd_decay_per_idle_day = float(ltd_decay_per_idle_day)
        self.time_compression = float(time_compression)
        self.prune_floor = float(prune_floor)
        self._edges: dict[tuple[int, int], _Edge] = {}
        # Outgoing-neighbour index for O(deg) spreading activation.
        self._out: dict[int, set[int]] = defaultdict(set)

    # ----------------------------------------------------------- mutate
    def reinforce(self, i: int, j: int, *, score: float, now: float) -> None:
        """Apply one reinforcement event to edge ``(i, j)``.

        ``score`` should be in ``[0, 1]`` (a recall confidence or a
        normalised co-activation strength). Self-loops are silently
        ignored.
        """
        if i == j:
            return
        if not 0.0 <= score <= 1.0:
            score = max(0.0, min(1.0, float(score)))
        key = (i, j)
        e = self._edges.get(key)
        if e is None:
            e = _Edge()
            self._edges[key] = e
            self._out[i].add(j)
        # 1. Decay transient bonus by REAL wall-clock since last set.
        if math.isfinite(e.bonus_set_at):
            elapsed = max(0.0, now - e.bonus_set_at)
            if elapsed > 0:
                e.bonus *= math.exp(-elapsed / self.e_ltp_decay_seconds)
        else:
            elapsed = math.inf
        # 2. In-window vs reset.
        if elapsed <= self.e_ltp_decay_seconds:
            e.hits_in_window += 1
        else:
            e.hits_in_window = 1
        # 3. Score-weighted increment, with savings if ever-habit.
        gain = score * (self.habit_savings_factor if e.ever_habit else 1.0)
        e.bonus += 0.5 * gain
        e.bonus_set_at = now
        # 4. L-LTP fold-in.
        if e.hits_in_window >= self.l_ltp_threshold_hits:
            e.weight += e.bonus
            e.bonus = 0.0
            e.hits_in_window = 0
        # 5. Lifetime / habit latch.
        e.total_hits += 1
        if e.total_hits >= self.habit_threshold_hits:
            e.ever_habit = True
        e.last_touch = now

    def ltd_pass(self, *, now: float) -> int:
        """Decay weights for non-recently-touched edges; prune dead ones.

        Returns the number of edges removed (always non-habits — habit
        edges are *never* removed even when weight hits zero, so the
        savings flag survives indefinite disuse).
        """
        rate = self.ltd_decay_per_idle_day
        if rate <= 0.0:
            return 0
        prune: list[tuple[int, int]] = []
        for key, e in self._edges.items():
            if not math.isfinite(e.last_touch):
                continue
            idle_real = max(0.0, now - e.last_touch)
            idle_days = idle_real * self.time_compression / 86400.0
            mu = self.habit_ltd_multiplier if e.ever_habit else 1.0
            decay_per_day = rate * mu
            if decay_per_day >= 1.0:
                e.weight = 0.0
            else:
                e.weight *= max(0.0, 1.0 - decay_per_day) ** idle_days
            # Bonus is transient: same wall-clock decay as reinforce-time.
            if math.isfinite(e.bonus_set_at):
                e.bonus *= math.exp(-max(0.0, now - e.bonus_set_at) / self.e_ltp_decay_seconds)
            if e.weight < self.prune_floor and not e.ever_habit:
                prune.append(key)
        for key in prune:
            i, _j = key
            self._edges.pop(key, None)
            self._out[i].discard(_j)
            if not self._out[i]:
                self._out.pop(i, None)
        return len(prune)

    # ------------------------------------------------------------- read
    def edge_weight(self, i: int, j: int) -> float:
        """Combined durable + transient strength for the directed edge."""
        e = self._edges.get((i, j))
        if e is None:
            return 0.0
        return e.weight + e.bonus

    def is_habit(self, i: int, j: int) -> bool:
        e = self._edges.get((i, j))
        return bool(e and e.ever_habit)

    def boost(self, anchor: int, ids: list[int]) -> dict[int, float]:
        """Return ``{j: SR[a,j] + W[a,j] + b[a,j]}`` for each j in ``ids``."""
        sr_part = self._sr.boost(anchor, ids) if self._sr is not None else {}
        out: dict[int, float] = {}
        for j in ids:
            out[j] = sr_part.get(j, 0.0) + self.edge_weight(anchor, j)
        return out

    def spreading(
        self,
        anchor: int,
        targets: list[int],
        *,
        hops: int,
        gamma: float,
    ) -> dict[int, float]:
        """Bounded-hop max-product BFS over durable weights.

        For each ``j in targets`` returns the best chain activation
        ``gamma^d * prod(W on path)`` from ``anchor`` to ``j`` over
        paths of length ``<= hops``. Returns 0 for ids unreachable
        within ``hops`` or when ``hops <= 0`` / ``gamma <= 0``.
        """
        if hops <= 0 or gamma <= 0.0 or not targets:
            return {j: 0.0 for j in targets}
        target_set = set(targets)
        # frontier: id -> best activation so far
        best: dict[int, float] = {anchor: 1.0}
        frontier: dict[int, float] = {anchor: 1.0}
        for _ in range(hops):
            next_frontier: dict[int, float] = {}
            for k, ak in frontier.items():
                for j in self._out.get(k, ()):
                    e = self._edges.get((k, j))
                    if e is None or e.weight <= 0.0:
                        continue
                    contrib = ak * gamma * e.weight
                    if contrib <= 0.0:
                        continue
                    if contrib > next_frontier.get(j, 0.0):
                        next_frontier[j] = contrib
            for j, v in next_frontier.items():
                if v > best.get(j, 0.0):
                    best[j] = v
            frontier = next_frontier
            if not frontier:
                break
        return {j: float(best.get(j, 0.0)) for j in targets if j != anchor or anchor in target_set}

    # --------------------------------------------------- introspection
    def stats(self) -> dict[str, float]:
        if not self._edges:
            return {"edges": 0.0, "habits": 0.0, "max_total_hits": 0.0, "max_weight": 0.0}
        habits = 0
        max_total = 0
        max_w = 0.0
        for e in self._edges.values():
            if e.ever_habit:
                habits += 1
            max_total = max(max_total, e.total_hits)
            max_w = max(max_w, e.weight)
        return {
            "edges": float(len(self._edges)),
            "habits": float(habits),
            "max_total_hits": float(max_total),
            "max_weight": float(max_w),
        }
