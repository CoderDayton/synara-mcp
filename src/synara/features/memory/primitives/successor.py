"""Successor representation (Stachenfeld, Botvinick & Gershman 2017).

A sparse, in-memory SR that boosts recall ranking with a temporal
co-occurrence signal. We track a transition tally ``T_{ij}`` across
episodes recalled within ``window_seconds`` of one another in the same
session, then maintain its discounted closure

    M = sum_{k>=0} gamma^k T^k = (I - gamma T)^{-1}

approximately via TD(0) updates rather than ever forming the inverse:
on each new edge (i -> j) we run

    M[i] <- M[i] + alpha * (e_j + gamma * M[j] - M[i])

which converges to the true SR row in expectation while staying O(|M[i]|
+ |M[j]|) per step. Recall blends the SR boost ``M[i*, j]`` (where
``i*`` is the best-cosine episodic anchor) into the rank score with
weight ``omega`` that ramps from 0 to ``omega_max`` once the population
of edges exceeds the population of episodes — this gates the signal
during cold start.

SR is intentionally non-durable. Schema consolidation owns the
persistent cross-episode structure; SR is a fast, per-process ranking
prior that decays naturally as the process restarts.
"""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass, field


@dataclass(slots=True)
class _Window:
    queue: deque[tuple[int, float]] = field(default_factory=deque)


@dataclass
class SuccessorRepresentation:
    gamma: float = 0.7
    alpha: float = 0.1
    window_seconds: float = 60.0
    omega_max: float = 0.3
    cold_start_ratio: float = 1.0

    def __post_init__(self) -> None:
        self._T_counts: dict[int, dict[int, float]] = defaultdict(lambda: defaultdict(float))
        self._T_row_sum: dict[int, float] = defaultdict(float)
        self._M: dict[int, dict[int, float]] = defaultdict(lambda: defaultdict(float))
        self._sessions: dict[str, _Window] = {}
        self._total_edges: float = 0.0

    def observe(self, session_id: str, ep_id: int, t: float) -> None:
        """Record episode access at time t in session.

        Co-occurrences with in-window prior accesses (excluding self)
        fold into T and propagate through M via TD(0).
        """
        win = self._sessions.setdefault(session_id, _Window())
        cutoff = t - self.window_seconds
        while win.queue and win.queue[0][1] < cutoff:
            win.queue.popleft()
        for prior_id, _t in win.queue:
            if prior_id == ep_id:
                continue
            self._record_edge(prior_id, ep_id)
        win.queue.append((ep_id, t))

    def observe_recall_set(
        self,
        session_id: str,
        anchor_id: int,
        other_ids: list[int],
        t: float,
    ) -> None:
        """Anchor-style co-occurrence: anchor -> each j in other_ids.

        Anchor enters the in-session window for future chain edges.
        Avoids n*(n-1)/2 edge inflation from pairwise co-occurrence.
        """
        win = self._sessions.setdefault(session_id, _Window())
        cutoff = t - self.window_seconds
        while win.queue and win.queue[0][1] < cutoff:
            win.queue.popleft()
        # Pre-existing in-window anchor from prior recalls earns one
        # cross-recall edge to the new anchor.
        for prior_id, _t in win.queue:
            if prior_id == anchor_id:
                continue
            self._record_edge(prior_id, anchor_id)
        # Anchor -> each other returned hit.
        for j in other_ids:
            if j == anchor_id:
                continue
            self._record_edge(anchor_id, j)
        win.queue.append((anchor_id, t))

    def _record_edge(self, i: int, j: int) -> None:
        self._T_counts[i][j] += 1.0
        self._T_row_sum[i] += 1.0
        self._total_edges += 1.0
        self._td_update(i, j)

    def _td_update(self, i: int, j: int) -> None:
        Mi = self._M[i]
        Mj = self._M[j]
        # Sparse update: keys touched by this transition.
        keys: set[int] = set(Mi.keys()) | set(Mj.keys()) | {j}
        a = self.alpha
        g = self.gamma
        for k in keys:
            target = (1.0 if k == j else 0.0) + g * Mj.get(k, 0.0)
            new_val = (1.0 - a) * Mi.get(k, 0.0) + a * target
            if new_val == 0.0:
                Mi.pop(k, None)
            else:
                Mi[k] = new_val

    def boost(self, anchor_id: int, ep_ids: list[int]) -> dict[int, float]:
        """Return SR boost scores {ep_id: M[anchor_id, ep_id]} for re-ranking."""
        Mi = self._M.get(anchor_id, {})
        return {j: float(Mi.get(j, 0.0)) for j in ep_ids}

    def omega(self, episode_count: int) -> float:
        """Cold-start ramp: 0 until edges/episodes >= cold_start_ratio,
        then linearly approaches omega_max as edges accumulate."""
        if episode_count <= 0 or self.omega_max <= 0.0:
            return 0.0
        denom = max(1.0, float(episode_count) * self.cold_start_ratio)
        ratio = self._total_edges / denom
        if ratio <= 0.0:
            return 0.0
        return self.omega_max * min(1.0, ratio)

    @property
    def total_edges(self) -> float:
        return self._total_edges
