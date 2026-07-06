"""Per-edge plasticity (E-LTP / L-LTP / habit / LTD) over the SR.

Edges are persisted to ``coll.edges`` with kind ``"plasticity"``. The
durable weight ``W[i,j]`` (column ``weight``) and transient bonus
``b[i,j]`` (column ``bonus``) accrue from reinforcement events; lifetime
hits live in column ``hits`` and ``ever_habit`` is derived as
``hits >= habit_threshold_hits``. Window state (``bonus_set_at``,
``hits_in_window``) lives in the metadata blob.

Reinforce ``(i, j)`` with score ``s ∈ [0, 1]`` at time ``t``::

    elapsed  = max(0, t - bonus_set_at)
    bonus   *= exp(-elapsed / tau_E)               # E-LTP decay
    n        = (n + 1) if elapsed <= tau_E else 1  # window
    gain     = s * (sigma if ever_habit else 1)    # savings boost
    bonus   += 0.5 * gain
    if n >= theta_L:                                # L-LTP fold
        weight += bonus; bonus, n = 0, 0
    hits += 1

LTD over real-elapsed ``dt`` (``time_compression`` accelerates it)::

    idle_days = dt * C / 86400
    mu        = mu_habit if ever_habit else 1
    weight   *= (1 - lambda * mu) ** idle_days

Edges with ``weight < prune_floor`` and not ``ever_habit`` are deleted.
Habits persist at zero weight so the savings flag survives long disuse.
"""

from __future__ import annotations

import asyncio
import math
import weakref
from typing import TYPE_CHECKING, Any

from .successor import SuccessorRepresentation

if TYPE_CHECKING:
    from simplevecdb import AsyncVectorCollection

DEFAULT_KIND = "plasticity"


class PlasticityGraph:
    """Async plasticity layer over a persistent edge table."""

    def __init__(
        self,
        collection: AsyncVectorCollection,
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
        kind: str = DEFAULT_KIND,
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
        self._coll = collection
        self._sr = sr
        self.kind = kind
        self.e_ltp_decay_seconds = float(e_ltp_decay_seconds)
        self.l_ltp_threshold_hits = int(l_ltp_threshold_hits)
        self.habit_threshold_hits = int(habit_threshold_hits)
        self.habit_ltd_multiplier = float(habit_ltd_multiplier)
        self.habit_savings_factor = float(habit_savings_factor)
        self.ltd_decay_per_idle_day = float(ltd_decay_per_idle_day)
        self.time_compression = float(time_compression)
        self.prune_floor = float(prune_floor)
        # Serialise the read-modify-write in ``reinforce`` per directed
        # edge: ``edges.upsert`` is a full-row overwrite, so two
        # concurrent recalls sharing an anchor would otherwise lose the
        # earlier increment. Lock creation has no ``await`` between the
        # lookup and the insert, so it is atomic under asyncio.
        # WeakValueDictionary so the lock entry disappears once no
        # coroutine holds it — otherwise the table would accumulate one
        # lock per ever-seen edge for the lifetime of the process.
        self._edge_locks: weakref.WeakValueDictionary[tuple[int, int], asyncio.Lock] = (
            weakref.WeakValueDictionary()
        )

    async def _upsert(
        self,
        src: int,
        dst: int,
        *,
        weight: float,
        bonus: float,
        hits: int,
        metadata: dict[str, Any],
    ) -> None:
        # Coupling note: AsyncVectorCollection exposes ``get_edges`` but no
        # async edge ``upsert``, so we reach through ``_collection.edges``
        # (the sync engine) under ``to_thread``; the catalog serialises on
        # its RLock. If simplevecdb adds an async upsert, switch to it and
        # drop this private access. (SR's flush does the same — see
        # ``hippocampus/successor.py``.)
        await asyncio.to_thread(
            self._coll._collection.edges.upsert,
            src,
            dst,
            kind=self.kind,
            weight=weight,
            bonus=bonus,
            hits=hits,
            metadata=metadata,
        )

    async def _read_one(self, i: int, j: int) -> Any | None:
        edges = await self._coll.get_edges(src=i, dst=j, kind=self.kind, limit=1)
        return edges[0] if edges else None

    async def reinforce(self, i: int, j: int, *, score: float, now: float) -> None:
        """Apply one reinforcement event to edge ``(i, j)``."""
        if i == j:
            return
        if not math.isfinite(now):
            raise ValueError("now must be finite")
        score_f = float(score)
        if not math.isfinite(score_f):
            score_f = 0.0
        score = max(0.0, min(1.0, score_f))
        lock = self._edge_locks.setdefault((i, j), asyncio.Lock())
        async with lock:
            await self._reinforce_locked(i, j, score=score, now=now)

    async def _reinforce_locked(self, i: int, j: int, *, score: float, now: float) -> None:
        e = await self._read_one(i, j)
        if e is None:
            cur_w, cur_b, cur_hits = 0.0, 0.0, 0
            md_in: dict[str, Any] = {}
        else:
            cur_w, cur_b, cur_hits = float(e.weight), float(e.bonus), int(e.hits)
            md_in = dict(e.metadata or {})
        bonus_set_at = float(md_in.get("bonus_set_at", float("-inf")))
        hits_in_window = int(md_in.get("hits_in_window", 0))
        if math.isfinite(bonus_set_at):
            elapsed = max(0.0, now - bonus_set_at)
            if elapsed > 0:
                cur_b *= math.exp(-elapsed / self.e_ltp_decay_seconds)
        else:
            elapsed = math.inf
        if elapsed <= self.e_ltp_decay_seconds:
            hits_in_window += 1
        else:
            hits_in_window = 1
        ever_habit = cur_hits >= self.habit_threshold_hits
        gain = score * (self.habit_savings_factor if ever_habit else 1.0)
        cur_b += 0.5 * gain
        if hits_in_window >= self.l_ltp_threshold_hits:
            cur_w += cur_b
            cur_b = 0.0
            hits_in_window = 0
        await self._upsert(
            i,
            j,
            weight=cur_w,
            bonus=cur_b,
            hits=cur_hits + 1,
            metadata={
                "hits_in_window": hits_in_window,
                "bonus_set_at": now,
                "last_touch_virt": now,
            },
        )

    async def ltd_pass(self, *, now: float, max_scan: int | None = None) -> int:
        """Decay weights for non-recently-touched edges; prune dead ones.

        Steady-state memory is bounded by ``prune_floor`` deletion:
        edges that fall below the floor (and are not habits) are removed
        each cycle, so the table is self-limiting in production.

        ``max_scan`` bounds the per-pass read-modify-write work — the
        expensive serial part; the key fetch is one query either way.
        The budget is spent on the *stalest* edges first (ordered by the
        catalog's ``last_touch``, a wall-clock approximation of the
        authoritative ``last_touch_virt``) because they carry the most
        pending decay; fresh edges can safely wait for a later pass, so
        the sweep converges across dreams instead of starving the very
        edges LTD targets. ``None`` (or 0) scans everything.
        """
        rate = self.ltd_decay_per_idle_day
        if rate <= 0.0:
            return 0
        # Snapshot the edge keys only; the per-edge state is re-read
        # inside the lock below so a concurrent ``reinforce`` cannot
        # interleave its read-modify-write with ours. Holding only the
        # key list keeps memory bounded and avoids stale-snapshot
        # decisions (prune vs. update) that would otherwise resurrect
        # an edge ``reinforce`` just bumped.
        edges = await self._coll.get_edges(kind=self.kind)
        if max_scan is not None and 0 < max_scan < len(edges):
            edges = sorted(edges, key=lambda e: float(e.last_touch))[:max_scan]
        edge_keys = [(int(e.src_id), int(e.dst_id)) for e in edges]
        pruned = 0
        for i, j in edge_keys:
            lock = self._edge_locks.setdefault((i, j), asyncio.Lock())
            async with lock:
                cur = await self._read_one(i, j)
                if cur is None:
                    # ``reinforce`` deleted or never wrote it; skip.
                    continue
                md = dict(cur.metadata or {})
                last_touch = float(md.get("last_touch_virt", float(cur.last_touch)))
                if not math.isfinite(last_touch):
                    continue
                idle_real = max(0.0, now - last_touch)
                idle_days = idle_real * self.time_compression / 86400.0
                ever_habit = int(cur.hits) >= self.habit_threshold_hits
                mu = self.habit_ltd_multiplier if ever_habit else 1.0
                decay_per_day = rate * mu
                if decay_per_day >= 1.0:
                    new_w = 0.0
                else:
                    new_w = float(cur.weight) * max(0.0, 1.0 - decay_per_day) ** idle_days
                bonus_set_at = float(md.get("bonus_set_at", float("-inf")))
                new_b = float(cur.bonus)
                if math.isfinite(bonus_set_at):
                    new_b *= math.exp(-max(0.0, now - bonus_set_at) / self.e_ltp_decay_seconds)
                # Prune on combined durable + still-decaying transient
                # potentiation: a bonus-only E-LTP edge (weight 0) is
                # not dead — it is mid-consolidation, and culling it
                # resets the in-window hit counter so off-policy dream
                # replay could never reach the L-LTP fold across
                # cadenced dreams.
                if new_w + new_b < self.prune_floor and not ever_habit:
                    await self._coll.delete_edge(i, j, kind=self.kind)
                    pruned += 1
                else:
                    await self._coll.update_edge(
                        i,
                        j,
                        kind=self.kind,
                        weight=new_w,
                        bonus=new_b,
                        metadata={**md, "last_touch_virt": last_touch},
                    )
        return pruned

    async def edge_weight(self, i: int, j: int) -> float:
        """Combined durable + transient strength for the directed edge."""
        e = await self._read_one(i, j)
        if e is None:
            return 0.0
        return float(e.weight) + float(e.bonus)

    async def is_habit(self, i: int, j: int) -> bool:
        e = await self._read_one(i, j)
        if e is None:
            return False
        return int(e.hits) >= self.habit_threshold_hits

    async def edge_state(self, i: int, j: int) -> dict[str, Any] | None:
        """Full edge state for tests/introspection."""
        e = await self._read_one(i, j)
        if e is None:
            return None
        md = dict(e.metadata or {})
        return {
            "weight": float(e.weight),
            "bonus": float(e.bonus),
            "hits": int(e.hits),
            "last_touch": float(md.get("last_touch_virt", float(e.last_touch))),
            "hits_in_window": int(md.get("hits_in_window", 0)),
            "bonus_set_at": float(md.get("bonus_set_at", float("-inf"))),
            "ever_habit": int(e.hits) >= self.habit_threshold_hits,
        }

    async def boost(self, anchor: int, ids: list[int]) -> dict[int, float]:
        """Return ``{j: SR[a,j] + W[a,j] + b[a,j]}`` for each j in ``ids``."""
        sr_part = self._sr.boost(anchor, ids) if self._sr is not None else {}
        if not ids:
            return {}
        edges = await self._coll.get_edges(src=anchor, kind=self.kind)
        edge_map = {int(e.dst_id): float(e.weight) + float(e.bonus) for e in edges}
        return {j: sr_part.get(j, 0.0) + edge_map.get(j, 0.0) for j in ids}

    async def spreading(
        self,
        anchor: int,
        targets: list[int],
        *,
        hops: int,
        gamma: float,
        max_fanout: int = 0,
    ) -> dict[int, float]:
        """Bounded-hop max-product BFS over durable weights.

        When ``max_fanout > 0``, each frontier node expands only its
        top-``max_fanout`` edges by weight per hop -- a hub-degree guard
        against a single high-out-degree episode flooding the frontier.
        ``0`` (default) leaves expansion unbounded.
        """
        if hops <= 0 or gamma <= 0.0 or not targets:
            return dict.fromkeys(targets, 0.0)
        target_set = set(targets)
        best: dict[int, float] = {anchor: 1.0}
        frontier: dict[int, float] = {anchor: 1.0}
        for _ in range(hops):
            # Fan out per-frontier-node ``get_edges`` calls concurrently.
            # simplevecdb's edge catalog has no IN-filter on ``src_id``,
            # so a single batched query isn't expressible — but each
            # call routes through ``to_thread``, and ``asyncio.gather``
            # lets the work pipeline through the executor instead of
            # paying per-node await latency. The catalog's RLock still
            # serializes the SQL, so this scales hop fan-out *latency*,
            # not throughput.
            keys = list(frontier.keys())
            results = await asyncio.gather(
                *(self._coll.get_edges(src=k, kind=self.kind) for k in keys),
            )
            next_frontier: dict[int, float] = {}
            for k, edges in zip(keys, results, strict=True):
                ak = frontier[k]
                node_edges = edges
                if max_fanout > 0 and len(node_edges) > max_fanout:
                    node_edges = sorted(node_edges, key=lambda e: float(e.weight), reverse=True)[
                        :max_fanout
                    ]
                for e in node_edges:
                    w = float(e.weight)
                    if w <= 0.0:
                        continue
                    contrib = ak * gamma * w
                    if contrib <= 0.0:
                        continue
                    dst = int(e.dst_id)
                    if contrib > next_frontier.get(dst, 0.0):
                        next_frontier[dst] = contrib
            for j, v in next_frontier.items():
                if v > best.get(j, 0.0):
                    best[j] = v
            frontier = next_frontier
            if not frontier:
                break
        return {j: float(best.get(j, 0.0)) for j in targets if j != anchor or anchor in target_set}

    async def stats(self) -> dict[str, float]:
        edges = await self._coll.get_edges(kind=self.kind)
        if not edges:
            return {"edges": 0.0, "habits": 0.0, "max_total_hits": 0.0, "max_weight": 0.0}
        habits = sum(1 for e in edges if int(e.hits) >= self.habit_threshold_hits)
        max_total = max(int(e.hits) for e in edges)
        max_w = max(float(e.weight) for e in edges)
        return {
            "edges": float(len(edges)),
            "habits": float(habits),
            "max_total_hits": float(max_total),
            "max_weight": float(max_w),
        }
