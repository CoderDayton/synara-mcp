"""Successor representation (Stachenfeld, Botvinick & Gershman 2017).

A sparse SR that boosts recall ranking with a temporal co-occurrence
signal. We track a transition tally ``T_{ij}`` across episodes recalled
within ``window_seconds`` of one another in the same session, then
maintain its discounted closure

    M = sum_{k>=0} gamma^k T^k = (I - gamma T)^{-1}

approximately via TD(0) updates rather than ever forming the inverse:
on each new edge (i -> j) we run

    M[i] <- M[i] + alpha * (e_j + gamma * M[j] - M[i])

which converges to the true SR row in expectation while staying
O(|M[i]| + |M[j]|) per step. Recall blends the SR boost ``M[i*, j]``
(where ``i*`` is the best-cosine episodic anchor) into the rank score
with weight ``omega`` that ramps from 0 to ``omega_max`` once the
population of edges exceeds the population of episodes — this gates the
signal during cold start.

Durability
----------
When an ``AsyncVectorCollection`` is attached via :meth:`attach`, the
transition tally ``T_{ij}`` is persisted to ``coll.edges`` under
``kind="sr"`` (count stored in the ``hits`` column). On startup
``load()`` rehydrates ``T`` from disk and rebuilds ``M`` by replaying
one TD pass per stored edge. ``M`` itself is not persisted — it is a
fast, derivable ranking prior, and persisting it would require an
upsert per row-key touched by every TD step. The in-memory window
state (``_sessions``) is intentionally not persisted: it is a
short-lived recency queue, not part of the relational graph.
"""

from __future__ import annotations

import asyncio
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from simplevecdb import AsyncVectorCollection

DEFAULT_KIND = "sr"


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
    kind: str = DEFAULT_KIND

    def __post_init__(self) -> None:
        self._T_counts: dict[int, dict[int, float]] = defaultdict(lambda: defaultdict(float))
        self._T_row_sum: dict[int, float] = defaultdict(float)
        self._M: dict[int, dict[int, float]] = defaultdict(lambda: defaultdict(float))
        self._sessions: dict[str, _Window] = {}
        self._total_edges: float = 0.0
        self._coll: AsyncVectorCollection | None = None
        self._loaded: bool = False
        self._load_lock: asyncio.Lock | None = None
        self._pending: set[tuple[int, int]] = set()

    # ------------------------------------------------------------ persistence

    def attach(self, coll: AsyncVectorCollection) -> None:
        """Bind a vector collection for durable T-count storage.

        Must be called before :meth:`load` / :meth:`flush`. Safe to call
        once; subsequent calls overwrite the bound collection.
        """
        self._coll = coll

    async def load(self) -> None:
        """Rehydrate ``T`` from ``coll.edges`` and rebuild ``M``.

        Idempotent. Concurrent callers serialise on an internal lock so
        the rebuild runs exactly once per process.
        """
        if self._loaded:
            return
        if self._load_lock is None:
            self._load_lock = asyncio.Lock()
        async with self._load_lock:
            if self._loaded:
                return
            if self._coll is None:
                self._loaded = True
                return
            edges = await self._coll.get_edges(kind=self.kind)
            for e in edges:
                i, j = int(e.src_id), int(e.dst_id)
                count = float(e.hits)
                if count <= 0.0:
                    continue
                self._T_counts[i][j] = count
                self._T_row_sum[i] += count
                self._total_edges += count
            # Rebuild M from T via one TD pass per stored edge. Order
            # is whatever the collection returned; M converges with
            # repeated exposures so a single pass is an approximation.
            for i, row in list(self._T_counts.items()):
                for j in list(row.keys()):
                    self._td_update(i, j)
            self._loaded = True

    async def flush(self) -> None:
        """Persist any pending edge updates to ``coll.edges``."""
        if self._coll is None or not self._pending:
            return
        pending = list(self._pending)
        self._pending.clear()
        coll = self._coll
        for i, j in pending:
            count = int(self._T_counts[i][j])
            await asyncio.to_thread(
                coll._collection.edges.upsert,
                i,
                j,
                kind=self.kind,
                weight=0.0,
                bonus=0.0,
                hits=count,
                metadata={},
            )

    # ------------------------------------------------------------- observation

    def observe(self, session_id: str, ep_id: int, t: float) -> None:
        """Record episode access at time ``t`` in ``session_id``.

        Co-occurrences with in-window prior accesses (excluding self)
        fold into T and propagate through M via TD(0). Edge changes are
        queued for the next :meth:`flush` call.
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
        """Anchor-style co-occurrence: anchor -> each j in ``other_ids``.

        Anchor enters the in-session window for future chain edges.
        Avoids the n*(n-1)/2 inflation a naive pairwise loop produces.
        """
        win = self._sessions.setdefault(session_id, _Window())
        cutoff = t - self.window_seconds
        while win.queue and win.queue[0][1] < cutoff:
            win.queue.popleft()
        for prior_id, _t in win.queue:
            if prior_id == anchor_id:
                continue
            self._record_edge(prior_id, anchor_id)
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
        if self._coll is not None:
            self._pending.add((i, j))

    def _td_update(self, i: int, j: int) -> None:
        Mi = self._M[i]
        Mj = self._M[j]
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

    # ------------------------------------------------------------------ readers

    def boost(self, anchor_id: int, ep_ids: list[int]) -> dict[int, float]:
        """Return SR boost scores ``{ep_id: M[anchor_id, ep_id]}``."""
        Mi = self._M.get(anchor_id, {})
        return {j: float(Mi.get(j, 0.0)) for j in ep_ids}

    def omega(self, episode_count: int) -> float:
        """Cold-start ramp: 0 until edges/episodes >= ``cold_start_ratio``,
        then linearly approaches ``omega_max`` as edges accumulate."""
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

    def _reset_for_tests(self, **kwargs: Any) -> None:  # pragma: no cover
        """Reset in-memory state (test helper, not used in production)."""
        self.__post_init__()
        for k, v in kwargs.items():
            setattr(self, k, v)
