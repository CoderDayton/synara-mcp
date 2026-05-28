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
two TD passes per stored edge. ``M`` itself is not persisted — it is a
fast, derivable ranking prior, and persisting it would require an
upsert per row-key touched by every TD step. The in-memory window
state is intentionally not persisted: it is a short-lived global
recency queue, not part of the relational graph.
"""

from __future__ import annotations

import asyncio
import logging
from collections import defaultdict, deque
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from simplevecdb import AsyncVectorCollection

DEFAULT_KIND = "sr"
_log = logging.getLogger(__name__)

# Co-occurrence is GLOBAL, not partitioned by session. All episode
# accesses share one observation window so episodes co-occurring across
# sessions fold into T by the exact same rule as intra-session ones —
# the store is one interconnected graph, not per-session islands. The
# ``session_id`` arg on observe* is retained for call-site clarity
# (recall passes the caller's session) but no longer keys the window.


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
        if not 0.0 <= self.gamma < 1.0:
            raise ValueError("gamma must be in [0, 1)")
        if not 0.0 < self.alpha <= 1.0:
            raise ValueError("alpha must be in (0, 1]")
        if self.window_seconds <= 0.0:
            raise ValueError("window_seconds must be positive")
        if self.omega_max < 0.0:
            raise ValueError("omega_max must be >= 0")
        if self.cold_start_ratio <= 0.0:
            raise ValueError("cold_start_ratio must be positive")
        self._T_counts: dict[int, dict[int, float]] = defaultdict(lambda: defaultdict(float))
        self._T_row_sum: dict[int, float] = defaultdict(float)
        self._M: dict[int, dict[int, float]] = defaultdict(lambda: defaultdict(float))
        self._window: _Window = _Window()
        self._total_edges: float = 0.0
        self._coll: AsyncVectorCollection | None = None
        self._loaded: bool = False
        # Eager Lock binding (Python 3.13: ``asyncio.Lock()`` does not
        # bind to an event loop until first use, so this is safe in
        # __post_init__). Avoids the lazy-create-then-acquire pattern
        # which is brittle to future refactors that insert ``await``
        # between the None-check and the assignment.
        #
        # ``_state_lock`` guards every mutation of ``_T_counts``,
        # ``_T_row_sum``, ``_M``, ``_window``, ``_pending``, and
        # ``_total_edges``. Observe/evict acquire it for their full
        # mutation; ``load()``'s rebuild and ``flush()``'s snapshot also
        # acquire it so a concurrent observe cannot tear in. Pure
        # asyncio gives sync code non-preemption, but several callers
        # (``forget``, ``flush``) interleave awaits between mutate and
        # IO — without the lock those interleavings reintroduce evicted
        # nodes or write stale counts. Coarse single lock is fine: the
        # contended sections are tiny and the leader is single-process.
        self._state_lock: asyncio.Lock = asyncio.Lock()
        self._load_lock: asyncio.Lock = asyncio.Lock()
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
        async with self._load_lock:
            if self._loaded:
                return
            if self._coll is None:
                self._loaded = True
                return
            edges = await self._coll.get_edges(kind=self.kind)
            # Hold ``_state_lock`` for the rebuild so a concurrent
            # ``observe()`` (which also takes the lock) cannot mutate
            # ``_T_counts`` / ``_M`` under the TD passes' feet. The
            # whole rebuild is sync once we are inside the lock, so the
            # event loop cannot dispatch another coroutine into the
            # mutating critical section.
            async with self._state_lock:
                for e in edges:
                    i, j = int(e.src_id), int(e.dst_id)
                    count = float(e.hits)
                    if count <= 0.0:
                        continue
                    self._T_counts[i][j] = count
                    self._T_row_sum[i] += count
                    self._total_edges += count
                # Rebuild M from T via two TD passes per stored edge.
                # Order is whatever the collection returned; M
                # converges with repeated exposures, so a second sweep
                # meaningfully tightens the approximation of the
                # discounted closure without the cost of a full Bellman
                # solve. Two passes is the empirical knee where
                # successive sweeps stop materially moving rows.
                pass_items = [(i, j) for i, row in self._T_counts.items() for j in row]
                for _ in range(2):
                    for i, j in pass_items:
                        self._td_update(i, j)
                self._loaded = True

    async def flush(self) -> None:
        """Persist any pending edge updates to ``coll.edges``.

        Per-edge upsert failures are tolerated: the failed (i, j) pair
        is re-added to ``_pending`` so the next flush retries it, and
        the exception is logged with edge context. Without this, a
        transient backend error would silently drop the edge from the
        durable T tally — exactly the kind of write that's hardest to
        spot missing.
        """
        if self._coll is None or not self._pending:
            return
        # Snapshot pending pairs + their current counts under the state
        # lock so observe() cannot interleave between the read and the
        # upsert. Without the snapshot, a concurrent observe could
        # bump _T_counts[i][j] after we read it, and the upsert would
        # persist a stale count for an edge that's already in the next
        # flush's pending set — fine for eventual consistency, but the
        # snapshot is cheap and removes the timing dependency entirely.
        async with self._state_lock:
            pending = self._pending
            self._pending = set()
            snapshot: list[tuple[int, int, int]] = [
                (i, j, int(self._T_counts[i][j])) for (i, j) in pending
            ]
        coll = self._coll
        failed: list[tuple[int, int]] = []
        for i, j, count in snapshot:
            # Re-validate endpoints: a state_freeze evict + delete may
            # have run between our snapshot capture and this iteration
            # (the snapshot escaped the _pending purge in _evict_inner).
            # Upserting an edge whose endpoint is no longer in
            # _T_counts would persist an FK-violating row.
            if i not in self._T_counts or j not in self._T_counts.get(i, {}):
                continue
            try:
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
            except Exception:
                _log.exception("SR edge upsert failed: src=%d dst=%d kind=%s", i, j, self.kind)
                failed.append((i, j))
        if failed:
            async with self._state_lock:
                self._pending.update(failed)

    # ------------------------------------------------------------- observation

    async def observe(self, session_id: str, ep_id: int, t: float) -> None:
        """Record episode access at time ``t`` in ``session_id``.

        Co-occurrences with in-window prior accesses (excluding self)
        fold into T and propagate through M via TD(0). Edge changes are
        queued for the next :meth:`flush` call. Async because it
        acquires ``_state_lock`` to gate concurrent observe / load /
        evict / flush callers under the leader's event loop.
        """
        async with self._state_lock:
            win = self._window
            cutoff = t - self.window_seconds
            while win.queue and win.queue[0][1] < cutoff:
                win.queue.popleft()
            for prior_id, _t in win.queue:
                if prior_id == ep_id:
                    continue
                self._record_edge(prior_id, ep_id)
            win.queue.append((ep_id, t))

    async def observe_recall_set(
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
        async with self._state_lock:
            win = self._window
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
        # dict.keys() is already set-like and supports `|` directly,
        # so we skip the two extra `set(...)` allocations and union
        # the views once. ``{j}`` ensures the e_j basis entry lands in
        # the key set even when neither row references j yet.
        keys: set[int] = Mi.keys() | Mj.keys() | {j}
        a = self.alpha
        g = self.gamma
        one_minus_a = 1.0 - a
        # Bind dict.get to a local: skips the per-iteration attribute
        # lookup, which dominates this Python-bound inner loop when
        # rows are dense (hundreds of entries).
        Mi_get = Mi.get
        Mj_get = Mj.get
        for k in keys:
            target = (1.0 if k == j else 0.0) + g * Mj_get(k, 0.0)
            new_val = one_minus_a * Mi_get(k, 0.0) + a * target
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

    # ------------------------------------------------------------------ removal

    async def evict_nodes(self, ids: set[int]) -> None:
        """Drop episode ids from in-memory SR state.

        Required when the underlying episode documents are deleted while
        the process is live: ``coll.edges`` has an ``ON DELETE CASCADE``
        FK to documents, so durable SR edges vanish with the doc — but a
        lingering ``_pending`` entry, ``_T`` row/column, or window entry
        referencing a now-deleted id would make the next
        :meth:`flush` upsert a FK-violating edge. Removes outgoing rows,
        incoming columns, the ``M`` row/column, queued pending pairs, and
        window entries; ``_total_edges`` is decremented by the removed
        tally (clamped at 0). Async + ``_state_lock``-guarded so that a
        concurrent observe cannot reintroduce an edge for a node we are
        in the middle of evicting.
        """
        if not ids:
            return
        async with self._state_lock:
            self.evict_nodes_locked(ids)

    @asynccontextmanager
    async def state_freeze(self) -> AsyncIterator[None]:
        """Acquire ``_state_lock`` for a multi-step critical section.

        Callers wrap an evict + durable-delete pair in this context so
        no concurrent :meth:`observe` interleaves between the in-memory
        eviction and the row delete (which would re-introduce edges for
        ids the caller is about to delete). Inside the ``async with``
        call :meth:`evict_nodes_locked` rather than :meth:`evict_nodes`
        — re-entering :meth:`evict_nodes` would deadlock on the
        non-reentrant lock.
        """
        async with self._state_lock:
            yield

    def evict_nodes_locked(self, ids: set[int]) -> None:
        """Sync body of :meth:`evict_nodes` for callers already holding
        ``_state_lock`` via :meth:`state_freeze`. Public so cross-module
        callers (``forget``, :meth:`MemoryService.delete_episode`) can
        coordinate an evict + delete inside one frozen section.
        """
        if not ids:
            return
        self._evict_inner(ids)

    def _evict_inner(self, ids: set[int]) -> None:
        for x in ids:
            row = self._T_counts.pop(x, None)
            if row is not None:
                self._total_edges -= sum(row.values())
            self._T_row_sum.pop(x, None)
            self._M.pop(x, None)
        for i, row in self._T_counts.items():
            for x in ids & row.keys():
                removed = row.pop(x)
                self._T_row_sum[i] -= removed
                self._total_edges -= removed
        for mrow in self._M.values():
            for x in ids & mrow.keys():
                mrow.pop(x, None)
        self._total_edges = max(self._total_edges, 0.0)
        self._pending = {(i, j) for (i, j) in self._pending if i not in ids and j not in ids}
        win = self._window
        kept = [(e, t) for (e, t) in win.queue if e not in ids]
        win.queue.clear()
        win.queue.extend(kept)

    async def snapshot_boost(self, anchor_id: int, ep_ids: list[int]) -> dict[int, float]:
        """Atomic ``boost`` under ``_state_lock``.

        Equivalent to :meth:`boost` but guaranteed not to see a
        half-mutated ``_M`` row — recall pipelines call this to lock in
        a deterministic rank prior before later awaits that can
        interleave with concurrent observe/load/evict.
        """
        async with self._state_lock:
            Mi = self._M.get(anchor_id, {})
            return {j: float(Mi.get(j, 0.0)) for j in ep_ids}

    def _reset_for_tests(self, **kwargs: Any) -> None:  # pragma: no cover
        """Reset in-memory state (test helper, not used in production)."""
        self.__post_init__()
        for k, v in kwargs.items():
            setattr(self, k, v)
