"""Concurrency hardening tests (per the resilient-memory plan).

These tests target the failure modes the hardening pass closed:

* Write-atomicity (§1): a failed id-patch must roll back the row so
  orphan documents are never visible to recall.
* Read-modify-write (§3a/b): the per-doc lock protects ``access_history``
  appends and ``drift_total`` cap enforcement under concurrent recalls.
* SR state lock (§2): ``observe`` cannot reintroduce edges for nodes
  that ``forget`` is in the middle of evicting/deleting.
* Plasticity per-edge lock (§2c): ``ltd_pass`` and ``reinforce`` on the
  same edge serialise — no resurrection of pruned edges.

Patterns mirror ``tests/features/memory/test_service.py``: deterministic
hash embedder, ``AsyncVectorDB(":memory:")``, no model download.
"""

from __future__ import annotations

import asyncio
import hashlib
from collections.abc import AsyncIterator

import numpy as np
import pytest
import pytest_asyncio
from simplevecdb import AsyncVectorDB

from synara.features.memory.hippocampus.recall import _accrue_drift
from synara.features.memory.hippocampus.segment import split_into_segments
from synara.features.memory.hippocampus.successor import SuccessorRepresentation
from synara.features.memory.service import MemoryConfig, MemoryService


def hash_embed(text: str, dim: int = 32) -> list[float]:
    seed_bytes = hashlib.sha256(text.encode("utf-8")).digest()[:8]
    seed = int.from_bytes(seed_bytes, "big", signed=False)
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(dim).astype(np.float32)
    n = float(np.linalg.norm(v))
    out = (v / n) if n > 0 else v
    return [float(x) for x in out.tolist()]


@pytest_asyncio.fixture
async def service() -> AsyncIterator[MemoryService]:
    db = AsyncVectorDB(":memory:")
    try:
        yield MemoryService(db, config=MemoryConfig(), embed_fn=hash_embed)
    finally:
        await db.close()


# ----------------------------------------------------------------- §1 atomicity


async def test_episode_insert_rolls_back_on_id_patch_failure(
    service: MemoryService,
) -> None:
    """If ``update_metadata`` fails after ``add_texts`` succeeds, the row
    must be deleted so no orphan (no ``id`` field) lingers."""
    # Monkey-patch update_metadata to raise on the first call only.
    original = service.episodic.update_metadata
    calls = {"n": 0}

    async def boom(updates: object) -> object:
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("boom")
        return await original(updates)

    service.episodic.update_metadata = boom

    with pytest.raises(RuntimeError, match="boom"):
        await service.encode_episode("orphan-prone", "s1", salience=0.5)

    # Restore and assert no orphan row remains visible.
    service.episodic.update_metadata = original
    rows = await service.episodic.get_documents({"session_id": "s1"})
    assert rows == [], f"orphan row left behind: {rows}"


async def test_segmented_insert_rolls_back_all_segments_on_id_patch_failure(
    service: MemoryService,
) -> None:
    """``_insert_segmented`` writes N sub-records then bulk-patches their
    ids/group. If the patch fails, every segment row must be deleted so
    no orphan group lingers."""
    # Long, multi-sentence content forces theta segmentation (> the
    # default 1024-char budget across several sentences).
    sentence = "This is a reasonably long sentence about memory systems. "
    content = sentence * 40
    segments = split_into_segments(
        content,
        max_chars=service.config.theta_segment_max_chars,
        max_items=service.config.theta_segment_max_items,
    )
    assert len(segments) > 1, "test content must actually segment"

    original = service.episodic.update_metadata

    async def boom(updates: object) -> object:
        raise RuntimeError("seg-boom")

    service.episodic.update_metadata = boom
    with pytest.raises(RuntimeError, match="seg-boom"):
        await service.encode_episode(content, "seg-sess", salience=0.5)

    service.episodic.update_metadata = original
    rows = await service.episodic.get_documents({"session_id": "seg-sess"})
    assert rows == [], f"segment rows left behind after rollback: {rows}"


async def test_episode_insert_preserves_original_error_when_cleanup_also_fails(
    service: MemoryService,
) -> None:
    """If both ``update_metadata`` and the rollback ``delete_by_ids`` raise,
    the *original* ``update_metadata`` error must surface — operators need
    the root cause, not the cleanup failure."""
    original_update = service.episodic.update_metadata
    original_delete = service.episodic.delete_by_ids

    async def fail_update(updates: object) -> object:
        raise RuntimeError("original-cause")

    async def fail_delete(ids: object) -> object:
        raise RuntimeError("cleanup-also-failed")

    service.episodic.update_metadata = fail_update
    service.episodic.delete_by_ids = fail_delete

    with pytest.raises(RuntimeError, match="original-cause"):
        await service.encode_episode("x", "s1", salience=0.5)

    service.episodic.update_metadata = original_update
    service.episodic.delete_by_ids = original_delete


async def test_semantic_insert_preserves_original_error_when_cleanup_also_fails(
    service: MemoryService,
) -> None:
    """Same root-cause-preservation contract on the semantic-store path."""
    original_update = service.semantic.update_metadata
    original_delete = service.semantic.delete_by_ids

    async def fail_update(updates: object) -> object:
        raise RuntimeError("original-cause")

    async def fail_delete(ids: object) -> object:
        raise RuntimeError("cleanup-also-failed")

    service.semantic.update_metadata = fail_update
    service.semantic.delete_by_ids = fail_delete

    with pytest.raises(RuntimeError, match="original-cause"):
        await service.store_semantic_memory("a fact", kind="fact", scope="global")

    service.semantic.update_metadata = original_update
    service.semantic.delete_by_ids = original_delete


async def test_semantic_insert_rolls_back_on_id_patch_failure(
    service: MemoryService,
) -> None:
    """Same rollback contract for the semantic-store insert path."""
    original = service.semantic.update_metadata
    calls = {"n": 0}

    async def boom(updates: object) -> object:
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("boom")
        return await original(updates)

    service.semantic.update_metadata = boom
    with pytest.raises(RuntimeError, match="boom"):
        await service.store_semantic_memory("a fact", kind="fact", scope="global")

    service.semantic.update_metadata = original
    rows = await service.semantic.get_documents({"kind": "fact"})
    assert rows == [], f"orphan semantic row: {rows}"


# ---------------------------------------------------------- §3a access_history


async def test_bump_retrieval_does_not_lose_appends_under_concurrent_calls(
    service: MemoryService,
) -> None:
    """N concurrent bumps on the same doc must extend access_history by
    exactly N — no append lost to a read-modify-write race."""
    r = await service.encode_episode("shared doc", "s1", salience=0.5)
    doc_id = int(r["id"])

    # Snapshot for each call — the function re-reads inside the lock,
    # so the snapshot's staleness must not corrupt the live list.
    rows = await service.episodic.get_documents({"id": doc_id})
    _, _, snapshot = rows[0]
    initial_len = len(snapshot.get("access_history") or [])

    n = 10
    await asyncio.gather(*[service.bump_retrieval(doc_id, snapshot) for _ in range(n)])

    rows = await service.episodic.get_documents({"id": doc_id})
    _, _, md = rows[0]
    assert len(md["access_history"]) == initial_len + n
    # retrieval_count is the atomic counter — must also be exactly n.
    assert int(md["retrieval_count"]) == n


# --------------------------------------------------------------- §3b drift cap


async def test_drift_total_never_exceeds_configured_cap_under_race() -> None:
    """Concurrent reconsolidations must respect the configured cap —
    the projected/actual mismatch the lock closes."""
    cfg = MemoryConfig(
        reconsolidation_alpha=0.5,
        reconsolidation_min_score=0.0,
        reconsolidation_max_total_drift=1.0,
        reconsolidation_window_seconds=1e9,
    )
    db = AsyncVectorDB(":memory:")
    service = MemoryService(db, config=cfg, embed_fn=hash_embed)

    r = await service.encode_episode("anchor", "s1", salience=0.5)
    doc_id = int(r["id"])
    rows = await service.episodic.get_documents({"id": doc_id})
    _, _, md = rows[0]

    row = {"id": doc_id, "distance": 0.0, "metadata": md}
    # Many concurrent accruals on the same doc.
    await asyncio.gather(*[_accrue_drift(service, row, t=100.0 + i) for i in range(8)])

    rows = await service.episodic.get_documents({"id": doc_id})
    _, _, md_final = rows[0]
    assert float(md_final.get("drift_total", 0.0)) <= cfg.reconsolidation_max_total_drift
    # And once locked, drift_locked must be set.
    assert md_final.get("drift_locked") is True
    await db.close()


# ----------------------------------------------------------- §2d forget freeze


async def test_forget_freeze_serialises_evict_with_observe(
    service: MemoryService,
) -> None:
    """While ``state_freeze`` is held, any pending ``observe`` queues
    on ``_state_lock`` and runs *after* the eviction completes. The
    invariant: the *pre-eviction* tally for the evicted id is gone
    from ``_T_counts`` and ``_pending`` once the freeze exits, even
    while a concurrent observe is racing.
    """
    assert service._sr is not None
    sr = service._sr

    # Seed the pre-eviction edge (1, 2). After eviction of {1}, this
    # particular edge must be gone from _T_counts and _pending.
    await sr.observe("s", 1, t=0.0)
    await sr.observe("s", 2, t=1.0)
    pre_edges = dict(sr._T_counts)
    assert 1 in pre_edges
    assert 2 in pre_edges[1]

    started = asyncio.Event()
    release = asyncio.Event()

    async def freezing_evict() -> None:
        async with sr.state_freeze():
            started.set()
            await release.wait()
            sr.evict_nodes_locked({1})

    async def attempted_observe() -> None:
        await started.wait()
        # Observe inside the freeze window: blocks on _state_lock.
        await sr.observe("s", 1, t=2.0)

    fz_task = asyncio.create_task(freezing_evict())
    await started.wait()
    obs_task = asyncio.create_task(attempted_observe())
    # Let the observe coroutine reach its await on _state_lock.
    await asyncio.sleep(0)
    release.set()
    await asyncio.gather(fz_task, obs_task)

    # Pre-eviction edge for id 1 is gone — the observe ran *after*
    # evict_locked scrubbed the in-memory state.
    assert 1 not in sr._T_counts or 2 not in sr._T_counts[1]
    # _pending must not contain a (1, _) or (_, 1) pair from the
    # pre-eviction tally (the only edges that could survive are
    # post-release observes, and observe(1, t=2.0) finds only id 2
    # in the window, so any new edge is (2, 1) — a *fresh* edge,
    # which is fine. The contract is that the *evicted* state
    # cannot leak past the freeze.)
    # The original (1, 2) pending entry must be absent.
    assert (1, 2) not in sr._pending


# ----------------------------------------------- §2c plasticity reinforce/ltd


async def test_plasticity_reinforce_and_ltd_pass_serialise_per_edge() -> None:
    """ltd_pass and reinforce on the same edge must not resurrect a
    just-deleted edge or lose a just-applied increment."""
    cfg = MemoryConfig(
        ltd_decay_per_idle_day=0.5,
        time_compression=1.0,
    )
    db = AsyncVectorDB(":memory:")
    service = MemoryService(db, config=cfg, embed_fn=hash_embed)
    plast = service._plasticity
    # Pre-seed an edge that will be on the LTD prune boundary: tiny
    # weight, no habit. ltd_pass should delete it; if reinforce runs
    # concurrently inside the same per-edge lock, the final state is
    # whichever ran last — but never a resurrected stale row.
    plast.prune_floor = 1e-3

    a = await service.encode_episode("a", "s", salience=0.5)
    b = await service.encode_episode("b", "s", salience=0.5)
    ai, bi = int(a["id"]), int(b["id"])

    # Initial reinforce so the edge exists with a tiny weight.
    await plast.reinforce(ai, bi, score=0.001, now=0.0)

    # Now race ltd_pass (idle long enough to prune) against a fresh
    # reinforce. Each task acquires the per-edge lock; one of them
    # wins last.
    async def reinforce_more() -> None:
        await plast.reinforce(ai, bi, score=1.0, now=1.0e9)

    async def decay() -> None:
        # Massive idle time; with rate=0.5 the prune floor will be hit.
        await plast.ltd_pass(now=1.0e12)

    await asyncio.gather(reinforce_more(), decay())

    # Whatever the winner, the state must be internally consistent:
    # either the edge is gone (ltd_pass won last) or it has a positive
    # weight + bonus (reinforce won last). It must NOT exist with
    # zero/negative weight AND zero bonus (the resurrected-stale case).
    state = await plast.edge_state(ai, bi)
    if state is not None:
        assert state["weight"] + state["bonus"] > 0.0, (
            f"edge exists but has no potentiation — resurrected stale row: {state}"
        )
    await db.close()


# ------------------------------------------------------------ §2a observe load


# ------------------------------------------------------------- §2b flush vs evict


class _FakeEdgesAPI:
    """Stand-in for ``coll._collection.edges``; records upsert calls."""

    def __init__(self) -> None:
        self.calls: list[tuple[int, int]] = []

    def upsert(self, src: int, dst: int, **_: object) -> None:
        self.calls.append((src, dst))


class _FakeColl:
    def __init__(self, edges: _FakeEdgesAPI) -> None:
        class _Inner:
            def __init__(self, e: _FakeEdgesAPI) -> None:
                self.edges = e

        self._collection = _Inner(edges)

    async def get_edges(self, kind: str) -> list[object]:
        return []


async def test_flush_holds_state_lock_across_upserts_blocking_concurrent_evict() -> None:
    """``flush`` holds ``_state_lock`` across the durable upserts, so a
    concurrent ``state_freeze`` evict+delete cannot interleave between
    the endpoint re-check and the upsert and persist an FK-violating row.

    We assert ordering: an evict task that contends for ``state_freeze``
    while flush is mid-upsert is blocked until flush releases the lock.
    """
    order: list[str] = []

    class _OrderingEdges(_FakeEdgesAPI):
        def __init__(self) -> None:
            super().__init__()
            self._evict_ready: asyncio.Event = asyncio.Event()

        def upsert(self, src: int, dst: int, **_: object) -> None:
            order.append(f"upsert:{src}->{dst}")
            super().upsert(src, dst)
            # Signal the evict task to try contending for the lock now.
            self._evict_ready.set()

    edges = _OrderingEdges()
    coll = _FakeColl(edges)
    sr = SuccessorRepresentation()
    sr.attach(coll)
    await sr.load()
    await sr.observe("s", 1, t=0.0)
    await sr.observe("s", 2, t=1.0)
    assert (1, 2) in sr._pending

    async def _evict_when_flush_running() -> None:
        # Wait until flush is inside its locked upsert section, then try
        # to acquire ``state_freeze`` (the same ``_state_lock``).
        await edges._evict_ready.wait()
        async with sr.state_freeze():
            order.append("evict")
            sr.evict_nodes_locked({2})

    evict_task = asyncio.create_task(_evict_when_flush_running())
    await sr.flush()
    await evict_task

    # The evict must run strictly after flush's upsert: flush held the
    # lock across the durable write, so the eviction could not interleave.
    assert order == ["upsert:1->2", "evict"], order
    # The edge was upserted (then the post-flush evict's delete cascades
    # it durably); no FK-violating interleave occurred.
    assert (1, 2) in edges.calls


async def test_observe_during_load_is_serialised_by_state_lock() -> None:
    """``load`` rebuild and concurrent ``observe`` must not tear ``_M``;
    after both complete, every row's values are finite floats."""

    class _Edge:
        def __init__(self, src: int, dst: int, hits: int) -> None:
            self.src_id = src
            self.dst_id = dst
            self.hits = hits

    class _LoadColl:
        """Fake collection whose ``get_edges`` yields a non-trivial T so
        ``load()`` runs the *real* rebuild + TD passes (not the
        no-collection short-circuit) concurrently with ``observe``."""

        def __init__(self) -> None:
            self._collection = None
            self._edges = [_Edge(i, (i + 1) % 5, hits=i + 1) for i in range(5)]

        async def get_edges(self, *, kind: str) -> list[object]:
            # Yield control so a concurrent observe can interleave around
            # the await, then return edges for the rebuild to fold in.
            await asyncio.sleep(0)
            return list(self._edges)

    sr = SuccessorRepresentation()
    sr.attach(_LoadColl())

    async def hammer_observe() -> None:
        for i in range(50):
            await sr.observe("s", i % 5, t=float(i))

    # load() now exercises the rebuild critical section (it acquires
    # _state_lock around the TD passes) against concurrent observes.
    await asyncio.gather(sr.load(), hammer_observe(), hammer_observe())
    assert sr._loaded

    # _M must contain only finite floats — no NaN/Inf from a torn TD
    # update.
    for row in sr._M.values():
        for val in row.values():
            assert isinstance(val, float)
            assert val == val  # noqa: PLR0124 -- explicit NaN check
            assert val not in (float("inf"), float("-inf"))
