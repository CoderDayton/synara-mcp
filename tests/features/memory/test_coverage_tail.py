"""Residual-branch coverage: completion, recall ranking, replay, embed."""

from __future__ import annotations

import hashlib
from collections.abc import AsyncIterator, Awaitable
from typing import Any

import numpy as np
import pytest_asyncio
from simplevecdb import AsyncVectorDB

from synara.features.memory.config import MemoryConfig
from synara.features.memory.hippocampus import complete as complete_mod
from synara.features.memory.hippocampus import recall as recall_mod
from synara.features.memory.hippocampus import replay as replay_mod
from synara.features.memory.hippocampus.separate import DGProjector
from synara.features.memory.service import MemoryService


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


# ---- complete.run / _gather_candidates -------------------------------


async def test_completion_run_zero_iters_returns_q0(service: MemoryService) -> None:
    res = await complete_mod.run(service, [1.0, 0.0], k_inner=4, iters=0, beta=8.0, eta0=0.6)
    assert res.converged is True
    assert res.scores == []
    assert res.query == [1.0, 0.0]


async def test_completion_run_breaks_on_empty_candidates(
    service: MemoryService,
) -> None:
    res = await complete_mod.run(service, [1.0, 0.0], k_inner=4, iters=3, beta=8.0, eta0=0.6)
    assert res.converged is False
    assert res.scores == []


async def test_gather_candidates_empty_stores_returns_empty(
    service: MemoryService,
) -> None:
    X = await complete_mod._gather_candidates(service, [0.1] * 32, k=4)
    assert X.shape == (0,)


async def test_completion_run_converges_early_on_stable_store() -> None:
    db = AsyncVectorDB(":memory:")
    try:
        cfg = MemoryConfig(recall_completion_iters=6)
        svc = MemoryService(db, config=cfg, embed_fn=hash_embed)
        for i in range(4):
            await svc.encode_episode(f"stable cluster member {i}", "s1")
        q = await svc.query_arg("stable cluster member 0")
        assert isinstance(q, list)
        res = await complete_mod.run(svc, q, k_inner=8, iters=6, beta=8.0, eta0=0.6)
        # Either it converged (delta < eps) or ran the full trace; both
        # exercise the iteration body and the score-delta check.
        assert res.scores
    finally:
        await db.close()


# ---- recall._merge_hits semantic leg + _sr_rank_keys -----------------


async def test_recall_hybrid_merges_semantic_hits(service: MemoryService) -> None:
    await service.store_semantic_memory("semantic fact alpha", kind="fact", scope="global")
    await service.encode_episode("episodic note alpha", "s1")
    out = await service.recall(query="alpha", session_id="s1", k=8, mode="hybrid")
    assert any(r["source"] == "semantic" for r in out)
    assert any(r["source"] == "episodic" for r in out)


async def test_sr_rank_keys_returns_empty_when_all_signals_off() -> None:
    db = AsyncVectorDB(":memory:")
    try:
        cfg = MemoryConfig(
            sr_enabled=False,
            spreading_activation_hops=0,
            spreading_activation_weight=0.0,
            same_session_bonus=0.0,
        )
        svc = MemoryService(db, config=cfg, embed_fn=hash_embed)
        await svc.encode_episode("ranked episode one", "s1")
        merged = [(1, "t", {"id": 1, "session_id": "s1"}, 0.2, "episodic")]
        keys = await recall_mod._sr_rank_keys(svc, merged, caller_sid=None)
        assert keys == {}
    finally:
        await db.close()


async def test_sr_rank_keys_skips_semantic_rows(service: MemoryService) -> None:
    await service.encode_episode("episodic anchor row", "s1")
    rows = await service.episodic.get_documents({"session_id": "s1"})
    ep_id = int(rows[0][0])
    merged: list[tuple[int, str, dict[str, Any], float, str]] = [
        (ep_id, "ep", {"id": ep_id, "session_id": "s1"}, 0.1, "episodic"),
        (99, "sem", {"id": 99}, 0.15, "semantic"),
    ]
    keys = await recall_mod._sr_rank_keys(service, merged, caller_sid="s1")
    # Semantic row is never assigned a rank key (src != "episodic").
    assert (99, "semantic") not in keys


# ---- replay: episode without session_id ------------------------------


async def test_replay_skips_episode_without_session_id(
    service: MemoryService,
) -> None:
    # Insert a raw episodic doc whose metadata lacks session_id so the
    # "no originating context" continue branch fires.
    await service.episodic.add_texts(
        texts=["orphan trace with no session"],
        embeddings=[hash_embed("orphan trace with no session")],
        ids=[1],
        metadatas=[{"id": 1, "salience": 0.9, "consolidated_into": 0}],
    )
    assert await replay_mod.run(service, now=10.0) == 0


# ---- separate: nonzero <= k short path -------------------------------


def test_dg_support_hits_small_active_set_path() -> None:
    # Tiny layer: scan seeds until exactly one unit is active for the
    # probe, exercising the ``nonzero <= self.k`` return branch.
    probe = [1.0, 0.0]
    for seed in range(200):
        p = DGProjector(dim=2, expansion=1, sparsity=0.5, seed=seed)
        h = p.W @ np.asarray(probe, dtype=np.float32)
        nonzero = int((h > 0).sum())
        if 0 < nonzero <= p.k:
            assert len(p.support(probe)) == nonzero
            break
    else:  # pragma: no cover - sanity guard
        raise AssertionError("no seed produced the small-active-set case")


# ---- service: async-returning sync fn + dimension probe --------------


async def test_embed_fn_returning_awaitable_is_honoured() -> None:
    async def _coro(text: str) -> list[float]:
        return hash_embed(text)

    def sync_returns_coro(text: str) -> Awaitable[list[float]]:
        # Not a coroutine function itself, but returns an awaitable.
        return _coro(text)

    db = AsyncVectorDB(":memory:")
    try:
        svc = MemoryService(
            db,
            config=MemoryConfig(),
            embed_fn=sync_returns_coro,
        )
        r = await svc.encode_episode("awaitable embed result", "s1")
        assert r["deduped"] is False
    finally:
        await db.close()


async def test_embedding_dimension_probes_on_first_call() -> None:
    db = AsyncVectorDB(":memory:")
    try:
        svc = MemoryService(db, config=MemoryConfig(), embed_fn=hash_embed)
        # No prior vectorise/query_arg -> the probe path runs here.
        assert await svc.embedding_dimension() == 32
        assert await svc.embedding_dimension() == 32  # cached second call
    finally:
        await db.close()
