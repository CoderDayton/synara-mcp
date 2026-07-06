"""MemoryService coordinator-level branch coverage."""

from __future__ import annotations

import hashlib
from collections.abc import AsyncIterator

import numpy as np
import pytest
import pytest_asyncio
from simplevecdb import AsyncVectorDB

from synara.core.errors import ValidationError
from synara.features.memory.config import MemoryConfig
from synara.features.memory.memory_types import MemoryType
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


async def test_no_embedder_service_defers_to_simplevecdb() -> None:
    db = AsyncVectorDB(":memory:")
    try:
        svc = MemoryService(db, config=MemoryConfig(), embed_fn=None)
        assert await svc.vectorise(["x"]) is None
        assert await svc.query_arg("x") == "x"
        assert await svc.embedding_dimension() is None
    finally:
        await db.close()


async def test_async_embed_fn_is_supported() -> None:
    async def aembed(text: str) -> list[float]:
        return hash_embed(text)

    db = AsyncVectorDB(":memory:")
    try:
        svc = MemoryService(db, config=MemoryConfig(), embed_fn=aembed)
        r = await svc.encode_episode("async embed path", "s1")
        assert r["deduped"] is False
        assert await svc.embedding_dimension() == 32
    finally:
        await db.close()


async def test_vectorise_dimension_drift_raises() -> None:
    calls = {"n": 0}

    def drifting(text: str) -> list[float]:
        calls["n"] += 1
        return [0.0] * (4 if calls["n"] == 1 else 8)

    db = AsyncVectorDB(":memory:")
    try:
        svc = MemoryService(db, config=MemoryConfig(), embed_fn=drifting)
        await svc.vectorise(["first"])  # caches dim 4
        with pytest.raises(ValidationError, match="embedding dimension drift"):
            await svc.vectorise(["second"])
    finally:
        await db.close()


async def test_query_arg_caches_dimension(service: MemoryService) -> None:
    assert service._embedding_dimension is None
    vec = await service.query_arg("seed the dim")
    assert isinstance(vec, list)
    assert service._embedding_dimension == 32


async def test_collection_for_unknown_kind_raises(service: MemoryService) -> None:
    assert service.collection_for(MemoryType.EPISODIC) is service.episodic
    # Drop a registered collection to force the KeyError -> ValidationError.
    service._collections.pop(MemoryType.SEMANTIC)
    with pytest.raises(ValidationError, match="not registered on this service"):
        service.collection_for(MemoryType.SEMANTIC)


async def test_bump_retrieval_trims_access_history() -> None:
    db = AsyncVectorDB(":memory:")
    try:
        cfg = MemoryConfig(access_history_cap=3)
        svc = MemoryService(db, config=cfg, embed_fn=hash_embed)
        await svc.encode_episode("trim my history", "s1")
        for _ in range(6):
            await svc.recall(query="trim my history", session_id="s1", k=1)
        rows = await svc.episodic.get_documents({"session_id": "s1"})
        assert len(rows[0][2]["access_history"]) == 3
    finally:
        await db.close()


async def test_fetch_episode_group_with_session_filter() -> None:
    db = AsyncVectorDB(":memory:")
    try:
        cfg = MemoryConfig(theta_segment_max_chars=20, theta_segment_max_items=4)
        svc = MemoryService(db, config=cfg, embed_fn=hash_embed)
        long_text = (
            "First sentence here. Second sentence follows. Third one as well. Fourth wraps it up."
        )
        r = await svc.encode_episode(long_text, "s1")
        gid = r["id"]
        items = await svc.fetch_episode_group(gid, session_id="s1")
        assert len(items) >= 2
        assert items == sorted(items, key=lambda d: d["position"])
        # session filter that excludes everything -> empty walk
        assert await svc.fetch_episode_group(gid, session_id="other") == []
    finally:
        await db.close()


async def test_recall_populates_last_trace_when_tracing_enabled() -> None:
    db = AsyncVectorDB(":memory:")
    try:
        cfg = MemoryConfig(tracing_enabled=True)
        svc = MemoryService(db, config=cfg, embed_fn=hash_embed)
        await svc.encode_episode("trace this recall", "s1")
        await svc.recall(query="trace this recall", session_id="s1", k=2)
        assert svc.last_trace is not None
        assert "spans" in svc.last_trace
        assert any(s["name"] == "merge_hits" for s in svc.last_trace["spans"])
    finally:
        await db.close()


async def test_recall_semantic_memory_breaks_at_k() -> None:
    db = AsyncVectorDB(":memory:")
    try:
        # Disable the relevance gate so this isolates the k-cap: an exact
        # hash-embed match leaves a lone peak the elbow would otherwise trim.
        cfg = MemoryConfig(recall_relevance_gate=False)
        svc = MemoryService(db, config=cfg, embed_fn=hash_embed)
        for i in range(6):
            await svc.store_semantic_memory(f"fact number {i}", kind="fact", scope="global")
        out = await svc.recall_semantic_memory(query="fact number 1", k=2)
        assert len(out) == 2
    finally:
        await db.close()


async def test_event_log_snapshot_round_trips(service: MemoryService) -> None:
    await service.encode_episode("an event happens", "s1")
    log = await service.event_log()
    assert any(e.kind == "encode" for e in log)
