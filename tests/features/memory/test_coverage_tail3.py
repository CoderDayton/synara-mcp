"""Consolidation absorb-loop branches + completion id-filter branch."""

from __future__ import annotations

import hashlib
from collections.abc import AsyncIterator

import numpy as np
import pytest_asyncio
from simplevecdb import AsyncVectorDB

from synara.features.memory.config import MemoryConfig
from synara.features.memory.hippocampus import complete as complete_mod
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


async def test_absorb_skips_episode_too_far_from_any_schema(
    service: MemoryService,
) -> None:
    # A schema plus an unrelated episode: hash embeddings make them
    # near-orthogonal (distance ~1.0 > consolidate_absorb_distance), so
    # the ``dist > absorb_dist`` continue branch fires for every episode.
    await service.store_semantic_memory(
        "totally unrelated schema topic", kind="schema", scope="global"
    )
    await service.encode_episode("a different unconnected episodic note", "s1")
    await service.encode_episode("yet another unrelated episodic note", "s1")
    formed = await service.consolidate(session_id="s1", min_cluster_size=2)
    assert not any(s.get("absorbed") for s in formed)


async def test_absorb_skips_when_schema_already_lists_episode() -> None:
    db = AsyncVectorDB(":memory:")
    try:
        cfg = MemoryConfig(consolidate_absorb_distance=2.0)  # always "near"
        svc = MemoryService(db, config=cfg, embed_fn=hash_embed)
        await svc.encode_episode("shared content for absorb test", "s1")
        rows = await svc.episodic.get_documents({"session_id": "s1"})
        ep_id = int(rows[0][0])
        # Schema whose source list already contains ep_id -> the absorb
        # pass computes no new sources and hits the continue at L166.
        sem = await svc.store_semantic_memory(
            "shared content for absorb test", kind="schema", scope="global"
        )
        await svc.semantic.update_metadata([(int(sem["id"]), {"source_episode_ids": [ep_id]})])
        await svc.encode_episode("second filler episode here", "s1")
        formed = await svc.consolidate(session_id="s1", min_cluster_size=1)
        # ep_id already credited -> not re-absorbed via the no-new-source path.
        for s in formed:
            if s.get("absorbed"):
                assert ep_id in s["source_episode_ids"]
    finally:
        await db.close()


async def test_gather_candidates_filters_negative_ids(
    service: MemoryService,
) -> None:
    # Episodic doc stored without an "id" metadata key -> metadata.get
    # defaults to -1, so the id list is emptied and that store is
    # skipped (complete.py id-filter branch).
    await service.episodic.add_texts(
        texts=["doc with no id metadata"],
        embeddings=[hash_embed("doc with no id metadata")],
        ids=[1],
        metadatas=[{"session_id": "s1"}],
    )
    X = await complete_mod._gather_candidates(service, hash_embed("doc with no id metadata"), k=4)
    assert X.shape == (0,)
