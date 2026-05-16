"""Recall + encode internal-branch coverage.

Targets the validation guards and reconsolidation/drift helper
branches not reached by the service-level happy-path tests.
"""

from __future__ import annotations

import hashlib
from collections.abc import AsyncIterator

import numpy as np
import pytest
import pytest_asyncio
from simplevecdb import AsyncVectorDB

from synara.core.errors import ValidationError
from synara.features.memory.amygdala.signals import SignalRegistry, SignalSpec
from synara.features.memory.config import MemoryConfig
from synara.features.memory.hippocampus import recall as recall_mod
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


# ---- recall.run guards ------------------------------------------------


async def test_recall_empty_query_raises(service: MemoryService) -> None:
    with pytest.raises(ValidationError, match="query must be non-empty"):
        await service.recall(query="   ")


async def test_recall_non_positive_k_returns_empty(service: MemoryService) -> None:
    await service.encode_episode("something", "s1")
    assert await service.recall(query="something", k=0) == []
    assert await service.recall(query="something", k=-3) == []


def test_cosine_score_from_distance_none_is_midpoint() -> None:
    assert recall_mod._cosine_score_from_distance(None) == 0.5
    assert recall_mod._cosine_score_from_distance(0.0) == 1.0
    assert recall_mod._cosine_score_from_distance(2.0) == 0.0


# ---- _sr_rank_keys early returns -------------------------------------


async def test_sr_rank_keys_empty_merged(service: MemoryService) -> None:
    assert await recall_mod._sr_rank_keys(service, [], caller_sid=None) == {}


async def test_sr_rank_keys_no_episodic_hits(service: MemoryService) -> None:
    merged = [(1, "t", {"id": 1}, 0.1, "semantic")]
    assert await recall_mod._sr_rank_keys(service, merged, caller_sid="x") == {}


# ---- _apply_drift_to_vector guards -----------------------------------


async def test_apply_drift_blend_non_positive_is_noop(service: MemoryService) -> None:
    await service.encode_episode("anchor text here", "s1")
    rows = await service.episodic.get_documents({"session_id": "s1"})
    doc_id = rows[0][0]
    # blend <= 0 -> early return before any vector fetch.
    await recall_mod._apply_drift_to_vector(service, doc_id, cue=[0.1], blend=0.0)


async def test_apply_drift_missing_vector_is_noop(service: MemoryService) -> None:
    await recall_mod._apply_drift_to_vector(service, 999999, cue=[0.1], blend=0.5)


async def test_apply_drift_shape_mismatch_is_noop(service: MemoryService) -> None:
    await service.encode_episode("vector shape mismatch case", "s1")
    rows = await service.episodic.get_documents({"session_id": "s1"})
    doc_id = rows[0][0]
    await recall_mod._apply_drift_to_vector(service, doc_id, cue=[0.1, 0.2, 0.3], blend=0.5)


async def test_apply_drift_zero_norm_blend_is_noop(service: MemoryService) -> None:
    await service.encode_episode("zero norm blend case", "s1")
    rows = await service.episodic.get_documents({"session_id": "s1"})
    doc_id = rows[0][0]
    embeds = await service.episodic.get_embeddings_by_ids([doc_id])
    v_old = np.asarray(embeds[doc_id], dtype=np.float64)
    # blend=0.5 and cue=-v_old -> blended is exactly the zero vector.
    cue = (-v_old).tolist()
    await recall_mod._apply_drift_to_vector(service, doc_id, cue=cue, blend=0.5)


# ---- _accrue_drift gates ---------------------------------------------


async def test_accrue_drift_min_score_gate(service: MemoryService) -> None:
    # distance 2.0 -> cosine score 0.0 < reconsolidation_min_score (0.4),
    # so the accrual returns before touching metadata.
    row = {"id": 1, "metadata": {}, "distance": 2.0}
    await recall_mod._accrue_drift(service, row, t=100.0)


async def test_accrue_drift_outside_window_resets_clock() -> None:
    db = AsyncVectorDB(":memory:")
    try:
        cfg = MemoryConfig(reconsolidation_window_seconds=10.0)
        svc = MemoryService(db, config=cfg, embed_fn=hash_embed)
        await svc.encode_episode("reconsolidate me later please", "s1")
        rows = await svc.episodic.get_documents({"session_id": "s1"})
        doc_id = rows[0][0]
        await svc.episodic.update_metadata([(int(doc_id), {"last_reconsolidated_at": 1.0})])
        row = {
            "id": doc_id,
            "metadata": {"last_reconsolidated_at": 1.0},
            "distance": 0.0,  # score 1.0 >= min
        }
        # t far beyond window -> clock-reset branch, no drift accrued.
        await recall_mod._accrue_drift(svc, row, t=10_000.0)
        after = await svc.episodic.get_documents({"session_id": "s1"})
        assert after[0][2].get("drift_total", 0.0) == 0.0
        assert after[0][2]["last_reconsolidated_at"] == 10_000.0
    finally:
        await db.close()


# ---- encode: dense-cosine dedup + signal-registry config -------------


async def test_encode_dense_cosine_dedup_path() -> None:
    """dg_pattern_separation off -> the cosine-threshold dedup branch."""
    db = AsyncVectorDB(":memory:")
    try:
        cfg = MemoryConfig(dg_pattern_separation=False)
        svc = MemoryService(db, config=cfg, embed_fn=hash_embed)
        r1 = await svc.encode_episode("the quick brown fox jumps", "s1")
        r2 = await svc.encode_episode("the quick brown fox jumps", "s1")
        assert r1["deduped"] is False
        assert r2["deduped"] is True
        assert r2["id"] == r1["id"]
    finally:
        await db.close()


async def test_encode_uses_custom_signal_registry() -> None:
    reg = SignalRegistry(
        specs=(SignalSpec(name="shouty", weight=0.4, compute=lambda c: c.isupper()),),
        include_legacy_structural=True,
    )
    db = AsyncVectorDB(":memory:")
    try:
        cfg = MemoryConfig(
            signal_registry=reg,
            auto_signal_metadata=True,
            auto_salience=True,
        )
        svc = MemoryService(db, config=cfg, embed_fn=hash_embed)
        await svc.encode_episode("THIS IS ALL CAPS SHOUTING", "s1")
        rows = await svc.episodic.get_documents({"session_id": "s1"})
        md = rows[0][2]
        assert md["shouty"] is True
        # registry.salience path consumed the custom weight.
        assert md["salience"] >= reg.base_salience + 0.4
    finally:
        await db.close()
