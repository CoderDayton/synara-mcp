"""Hippocampus service tests.

Tests use a deterministic hash-based embedder (no network) so they do not
depend on the SentenceTransformer download. Identical text always
embeds to the same vector, which is enough to exercise dedup,
recall plumbing, and consolidation.
"""

from __future__ import annotations

import hashlib
from collections.abc import AsyncIterator

import numpy as np
import pytest
import pytest_asyncio
from simplevecdb import AsyncVectorDB

from synara.core.errors import ValidationError
from synara.features.hippocampus.consolidate import build_gist
from synara.features.hippocampus.forget import memory_strength
from synara.features.hippocampus.service import (
    UNCONSOLIDATED,
    HippocampusConfig,
    HippocampusService,
)


def hash_embed(text: str, dim: int = 32) -> list[float]:
    """Deterministic, normalised embedding seeded by text content."""
    seed_bytes = hashlib.sha256(text.encode("utf-8")).digest()[:8]
    seed = int.from_bytes(seed_bytes, "big", signed=False)
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(dim).astype(np.float32)
    n = float(np.linalg.norm(v))
    out = (v / n) if n > 0 else v
    return [float(x) for x in out.tolist()]


@pytest_asyncio.fixture
async def service() -> AsyncIterator[HippocampusService]:
    db = AsyncVectorDB(":memory:")
    try:
        yield HippocampusService(db, config=HippocampusConfig(), embed_fn=hash_embed)
    finally:
        await db.close()


async def test_encode_assigns_id_and_metadata(service: HippocampusService) -> None:
    r = await service.encode_episode("hello world", "s1", tags=["greet"], salience=0.7)
    assert r["deduped"] is False
    assert r["id"] >= 0
    rows = await service.episodic.get_documents({"session_id": "s1"})
    assert len(rows) == 1
    _, _text, md = rows[0]
    assert md["session_id"] == "s1"
    assert md["tags"] == ["greet"]
    assert md["salience"] == pytest.approx(0.7)
    assert md["consolidated_into"] == UNCONSOLIDATED
    assert md["id"] == r["id"]


async def test_encode_dedup_within_session(service: HippocampusService) -> None:
    r1 = await service.encode_episode("the cat sat on the mat", "s1", salience=0.8)
    r2 = await service.encode_episode("the cat sat on the mat", "s1", salience=0.5)
    assert r1["deduped"] is False
    assert r2["deduped"] is True
    assert r2["id"] == r1["id"]
    # The dedup hit should also bump retrieval_count on the original.
    rows = await service.episodic.get_documents({"session_id": "s1"})
    assert len(rows) == 1
    assert rows[0][2]["retrieval_count"] == 1


async def test_encode_does_not_dedup_across_sessions(
    service: HippocampusService,
) -> None:
    r1 = await service.encode_episode("identical content", "s1")
    r2 = await service.encode_episode("identical content", "s2")
    assert r2["deduped"] is False
    assert r1["id"] != r2["id"]


async def test_encode_rejects_bad_input(service: HippocampusService) -> None:
    with pytest.raises(ValidationError):
        await service.encode_episode("", "s1")
    with pytest.raises(ValidationError):
        await service.encode_episode("ok", "")
    with pytest.raises(ValidationError):
        await service.encode_episode("ok", "s1", salience=1.5)


async def test_recall_returns_results_and_bumps_retrieval(
    service: HippocampusService,
) -> None:
    await service.encode_episode("alpha note", "s1", salience=0.9)
    await service.encode_episode("beta note", "s1", salience=0.9)
    hits = await service.recall("alpha note", session_id="s1", k=5)
    assert hits, "expected at least one recall result"
    assert all(h["source"] in {"episodic", "semantic"} for h in hits)
    # Episodic hits must bump retrieval_count.
    rows = await service.episodic.get_documents({"session_id": "s1"})
    total_retrievals = sum(int(md.get("retrieval_count", 0)) for _, _, md in rows)
    assert total_retrievals >= 1


async def test_recall_unknown_mode_raises(service: HippocampusService) -> None:
    with pytest.raises(ValidationError):
        await service.recall("x", mode="bogus")


async def test_consolidate_forms_schemas_and_links_episodes(
    service: HippocampusService,
) -> None:
    for i in range(4):
        await service.encode_episode(f"item-A-{i}", "s1", salience=0.5, tags=["a"])
    for i in range(4):
        await service.encode_episode(f"item-B-{i}", "s1", salience=0.5, tags=["b"])
    formed = await service.consolidate(session_id="s1", n_clusters=2, min_cluster_size=2)
    assert formed, "expected at least one schema"
    stats = await service.stats()
    assert stats["semantic_count"] >= 1
    # All clustered episodes should now point at a positive schema id.
    rows = await service.episodic.get_documents({"session_id": "s1"})
    consolidated = [md["consolidated_into"] for _, _, md in rows if md["consolidated_into"]]
    assert len(consolidated) >= 4


async def test_consolidate_returns_empty_when_too_few_candidates(
    service: HippocampusService,
) -> None:
    await service.encode_episode("only one", "s1")
    assert await service.consolidate(min_cluster_size=2) == []


async def test_forget_dry_run_flags_old_low_salience(
    service: HippocampusService,
) -> None:
    r = await service.encode_episode("ephemeral note", "s1", salience=0.0)
    out = await service.forget(strength_floor=0.5, decay_tau_seconds=1.0, dry_run=True)
    assert r["id"] in out["candidate_ids"]
    assert out["removed"] == 0
    assert out["dry_run"] is True
    # Episode is still present in dry_run mode.
    stats = await service.stats()
    assert stats["episodic_count"] == 1


async def test_forget_real_run_deletes(service: HippocampusService) -> None:
    r = await service.encode_episode("ephemeral note", "s1", salience=0.0)
    out = await service.forget(strength_floor=0.5, decay_tau_seconds=1.0, dry_run=False)
    assert out["removed"] >= 1
    assert r["id"] in out["candidate_ids"]
    stats = await service.stats()
    assert stats["episodic_count"] == 0


async def test_reflect_returns_recent_episodes(service: HippocampusService) -> None:
    a = await service.encode_episode("first", "s1", salience=0.9, tags=["x"])
    await service.encode_episode("second", "s1", salience=0.9, tags=["x"])
    out = await service.reflect(session_id="s1", k=2)
    assert out["session_id"] == "s1"
    ids = {ep["id"] for ep in out["recent_episodes"]}
    assert a["id"] in ids


async def test_stats_starts_empty(service: HippocampusService) -> None:
    assert await service.stats() == {"episodic_count": 0, "semantic_count": 0}


def test_memory_strength_decays_with_age() -> None:
    fresh = memory_strength(
        salience=1.0,
        age_seconds=0.0,
        retrievals=0,
        tau_seconds=10.0,
        retrieval_boost=0.05,
    )
    aged = memory_strength(
        salience=1.0,
        age_seconds=20.0,
        retrievals=0,
        tau_seconds=10.0,
        retrieval_boost=0.05,
    )
    assert fresh > aged > 0.0


def test_memory_strength_invalid_tau() -> None:
    with pytest.raises(ValidationError):
        memory_strength(
            salience=1.0,
            age_seconds=1.0,
            retrievals=0,
            tau_seconds=0.0,
            retrieval_boost=0.0,
        )


def test_build_gist_with_only_headline() -> None:
    assert build_gist("only headline", ["only headline"]) == "only headline"


def test_build_gist_with_siblings() -> None:
    out = build_gist("headline", ["headline", "sibling one", "sibling two"])
    assert out.startswith("headline")
    assert "sibling one" in out
    assert "sibling two" in out
