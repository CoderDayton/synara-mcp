"""Hippocampus service tests.

Tests use a deterministic hash-based embedder (no network) so they do not
depend on the SentenceTransformer download. Identical text always
embeds to the same vector, which is enough to exercise dedup,
recall plumbing, and consolidation.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import AsyncIterator

import numpy as np
import pytest
import pytest_asyncio
from simplevecdb import AsyncVectorDB

from synara.core.errors import ValidationError
from synara.features.hippocampus.complete import attractor_step, completion_score
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
    fresh = memory_strength(salience=1.0, access_times=[100.0], now=100.0)
    aged = memory_strength(salience=1.0, access_times=[100.0], now=120.0)
    assert fresh > aged > 0.0


def test_memory_strength_power_law_slower_than_exponential_for_old_traces() -> None:
    """Power-law retention dominates exponential at long lags (Jost's law)."""
    s, age = 1.0, 1000.0
    power = memory_strength(salience=s, access_times=[0.0], now=age, d=0.5)
    exp_eq = s * math.exp(-age / 10.0)
    assert power > exp_eq


def test_memory_strength_retrievals_increase_strength() -> None:
    one = memory_strength(salience=1.0, access_times=[100.0], now=100.0)
    many = memory_strength(salience=1.0, access_times=[100.0] * 5, now=100.0)
    assert many > one


def test_memory_strength_invalid_d() -> None:
    with pytest.raises(ValidationError):
        memory_strength(salience=1.0, access_times=[0.0], now=1.0, d=0.0)


def _bucket_embed(text: str, dim: int = 32) -> list[float]:
    """Prefix-bucketed embedder: shared content prefix ⇒ near-collinear
    vectors (cluster), differing prefixes ⇒ near-orthogonal. Lets the
    absorption test exercise schema-fit logic with a deterministic
    content-aware similarity model.
    """
    head = text.strip().split("-", 1)[0].split()[0].lower() if text.strip() else ""
    bucket_seed = int(hashlib.sha256(head.encode()).hexdigest()[:8], 16)
    base = np.random.default_rng(bucket_seed).standard_normal(dim).astype(np.float32)
    text_seed = int(hashlib.sha256(text.encode()).hexdigest()[:8], 16)
    # 0.5 noise scale puts within-bucket cosine distance ~0.1-0.2, above
    # dedup_distance (0.05) so distinct texts encode separately, but below
    # consolidate_absorb_distance (0.4) so the absorb path triggers.
    noise = 0.5 * np.random.default_rng(text_seed).standard_normal(dim).astype(np.float32)
    v = base + noise
    n = float(np.linalg.norm(v))
    out = (v / n) if n > 0 else v
    return [float(x) for x in out.tolist()]


async def test_consolidate_absorbs_into_existing_schema() -> None:
    """A second consolidation pass should absorb new fitting episodes into
    the existing schema instead of forming a parallel one."""
    db = AsyncVectorDB(":memory:")
    try:
        svc = HippocampusService(db, config=HippocampusConfig(), embed_fn=_bucket_embed)
        for i in range(3):
            await svc.encode_episode(f"alpha-{i}", "s1", salience=0.5, tags=["a"])
        formed_first = await svc.consolidate(session_id="s1", n_clusters=1, min_cluster_size=2)
        assert formed_first
        assert not formed_first[0]["absorbed"]
        sch_id = formed_first[0]["id"]

        # New "alpha-*" episodes share the prefix bucket and should be
        # absorbed by the existing schema rather than spawn a parallel one.
        for i in range(3, 6):
            await svc.encode_episode(f"alpha-{i}", "s1", salience=0.5, tags=["a"])
        formed_second = await svc.consolidate(session_id="s1", min_cluster_size=2)

        absorbed = [s for s in formed_second if s.get("absorbed")]
        assert absorbed, "expected absorption into the existing alpha schema"
        assert any(s["id"] == sch_id for s in absorbed)
    finally:
        await db.close()


async def test_encode_records_access_history(service: HippocampusService) -> None:
    r = await service.encode_episode("hello", "s1", salience=0.5)
    rows = await service.episodic.get_documents({"session_id": "s1"})
    _, _, md = next(row for row in rows if row[0] == r["id"])
    assert isinstance(md.get("access_history"), list)
    assert len(md["access_history"]) == 1


async def test_recall_appends_access_history(service: HippocampusService) -> None:
    r = await service.encode_episode("hello access", "s1", salience=0.5)
    await service.recall(query="hello access", session_id="s1", k=1, mode="episodic")
    rows = await service.episodic.get_documents({"session_id": "s1"})
    _, _, md = next(row for row in rows if row[0] == r["id"])
    assert len(md["access_history"]) >= 2


def test_completion_score_increases_along_attractor_step() -> None:
    """Modern Hopfield update is provably non-decreasing in the
    log-sum-exp completion score when the anchor is fully released."""
    rng = np.random.default_rng(0)
    cluster = rng.standard_normal((6, 16)).astype(np.float64)
    cluster /= np.linalg.norm(cluster, axis=1, keepdims=True)
    # Off-cluster query: random direction near the cluster mean
    centroid = cluster.mean(axis=0)
    centroid /= np.linalg.norm(centroid)
    q0 = centroid + 0.5 * rng.standard_normal(16)
    q0 /= np.linalg.norm(q0)

    s_before = completion_score(q0, cluster, beta=8.0)
    q1, s_step = attractor_step(q0, cluster, beta=8.0, q0=q0, eta0=1.0)
    s_after = completion_score(q1, cluster, beta=8.0)

    # Score reported by the step matches recomputation at q0; the next
    # iterate scores >= the previous (modulo numerical noise).
    assert abs(s_step - s_before) < 1e-9
    assert s_after >= s_before - 1e-9


def test_completion_score_bounded_by_max_similarity() -> None:
    """C(q) <= max_i <q, x_i> + log(N)/beta (log-sum-exp upper bound)."""
    rng = np.random.default_rng(1)
    X = rng.standard_normal((10, 8))
    X /= np.linalg.norm(X, axis=1, keepdims=True)
    q = rng.standard_normal(8)
    q /= np.linalg.norm(q)

    sims = X @ q
    score = completion_score(q, X, beta=8.0)
    # log-sum-exp upper bound: max + (log N)/beta
    assert score <= float(sims.max()) + math.log(len(X)) / 8.0 + 1e-9
    assert score >= float(sims.max()) - 1e-9


async def test_recall_with_completion_iters_returns_results() -> None:
    """End-to-end: recall with completion_iters > 0 routes through the
    Hopfield iteration and still returns valid hits."""
    db = AsyncVectorDB(":memory:")
    try:
        cfg = HippocampusConfig(
            recall_completion_iters=2,
            recall_completion_beta=8.0,
            recall_completion_anchor=0.6,
        )
        svc = HippocampusService(db, config=cfg, embed_fn=_bucket_embed)
        for i in range(4):
            await svc.encode_episode(f"alpha-{i}", "s1", salience=0.5, tags=["a"])
        for i in range(4):
            await svc.encode_episode(f"beta-{i}", "s1", salience=0.5, tags=["b"])
        hits = await svc.recall(query="alpha-99", session_id="s1", k=3, mode="episodic")
        assert hits, "iterative recall should return results"
        # Bucket-embedder puts all alpha-* in one bucket; the refined
        # query should pull alpha hits over beta hits.
        top_texts = [h["content"] for h in hits]
        assert any(t.startswith("alpha") for t in top_texts)
    finally:
        await db.close()


def test_build_gist_with_only_headline() -> None:
    assert build_gist("only headline", ["only headline"]) == "only headline"


def test_build_gist_with_siblings() -> None:
    out = build_gist("headline", ["headline", "sibling one", "sibling two"])
    assert out.startswith("headline")
    assert "sibling one" in out
    assert "sibling two" in out
