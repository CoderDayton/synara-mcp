"""Memory service tests.

Tests use a deterministic hash-based embedder (no network) so they do not
depend on the SentenceTransformer download. Identical text always
embeds to the same vector, which is enough to exercise dedup,
recall plumbing, and consolidation.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import AsyncIterator
from pathlib import Path

import numpy as np
import pytest
import pytest_asyncio
from simplevecdb import AsyncVectorDB

from synara.core.errors import ValidationError
from synara.features.memory.hippocampus.complete import attractor_step, completion_score
from synara.features.memory.hippocampus.segment import split_into_segments
from synara.features.memory.hippocampus.separate import DGProjector, jaccard
from synara.features.memory.hippocampus.successor import SuccessorRepresentation
from synara.features.memory.neocortex import consolidate as consolidate_mod
from synara.features.memory.neocortex.consolidate import build_gist
from synara.features.memory.neocortex.forget import memory_strength
from synara.features.memory.service import (
    UNCONSOLIDATED,
    MemoryConfig,
    MemoryService,
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
async def service() -> AsyncIterator[MemoryService]:
    db = AsyncVectorDB(":memory:")
    try:
        yield MemoryService(db, config=MemoryConfig(), embed_fn=hash_embed)
    finally:
        await db.close()


async def test_encode_assigns_id_and_metadata(service: MemoryService) -> None:
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


async def test_encode_dedup_within_session(service: MemoryService) -> None:
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
    service: MemoryService,
) -> None:
    r1 = await service.encode_episode("identical content", "s1")
    r2 = await service.encode_episode("identical content", "s2")
    assert r2["deduped"] is False
    assert r1["id"] != r2["id"]


async def test_encode_rejects_bad_input(service: MemoryService) -> None:
    with pytest.raises(ValidationError):
        await service.encode_episode("", "s1")
    with pytest.raises(ValidationError):
        await service.encode_episode("ok", "")
    with pytest.raises(ValidationError):
        await service.encode_episode("ok", "s1", salience=1.5)


async def test_encode_merges_signal_metadata(service: MemoryService) -> None:
    content = (
        "Traceback (most recent call last):\n"
        "ValueError: bad value\n"
        "see src/foo.py and `bar` for context"
    )
    await service.encode_episode(content, "s1", salience=0.5)
    rows = await service.episodic.get_documents({"session_id": "s1"})
    md = rows[0][2]
    assert md["has_traceback"] is True
    assert md["has_diff_markers"] is False
    assert "src/foo.py" in md["references"]
    assert "bar" in md["references"]
    assert md["length_class"] in {"short", "medium"}


async def test_auto_salience_off_keeps_default_when_omitted() -> None:
    db = AsyncVectorDB(":memory:")
    try:
        svc = MemoryService(
            db,
            config=MemoryConfig(auto_salience=False),
            embed_fn=hash_embed,
        )
        await svc.encode_episode("plain note", "s1")  # no salience kwarg
        rows = await svc.episodic.get_documents({"session_id": "s1"})
        assert rows[0][2]["salience"] == pytest.approx(0.5)
    finally:
        await db.close()


async def test_auto_salience_on_uses_derived_when_omitted() -> None:
    db = AsyncVectorDB(":memory:")
    try:
        svc = MemoryService(
            db,
            config=MemoryConfig(auto_salience=True, auto_salience_base=0.3),
            embed_fn=hash_embed,
        )
        # Traceback + diff content should derive well above the 0.3 base.
        content = (
            "Traceback (most recent call last):\nValueError: x\n"
            "--- a/x.py\n+++ b/x.py\n@@ -1 +1 @@\n-old\n+new\n"
        )
        await svc.encode_episode(content, "s1")
        rows = await svc.episodic.get_documents({"session_id": "s1"})
        assert rows[0][2]["salience"] > 0.5
    finally:
        await db.close()


async def test_signal_metadata_disabled_by_config() -> None:
    db = AsyncVectorDB(":memory:")
    try:
        svc = MemoryService(
            db,
            config=MemoryConfig(auto_signal_metadata=False),
            embed_fn=hash_embed,
        )
        await svc.encode_episode("RuntimeError: nope", "s1", salience=0.5)
        rows = await svc.episodic.get_documents({"session_id": "s1"})
        md = rows[0][2]
        assert "has_traceback" not in md
        assert "references" not in md
    finally:
        await db.close()


async def test_recall_returns_results_and_bumps_retrieval(
    service: MemoryService,
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


async def test_recall_unknown_mode_raises(service: MemoryService) -> None:
    with pytest.raises(ValidationError):
        await service.recall("x", mode="bogus")


async def test_consolidate_forms_schemas_and_links_episodes() -> None:
    db = AsyncVectorDB(":memory:")
    try:
        # Schema-eligibility gates default to ON; this test predates them
        # and assumes immediate consolidation, so disable both gates here.
        # The recurrence gate also defaults ON now -- set min_recurrence=1
        # so this test keeps measuring the stage-1/stage-2 mechanics.
        cfg = MemoryConfig(
            consolidate_min_age_seconds=0.0,
            consolidate_min_retrievals=0,
            consolidate_min_recurrence=1,
        )
        service = MemoryService(db, config=cfg, embed_fn=hash_embed)
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
    finally:
        await db.close()


async def test_consolidate_returns_empty_when_too_few_candidates(
    service: MemoryService,
) -> None:
    await service.encode_episode("only one", "s1")
    assert await service.consolidate(min_cluster_size=2) == []


async def test_forget_dry_run_flags_old_low_salience(
    service: MemoryService,
) -> None:
    r = await service.encode_episode("ephemeral note", "s1", salience=0.0)
    out = await service.forget(strength_floor=0.5, decay_tau_seconds=1.0, dry_run=True)
    assert r["id"] in out["candidate_ids"]
    assert out["removed"] == 0
    assert out["dry_run"] is True
    # Episode is still present in dry_run mode.
    stats = await service.stats()
    assert stats["episodic_count"] == 1


async def test_forget_real_run_deletes(service: MemoryService) -> None:
    r = await service.encode_episode("ephemeral note", "s1", salience=0.0)
    out = await service.forget(strength_floor=0.5, decay_tau_seconds=1.0, dry_run=False)
    assert out["removed"] >= 1
    assert r["id"] in out["candidate_ids"]
    stats = await service.stats()
    assert stats["episodic_count"] == 0


async def test_reflect_returns_recent_episodes(service: MemoryService) -> None:
    a = await service.encode_episode("first", "s1", salience=0.9, tags=["x"])
    await service.encode_episode("second", "s1", salience=0.9, tags=["x"])
    out = await service.reflect(session_id="s1", k=2)
    assert out["session_id"] == "s1"
    ids = {ep["id"] for ep in out["recent_episodes"]}
    assert a["id"] in ids


async def test_stats_starts_empty(service: MemoryService) -> None:
    s = await service.stats()
    assert s["episodic_count"] == 0
    assert s["semantic_count"] == 0
    assert s["schema_candidate_count"] == 0


async def test_store_semantic_memory_persists_with_metadata(
    service: MemoryService,
) -> None:
    out = await service.store_semantic_memory(
        "Prefer pytest fixtures over manual setup.",
        kind="preference",
        tags=["testing", "python"],
        confidence=0.9,
    )
    assert out["id"] >= 0
    assert out["kind"] == "preference"
    assert out["tags"] == ["python", "testing"]  # sorted, deduped
    assert out["confidence"] == pytest.approx(0.9)
    stats = await service.stats()
    assert stats["semantic_count"] == 1
    assert stats["episodic_count"] == 0  # bypasses episodic store
    rows = await service.semantic.get_documents({"id": out["id"]})
    assert len(rows) == 1
    md = rows[0][2]
    assert md["kind"] == "preference"
    assert md["authored"] is True
    assert md["source_episode_ids"] == []


async def test_store_semantic_memory_rejects_bad_input(
    service: MemoryService,
) -> None:
    with pytest.raises(ValidationError):
        await service.store_semantic_memory("", kind="fact")
    with pytest.raises(ValidationError):
        await service.store_semantic_memory("ok", kind="")
    with pytest.raises(ValidationError):
        await service.store_semantic_memory("ok", confidence=1.5)


async def test_recall_semantic_memory_returns_only_semantic_hits(
    service: MemoryService,
) -> None:
    # Episode in episodic store + same-text semantic memory in semantic store.
    # recall_semantic_memory must only see the semantic side.
    await service.encode_episode("alpha episode", "s1", salience=0.5)
    sem = await service.store_semantic_memory("alpha truth", kind="fact")
    hits = await service.recall_semantic_memory("alpha", k=5)
    assert hits, "expected at least one semantic hit"
    assert all("kind" in h["metadata"] for h in hits)
    assert sem["id"] in {h["id"] for h in hits}


async def test_recall_semantic_memory_filters_by_kind(
    service: MemoryService,
) -> None:
    await service.store_semantic_memory("alpha fact", kind="fact")
    pref = await service.store_semantic_memory("alpha preference", kind="preference")
    hits = await service.recall_semantic_memory("alpha", k=5, kind="preference")
    assert hits
    assert {h["id"] for h in hits} == {pref["id"]}


async def test_recall_semantic_memory_empty_store_returns_empty(
    service: MemoryService,
) -> None:
    assert await service.recall_semantic_memory("anything", k=5) == []


async def test_recall_semantic_memory_rejects_bad_input(
    service: MemoryService,
) -> None:
    with pytest.raises(ValidationError):
        await service.recall_semantic_memory("", k=5)
    assert await service.recall_semantic_memory("ok", k=0) == []


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
        # Disable age/retrieval gates so the absorb path triggers on
        # freshly-encoded, never-recalled episodes. Set min_recurrence=1
        # so the first consolidate actually forms the schema that the
        # second consolidate is supposed to absorb into.
        cfg = MemoryConfig(
            consolidate_min_age_seconds=0.0,
            consolidate_min_retrievals=0,
            consolidate_min_recurrence=1,
        )
        svc = MemoryService(db, config=cfg, embed_fn=_bucket_embed)
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


async def test_encode_records_access_history(service: MemoryService) -> None:
    r = await service.encode_episode("hello", "s1", salience=0.5)
    rows = await service.episodic.get_documents({"session_id": "s1"})
    _, _, md = next(row for row in rows if row[0] == r["id"])
    assert isinstance(md.get("access_history"), list)
    assert len(md["access_history"]) == 1


async def test_recall_appends_access_history(service: MemoryService) -> None:
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
        cfg = MemoryConfig(
            recall_completion_iters=2,
            recall_completion_beta=8.0,
            recall_completion_anchor=0.6,
        )
        svc = MemoryService(db, config=cfg, embed_fn=_bucket_embed)
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


def test_dg_projector_supports_have_correct_size() -> None:
    proj = DGProjector(dim=32, expansion=4, sparsity=0.05, seed=0)
    rng = np.random.default_rng(0)
    x = rng.standard_normal(32)
    x /= np.linalg.norm(x)
    s = proj.support(x.tolist())
    assert len(s) == proj.k
    assert all(0 <= i < proj.M for i in s)
    assert s == tuple(sorted(s))


def test_dg_jaccard_orthogonalises_similar_inputs() -> None:
    """High-cosine inputs map to a Jaccard strictly below their cosine —
    the pattern-separation property: small input differences produce
    larger representation-space differences."""
    proj = DGProjector(dim=64, expansion=8, sparsity=0.05, seed=0)
    rng = np.random.default_rng(0)
    x = rng.standard_normal(64)
    x /= np.linalg.norm(x)
    # Noise scale 0.06 gives cos ~0.9 — the high-similarity regime
    # where pattern separation matters.
    y = x + 0.06 * rng.standard_normal(64)
    y /= np.linalg.norm(y)
    cos = float(x @ y)
    j = jaccard(proj.support(x.tolist()), proj.support(y.tolist()))
    assert 0.85 < cos < 1.0
    # Pattern separation: representation overlap strictly below input overlap.
    assert j < cos


def test_dg_jaccard_separates_unrelated_inputs() -> None:
    """Independent random inputs have near-zero Jaccard overlap."""
    proj = DGProjector(dim=64, expansion=8, sparsity=0.05, seed=0)
    rng = np.random.default_rng(0)
    a = rng.standard_normal(64)
    a /= np.linalg.norm(a)
    b = rng.standard_normal(64)
    b /= np.linalg.norm(b)
    j = jaccard(proj.support(a.tolist()), proj.support(b.tolist()))
    assert j < 0.2


def _paraphrase_embed(text: str, dim: int = 64) -> list[float]:
    """Tight-bucket embedder for the DG paraphrase test: same prefix ⇒
    cosine ~0.96 (genuine paraphrase regime), different prefix ⇒
    near-orthogonal."""
    head = text.strip().split("-", 1)[0].split()[0].lower() if text.strip() else ""
    bucket_seed = int(hashlib.sha256(head.encode()).hexdigest()[:8], 16)
    base = np.random.default_rng(bucket_seed).standard_normal(dim).astype(np.float32)
    text_seed = int(hashlib.sha256(text.encode()).hexdigest()[:8], 16)
    noise = 0.2 * np.random.default_rng(text_seed).standard_normal(dim).astype(np.float32)
    v = base + noise
    n = float(np.linalg.norm(v))
    out = (v / n) if n > 0 else v
    return [float(x) for x in out.tolist()]


async def test_encode_dg_dedup_catches_paraphrase() -> None:
    """With DG pattern separation enabled, near-identical embeddings
    that fall just outside the cosine dedup_distance still get caught
    by the Jaccard threshold."""
    db = AsyncVectorDB(":memory:")
    try:
        # Tight cosine threshold so cosine alone wouldn't dedup; Jaccard
        # threshold tuned to catch the paraphrase regime (cos ~0.96).
        cfg = MemoryConfig(
            dedup_distance=0.001,
            dg_pattern_separation=True,
            dg_expansion=8,
            dg_jaccard_threshold=0.4,
            dg_dedup_candidates=4,
        )
        svc = MemoryService(db, config=cfg, embed_fn=_paraphrase_embed)
        r1 = await svc.encode_episode("alpha-original", "s1", salience=0.5)
        r2 = await svc.encode_episode("alpha-paraphrase", "s1", salience=0.5)
        assert r1["deduped"] is False
        assert r2["deduped"] is True
        assert r2["id"] == r1["id"]
        assert "jaccard" in r2
        assert r2["jaccard"] >= 0.4
    finally:
        await db.close()


def _crowding_embed(text: str, dim: int = 16) -> list[float]:
    """Models short-text embedding crowding: genuinely distinct tokens
    of <=4 chars collapse onto one shared embedding (the regime where
    neither cosine nor DG can separate them); longer text embeds by
    content hash."""
    key = "__short__" if len(text.strip()) <= 4 else text
    seed = int(hashlib.sha256(key.encode()).hexdigest()[:8], 16)
    v = np.random.default_rng(seed).standard_normal(dim).astype(np.float32)
    n = float(np.linalg.norm(v))
    out = (v / n) if n > 0 else v
    return [float(x) for x in out.tolist()]


async def test_encode_does_not_false_merge_distinct_short_episodes() -> None:
    """Two genuinely distinct short episodes that crowd onto the same
    embedding must NOT be merged: below the dedup content-length floor,
    embedding-based dedup is unreliable and a false merge is
    irreversible data loss, so each is stored."""
    db = AsyncVectorDB(":memory:")
    try:
        cfg = MemoryConfig(dg_pattern_separation=True, min_dedup_chars=8)
        svc = MemoryService(db, config=cfg, embed_fn=_crowding_embed)
        r1 = await svc.encode_episode("abc", "s1", salience=0.5)
        r2 = await svc.encode_episode("xyz", "s1", salience=0.5)
        assert r1["deduped"] is False
        assert r2["deduped"] is False
        assert r2["id"] != r1["id"]
        # Control: above the floor, identical embeddings still dedup, so
        # the floor narrows dedup rather than disabling it.
        r3 = await svc.encode_episode("longcontent one", "s2", salience=0.5)
        r4 = await svc.encode_episode("longcontent one", "s2", salience=0.5)
        assert r3["deduped"] is False
        assert r4["deduped"] is True
        assert r4["id"] == r3["id"]
    finally:
        await db.close()


async def test_encode_stores_dg_support_when_enabled() -> None:
    db = AsyncVectorDB(":memory:")
    try:
        cfg = MemoryConfig(dg_pattern_separation=True)
        svc = MemoryService(db, config=cfg, embed_fn=hash_embed)
        r = await svc.encode_episode("hello", "s1", salience=0.5)
        rows = await svc.episodic.get_documents({"session_id": "s1"})
        _, _, md = next(row for row in rows if row[0] == r["id"])
        assert isinstance(md.get("dg_support"), list)
        assert len(md["dg_support"]) > 0
    finally:
        await db.close()


def test_build_gist_with_only_headline() -> None:
    assert build_gist("only headline", ["only headline"]) == "only headline"


def test_build_gist_with_siblings() -> None:
    out = build_gist("headline", ["headline", "sibling one", "sibling two"])
    assert out.startswith("headline")
    assert "sibling one" in out
    assert "sibling two" in out


# ---------------------------------------------------------------- segment (#6)
def test_split_into_segments_passthrough_for_short_content() -> None:
    text = "Short note."
    assert split_into_segments(text, max_chars=1024, max_items=7) == [text]


def test_split_into_segments_disabled_returns_single_item() -> None:
    text = "x" * 5000
    assert split_into_segments(text, max_chars=0, max_items=7) == [text]
    assert split_into_segments(text, max_chars=1024, max_items=1) == [text]


def test_split_into_segments_breaks_on_sentences() -> None:
    text = "First sentence. Second sentence. Third sentence."
    out = split_into_segments(text, max_chars=20, max_items=7)
    assert len(out) >= 2
    # Concatenation preserves all original tokens (modulo separator
    # whitespace).
    flat = " ".join(out)
    for token in ("First", "Second", "Third"):
        assert token in flat


def test_split_into_segments_caps_count_at_max_items() -> None:
    sentences = [f"S{i}." for i in range(20)]
    text = " ".join(sentences)
    out = split_into_segments(text, max_chars=4, max_items=5)
    assert 1 <= len(out) <= 5
    flat = " ".join(out)
    for tok in ("S0", "S19"):
        assert tok in flat


def test_split_into_segments_windows_overlong_sentence() -> None:
    text = "x" * 100  # No sentence boundary inside.
    out = split_into_segments(text, max_chars=30, max_items=7)
    assert len(out) >= 2
    assert "".join(out) == text


async def test_encode_long_content_creates_segments() -> None:
    db = AsyncVectorDB(":memory:")
    try:
        cfg = MemoryConfig(theta_segment_max_chars=40, theta_segment_max_items=5)
        svc = MemoryService(db, config=cfg, embed_fn=hash_embed)
        long_content = (
            "First sentence about alpha. Second sentence about beta. "
            "Third sentence about gamma. Fourth sentence about delta."
        )
        r = await svc.encode_episode(long_content, "s1", salience=0.5)
        assert r["deduped"] is False
        assert "group_id" in r
        assert "segment_ids" in r
        assert len(r["segment_ids"]) >= 2
        # First segment id is the group id (and the published id).
        assert r["id"] == r["segment_ids"][0]
        assert r["group_id"] == r["segment_ids"][0]

        rows = await svc.episodic.get_documents({"session_id": "s1"})
        assert len(rows) == len(r["segment_ids"])
        positions = sorted(int(md["position_in_episode"]) for _, _, md in rows)
        assert positions == list(range(len(r["segment_ids"])))
        assert all(int(md["segment_count"]) == len(r["segment_ids"]) for _, _, md in rows)
    finally:
        await db.close()


async def test_fetch_episode_group_returns_ordered_segments() -> None:
    db = AsyncVectorDB(":memory:")
    try:
        cfg = MemoryConfig(theta_segment_max_chars=30, theta_segment_max_items=4)
        svc = MemoryService(db, config=cfg, embed_fn=hash_embed)
        text = "Alpha one. Beta two. Gamma three. Delta four."
        r = await svc.encode_episode(text, "s1", salience=0.5)
        group = await svc.fetch_episode_group(r["group_id"])
        assert len(group) == len(r["segment_ids"])
        # Positions are strictly ascending.
        positions = [item["position"] for item in group]
        assert positions == sorted(positions)
        assert positions[0] == 0
    finally:
        await db.close()


async def test_encode_short_content_keeps_legacy_shape() -> None:
    db = AsyncVectorDB(":memory:")
    try:
        # Threshold high: short note never splits.
        cfg = MemoryConfig(theta_segment_max_chars=1024, theta_segment_max_items=7)
        svc = MemoryService(db, config=cfg, embed_fn=hash_embed)
        r = await svc.encode_episode("just a short note", "s1", salience=0.5)
        assert "group_id" not in r
        assert "segment_ids" not in r
    finally:
        await db.close()


# ----------------------------------------------------------- successor SR (#5)
async def test_successor_observe_within_window_creates_edge() -> None:
    sr = SuccessorRepresentation(window_seconds=10.0)
    await sr.observe("s1", 1, t=0.0)
    await sr.observe("s1", 2, t=1.0)
    assert sr.total_edges == 1.0
    boost = sr.boost(1, [2])
    # First TD step on (1->2) sets M[1][2] = alpha * (1 + gamma * M[2][2])
    # = 0.1 * (1 + 0) = 0.1.
    assert boost[2] > 0.0


async def test_successor_observe_outside_window_skips_edge() -> None:
    sr = SuccessorRepresentation(window_seconds=5.0)
    await sr.observe("s1", 1, t=0.0)
    await sr.observe("s1", 2, t=100.0)
    assert sr.total_edges == 0.0
    assert sr.boost(1, [2]) == {2: 0.0}


async def test_successor_observe_is_global_across_sessions() -> None:
    # The observation window is GLOBAL, not partitioned by session_id:
    # episodes co-occurring within window_seconds fold into T by the same
    # rule even when seen under different session ids (one interconnected
    # graph, not per-session islands).
    sr = SuccessorRepresentation(window_seconds=10.0)
    await sr.observe("s1", 1, t=0.0)
    await sr.observe("s2", 2, t=1.0)
    assert sr.total_edges == 1.0
    assert sr.boost(1, [2])[2] > 0.0
    # Temporal gating itself is unchanged: outside the window no edge
    # forms, regardless of session.
    await sr.observe("s3", 3, t=100.0)
    assert sr.total_edges == 1.0


async def test_successor_omega_cold_start_ramp() -> None:
    sr = SuccessorRepresentation(
        omega_max=0.3,
        cold_start_ratio=1.0,
        window_seconds=1.5,
    )
    # No edges -> omega 0.
    assert sr.omega(episode_count=10) == 0.0
    # Window of 1.5s and t-spacing of 1s: each new event keeps exactly
    # one prior in the window, so we get exactly 4 edges across 5 events.
    for i in range(5):
        await sr.observe("s1", i, t=float(i))
    assert sr.total_edges == 4.0
    # 4 edges across 10 episodes -> ratio 0.4, partial ramp.
    omega = sr.omega(episode_count=10)
    assert 0.0 < omega < 0.3
    # Plenty of edges past the ratio -> plateau at omega_max.
    omega_full = sr.omega(episode_count=2)
    assert omega_full == pytest.approx(0.3)


async def test_successor_td_propagates_through_chain() -> None:
    """A->B->C should give M[A][C] > 0 once chain edges exist (gamma>0)."""
    sr = SuccessorRepresentation(window_seconds=10.0, gamma=0.7, alpha=0.5)
    # Train the chain a few times so TD has converged enough to propagate.
    for trial in range(20):
        base = trial * 100.0
        await sr.observe("s1", 1, t=base)
        await sr.observe("s1", 2, t=base + 1.0)
        await sr.observe("s1", 3, t=base + 2.0)
    boost = sr.boost(1, [3])
    assert boost[3] > 0.0


async def test_recall_observes_cooccurrences_into_sr() -> None:
    db = AsyncVectorDB(":memory:")
    try:
        # Mechanics test: disable the relevance gate so both co-recalls
        # return and fold an SR edge.
        svc = MemoryService(
            db, config=MemoryConfig(recall_relevance_gate=False), embed_fn=hash_embed
        )
        await svc.encode_episode("alpha note", "s1", salience=0.5)
        await svc.encode_episode("beta note", "s1", salience=0.5)
        # First recall returns both, which folds an edge into SR.
        await svc.recall("alpha note", session_id="s1", k=5, mode="episodic")
        assert svc._sr is not None
        assert svc._sr.total_edges >= 1.0
    finally:
        await db.close()


async def test_recall_anchor_model_does_not_inflate_edges() -> None:
    """A single recall returning n hits adds exactly n-1 edges, not
    n*(n-1)/2 — confirming the anchor-model fold prevents the
    pairwise-coincident inflation a naive observe-each loop produces."""
    db = AsyncVectorDB(":memory:")
    try:
        # Disable the relevance gate so all four hits return: this isolates
        # the anchor-fold edge count, not the elbow's plateau trimming.
        cfg = MemoryConfig(recall_relevance_gate=False)
        svc = MemoryService(db, config=cfg, embed_fn=hash_embed)
        for tag in ("alpha", "beta", "gamma", "delta"):
            await svc.encode_episode(f"{tag} note", "s1", salience=0.5)
        await svc.recall("alpha note", session_id="s1", k=4, mode="episodic")
        assert svc._sr is not None
        # Exactly k-1 edges out of the anchor (no pairwise inflation).
        assert svc._sr.total_edges == 3.0
    finally:
        await db.close()


async def test_successor_observe_recall_set_anchor_only_edges() -> None:
    sr = SuccessorRepresentation(window_seconds=10.0)
    await sr.observe_recall_set("s1", anchor_id=1, other_ids=[2, 3, 4], t=0.0)
    # Anchor -> each other = 3 edges (no pairwise between 2,3,4).
    assert sr.total_edges == 3.0
    boost = sr.boost(1, [2, 3, 4])
    assert all(v > 0.0 for v in boost.values())
    # 2 has no outgoing edges from this single recall.
    assert sr.boost(2, [3, 4]) == {3: 0.0, 4: 0.0}


async def test_successor_recall_set_chains_across_recalls() -> None:
    """A second recall in the same session should chain off the prior
    anchor by adding one cross-recall edge into the new anchor."""
    sr = SuccessorRepresentation(window_seconds=10.0)
    await sr.observe_recall_set("s1", anchor_id=1, other_ids=[2], t=0.0)
    edges_after_first = sr.total_edges
    await sr.observe_recall_set("s1", anchor_id=3, other_ids=[4], t=1.0)
    # Second recall adds 1 within-recall edge (3 -> 4) + 1 cross-recall
    # edge (1 -> 3) since prior anchor 1 was still in the window.
    assert sr.total_edges == edges_after_first + 2.0


async def test_recall_scopes_to_session_by_default() -> None:
    """A supplied session_id scopes recall to that session by default;
    scope_session=False opts back into cross-session results."""
    db = AsyncVectorDB(":memory:")
    try:
        svc = MemoryService(db, config=MemoryConfig(), embed_fn=hash_embed)
        await svc.encode_episode("shared cue alpha", "old", salience=0.5)
        await svc.encode_episode("shared cue beta", "now", salience=0.5)
        scoped = await svc.recall("shared cue", session_id="now", k=5, mode="episodic")
        assert {h["metadata"].get("session_id") for h in scoped} == {"now"}
        cross = await svc.recall(
            "shared cue", session_id="now", k=5, mode="episodic", scope_session=False
        )
        assert {"old", "now"} <= {h["metadata"].get("session_id") for h in cross}
    finally:
        await db.close()


async def test_recall_biases_same_session_at_equal_cosine() -> None:
    """Same-session episodes win ties via the contextual bonus, but
    cross-session episodes are still in the result set."""
    db = AsyncVectorDB(":memory:")
    try:
        cfg = MemoryConfig(same_session_bonus=0.5, sr_enabled=False)
        svc = MemoryService(db, config=cfg, embed_fn=hash_embed)
        await svc.encode_episode("identical content", "old", salience=0.5)
        await svc.encode_episode("identical content", "now", salience=0.5)
        # scope_session=False keeps both sessions in play so the contextual
        # bonus (not a hard filter) decides the tie-break order.
        hits = await svc.recall(
            "identical content", session_id="now", k=2, mode="episodic", scope_session=False
        )
        assert len(hits) == 2
        assert hits[0]["metadata"]["session_id"] == "now"
        assert hits[1]["metadata"]["session_id"] == "old"
    finally:
        await db.close()


async def test_recall_cross_session_no_caller_session() -> None:
    db = AsyncVectorDB(":memory:")
    try:
        # Mechanics test: disable the relevance gate so cross-session recall
        # returns both sessions' hits regardless of the synthetic distances.
        svc = MemoryService(
            db, config=MemoryConfig(recall_relevance_gate=False), embed_fn=hash_embed
        )
        await svc.encode_episode("alpha", "a", salience=0.5)
        await svc.encode_episode("alpha note", "b", salience=0.5)
        hits = await svc.recall("alpha", k=5, mode="episodic")
        assert {h["metadata"].get("session_id") for h in hits} == {"a", "b"}
    finally:
        await db.close()


async def test_semantic_memory_scope_and_global() -> None:
    """A session-scoped semantic memory surfaces only in its own session;
    a global one surfaces from any session (and an absent caller session
    disables scoping entirely)."""
    db = AsyncVectorDB(":memory:")
    try:
        svc = MemoryService(db, config=MemoryConfig(), embed_fn=hash_embed)
        await svc.store_semantic_memory(
            "widget policy detail", scope="session", session_id="proj-a"
        )
        await svc.store_semantic_memory("widget policy detail", scope="global")
        # In proj-a: the session-scoped fact and the global one are visible.
        a = await svc.recall_semantic_memory("widget policy detail", session_id="proj-a")
        assert sorted(h["metadata"].get("scope") for h in a) == ["global", "session"]
        # In proj-b: only the global fact.
        b = await svc.recall_semantic_memory("widget policy detail", session_id="proj-b")
        assert [h["metadata"].get("scope") for h in b] == ["global"]
        # No caller session: no scoping, both visible.
        n = await svc.recall_semantic_memory("widget policy detail")
        assert len(n) == 2
        # Explicit opt-out also returns both even with a session.
        c = await svc.recall_semantic_memory(
            "widget policy detail", session_id="proj-b", scope_session=False
        )
        assert len(c) == 2
    finally:
        await db.close()


async def test_store_semantic_memory_scope_defaults_and_validation() -> None:
    """scope is inferred from session_id when unset; an explicit 'session'
    scope without a session_id is rejected."""
    db = AsyncVectorDB(":memory:")
    try:
        svc = MemoryService(db, config=MemoryConfig(), embed_fn=hash_embed)
        with_sid = await svc.store_semantic_memory("fact one", session_id="proj-a")
        assert with_sid["scope"] == "session"
        no_sid = await svc.store_semantic_memory("fact two")
        assert no_sid["scope"] == "global"
        with pytest.raises(ValidationError):
            await svc.store_semantic_memory("fact three", scope="session")
    finally:
        await db.close()


async def test_sr_transitions_persist_across_restart(tmp_path: Path) -> None:
    """T-counts written by recall in process A must rehydrate in process B."""
    db_path = str(tmp_path / "sr.db")

    db1 = AsyncVectorDB(db_path)
    try:
        svc1 = MemoryService(
            db1, config=MemoryConfig(recall_relevance_gate=False), embed_fn=hash_embed
        )
        for tag in ("alpha", "beta", "gamma"):
            await svc1.encode_episode(f"{tag} note", "s1", salience=0.5)
        await svc1.recall("alpha note", session_id="s1", k=3, mode="episodic")
        assert svc1._sr is not None
        edges_first = svc1._sr.total_edges
        assert edges_first >= 1.0
    finally:
        await db1.close()

    db2 = AsyncVectorDB(db_path)
    try:
        svc2 = MemoryService(
            db2, config=MemoryConfig(recall_relevance_gate=False), embed_fn=hash_embed
        )
        assert svc2._sr is not None
        # Force the lazy load that normally happens on first async op.
        await svc2._ensure_sr_loaded()
        assert svc2._sr.total_edges == edges_first
        # M was rebuilt from T, so boost from any anchor with an outgoing
        # edge is non-zero.
        any_anchor = next(iter(svc2._sr._T_counts))
        any_target = next(iter(svc2._sr._T_counts[any_anchor]))
        assert svc2._sr.boost(any_anchor, [any_target])[any_target] > 0.0
    finally:
        await db2.close()


async def test_recall_sr_disabled_when_config_off() -> None:
    db = AsyncVectorDB(":memory:")
    try:
        cfg = MemoryConfig(sr_enabled=False)
        svc = MemoryService(db, config=cfg, embed_fn=hash_embed)
        assert svc._sr is None
        await svc.encode_episode("alpha", "s1")
        await svc.recall("alpha", session_id="s1", k=3, mode="episodic")
    finally:
        await db.close()


# --------------------------------------------------- input caps (Important #8)
async def test_encode_rejects_oversized_content() -> None:
    db = AsyncVectorDB(":memory:")
    try:
        svc = MemoryService(db, config=MemoryConfig(max_content_chars=20), embed_fn=hash_embed)
        await svc.encode_episode("short enough", "s1")  # under the cap: OK
        with pytest.raises(ValidationError, match="max_content_chars"):
            await svc.encode_episode("x" * 21, "s1")
    finally:
        await db.close()


async def test_encode_rejects_too_many_tags() -> None:
    db = AsyncVectorDB(":memory:")
    try:
        svc = MemoryService(db, config=MemoryConfig(max_tags=2), embed_fn=hash_embed)
        await svc.encode_episode("ok", "s1", tags=["a", "b"])  # at the cap: OK
        with pytest.raises(ValidationError, match="too many tags"):
            await svc.encode_episode("ok2", "s1", tags=["a", "b", "c"])
    finally:
        await db.close()


# ------------------------------------- forget salience default (Critical C5)
async def test_forget_does_not_prune_episode_missing_salience() -> None:
    """An episode whose metadata lacks ``salience`` must not be a
    guaranteed prune candidate: absent salience falls back to a neutral
    base, not 0.0 (which would force strength to 0)."""
    db = AsyncVectorDB(":memory:")
    try:
        svc = MemoryService(db, config=MemoryConfig(), embed_fn=hash_embed)
        # Write directly, bypassing encode (which always derives salience)
        # to simulate a direct DB write / pre-field / fixture episode.
        ids = await svc.episodic.add_texts(
            ["orphan trace with no salience field"],
            metadatas=[{"session_id": "s1"}],
            embeddings=[hash_embed("orphan trace with no salience field")],
        )
        orphan_id = int(ids[0])
        result = await svc.forget(dry_run=True)
        assert orphan_id not in result["candidate_ids"]
    finally:
        await db.close()


# ----------------------------- reactor isolation + counter reset (C3 / C4)
async def test_failed_reactor_consolidate_is_isolated_and_resets_counter(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A raising background consolidation must not surface on the user's
    encode call (C3), and the novel-encode counter must be reset so the
    next encode does not immediately re-fire the failing pass (C4)."""

    async def boom(*_args: object, **_kwargs: object) -> list[dict[str, object]]:
        raise RuntimeError("consolidation backend down")

    monkeypatch.setattr(consolidate_mod, "run", boom)

    db = AsyncVectorDB(":memory:")
    try:
        cfg = MemoryConfig(
            reactor_consolidate_after_novel=1,
            reactor_consolidate_cooldown_seconds=0.0,
        )
        svc = MemoryService(db, config=cfg, embed_fn=hash_embed)
        # The encode trips the consolidate trigger; the callback raises
        # but the user-facing call must still succeed.
        result = await svc.encode_episode("trigger reactor", "s1")
        assert result["id"] >= 0
        # C4: counter was advanced before the failing work, so it is not
        # left saturated (which would re-trigger on every later encode).
        assert svc._bus.state.novel_encodes_since_consolidate == 0
    finally:
        await db.close()


# ----------------------------- reactor state rehydration across restart
async def test_reactor_state_rehydrates_novel_encodes_from_persisted_log() -> None:
    """Regression: the novel-encode counter must be rebuilt from the durable
    event log on a fresh process. Under stdio-per-session the server is
    relaunched every session, so a counter that resets to 0 each launch can
    never reach the consolidate threshold and ``consolidate.run`` never fires.
    """
    from synara.features.memory.basal_ganglia.events import now_seconds  # noqa: PLC0415

    db = AsyncVectorDB(":memory:")
    try:
        threshold = 5
        # First "process": self-learning OFF so the reactor doesn't consume
        # the counter itself — we only want it to populate the durable log.
        svc1 = MemoryService(
            db,
            config=MemoryConfig(
                self_learning_enabled=False,
                reactor_consolidate_after_novel=threshold,
            ),
            embed_fn=hash_embed,
        )
        for i in range(threshold):
            r = await svc1.encode_episode(f"unique episode {i} on subject {i}", "s1")
            assert r["deduped"] is False

        # Simulated restart: fresh service + fresh ReactorState over the same
        # durable collection. The counter starts at 0...
        svc2 = MemoryService(
            db,
            config=MemoryConfig(reactor_consolidate_after_novel=threshold),
            embed_fn=hash_embed,
        )
        assert svc2._bus.state.novel_encodes_since_consolidate == 0

        # ...and must be rebuilt from the persisted event log on first touch.
        await svc2._bus.ensure_state_loaded()
        assert svc2._bus.state.novel_encodes_since_consolidate >= threshold
        assert svc2._bus.policy.consolidate_due(svc2._bus.state, now_seconds())
    finally:
        await db.close()


# ----------------------------- concurrent consolidate serialisation
async def test_concurrent_consolidate_calls_serialise(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Two concurrent ``consolidate`` calls must not overlap.

    The service guards consolidation behind ``_consolidate_lock`` so the
    reactor trigger and an explicit call cannot interleave their
    ``consolidated_into`` metadata writes on the same episodes. We force
    a recordable critical section into the patched ``run`` and assert
    its start/end windows are disjoint.
    """
    import asyncio  # noqa: PLC0415

    in_flight = 0
    max_in_flight = 0

    async def slow_run(*_args: object, **_kwargs: object) -> list[dict[str, object]]:
        nonlocal in_flight, max_in_flight
        in_flight += 1
        max_in_flight = max(max_in_flight, in_flight)
        # Yield to the loop so a second waiter has a real chance to
        # observe the lock if it leaked.
        await asyncio.sleep(0.01)
        in_flight -= 1
        return []

    monkeypatch.setattr(consolidate_mod, "run", slow_run)

    db = AsyncVectorDB(":memory:")
    try:
        svc = MemoryService(db, config=MemoryConfig(), embed_fn=hash_embed)
        await asyncio.gather(svc.consolidate(), svc.consolidate(), svc.consolidate())
        assert max_in_flight == 1, "consolidate calls overlapped despite the lock"
    finally:
        await db.close()
