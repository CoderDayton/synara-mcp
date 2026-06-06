"""Ontology-hygiene gates layered on top of the candidate buffer.

Coverage targets each new knob independently AND pins the "default = off"
invariants that regression-tested a real bug: when the diversity knobs
are at their off defaults, promotion must NOT be gated by what a single
hit carries (e.g. session_id=None from a tied cluster).


Each test asserts ONE knob's invariant: same-session repetition cannot
promote, same-epoch repetition cannot promote, sub-floor clusters are
rejected, confidence floor is enforced, hit decay shrinks parked hits,
and the cold-schema eviction path in ``forget`` deletes schemas whose
``last_hit_at`` is older than the threshold.
"""

from __future__ import annotations

import hashlib
from collections.abc import AsyncIterator

import numpy as np
import pytest
import pytest_asyncio
from simplevecdb import AsyncVectorDB

from synara.features.memory.config import MemoryConfig
from synara.features.memory.neocortex import consolidate as cm
from synara.features.memory.neocortex import forget as fm
from synara.features.memory.service import MemoryService, now_seconds


def _embed(text: str, dim: int = 32) -> list[float]:
    seed = int.from_bytes(hashlib.sha256(text.encode("utf-8")).digest()[:8], "big")
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(dim).astype(np.float32)
    n = float(np.linalg.norm(v))
    return [float(x) for x in (v / n).tolist()] if n > 0 else [float(x) for x in v.tolist()]


async def _seed_candidate(
    svc: MemoryService,
    emb: list[float],
    *,
    hits: int,
    age: int,
    sessions: list[str] | None = None,
    epochs: list[int] | None = None,
) -> int:
    md: dict[str, object] = {
        "hits": hits,
        "age": age,
        "sessions": list(sessions or []),
        "epochs": list(epochs or []),
    }
    new_ids = await svc.schema_candidates.add_texts(["seed"], metadatas=[md], embeddings=[emb])
    new_id = int(new_ids[0])
    await svc.schema_candidates.update_metadata([(new_id, {"id": new_id})])
    return new_id


@pytest_asyncio.fixture
async def svc_sessions_2() -> AsyncIterator[MemoryService]:
    db = AsyncVectorDB(":memory:")
    try:
        cfg = MemoryConfig(
            consolidate_min_age_seconds=0.0,
            consolidate_min_retrievals=0,
            consolidate_min_recurrence=2,
            consolidate_min_hit_sessions=2,
        )
        yield MemoryService(db, config=cfg, embed_fn=_embed)
    finally:
        await db.close()


@pytest_asyncio.fixture
async def svc_epochs_2() -> AsyncIterator[MemoryService]:
    db = AsyncVectorDB(":memory:")
    try:
        cfg = MemoryConfig(
            consolidate_min_age_seconds=0.0,
            consolidate_min_retrievals=0,
            consolidate_min_recurrence=2,
            consolidate_min_hit_epochs=2,
        )
        yield MemoryService(db, config=cfg, embed_fn=_embed)
    finally:
        await db.close()


# ============================================================ session diversity
@pytest.mark.asyncio
async def test_session_diversity_blocks_same_session_promotion(
    svc_sessions_2: MemoryService,
) -> None:
    # A candidate hits the recurrence count from one session only.
    # Without cross-session diversity it would promote; with the gate
    # on, promotion is held back and a rejection counter ticks.
    anchor = _embed("alpha headline guard sample 0")
    await _seed_candidate(
        svc_sessions_2,
        anchor,
        hits=1,
        age=0,
        sessions=["s1"],
        epochs=[0],
    )
    promoted = await cm._v2_promote_or_park(
        svc_sessions_2,
        summary="alpha headline guard sample 0",
        summary_emb=anchor,
        merge_dist=svc_sessions_2.config.consolidate_schema_merge_distance,
        min_recurrence=2,
        session_id="s1",
        epoch=1,
    )
    assert promoted is False
    rejected = svc_sessions_2._hygiene_counters["candidates_rejected_session_diversity"]
    assert rejected >= 1


@pytest.mark.asyncio
async def test_session_diversity_promotes_when_two_sessions_hit(
    svc_sessions_2: MemoryService,
) -> None:
    anchor = _embed("beta headline guard sample")
    await _seed_candidate(
        svc_sessions_2,
        anchor,
        hits=1,
        age=0,
        sessions=["s1"],
        epochs=[0],
    )
    promoted = await cm._v2_promote_or_park(
        svc_sessions_2,
        summary="beta headline guard sample",
        summary_emb=anchor,
        merge_dist=svc_sessions_2.config.consolidate_schema_merge_distance,
        min_recurrence=2,
        session_id="s2",  # second distinct session
        epoch=1,
    )
    assert promoted is True


# ============================================================ epoch diversity
@pytest.mark.asyncio
async def test_epoch_diversity_blocks_same_epoch_promotion(
    svc_epochs_2: MemoryService,
) -> None:
    anchor = _embed("gamma headline guard sample")
    await _seed_candidate(
        svc_epochs_2,
        anchor,
        hits=1,
        age=0,
        sessions=["s1"],
        epochs=[5],
    )
    promoted = await cm._v2_promote_or_park(
        svc_epochs_2,
        summary="gamma headline guard sample",
        summary_emb=anchor,
        merge_dist=svc_epochs_2.config.consolidate_schema_merge_distance,
        min_recurrence=2,
        session_id="s1",
        epoch=5,  # same epoch -- diversity fails
    )
    assert promoted is False
    rejected = svc_epochs_2._hygiene_counters["candidates_rejected_epoch_diversity"]
    assert rejected >= 1


# ============================================================ hit decay
@pytest.mark.asyncio
async def test_hit_decay_shrinks_parked_hits() -> None:
    db = AsyncVectorDB(":memory:")
    try:
        cfg = MemoryConfig(
            consolidate_min_recurrence=2,
            consolidate_candidate_max_age=0,  # disable expiry to isolate decay
            consolidate_candidate_hit_decay=0.5,
        )
        svc = MemoryService(db, config=cfg, embed_fn=_embed)
        await _seed_candidate(svc, _embed("decay-me"), hits=10, age=0)
        await cm._age_schema_candidates(svc)
        rows = await svc.schema_candidates.get_documents(filter_dict=None)
        assert rows
        _, _, md = rows[0]
        # 10 * (1 - 0.5) = 5
        assert int(md["hits"]) == 5
    finally:
        await db.close()


# ============================================================ min-schema-size
@pytest.mark.asyncio
async def test_min_schema_size_rejects_undersized_cluster() -> None:
    db = AsyncVectorDB(":memory:")
    try:
        cfg = MemoryConfig(
            consolidate_min_age_seconds=0.0,
            consolidate_min_retrievals=0,
            consolidate_min_recurrence=2,
            consolidate_min_schema_size=10,  # impossible floor for small workload
        )
        svc = MemoryService(db, config=cfg, embed_fn=_embed)
        for i in range(2):
            await svc.encode_episode(f"zeta hygiene sample {i}", session_id="s1", salience=0.6)
        # pre-seed a matching candidate so the second consolidate pass
        # would promote -- but the size floor must reject.
        await svc.consolidate(min_cluster_size=2)  # parks a candidate
        # Manually seed hits to clear recurrence threshold.
        rows = await svc.schema_candidates.get_documents(filter_dict=None)
        if rows:
            cid, _, _ = rows[0]
            await svc.schema_candidates.update_metadata(
                [(int(cid), {"hits": 5, "sessions": ["s1", "s2"], "epochs": [1, 2]})]
            )
        for i in range(2):
            await svc.encode_episode(
                f"zeta hygiene sample {i + 100}", session_id="s2", salience=0.6
            )
        await svc.consolidate(min_cluster_size=2)
        # No schema must materialise (size floor blocks promotion). To
        # confirm the floor actually fired -- and the test is not just
        # passing because nothing reached the promotion site -- assert
        # the size-rejection counter is non-zero when the cluster does
        # match the seeded candidate.
        assert await svc.semantic.count() == 0
        matched = svc._hygiene_counters["candidates_rejected_size"] >= 1
        parked_fresh = await svc.schema_candidates.count() >= 1
        # Either the cluster matched the seeded candidate (rejected by
        # size) or K-Means produced a different gist that parked fresh.
        # Both are valid "no schema formed" outcomes, but at least one
        # of the two signals MUST hold for the test to be meaningful.
        assert matched or parked_fresh
    finally:
        await db.close()


# ============================================================ cold-schema eviction
@pytest.mark.asyncio
async def test_forget_evicts_unused_schemas() -> None:
    db = AsyncVectorDB(":memory:")
    try:
        cfg = MemoryConfig(
            forget_schema_unused_seconds=1.0,  # very short for test
        )
        svc = MemoryService(db, config=cfg, embed_fn=_embed)
        old = now_seconds() - 1000.0
        recent = now_seconds()
        await svc.semantic.add_texts(
            ["stale gist"],
            metadatas=[{"last_hit_at": old, "confidence": 0.5}],
            embeddings=[_embed("stale")],
        )
        await svc.semantic.add_texts(
            ["fresh gist"],
            metadatas=[{"last_hit_at": recent, "confidence": 0.5}],
            embeddings=[_embed("fresh")],
        )
        await svc.semantic.add_texts(
            ["legacy gist"],
            metadatas=[{"confidence": 0.5}],  # no last_hit_at -- preserved
            embeddings=[_embed("legacy")],
        )
        result = await fm.run(svc, dry_run=False)
        assert result["schemas_removed"] == 1
        remaining = await svc.semantic.count()
        # Started with 3, removed 1 stale; 2 should remain.
        assert remaining == 2
        assert svc._hygiene_counters["schemas_evicted_unused"] == 1
    finally:
        await db.close()


@pytest.mark.asyncio
async def test_forget_schema_eviction_off_by_default() -> None:
    db = AsyncVectorDB(":memory:")
    try:
        svc = MemoryService(db, config=MemoryConfig(), embed_fn=_embed)
        old = now_seconds() - 999_999.0
        await svc.semantic.add_texts(
            ["ancient gist"],
            metadatas=[{"last_hit_at": old}],
            embeddings=[_embed("ancient")],
        )
        result = await fm.run(svc, dry_run=False)
        assert result["schemas_removed"] == 0
        assert await svc.semantic.count() == 1
    finally:
        await db.close()


# ============================================================ stats surface
# ============================================================ diversity-gate inert at default
@pytest.mark.asyncio
async def test_diversity_gate_inert_at_default_promotes_tied_session_cluster() -> None:
    """Regression: a cluster with no recorded session (``_infer_cluster_session``
    returned None due to a tie or missing metadata) must still promote
    once recurrence is met when the diversity knob is at its off default.

    Pre-fix the gate checked ``distinct_sessions >= max(1, knob)``, which
    silently blocked promotion forever for tied clusters. The sim caught
    this as a 50% drop in schemas formed.
    """
    db = AsyncVectorDB(":memory:")
    try:
        cfg = MemoryConfig(
            consolidate_min_recurrence=2,
            # both diversity knobs explicitly at off-default
            consolidate_min_hit_sessions=1,
            consolidate_min_hit_epochs=1,
        )
        svc = MemoryService(db, config=cfg, embed_fn=_embed)
        anchor = _embed("ontology-tie cluster")
        await _seed_candidate(svc, anchor, hits=1, age=0, sessions=[], epochs=[1])
        promoted = await cm._v2_promote_or_park(
            svc,
            summary="ontology-tie cluster",
            summary_emb=anchor,
            merge_dist=cfg.consolidate_schema_merge_distance,
            min_recurrence=2,
            session_id=None,  # tied cluster -> no plurality winner
            epoch=2,
        )
        assert promoted is True, (
            "diversity gate must be inert at default; tied-session cluster blocked"
        )
    finally:
        await db.close()


@pytest.mark.asyncio
async def test_fast_promote_on_park_works_for_anonymous_caller() -> None:
    """min_recurrence=1 + diversity knobs at default must fast-promote on
    park even when session_id is None. Pre-fix the fast-promote condition
    was keyed on ``distinct_sessions_seed >= knob`` which evaluated False
    when session_id was None and the knob was at its default of 1.
    """
    db = AsyncVectorDB(":memory:")
    try:
        cfg = MemoryConfig(consolidate_min_recurrence=1)
        svc = MemoryService(db, config=cfg, embed_fn=_embed)
        promoted = await cm._v2_promote_or_park(
            svc,
            summary="anonymous fast-promote",
            summary_emb=_embed("anonymous fast-promote"),
            merge_dist=cfg.consolidate_schema_merge_distance,
            min_recurrence=1,
            session_id=None,
            epoch=1,
        )
        assert promoted is True
        # Fast-promote takes the no-write branch -- nothing was parked.
        assert await svc.schema_candidates.count() == 0
    finally:
        await db.close()


# ============================================================ _infer_cluster_session
def test_infer_cluster_session_returns_plurality_winner() -> None:
    members = [
        ("t", {"session_id": "s1"}),
        ("t", {"session_id": "s1"}),
        ("t", {"session_id": "s2"}),
    ]
    assert cm._infer_cluster_session(members) == "s1"


def test_infer_cluster_session_returns_none_on_tie() -> None:
    members = [
        ("t", {"session_id": "s1"}),
        ("t", {"session_id": "s2"}),
    ]
    assert cm._infer_cluster_session(members) is None


def test_infer_cluster_session_returns_none_when_empty() -> None:
    assert cm._infer_cluster_session([]) is None
    assert cm._infer_cluster_session([("t", {})]) is None
    assert cm._infer_cluster_session([("t", {"session_id": ""})]) is None


# ============================================================ min-promotion-confidence
@pytest.mark.asyncio
async def test_min_promotion_confidence_rejects_low_confidence_cluster() -> None:
    """A 2-source cluster yields confidence 0.4 (2/5). A floor of 0.5
    must reject it; the floor of 0 (default) must accept it."""
    db = AsyncVectorDB(":memory:")
    try:
        cfg = MemoryConfig(
            consolidate_min_age_seconds=0.0,
            consolidate_min_retrievals=0,
            consolidate_min_recurrence=1,  # fast-promote on park
            consolidate_min_promotion_confidence=0.5,  # 2/5 = 0.4 fails
        )
        svc = MemoryService(db, config=cfg, embed_fn=_embed)
        for i in range(2):
            await svc.encode_episode(f"theta confidence floor {i}", session_id="s1", salience=0.6)
        await svc.consolidate(min_cluster_size=2)
        assert await svc.semantic.count() == 0
        assert svc._hygiene_counters["candidates_rejected_confidence"] >= 1
    finally:
        await db.close()


# ========================================================= last_accessed lifecycle
@pytest.mark.asyncio
async def test_schema_last_accessed_set_on_creation() -> None:
    """Newly promoted schemas must carry ``last_accessed`` so cold-schema
    eviction has a timestamp to compare against."""
    db = AsyncVectorDB(":memory:")
    try:
        cfg = MemoryConfig(
            consolidate_min_age_seconds=0.0,
            consolidate_min_retrievals=0,
            consolidate_min_recurrence=1,
        )
        svc = MemoryService(db, config=cfg, embed_fn=_embed)
        for i in range(2):
            await svc.encode_episode(f"iota create-stamp {i}", session_id="s1", salience=0.6)
        await svc.consolidate(min_cluster_size=2)
        rows = await svc.semantic.get_documents(filter_dict=None)
        assert rows, "expected at least one schema"
        for _sid, _text, md in rows:
            assert "last_accessed" in md
            assert isinstance(md["last_accessed"], (int, float))
    finally:
        await db.close()


@pytest.mark.asyncio
async def test_schema_last_accessed_bumped_on_absorb_merge() -> None:
    """``_merge_into_schema`` must bump ``last_accessed`` so absorbed
    episodes count as schema activity."""
    db = AsyncVectorDB(":memory:")
    try:
        cfg = MemoryConfig(
            consolidate_min_age_seconds=0.0,
            consolidate_min_retrievals=0,
            consolidate_min_recurrence=1,
            consolidate_absorb_distance=2.0,  # absorb everything
        )
        svc = MemoryService(db, config=cfg, embed_fn=_embed)
        for i in range(2):
            await svc.encode_episode(f"kappa merge-stamp {i}", session_id="s1", salience=0.6)
        await svc.consolidate(min_cluster_size=2)
        rows = await svc.semantic.get_documents(filter_dict=None)
        assert rows
        sch_id = int(rows[0][0])
        first_stamp = float(rows[0][2]["last_accessed"])
        # Manually back-date the stamp, then trigger a merge.
        await svc.semantic.update_metadata([(sch_id, {"last_accessed": first_stamp - 1000.0})])
        await svc.encode_episode("kappa merge-stamp extra", session_id="s1", salience=0.6)
        await svc.recall("kappa merge-stamp extra", session_id="s1", k=1)
        await svc.consolidate(min_cluster_size=1)
        rows2 = await svc.semantic.get_documents({"id": sch_id}, limit=1)
        assert rows2
        new_stamp = float(rows2[0][2]["last_accessed"])
        assert new_stamp > first_stamp - 1000.0, "merge did not bump last_accessed"
    finally:
        await db.close()


# ============================================================ recall write-back gating
@pytest.mark.asyncio
async def test_recall_semantic_memory_does_not_bump_when_eviction_off() -> None:
    """With ``forget_schema_unused_seconds=0`` (default), recall must
    NOT write to the semantic store -- recall stays read-only."""
    db = AsyncVectorDB(":memory:")
    try:
        cfg = MemoryConfig()  # eviction off (default)
        svc = MemoryService(db, config=cfg, embed_fn=_embed)
        stamp = now_seconds() - 5000.0
        await svc.semantic.add_texts(
            ["lambda gist"],
            metadatas=[{"last_hit_at": stamp, "confidence": 0.5}],
            embeddings=[_embed("lambda gist")],
        )
        results = await svc.recall_semantic_memory("lambda gist", k=1)
        assert results
        rows = await svc.semantic.get_documents(filter_dict=None)
        # last_hit_at must be unchanged (no write-back when eviction is off).
        assert float(rows[0][2]["last_hit_at"]) == pytest.approx(stamp)
    finally:
        await db.close()


@pytest.mark.asyncio
async def test_recall_semantic_memory_bumps_when_eviction_enabled() -> None:
    """With ``forget_schema_unused_seconds > 0``, returned schemas must
    have ``last_accessed`` bumped so a fresh recall keeps the schema alive.
    Seeds the legacy ``last_hit_at`` key to also prove a pre-unification
    schema gets a canonical stamp on its next recall."""
    db = AsyncVectorDB(":memory:")
    try:
        cfg = MemoryConfig(forget_schema_unused_seconds=60.0)
        svc = MemoryService(db, config=cfg, embed_fn=_embed)
        old = now_seconds() - 5000.0
        new_ids = await svc.semantic.add_texts(
            ["mu gist"],
            metadatas=[{"last_hit_at": old, "confidence": 0.5}],
            embeddings=[_embed("mu gist")],
        )
        # Production code stamps ``id`` after add_texts; recall keys the
        # write-back on it, so mirror the contract here.
        await svc.semantic.update_metadata([(int(new_ids[0]), {"id": int(new_ids[0])})])
        results = await svc.recall_semantic_memory("mu gist", k=1)
        assert results
        rows = await svc.semantic.get_documents(filter_dict=None)
        new_stamp = float(rows[0][2]["last_accessed"])
        assert new_stamp > old, "recall must bump last_accessed when eviction is enabled"
    finally:
        await db.close()


# ============================================================ eviction robustness
@pytest.mark.asyncio
async def test_forget_eviction_skips_malformed_last_hit_at() -> None:
    """A schema with a non-numeric ``last_hit_at`` must not crash the
    forget pass; it is skipped and preserved."""
    db = AsyncVectorDB(":memory:")
    try:
        cfg = MemoryConfig(forget_schema_unused_seconds=1.0)
        svc = MemoryService(db, config=cfg, embed_fn=_embed)
        await svc.semantic.add_texts(
            ["broken gist"],
            metadatas=[{"last_hit_at": "not-a-number", "confidence": 0.5}],
            embeddings=[_embed("broken")],
        )
        out = await fm.run(svc, dry_run=False)
        assert out["schemas_removed"] == 0
        assert await svc.semantic.count() == 1
    finally:
        await db.close()


# ============================================================ epoch monotonicity
@pytest.mark.asyncio
async def test_consolidate_epoch_advances_monotonically() -> None:
    db = AsyncVectorDB(":memory:")
    try:
        svc = MemoryService(
            db,
            config=MemoryConfig(
                consolidate_min_age_seconds=0.0,
                consolidate_min_retrievals=0,
            ),
            embed_fn=_embed,
        )
        assert svc._consolidate_epoch == 0
        await svc.consolidate(min_cluster_size=2)
        e1 = svc._consolidate_epoch
        await svc.consolidate(min_cluster_size=2)
        e2 = svc._consolidate_epoch
        assert e1 == 1
        assert e2 == 2
    finally:
        await db.close()


# ============================================================ stats surface
@pytest.mark.asyncio
async def test_stats_surface_includes_hygiene_counters() -> None:
    db = AsyncVectorDB(":memory:")
    try:
        svc = MemoryService(db, config=MemoryConfig(), embed_fn=_embed)
        s = await svc.stats()
        for key in (
            "schemas_promoted",
            "candidates_parked",
            "candidates_rejected_size",
            "candidates_rejected_confidence",
            "candidates_rejected_session_diversity",
            "candidates_rejected_epoch_diversity",
            "schemas_evicted_unused",
            "schema_candidate_count",
            "consolidate_epoch",
        ):
            assert key in s
    finally:
        await db.close()
