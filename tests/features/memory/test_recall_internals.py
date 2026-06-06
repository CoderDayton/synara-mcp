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


async def test_recall_empty_query_raises_even_when_k_invalid(
    service: MemoryService,
) -> None:
    """An empty query must always raise — even when k <= 0 would
    otherwise short-circuit. A bad query is a programmer error
    regardless of the result budget.
    """
    with pytest.raises(ValidationError, match="query must be non-empty"):
        await service.recall(query="", k=0)
    with pytest.raises(ValidationError, match="query must be non-empty"):
        await service.recall(query="   ", k=-5)


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


# --------------------------------------------------------------- elbow gate


def test_elbow_cutoff_sharp_cliff_keeps_only_peak() -> None:
    # One strong hit then a noise plateau (the live "timestamp fields" case:
    # episodic distances 0.388, 0.706, 0.726, 0.729).
    dists = [0.388, 0.706, 0.726, 0.729]
    cutoff = recall_mod.elbow_cutoff(dists)
    assert cutoff is not None
    assert cutoff == pytest.approx(0.388)
    assert [d for d in dists if d <= cutoff + 1e-9] == [0.388]


def test_elbow_cutoff_keeps_relevant_ramp() -> None:
    # Three relevant hits, then a plateau -> keep the three.
    dists = [0.30, 0.35, 0.40, 0.72, 0.74]
    cutoff = recall_mod.elbow_cutoff(dists)
    assert cutoff is not None
    assert cutoff == pytest.approx(0.40)
    assert [d for d in dists if d <= cutoff + 1e-9] == [0.30, 0.35, 0.40]


def test_elbow_cutoff_linear_ramp_no_cut() -> None:
    # A straight ramp has no elbow -> keep everything.
    assert recall_mod.elbow_cutoff([0.30, 0.40, 0.50, 0.60, 0.70]) is None


def test_elbow_cutoff_gradual_plateau_no_cut() -> None:
    # All candidates uniformly mediocre (no steep ramp) -> no defensible
    # knee, so nothing is cut (the live semantic "timestamp fields" case).
    dists = [0.771, 0.778, 0.781, 0.793, 0.809, 0.810, 0.824, 0.841]
    assert recall_mod.elbow_cutoff(dists) is None


def test_elbow_cutoff_too_few_candidates() -> None:
    assert recall_mod.elbow_cutoff([0.1, 0.9]) is None
    assert recall_mod.elbow_cutoff([]) is None


def test_elbow_cutoff_low_spread_no_cut() -> None:
    # Every candidate within the min-spread band -> no structure to cut.
    assert recall_mod.elbow_cutoff([0.50, 0.51, 0.52, 0.53]) is None


def test_elbow_cutoff_ignores_non_finite() -> None:
    # A NaN/inf distance (degenerate stored vector) must not break the fit.
    cutoff = recall_mod.elbow_cutoff([0.388, 0.706, 0.726, 0.729, float("inf")])
    assert cutoff == pytest.approx(0.388)


def test_elbow_cutoff_sensitivity_gates_shallow_knee() -> None:
    # A bend (knee prominence ~0.083) is cut at low sensitivity but spared
    # at high sensitivity.
    dists = [0.30, 0.35, 0.40, 0.55, 0.60]
    assert recall_mod.elbow_cutoff(dists, sensitivity=0.04) is not None
    assert recall_mod.elbow_cutoff(dists, sensitivity=0.2) is None


def test_dynamic_ceiling_keeps_standout_near_hit() -> None:
    # One near hit then a far cluster (the live episodic "timestamp fields"
    # case): p90 ~0.728, alpha 0.8 -> ceiling ~0.582, so only the near hit
    # survives.
    dists = [0.388, 0.706, 0.726, 0.729]
    cut = recall_mod.dynamic_ceiling(dists)
    assert cut is not None
    assert [d for d in dists if d <= cut + 1e-9] == [0.388]


def test_dynamic_ceiling_all_weak_empties() -> None:
    # A uniformly mediocre cluster (the live "weather on mars" case): every
    # candidate sits beyond alpha * p90, so there is no near-field hit to
    # keep. With no floor, the off-topic query empties instead of returning
    # its least-bad hit.
    dists = [0.771, 0.778, 0.781, 0.793, 0.809, 0.810, 0.824, 0.841]
    cut = recall_mod.dynamic_ceiling(dists)
    assert cut is not None
    assert [d for d in dists if d <= cut + 1e-9] == []


def test_dynamic_ceiling_keeps_lone_standout() -> None:
    # A single genuine match in an otherwise-far cloud: d_min sits just above
    # alpha * p90 (the bare ceiling would empty it), but it is separated from
    # the background by more than standout_gap -- a real relevance cliff -- so
    # it is floored back in and kept.
    dists = [0.82, 1.0, 1.0, 1.0, 1.0]
    cut = recall_mod.dynamic_ceiling(dists, standout_gap=0.15)
    assert cut is not None
    assert [d for d in dists if d <= cut + 1e-9] == [0.82]
    # Without the standout exception the same lone hit empties.
    bare = recall_mod.dynamic_ceiling(dists)
    assert bare is not None
    assert [d for d in dists if d <= bare + 1e-9] == []


def test_dynamic_ceiling_standout_gap_still_empties_uniform_cloud() -> None:
    # The off-topic case: every neighbour is within standout_gap of the next,
    # so there is no cliff to protect -- the query still empties even with the
    # standout exception enabled.
    dists = [0.79, 0.80, 0.81, 0.82, 0.83, 0.84]
    cut = recall_mod.dynamic_ceiling(dists, standout_gap=0.15)
    assert cut is not None
    assert [d for d in dists if d <= cut + 1e-9] == []


def test_dynamic_ceiling_keeps_strong_match_at_any_scale() -> None:
    # Hash-stub scale: an exact match sits at ~0 and the rest at ~1. The
    # ceiling (~0.8) tracks the distribution, so the match survives without
    # any hardcoded 0.7 -- the model-dependence footgun is gone.
    dists = [0.0, 1.0, 1.0, 1.0]
    cut = recall_mod.dynamic_ceiling(dists)
    assert cut is not None
    assert [d for d in dists if d <= cut + 1e-9] == [0.0]


def test_dynamic_ceiling_too_few_candidates() -> None:
    # Below min_candidates the sample can't estimate a reference -> no gate
    # (this is what spares small synthetic corpora).
    assert recall_mod.dynamic_ceiling([0.1, 0.5, 0.9]) is None
    assert recall_mod.dynamic_ceiling([]) is None


def test_gap_cutoff_cuts_at_first_large_jump() -> None:
    # The live episodic-head vs semantic-tail case: a ~0.41 jump ends the run.
    assert recall_mod.gap_cutoff([0.364, 0.771], min_gap=0.35) == pytest.approx(0.364)
    vals = [0.30, 0.34, 0.36, 0.71]
    cut = recall_mod.gap_cutoff(vals, min_gap=0.35)
    assert cut is not None
    assert cut == pytest.approx(0.36)
    assert [d for d in vals if d <= cut + 1e-9] == [0.30, 0.34, 0.36]


def test_gap_cutoff_no_jump_keeps_all() -> None:
    # Tight cluster (the live semantic case): no jump >= min_gap.
    assert recall_mod.gap_cutoff([0.771, 0.778, 0.781, 0.793], min_gap=0.35) is None
    assert recall_mod.gap_cutoff([0.30], min_gap=0.35) is None
    assert recall_mod.gap_cutoff([], min_gap=0.35) is None


def test_gate_merged_gap_cut_drops_cross_source_tail() -> None:
    # Episodic head at 0.36, semantic tail at 0.77: a 0.41 cross-source jump
    # that the per-source ceiling/elbow can't see. The final gap cut drops it.
    merged: list[recall_mod._Hit] = [
        (1, "", {}, 0.36, "episodic"),
        (2, "", {}, 0.77, "semantic"),
    ]
    cfg = MemoryConfig()
    kept = recall_mod._gate_merged(list(merged), cfg)
    assert [row[0] for row in kept] == [1]


def test_gate_relevance_ceiling_disabled_keeps_all() -> None:
    hits = [{"distance": d} for d in (0.77, 0.95)]
    cfg = MemoryConfig(recall_elbow_cutoff=False, recall_max_distance_alpha=0.0)
    kept = recall_mod.gate_relevance([dict(h) for h in hits], cfg)
    assert [h["distance"] for h in kept] == [0.77, 0.95]


def test_gate_merged_dynamic_ceiling_then_elbow_per_source() -> None:
    # (doc_id, text, md, distance, source). The dynamic ceiling is computed
    # per source: it drops the far episodic (0.80) and semantic (0.95) hits;
    # the elbow then trims each remaining plateau to its peak.
    merged: list[recall_mod._Hit] = [
        (1, "", {}, 0.30, "episodic"),
        (2, "", {}, 0.66, "episodic"),
        (3, "", {}, 0.68, "episodic"),
        (4, "", {}, 0.69, "episodic"),
        (5, "", {}, 0.80, "episodic"),
        (6, "", {}, 0.55, "semantic"),
        (7, "", {}, 0.60, "semantic"),
        (8, "", {}, 0.62, "semantic"),
        (9, "", {}, 0.95, "semantic"),
    ]
    cfg = MemoryConfig()  # dynamic ceiling + elbow both on by default
    kept = recall_mod._gate_merged(list(merged), cfg)
    ids = [row[0] for row in kept]
    assert 5 not in ids  # far episodic hit, dropped by the dynamic ceiling
    assert 9 not in ids  # far semantic hit, dropped by the dynamic ceiling
    assert ids == [1, 6]
