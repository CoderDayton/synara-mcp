"""Plasticity + event bus + reactor tests."""

from __future__ import annotations

import hashlib
import math
from collections.abc import AsyncIterator

import numpy as np
import pytest
import pytest_asyncio
from simplevecdb import AsyncVectorDB

from synara.features.memory.basal_ganglia.events import (
    EventBus,
    InteractionEvent,
    ReactorState,
    TriggerPolicy,
)
from synara.features.memory.config import MemoryConfig
from synara.features.memory.hippocampus.plasticity import PlasticityGraph
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


@pytest_asyncio.fixture
async def db() -> AsyncIterator[AsyncVectorDB]:
    d = AsyncVectorDB(":memory:")
    try:
        yield d
    finally:
        await d.close()


def _make_graph(coll: object, **overrides: float | int) -> PlasticityGraph:
    base = {
        "e_ltp_decay_seconds": 7200.0,
        "l_ltp_threshold_hits": 3,
        "habit_threshold_hits": 5,
        "habit_ltd_multiplier": 0.18,
        "habit_savings_factor": 3.0,
        "ltd_decay_per_idle_day": 0.02,
        "time_compression": 24.0,
        "prune_floor": 1e-3,
    }
    base.update(overrides)
    return PlasticityGraph(collection=coll, sr=None, **base)  # type: ignore[arg-type]


async def _seed_docs(coll: object, ids: list[int]) -> None:
    """Insert placeholder docs so the edge FK constraint is satisfied."""
    texts = [f"doc-{i}" for i in ids]
    embeds = [[float(i % 7), float((i + 1) % 5), 0.0, 0.0] for i in ids]
    await coll.add_texts(texts=texts, embeddings=embeds, ids=list(ids))  # type: ignore[attr-defined]


# ----- PlasticityGraph unit tests --------------------------------------


async def test_plasticity_e_ltp_bonus_decays_between_reinforcements(
    db: AsyncVectorDB,
) -> None:
    coll = db.collection("ep")
    await _seed_docs(coll, [1, 2])
    g = _make_graph(coll)
    await g.reinforce(1, 2, score=1.0, now=0.0)
    s0 = await g.edge_state(1, 2)
    assert s0 is not None
    bonus0 = s0["bonus"]
    # Reinforce again exactly tau_E later with zero score — that adds
    # nothing but forces a decay-and-fold pass.
    await g.reinforce(1, 2, score=0.0, now=7200.0)
    s1 = await g.edge_state(1, 2)
    assert s1 is not None
    # First bonus (= 0.5) decayed by exp(-1); the zero-score event adds 0.
    assert s1["bonus"] == pytest.approx(bonus0 / math.e, rel=1e-3)


async def test_plasticity_l_ltp_threshold_folds_bonus_to_weight(
    db: AsyncVectorDB,
) -> None:
    coll = db.collection("ep")
    await _seed_docs(coll, [1, 2])
    g = _make_graph(coll)
    for k in range(3):
        await g.reinforce(1, 2, score=1.0, now=float(k))
    s = await g.edge_state(1, 2)
    assert s is not None
    assert s["weight"] > 0.0
    assert s["bonus"] == 0.0
    assert s["hits_in_window"] == 0


async def test_plasticity_habit_latches_after_threshold(db: AsyncVectorDB) -> None:
    coll = db.collection("ep")
    await _seed_docs(coll, [1, 2])
    g = _make_graph(coll, habit_threshold_hits=4)
    for k in range(5):
        await g.reinforce(1, 2, score=0.5, now=float(k))
    assert await g.is_habit(1, 2)


async def test_plasticity_habit_savings_amplifies_post_lapse_relearn(
    db: AsyncVectorDB,
) -> None:
    coll = db.collection("ep")
    await _seed_docs(coll, [1, 2])
    g = _make_graph(coll, habit_threshold_hits=3, l_ltp_threshold_hits=2)
    # Build habit fast (3 hits to latch).
    for k in range(3):
        await g.reinforce(1, 2, score=1.0, now=float(k))
    # Long disuse to drop bonus, then one more reinforcement with savings.
    await g.reinforce(1, 2, score=1.0, now=10_000.0)
    s = await g.edge_state(1, 2)
    assert s is not None
    # Compare against a graph with the habit threshold out of reach.
    coll2 = db.collection("ep2")
    await _seed_docs(coll2, [1, 2])
    g2 = _make_graph(coll2, habit_threshold_hits=999, l_ltp_threshold_hits=2)
    for k in range(3):
        await g2.reinforce(1, 2, score=1.0, now=float(k))
    await g2.reinforce(1, 2, score=1.0, now=10_000.0)
    s2 = await g2.edge_state(1, 2)
    assert s2 is not None
    assert (s["weight"] + s["bonus"]) > (s2["weight"] + s2["bonus"])


async def test_plasticity_ltd_decays_non_habit_faster_than_habit(
    db: AsyncVectorDB,
) -> None:
    coll = db.collection("ep")
    await _seed_docs(coll, [1, 2, 3, 4])
    g = _make_graph(
        coll,
        habit_threshold_hits=3,
        l_ltp_threshold_hits=1,
        ltd_decay_per_idle_day=0.5,
    )
    for k in range(3):
        await g.reinforce(1, 2, score=1.0, now=float(k))
    await g.reinforce(3, 4, score=1.0, now=0.0)
    await g.reinforce(3, 4, score=1.0, now=1.0)
    assert await g.is_habit(1, 2)
    assert not await g.is_habit(3, 4)
    s_h_before = await g.edge_state(1, 2)
    s_n_before = await g.edge_state(3, 4)
    assert s_h_before is not None
    assert s_n_before is not None
    assert s_h_before["weight"] > 0
    assert s_n_before["weight"] > 0
    # 1 real day idle (* 24 compression = 24 compressed days).
    await g.ltd_pass(now=86400.0 + 10.0)
    s_h_after = await g.edge_state(1, 2)
    s_n_after = await g.edge_state(3, 4)
    assert s_h_after is not None
    if s_n_after is not None:
        rel_h = s_h_after["weight"] / s_h_before["weight"]
        rel_n = s_n_after["weight"] / s_n_before["weight"]
        assert rel_n < rel_h


async def test_plasticity_prune_floor_removes_weak_non_habits(db: AsyncVectorDB) -> None:
    coll = db.collection("ep")
    await _seed_docs(coll, [5, 6])
    g = _make_graph(
        coll,
        habit_threshold_hits=99,
        l_ltp_threshold_hits=1,
        ltd_decay_per_idle_day=0.99,
    )
    await g.reinforce(5, 6, score=1.0, now=0.0)
    await g.reinforce(5, 6, score=1.0, now=0.5)
    assert await g.edge_state(5, 6) is not None
    pruned = await g.ltd_pass(now=86400.0 * 30.0)
    assert pruned >= 1
    assert await g.edge_state(5, 6) is None


async def test_plasticity_habit_edge_persists_at_zero_weight(db: AsyncVectorDB) -> None:
    coll = db.collection("ep")
    await _seed_docs(coll, [7, 8])
    g = _make_graph(
        coll,
        habit_threshold_hits=3,
        l_ltp_threshold_hits=1,
        ltd_decay_per_idle_day=0.99,
    )
    for k in range(3):
        await g.reinforce(7, 8, score=1.0, now=float(k))
    assert await g.is_habit(7, 8)
    await g.ltd_pass(now=86400.0 * 365.0)
    # Habit edge retained even if weight collapses to zero.
    assert await g.edge_state(7, 8) is not None


async def test_plasticity_spreading_returns_zero_when_disabled(db: AsyncVectorDB) -> None:
    coll = db.collection("ep")
    await _seed_docs(coll, [1, 2])
    g = _make_graph(coll)
    await g.reinforce(1, 2, score=1.0, now=0.0)
    out = await g.spreading(1, [2, 3], hops=0, gamma=0.5)
    assert out == {2: 0.0, 3: 0.0}


async def test_plasticity_spreading_one_hop_picks_up_durable_neighbour(
    db: AsyncVectorDB,
) -> None:
    coll = db.collection("ep")
    await _seed_docs(coll, [1, 2, 3])
    g = _make_graph(coll, l_ltp_threshold_hits=1)
    await g.reinforce(1, 2, score=1.0, now=0.0)
    await g.reinforce(1, 2, score=1.0, now=1.0)  # consolidates to weight
    out = await g.spreading(1, [2, 3], hops=1, gamma=0.5)
    assert out[2] > 0.0
    assert out[3] == 0.0


async def test_plasticity_persists_across_graph_instances(db: AsyncVectorDB) -> None:
    """A second PlasticityGraph over the same collection sees stored edges."""
    coll = db.collection("ep")
    await _seed_docs(coll, [11, 12])
    g = _make_graph(coll, l_ltp_threshold_hits=1)
    for k in range(2):
        await g.reinforce(11, 12, score=1.0, now=float(k))
    g2 = _make_graph(coll, l_ltp_threshold_hits=1)
    s = await g2.edge_state(11, 12)
    assert s is not None
    assert s["weight"] > 0


# ----- EventBus / TriggerPolicy unit tests -----------------------------


def test_trigger_policy_consolidate_due_after_threshold() -> None:
    p = TriggerPolicy(consolidate_after_novel_encodes=4, consolidate_cooldown_seconds=10.0)
    s = ReactorState(novel_encodes_since_consolidate=4, last_consolidate_at=0.0)
    assert p.consolidate_due(s, now=20.0)
    s.novel_encodes_since_consolidate = 3
    assert not p.consolidate_due(s, now=20.0)


def test_trigger_policy_consolidate_respects_cooldown() -> None:
    p = TriggerPolicy(consolidate_after_novel_encodes=1, consolidate_cooldown_seconds=60.0)
    s = ReactorState(novel_encodes_since_consolidate=1, last_consolidate_at=100.0)
    assert not p.consolidate_due(s, now=120.0)
    assert p.consolidate_due(s, now=200.0)


async def test_event_bus_records_and_caps_log() -> None:
    bus = EventBus(log_capacity=3)
    for i in range(5):
        await bus.record(
            InteractionEvent(
                kind="encode",
                timestamp=float(i),
                session_id="s",
                payload={"deduped": False},
            )
        )
    log = await bus.log()
    assert len(log) == 3
    assert [e.timestamp for e in log] == [2.0, 3.0, 4.0]
    assert bus.state.total_events == 5
    assert bus.state.novel_encodes_since_consolidate == 5


async def test_event_bus_react_skips_reactor_kinds() -> None:
    bus = EventBus()
    fired: list[str] = []

    async def _on_consolidate(_e: InteractionEvent) -> None:
        fired.append("c")

    async def _on_dream(_e: InteractionEvent) -> None:
        fired.append("d")

    bus.on_consolidate = _on_consolidate
    bus.on_dream = _on_dream
    bus.policy.consolidate_after_novel_encodes = 1
    bus.policy.consolidate_cooldown_seconds = 0.0
    bus.policy.dream_after_events = 1
    bus.policy.dream_after_idle_seconds = 0.0
    e_user = InteractionEvent(
        kind="encode",
        timestamp=10.0,
        session_id=None,
        payload={"deduped": False},
    )
    await bus.record(e_user)
    await bus.react(e_user)
    e_reactor = InteractionEvent(
        kind="consolidate",
        timestamp=11.0,
        session_id=None,
        payload={},
    )
    await bus.record(e_reactor)
    await bus.react(e_reactor)
    assert "c" in fired
    assert fired.count("c") == 1


# ----- Service-integration tests ---------------------------------------


async def test_service_emits_event_per_op(service: MemoryService) -> None:
    await service.encode_episode("hello", "s1")
    await service.recall("hello", session_id="s1", k=3)
    log = await service.event_log()
    kinds = [e.kind for e in log]
    assert "encode" in kinds
    assert "recall" in kinds


async def test_service_records_plasticity_after_co_recall(
    service: MemoryService,
) -> None:
    await service.encode_episode("alpha", "s1")
    await service.encode_episode("beta", "s1")
    await service.encode_episode("gamma", "s1")
    await service.recall("alpha", session_id="s1", k=3)
    await service.recall("alpha", session_id="s1", k=3)
    stats = await service._plasticity.stats()
    assert stats["edges"] >= 1.0


async def test_reactor_consolidate_fires_after_threshold() -> None:
    db = AsyncVectorDB(":memory:")
    try:
        cfg = MemoryConfig(
            reactor_consolidate_after_novel=3,
            reactor_consolidate_cooldown_seconds=0.0,
        )
        svc = MemoryService(db, config=cfg, embed_fn=hash_embed)
        for i in range(4):
            await svc.encode_episode(f"item-{i}", "s1")
        log = await svc.event_log()
        kinds = [e.kind for e in log]
        assert kinds.count("consolidate") >= 1
    finally:
        await db.close()


async def test_self_learning_disabled_skips_reactor() -> None:
    db = AsyncVectorDB(":memory:")
    try:
        cfg = MemoryConfig(self_learning_enabled=False, reactor_consolidate_after_novel=1)
        svc = MemoryService(db, config=cfg, embed_fn=hash_embed)
        for i in range(5):
            await svc.encode_episode(f"item-{i}", "s1")
        log = await svc.event_log()
        kinds = [e.kind for e in log]
        assert "consolidate" not in kinds
    finally:
        await db.close()


async def test_surprise_salience_boosts_when_enabled() -> None:
    db = AsyncVectorDB(":memory:")
    try:
        cfg = MemoryConfig(surprise_salience_boost=0.3, surprise_distance_floor=0.0)
        svc = MemoryService(db, config=cfg, embed_fn=hash_embed)
        # Seed a baseline so the next encode has an existing neighbour to
        # be surprised against (empty namespace = no prediction error).
        await svc.encode_episode("baseline", "s1", salience=0.5)
        r = await svc.encode_episode("anything", "s1", salience=0.4)
        rows = await svc.episodic.get_documents({"id": r["id"]}, limit=1)
        _, _, md = rows[0]
        assert float(md["salience"]) >= 0.7 - 1e-6
    finally:
        await db.close()


async def test_reconsolidation_drift_accounted_when_enabled() -> None:
    db = AsyncVectorDB(":memory:")
    try:
        cfg = MemoryConfig(reconsolidation_alpha=0.1, reconsolidation_min_score=0.0)
        svc = MemoryService(db, config=cfg, embed_fn=hash_embed)
        await svc.encode_episode("alpha", "s1")
        await svc.encode_episode("beta", "s1")
        for _ in range(5):
            await svc.recall("alpha", session_id="s1", k=2)
        rows = await svc.episodic.get_documents({"session_id": "s1"})
        any_drift = any(float(md.get("drift_total", 0.0)) > 0.0 for _, _, md in rows)
        assert any_drift
    finally:
        await db.close()


async def test_reconsolidation_pulls_vector_toward_cue() -> None:
    """Reconsolidation buffers a vector update so post-flush retrieval
    distance to the recall cue decreases."""
    db = AsyncVectorDB(":memory:")
    try:
        cfg = MemoryConfig(
            reconsolidation_alpha=0.4,
            reconsolidation_min_score=0.0,
            reconsolidation_max_total_drift=1.0,
        )
        svc = MemoryService(db, config=cfg, embed_fn=hash_embed)
        await svc.encode_episode("encoded-text", "s1")
        await svc.encode_episode("other-episode", "s1")
        cue_query = "different-cue"
        cue_vec = hash_embed(cue_query)
        # Discover which episode the recall picks as anchor; drift will
        # land on that one.
        first = await svc.recall(cue_query, session_id="s1", k=2)
        anchor_id = int(first[0]["id"])
        # Baseline distance from cue to the anchor episode before drift.
        hits_before = await svc.episodic.similarity_search(cue_vec, k=5)
        d_before = next(
            float(d) for doc, d in hits_before if int(doc.metadata.get("id", -1)) == anchor_id
        )
        for _ in range(6):
            await svc.recall(cue_query, session_id="s1", k=2)
        flushed = await svc.episodic.flush_pending()
        assert flushed > 0
        hits_after = await svc.episodic.similarity_search(cue_vec, k=5)
        d_after = next(
            float(d) for doc, d in hits_after if int(doc.metadata.get("id", -1)) == anchor_id
        )
        # Drift moved the stored vector toward the cue: distance shrinks.
        assert d_after < d_before
    finally:
        await db.close()


# ----- Default-on knob coverage ----------------------------------------


async def test_surprise_no_boost_when_namespace_is_empty() -> None:
    """First episode in an empty namespace has no neighbour to predict
    against, so default surprise boost must NOT fire."""
    db = AsyncVectorDB(":memory:")
    try:
        # Default config: surprise_salience_boost=0.1, floor=0.6.
        svc = MemoryService(db, config=MemoryConfig(), embed_fn=hash_embed)
        r = await svc.encode_episode("first ever", "s1", salience=0.5)
        rows = await svc.episodic.get_documents({"id": r["id"]}, limit=1)
        _, _, md = rows[0]
        assert float(md["salience"]) == pytest.approx(0.5)
    finally:
        await db.close()


async def test_surprise_default_boosts_distant_second_encode() -> None:
    """With defaults on, a second far-away encode in a populated session
    crosses the floor and gets the salience bump."""
    db = AsyncVectorDB(":memory:")
    try:
        # hash_embed gives near-orthogonal vectors for distinct text, so
        # cosine distance between unrelated entries is ~1.0 >> floor 0.6.
        svc = MemoryService(db, config=MemoryConfig(), embed_fn=hash_embed)
        await svc.encode_episode("baseline note", "s1", salience=0.5)
        r = await svc.encode_episode("totally different", "s1", salience=0.5)
        rows = await svc.episodic.get_documents({"id": r["id"]}, limit=1)
        _, _, md = rows[0]
        # Default boost is 0.1: 0.5 -> 0.6.
        assert float(md["salience"]) == pytest.approx(0.6)
    finally:
        await db.close()


async def test_consolidate_min_age_gate_skips_young_episodes() -> None:
    """Default min_age (60s) hides freshly-encoded episodes from
    consolidation; passing the override allows immediate consolidation."""
    db = AsyncVectorDB(":memory:")
    try:
        # Default: consolidate_min_age_seconds=60.0.
        svc = MemoryService(db, config=MemoryConfig(), embed_fn=hash_embed)
        for i in range(4):
            await svc.encode_episode(f"alpha-{i}", "s1", salience=0.5)
            await svc.recall(f"alpha-{i}", session_id="s1", k=1)
        # Young episodes -> nothing eligible.
        formed = await svc.consolidate(session_id="s1", min_cluster_size=2)
        assert formed == []

        # Same data, gate disabled -> consolidation runs.
        cfg2 = MemoryConfig(
            consolidate_min_age_seconds=0.0,
            consolidate_min_retrievals=0,
        )
        # Reuse the same DB; svc2 sees the same episodes.
        svc2 = MemoryService(db, config=cfg2, embed_fn=hash_embed)
        formed2 = await svc2.consolidate(session_id="s1", min_cluster_size=2)
        assert formed2, "expected schemas once age gate is removed"
    finally:
        await db.close()


async def test_consolidate_min_retrievals_gate_skips_unaccessed() -> None:
    """An episode that has never been recalled is below the default
    retrieval-count gate (1) and is excluded from consolidation."""
    db = AsyncVectorDB(":memory:")
    try:
        # Disable age gate to isolate the retrievals gate.
        cfg = MemoryConfig(
            consolidate_min_age_seconds=0.0,
            consolidate_min_retrievals=1,
        )
        svc = MemoryService(db, config=cfg, embed_fn=hash_embed)
        for i in range(4):
            await svc.encode_episode(f"alpha-{i}", "s1", salience=0.5)
        # Never recalled -> retrieval_count == 0 < 1 -> filtered out.
        formed_before = await svc.consolidate(session_id="s1", min_cluster_size=2)
        assert formed_before == []

        # Bump retrieval_count on each by recalling its content.
        for i in range(4):
            await svc.recall(f"alpha-{i}", session_id="s1", k=1, mode="episodic")
        formed_after = await svc.consolidate(session_id="s1", min_cluster_size=2)
        assert formed_after, "expected schemas once retrievals >= gate"
    finally:
        await db.close()


async def test_spreading_activation_boosts_neighbour_with_durable_edge() -> None:
    """With spreading_activation_hops=1 (default), a recall whose anchor
    has a durable plasticity edge to a candidate must rank that candidate
    above an equally-distant distractor."""
    db = AsyncVectorDB(":memory:")
    try:
        # Custom embedder: query and anchor collinear; neighbour and
        # distractor share equal (large) cosine distance to the query.
        vectors = {
            "anchor": [1.0, 0.0, 0.0, 0.0],
            "neighbour": [0.0, 1.0, 0.0, 0.0],
            "distractor": [0.0, 0.0, 1.0, 0.0],
            "cue": [1.0, 0.0, 0.0, 0.0],
        }

        def stub_embed(text: str) -> list[float]:
            return list(vectors[text])

        cfg = MemoryConfig(
            # Defaults already enable spreading; tighten weight so its
            # contribution dominates the rank-key tie-break.
            spreading_activation_hops=1,
            spreading_activation_decay=0.9,
            spreading_activation_weight=1.0,
            # Disable consolidation gates so the test stays focused.
            consolidate_min_age_seconds=0.0,
            consolidate_min_retrievals=0,
            # Avoid surprise side-effects on stored salience.
            surprise_salience_boost=0.0,
            # Single-hop e->w fold: any score reinforcement beyond the
            # threshold flips bonus into a durable weight.
            l_ltp_threshold_hits=1,
            sr_enabled=False,
        )
        svc = MemoryService(db, config=cfg, embed_fn=stub_embed)
        r_a = await svc.encode_episode("anchor", "s1")
        r_n = await svc.encode_episode("neighbour", "s1")
        r_d = await svc.encode_episode("distractor", "s1")

        # Build a durable plasticity edge anchor -> neighbour.
        await svc._plasticity.reinforce(r_a["id"], r_n["id"], score=1.0, now=0.0)
        await svc._plasticity.reinforce(r_a["id"], r_n["id"], score=1.0, now=1.0)
        state = await svc._plasticity.edge_state(r_a["id"], r_n["id"])
        assert state is not None
        assert state["weight"] > 0.0

        hits = await svc.recall("cue", session_id="s1", k=3, mode="episodic")
        ids_in_order = [int(h["id"]) for h in hits]
        # Anchor wins on raw cosine; the contested rank is between
        # neighbour (with spread boost) and distractor (without).
        assert ids_in_order.index(r_n["id"]) < ids_in_order.index(r_d["id"])
    finally:
        await db.close()
