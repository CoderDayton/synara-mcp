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


async def test_plasticity_dream_cadence_folds_repeated_replay_durable(
    db: AsyncVectorDB,
) -> None:
    """Repeated dream replay of one trace must fold it durable (L-LTP).

    Models the ``_reactor_dream`` ordering: ``ltd_pass`` runs before
    each off-policy replay ``reinforce``. A replay edge is bonus-only
    (weight 0) until ``l_ltp_threshold_hits`` in-window reinforcements
    fold it. The intervening LTD pass must not cull a still-potentiated
    (E-LTP bonus above floor) edge, or the in-window counter resets
    every dream and the fold can never fire.
    """
    coll = db.collection("ep")
    await _seed_docs(coll, [1, 2])
    g = _make_graph(
        coll,
        habit_threshold_hits=99,  # never a habit -> only weight protects it
        l_ltp_threshold_hits=3,
        e_ltp_decay_seconds=7200.0,
        ltd_decay_per_idle_day=0.02,
    )
    # Three dreams 1800s apart (idle cadence; well inside the 2h E-LTP
    # window), each: LTD pass, then one off-policy replay reinforce.
    for k in range(3):
        t = float(k) * 1800.0
        await g.ltd_pass(now=t)
        await g.reinforce(1, 2, score=0.5, now=t)
    state = await g.edge_state(1, 2)
    assert state is not None, "replay edge culled before L-LTP could fold it"
    assert state["weight"] > 0.0, "repeated replay never folded to durable weight"


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
    triggered = await bus.react(e_user)
    # The user event is due for *both* consolidate and dream; react must
    # fire both (and report them) on the single user event.
    assert "c" in fired
    assert "d" in fired
    assert triggered == ["consolidate", "dream"]
    e_reactor = InteractionEvent(
        kind="consolidate",
        timestamp=11.0,
        session_id=None,
        payload={},
    )
    await bus.record(e_reactor)
    # A reactor-kind event must not re-trigger.
    assert await bus.react(e_reactor) == []
    assert fired.count("c") == 1


# ----- Service-integration tests ---------------------------------------


async def test_service_emits_event_per_op(service: MemoryService) -> None:
    await service.encode_episode("hello", "s1")
    await service.recall("hello", session_id="s1", k=3)
    log = await service.event_log()
    kinds = [e.kind for e in log]
    assert "encode" in kinds
    assert "recall" in kinds


async def test_service_records_plasticity_after_co_recall() -> None:
    # Mechanics test: the relevance gate would drop the orthogonal hash-embed
    # co-recalls, so disable it and verify co-recall writes plasticity edges.
    db = AsyncVectorDB(":memory:")
    try:
        cfg = MemoryConfig(recall_relevance_gate=False)
        service = MemoryService(db, config=cfg, embed_fn=hash_embed)
        await service.encode_episode("alpha", "s1")
        await service.encode_episode("beta", "s1")
        await service.encode_episode("gamma", "s1")
        await service.recall("alpha", session_id="s1", k=3)
        await service.recall("alpha", session_id="s1", k=3)
        stats = await service._plasticity.stats()
        assert stats["edges"] >= 1.0
    finally:
        await db.close()


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
        # Consolidation runs as a background task; drain before reading
        # the log so the assertion is deterministic, not a scheduling race.
        await svc.drain_reactor_tasks()
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
            # Fast-promote-on-park: this test is about the age gate,
            # not the recurrence gate.
            consolidate_min_recurrence=1,
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
            # Fast-promote-on-park: this test is about the retrieval-count
            # gate, not the recurrence gate.
            consolidate_min_recurrence=1,
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
            # Mechanics test: isolate ranking from the relevance gate, which
            # would otherwise drop the equidistant neighbour and distractor.
            recall_relevance_gate=False,
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


async def test_dream_replay_reinforces_offpolicy_associations() -> None:
    """The dream reactor rehearses high-priority unconsolidated episodes
    off-policy: within-session associations are reinforced without any
    live recall, and the dream event reports the replayed count."""
    db = AsyncVectorDB(":memory:")
    try:
        # High consolidate threshold keeps episodes UNCONSOLIDATED so the
        # replay pass has a population to rehearse.
        cfg = MemoryConfig(
            reactor_consolidate_after_novel=999,
            dg_pattern_separation=False,  # keep 3 distinct episodes
            dream_replay_top_k=16,
            dream_replay_gain=0.3,
        )
        svc = MemoryService(db, config=cfg, embed_fn=hash_embed)
        for word in ("alpha", "bravo", "charlie"):
            await svc.encode_episode(word, "s1", salience=0.9)

        # No recall has happened -> no plasticity edges exist yet.
        assert (await svc._plasticity.stats())["edges"] == 0.0

        await svc._reactor_dream(None)

        # Off-policy rehearsal created the within-session edges.
        assert (await svc._plasticity.stats())["edges"] >= 1.0
        log = await svc.event_log()
        dream = next(e for e in reversed(log) if e.kind == "dream")
        assert dream.payload["replayed"] == 2  # anchor + 2 targets
    finally:
        await db.close()


async def test_dream_replay_disabled_when_top_k_zero() -> None:
    """``dream_replay_top_k=0`` restores the legacy LTD-only dream."""
    db = AsyncVectorDB(":memory:")
    try:
        cfg = MemoryConfig(
            reactor_consolidate_after_novel=999,
            dg_pattern_separation=False,
            dream_replay_top_k=0,
        )
        svc = MemoryService(db, config=cfg, embed_fn=hash_embed)
        for word in ("alpha", "bravo", "charlie"):
            await svc.encode_episode(word, "s1", salience=0.9)

        await svc._reactor_dream(None)

        assert (await svc._plasticity.stats())["edges"] == 0.0
        log = await svc.event_log()
        dream = next(e for e in reversed(log) if e.kind == "dream")
        assert dream.payload["replayed"] == 0
    finally:
        await db.close()


# ----- Sim-parity profile regression ----------------------------------
#
# scripts/sim/plasticity_sim.py validates the *intended* dynamics of the
# shipped MemoryConfig constants in a standalone Python model. These
# tests assert the same four qualitative sanity properties hold against
# the real async PlasticityGraph driven with the real default constants
# (esp. habit_threshold_hits=66), so a constant change that breaks the
# sim is also caught in the runtime. Hits are concentrated on one edge
# (the sim spreads them over ~15) to keep the cumulative threshold
# reachable at unit-test volume; `now` is in the sim's app-second units
# (sec_per_turn_app = 86400 / time_compression / turns_per_day).


def _cfg_graph(coll: object) -> PlasticityGraph:
    """PlasticityGraph wired from the shipped MemoryConfig defaults."""
    c = MemoryConfig()
    return PlasticityGraph(
        collection=coll,
        sr=None,
        e_ltp_decay_seconds=c.e_ltp_decay_seconds,
        l_ltp_threshold_hits=c.l_ltp_threshold_hits,
        habit_threshold_hits=c.habit_threshold_hits,
        habit_ltd_multiplier=c.habit_ltd_multiplier,
        habit_savings_factor=c.habit_savings_factor,
        ltd_decay_per_idle_day=c.ltd_decay_per_idle_day,
        time_compression=c.time_compression,
    )


def _step_app(turns_per_day: int) -> float:
    return (86400.0 / MemoryConfig().time_compression) / turns_per_day


def _idle_app(real_days: float) -> float:
    return real_days * 86400.0 / MemoryConfig().time_compression


async def test_sim_steady_profile_forms_a_habit(db: AsyncVectorDB) -> None:
    """steady (dense, no gaps) crosses the cumulative habit threshold."""
    coll = db.collection("ep")
    await _seed_docs(coll, [1, 2])
    g = _cfg_graph(coll)
    thr = MemoryConfig().habit_threshold_hits
    step = _step_app(30)  # 30 turns/day, as profile_steady
    for k in range(thr + 8):
        await g.reinforce(1, 2, score=0.85, now=k * step)
    assert await g.is_habit(1, 2)
    s = await g.edge_state(1, 2)
    assert s is not None
    assert s["hits"] >= thr
    assert s["weight"] > 0.0  # L-LTP folded within the E-LTP window


async def test_sim_bursty_profile_forms_a_habit_despite_gaps(
    db: AsyncVectorDB,
) -> None:
    """Dense bursts separated by 3 real-day idle gaps still latch a
    habit: the hit count is cumulative and gap-independent, and folded
    weight survives the gap via habit-protected LTD. This is the runtime
    analogue of the sim's bursty profile after the density fix."""
    coll = db.collection("ep")
    await _seed_docs(coll, [1, 2])
    g = _cfg_graph(coll)
    thr = MemoryConfig().habit_threshold_hits
    step = _step_app(120)  # dense block, as the fixed profile_bursty
    now = 0.0
    blocks = 5
    per_block = thr // blocks + 4  # total > thr across the gaps
    for _ in range(blocks):
        for _ in range(per_block):
            await g.reinforce(1, 2, score=0.85, now=now)
            now += step
        now += _idle_app(3.0)  # 3 real-day gap
        await g.ltd_pass(now=now)
    assert await g.is_habit(1, 2)
    s = await g.edge_state(1, 2)
    assert s is not None  # not pruned despite the gaps
    assert s["weight"] > 0.0


async def test_sim_sparse_profile_never_forms_a_habit(
    db: AsyncVectorDB,
) -> None:
    """Sparse usage (few, widely spaced reinforcements) stays well under
    the cumulative threshold and never latches a habit."""
    coll = db.collection("ep")
    await _seed_docs(coll, [1, 2])
    g = _cfg_graph(coll)
    now = 0.0
    for _ in range(12):  # << habit_threshold_hits (66)
        await g.reinforce(1, 2, score=0.85, now=now)
        now += _idle_app(2.0)
        await g.ltd_pass(now=now)
    assert not await g.is_habit(1, 2)


async def test_sim_disuse_then_resume_habit_survives_gap(
    db: AsyncVectorDB,
) -> None:
    """A latched habit survives long disuse (habit-protected LTD keeps
    weight > 0 and the savings flag), while a parallel weak non-habit
    edge is pruned over the same idle. Mirrors the sim's
    disuse-then-resume check (habW > 0)."""
    coll = db.collection("ep")
    await _seed_docs(coll, [1, 2, 3, 4])
    g = _cfg_graph(coll)
    thr = MemoryConfig().habit_threshold_hits
    step = _step_app(30)
    now = 0.0
    for _ in range(thr + 4):
        await g.reinforce(1, 2, score=0.85, now=now)
        now += step
    # Weak, never-consolidated companion edge.
    await g.reinforce(3, 4, score=0.85, now=now)
    await g.reinforce(3, 4, score=0.85, now=now + step)
    assert await g.is_habit(1, 2)

    # 90 real days of disuse, then the offline LTD pass.
    now += _idle_app(90.0)
    await g.ltd_pass(now=now)

    s_habit = await g.edge_state(1, 2)
    assert s_habit is not None  # habit edge persists
    assert s_habit["ever_habit"]
    assert s_habit["weight"] > 0.0  # habit_ltd_multiplier protected it
    assert await g.edge_state(3, 4) is None  # weak edge pruned
    w_trough = s_habit["weight"]

    # Resume: relearning lifts the habit back above its post-disuse
    # trough (savings vs a non-habit edge is covered separately by
    # test_plasticity_habit_savings_amplifies_post_lapse_relearn).
    for _ in range(MemoryConfig().l_ltp_threshold_hits):
        await g.reinforce(1, 2, score=0.85, now=now)
        now += step
    w_after = (await g.edge_state(1, 2))["weight"]  # type: ignore[index]
    assert w_after > w_trough
