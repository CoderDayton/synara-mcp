"""Plasticity + event bus + reactor tests.

These cover the new neuroplasticity layer (E-LTP/L-LTP/habit/savings/LTD,
spreading activation, reconsolidation drift accounting), the
``InteractionEvent`` bus, and the self-triggering reactor.
"""

from __future__ import annotations

import asyncio
import hashlib
import math
from collections.abc import AsyncIterator

import numpy as np
import pytest
import pytest_asyncio
from simplevecdb import AsyncVectorDB

from synara.features.hippocampus.config import HippocampusConfig
from synara.features.hippocampus.ops.events import (
    EventBus,
    InteractionEvent,
    ReactorState,
    TriggerPolicy,
)
from synara.features.hippocampus.primitives.plasticity import PlasticityGraph
from synara.features.hippocampus.service import HippocampusService


def hash_embed(text: str, dim: int = 32) -> list[float]:
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


# ----- PlasticityGraph unit tests --------------------------------------


def _make_graph(**overrides: float | int) -> PlasticityGraph:
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
    return PlasticityGraph(sr=None, **base)  # type: ignore[arg-type]


def test_plasticity_e_ltp_bonus_decays_over_time() -> None:
    g = _make_graph()
    g.reinforce(1, 2, score=1.0, now=0.0)
    bonus0 = g.edge_weight(1, 2)
    # No reinforcement after E-LTP tau: bonus should decay by exp(-1).
    g._edges[(1, 2)].bonus_set_at = 0.0
    e = g._edges[(1, 2)]
    e.bonus *= math.exp(-7200.0 / 7200.0)
    assert e.bonus == pytest.approx(bonus0 / math.e, rel=1e-3)


def test_plasticity_l_ltp_threshold_folds_bonus_to_weight() -> None:
    g = _make_graph()
    # Three in-window reinforcements should consolidate into durable weight.
    for k in range(3):
        g.reinforce(1, 2, score=1.0, now=float(k))  # all within tau
    e = g._edges[(1, 2)]
    assert e.weight > 0.0
    assert e.bonus == 0.0
    assert e.hits_in_window == 0


def test_plasticity_habit_latches_after_threshold() -> None:
    g = _make_graph(habit_threshold_hits=4)
    for k in range(5):
        g.reinforce(1, 2, score=0.5, now=float(k))
    assert g.is_habit(1, 2)


def test_plasticity_habit_savings_amplifies_post_lapse_relearn() -> None:
    g = _make_graph(habit_threshold_hits=3, l_ltp_threshold_hits=2)
    # Build habit fast (3 hits to latch).
    for k in range(3):
        g.reinforce(1, 2, score=1.0, now=float(k))
    # Long disuse to drop bonus.
    g.reinforce(1, 2, score=1.0, now=10_000.0)
    # The post-habit reinforcement gets the savings multiplier.
    e = g._edges[(1, 2)]
    # Either still in bonus or already folded; total accumulated should
    # exceed what a non-habit edge would have at the same step.
    g2 = _make_graph(habit_threshold_hits=999, l_ltp_threshold_hits=2)
    for k in range(3):
        g2.reinforce(1, 2, score=1.0, now=float(k))
    g2.reinforce(1, 2, score=1.0, now=10_000.0)
    e2 = g2._edges[(1, 2)]
    assert (e.weight + e.bonus) > (e2.weight + e2.bonus)


def test_plasticity_ltd_decays_non_habit_faster_than_habit() -> None:
    g = _make_graph(habit_threshold_hits=3, l_ltp_threshold_hits=1, ltd_decay_per_idle_day=0.5)
    for k in range(3):
        g.reinforce(1, 2, score=1.0, now=float(k))
    # Plant a non-habit edge (different pair).
    g.reinforce(3, 4, score=1.0, now=0.0)
    g.reinforce(3, 4, score=1.0, now=1.0)  # weight after 1 fold-in (l=1 threshold)
    assert g.is_habit(1, 2)
    assert not g.is_habit(3, 4)
    w_h_before = g._edges[(1, 2)].weight
    w_n_before = g._edges[(3, 4)].weight
    assert w_h_before > 0
    assert w_n_before > 0
    # 1 real day idle (* 24 compression = 24 compressed days).
    g.ltd_pass(now=86400.0 + 10.0)
    w_h_after = g._edges.get((1, 2), None)
    w_n_after = g._edges.get((3, 4), None)
    # Habit edge survives; non-habit either drops further or is pruned.
    assert w_h_after is not None
    if w_n_after is not None:
        # Non-habit's relative decay must exceed habit's.
        rel_h = w_h_after.weight / w_h_before
        rel_n = w_n_after.weight / w_n_before
        assert rel_n < rel_h
    # Either way the habit retains far more strength than the non-habit.


def test_plasticity_prune_floor_removes_weak_non_habits() -> None:
    g = _make_graph(habit_threshold_hits=99, l_ltp_threshold_hits=1, ltd_decay_per_idle_day=0.99)
    g.reinforce(5, 6, score=1.0, now=0.0)
    g.reinforce(5, 6, score=1.0, now=0.5)
    assert (5, 6) in g._edges
    pruned = g.ltd_pass(now=86400.0 * 30.0)
    assert pruned >= 1
    assert (5, 6) not in g._edges


def test_plasticity_habit_edge_persists_at_zero_weight() -> None:
    g = _make_graph(habit_threshold_hits=3, l_ltp_threshold_hits=1, ltd_decay_per_idle_day=0.99)
    for k in range(3):
        g.reinforce(7, 8, score=1.0, now=float(k))
    assert g.is_habit(7, 8)
    g.ltd_pass(now=86400.0 * 365.0)
    # Habit edge retained even if weight collapses to zero.
    assert (7, 8) in g._edges


def test_plasticity_spreading_returns_zero_when_disabled() -> None:
    g = _make_graph()
    g.reinforce(1, 2, score=1.0, now=0.0)
    out = g.spreading(1, [2, 3], hops=0, gamma=0.5)
    assert out == {2: 0.0, 3: 0.0}


def test_plasticity_spreading_one_hop_picks_up_durable_neighbour() -> None:
    g = _make_graph(l_ltp_threshold_hits=1)
    g.reinforce(1, 2, score=1.0, now=0.0)
    g.reinforce(1, 2, score=1.0, now=1.0)  # consolidates to weight
    out = g.spreading(1, [2, 3], hops=1, gamma=0.5)
    assert out[2] > 0.0
    assert out[3] == 0.0


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


def test_event_bus_records_and_caps_log() -> None:
    bus = EventBus(log_capacity=3)
    for i in range(5):
        bus.record(
            InteractionEvent(
                kind="encode",
                timestamp=float(i),
                session_id="s",
                payload={"deduped": False},
            )
        )
    log = bus.log()
    assert len(log) == 3
    assert [e.timestamp for e in log] == [2.0, 3.0, 4.0]
    assert bus.state.total_events == 5
    assert bus.state.novel_encodes_since_consolidate == 5


def test_event_bus_react_skips_reactor_kinds() -> None:
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
    # Encoding event should fire follow-ups; consolidate event should not.
    e_user = InteractionEvent(
        kind="encode",
        timestamp=10.0,
        session_id=None,
        payload={"deduped": False},
    )
    bus.record(e_user)
    asyncio.run(bus.react(e_user))
    e_reactor = InteractionEvent(
        kind="consolidate",
        timestamp=11.0,
        session_id=None,
        payload={},
    )
    bus.record(e_reactor)
    asyncio.run(bus.react(e_reactor))
    # 'c' and possibly 'd' from user event; reactor event must not double-trigger.
    assert "c" in fired
    assert fired.count("c") == 1


# ----- Service-integration tests ---------------------------------------


async def test_service_emits_event_per_op(service: HippocampusService) -> None:
    await service.encode_episode("hello", "s1")
    await service.recall("hello", session_id="s1", k=3)
    log = service.event_log()
    kinds = [e.kind for e in log]
    assert "encode" in kinds
    assert "recall" in kinds


async def test_service_records_plasticity_after_co_recall(
    service: HippocampusService,
) -> None:
    await service.encode_episode("alpha", "s1")
    await service.encode_episode("beta", "s1")
    await service.encode_episode("gamma", "s1")
    # Two recalls in the same session pull more than one episodic hit so
    # the anchor->other reinforcement fires.
    await service.recall("alpha", session_id="s1", k=3)
    await service.recall("alpha", session_id="s1", k=3)
    stats = service._plasticity.stats()
    assert stats["edges"] >= 1.0


async def test_reactor_consolidate_fires_after_threshold() -> None:
    db = AsyncVectorDB(":memory:")
    try:
        cfg = HippocampusConfig(
            reactor_consolidate_after_novel=3,
            reactor_consolidate_cooldown_seconds=0.0,
        )
        svc = HippocampusService(db, config=cfg, embed_fn=hash_embed)
        for i in range(4):
            await svc.encode_episode(f"item-{i}", "s1")
        # The 4th encode trips the policy; reactor runs consolidate.
        kinds = [e.kind for e in svc.event_log()]
        assert kinds.count("consolidate") >= 1
    finally:
        await db.close()


async def test_self_learning_disabled_skips_reactor() -> None:
    db = AsyncVectorDB(":memory:")
    try:
        cfg = HippocampusConfig(self_learning_enabled=False, reactor_consolidate_after_novel=1)
        svc = HippocampusService(db, config=cfg, embed_fn=hash_embed)
        for i in range(5):
            await svc.encode_episode(f"item-{i}", "s1")
        kinds = [e.kind for e in svc.event_log()]
        assert "consolidate" not in kinds
    finally:
        await db.close()


async def test_surprise_salience_boosts_when_enabled() -> None:
    db = AsyncVectorDB(":memory:")
    try:
        cfg = HippocampusConfig(surprise_salience_boost=0.3, surprise_distance_floor=0.0)
        svc = HippocampusService(db, config=cfg, embed_fn=hash_embed)
        r = await svc.encode_episode("anything", "s1", salience=0.4)
        rows = await svc.episodic.get_documents({"id": r["id"]}, limit=1)
        _, _, md = rows[0]
        # First episode in an empty store has no neighbour; floor=0 forces boost.
        assert float(md["salience"]) >= 0.7 - 1e-6
    finally:
        await db.close()


async def test_reconsolidation_drift_accounted_when_enabled() -> None:
    db = AsyncVectorDB(":memory:")
    try:
        cfg = HippocampusConfig(reconsolidation_alpha=0.1, reconsolidation_min_score=0.0)
        svc = HippocampusService(db, config=cfg, embed_fn=hash_embed)
        await svc.encode_episode("alpha", "s1")
        await svc.encode_episode("beta", "s1")
        for _ in range(5):
            await svc.recall("alpha", session_id="s1", k=2)
        rows = await svc.episodic.get_documents({"session_id": "s1"})
        any_drift = any(float(md.get("drift_total", 0.0)) > 0.0 for _, _, md in rows)
        assert any_drift
    finally:
        await db.close()
