"""PlasticityGraph guards + EventBus policy/prune + consolidate scoring."""

from __future__ import annotations

from collections.abc import AsyncIterator

import pytest
import pytest_asyncio
from simplevecdb import AsyncVectorDB

from synara.features.memory.basal_ganglia.events import (
    EventBus,
    InteractionEvent,
    ReactorState,
    TriggerPolicy,
)
from synara.features.memory.hippocampus.plasticity import PlasticityGraph
from synara.features.memory.neocortex.consolidate import _replay_score


@pytest_asyncio.fixture
async def db() -> AsyncIterator[AsyncVectorDB]:
    d = AsyncVectorDB(":memory:")
    try:
        yield d
    finally:
        await d.close()


def _make_graph(coll: object, **overrides: float | int) -> PlasticityGraph:
    base: dict[str, float | int] = {
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


async def _seed(coll: object, ids: list[int]) -> None:
    await coll.add_texts(  # type: ignore[attr-defined]
        texts=[f"d{i}" for i in ids],
        embeddings=[[float(i), 0.0, 0.0, 0.0] for i in ids],
        ids=list(ids),
    )


# ---- constructor guards ----------------------------------------------


@pytest.mark.parametrize(
    ("override", "match"),
    [
        ({"e_ltp_decay_seconds": 0.0}, "e_ltp_decay_seconds must be positive"),
        ({"l_ltp_threshold_hits": 0}, "l_ltp_threshold_hits must be positive"),
        ({"habit_threshold_hits": 0}, "habit_threshold_hits must be positive"),
        ({"habit_ltd_multiplier": 2.0}, "habit_ltd_multiplier must be in"),
        ({"habit_savings_factor": 0.5}, "habit_savings_factor must be >= 1"),
        ({"ltd_decay_per_idle_day": 2.0}, "ltd_decay_per_idle_day must be in"),
        ({"time_compression": 0.0}, "time_compression must be positive"),
    ],
)
def test_plasticity_ctor_rejects_bad_params(override: dict[str, float | int], match: str) -> None:
    with pytest.raises(ValueError, match=match):
        _make_graph(object(), **override)


# ---- reinforce / readers --------------------------------------------


async def test_reinforce_self_edge_is_noop(db: AsyncVectorDB) -> None:
    coll = db.collection("ep")
    await _seed(coll, [1])
    g = _make_graph(coll)
    await g.reinforce(1, 1, score=1.0, now=0.0)
    assert await g.edge_state(1, 1) is None


async def test_edge_weight_and_is_habit_on_missing_edge(db: AsyncVectorDB) -> None:
    coll = db.collection("ep")
    await _seed(coll, [1, 2])
    g = _make_graph(coll)
    assert await g.edge_weight(1, 2) == 0.0
    assert await g.is_habit(1, 2) is False
    await g.reinforce(1, 2, score=1.0, now=0.0)
    assert await g.edge_weight(1, 2) > 0.0


async def test_boost_combines_sr_absent_with_durable_edge(db: AsyncVectorDB) -> None:
    coll = db.collection("ep")
    await _seed(coll, [1, 2, 3])
    g = _make_graph(coll, l_ltp_threshold_hits=1)
    await g.reinforce(1, 2, score=1.0, now=0.0)
    out = await g.boost(1, [2, 3])
    assert out[2] > 0.0
    assert out[3] == 0.0
    assert await g.boost(1, []) == {}


async def test_ltd_pass_disabled_when_rate_zero(db: AsyncVectorDB) -> None:
    coll = db.collection("ep")
    await _seed(coll, [1, 2])
    g = _make_graph(coll, ltd_decay_per_idle_day=0.0)
    await g.reinforce(1, 2, score=1.0, now=0.0)
    assert await g.ltd_pass(now=10_000.0) == 0


async def test_ltd_full_decay_when_rate_one(db: AsyncVectorDB) -> None:
    coll = db.collection("ep")
    await _seed(coll, [1, 2])
    # rate 1.0, non-habit -> decay_per_day >= 1.0 -> weight zeroed, then
    # pruned because weight+bonus < prune_floor.
    g = _make_graph(coll, ltd_decay_per_idle_day=1.0, l_ltp_threshold_hits=1)
    await g.reinforce(1, 2, score=1.0, now=0.0)
    pruned = await g.ltd_pass(now=86_400.0)
    assert pruned == 1
    assert await g.edge_state(1, 2) is None


# ---- TriggerPolicy guards -------------------------------------------


def test_consolidate_due_disabled_when_threshold_non_positive() -> None:
    pol = TriggerPolicy(consolidate_after_novel_encodes=0)
    st = ReactorState(novel_encodes_since_consolidate=999)
    assert pol.consolidate_due(st, now=1e9) is False


def test_dream_due_false_without_events() -> None:
    pol = TriggerPolicy()
    assert pol.dream_due(ReactorState(events_since_dream=0), now=1e9) is False


def test_event_bus_rejects_non_positive_capacity() -> None:
    with pytest.raises(ValueError, match="log_capacity must be positive"):
        EventBus(log_capacity=0)


async def test_event_bus_prunes_persistent_log(db: AsyncVectorDB) -> None:
    coll = db.collection("ep")
    bus = EventBus(collection=coll, log_capacity=64)
    # 130 records -> two _maybe_prune cycles; the second has
    # last_seq - capacity > 0 so the real prune branch executes.
    for i in range(130):
        await bus.record(InteractionEvent(kind="encode", timestamp=float(i), session_id="s"))
    log = await bus.log()
    assert len(log) <= 64


# ---- consolidate replay score legacy path ---------------------------


def test_replay_score_legacy_meta_without_access_history() -> None:
    md = {"salience": 0.8, "encoded_at": 0.0, "last_accessed": 0.0, "retrieval_count": 2}
    s = _replay_score(md, d_near=1.0, margin=0.0, beta=0.0, now=10.0, d=0.5)
    assert s > 0.0


def test_replay_score_margin_suppresses_stable_episode() -> None:
    md = {"access_history": [0.0], "salience": 1.0}
    near = _replay_score(md, d_near=1.0, margin=0.0, beta=4.0, now=1.0, d=0.5)
    stable = _replay_score(md, d_near=1.0, margin=0.9, beta=4.0, now=1.0, d=0.5)
    assert stable < near
    assert stable >= 1e-9
