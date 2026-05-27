"""ToolMetrics: declaration, recording, rolling latency, snapshot shape."""

from __future__ import annotations

import time

import pytest

from synara.features.memory.metrics import ToolMetrics


def test_declare_creates_zero_row_in_snapshot() -> None:
    m = ToolMetrics()
    m.declare("store_episode", headline="encode an episodic trace")

    snap = m.snapshot()
    assert len(snap) == 1
    row = snap[0]
    assert row.name == "store_episode"
    assert row.headline == "encode an episodic trace"
    assert row.count == 0
    assert row.error_count == 0
    assert row.last_called_at is None
    assert row.last_duration_seconds is None
    assert row.p50_ms is None
    assert row.p95_ms is None


def test_declare_is_idempotent_and_updates_headline() -> None:
    m = ToolMetrics()
    m.declare("recall_episodes", headline="first")
    m.declare("recall_episodes", headline="second")

    snap = m.snapshot()
    assert len(snap) == 1
    assert snap[0].headline == "second"


def test_record_updates_count_and_last_called() -> None:
    m = ToolMetrics()
    m.declare("store_episode", headline="encode an episodic trace")

    before = time.time()
    m.record("store_episode", 0.012, ok=True)
    after = time.time()

    row = m.snapshot()[0]
    assert row.count == 1
    assert row.error_count == 0
    assert row.last_duration_seconds == pytest.approx(0.012)
    assert row.last_called_at is not None
    assert before <= row.last_called_at <= after


def test_record_failure_increments_error_count() -> None:
    m = ToolMetrics()
    m.declare("forget_episodes", headline="prune")
    m.record("forget_episodes", 0.001, ok=True)
    m.record("forget_episodes", 0.002, ok=False)
    m.record("forget_episodes", 0.003, ok=False)

    row = m.snapshot()[0]
    assert row.count == 3
    assert row.error_count == 2


def test_record_late_declares_unknown_tool() -> None:
    """Calls for an undeclared tool still land, with name as headline fallback."""
    m = ToolMetrics()
    m.record("mystery_tool", 0.005, ok=True)

    row = m.snapshot()[0]
    assert row.name == "mystery_tool"
    assert row.headline == "mystery_tool"
    assert row.count == 1


def test_percentiles_use_nearest_rank() -> None:
    m = ToolMetrics()
    m.declare("recall_episodes", headline="recall")
    # 100 samples 1..100 ms — nearest-rank p50 picks the 50th sorted value,
    # p95 picks the 95th. Verifying both pins the algorithm choice.
    for i in range(1, 101):
        m.record("recall_episodes", i / 1000.0, ok=True)

    row = m.snapshot()[0]
    assert row.p50_ms == pytest.approx(50.0)
    assert row.p95_ms == pytest.approx(95.0)


def test_rolling_window_drops_oldest_samples() -> None:
    m = ToolMetrics(window_size=3)
    m.declare("memory_stats", headline="stats")
    for d in (10.0, 20.0, 30.0, 40.0):
        m.record("memory_stats", d / 1000.0, ok=True)

    # Window holds last 3 (20, 30, 40); p95 is the max under nearest-rank.
    row = m.snapshot()[0]
    assert row.count == 4  # count is total, not windowed
    assert row.p50_ms == pytest.approx(30.0)
    assert row.p95_ms == pytest.approx(40.0)


def test_window_size_must_be_positive() -> None:
    with pytest.raises(ValueError, match="window_size must be positive"):
        ToolMetrics(window_size=0)


def test_as_dict_round_trip_shape() -> None:
    m = ToolMetrics()
    m.declare("store_episode", headline="encode an episodic trace")
    m.record("store_episode", 0.05, ok=True)

    d = m.snapshot()[0].as_dict()
    assert set(d.keys()) == {
        "name",
        "headline",
        "count",
        "error_count",
        "last_called_at",
        "last_duration_seconds",
        "p50_ms",
        "p95_ms",
    }
    assert d["count"] == 1
    assert d["p50_ms"] == pytest.approx(50.0)
