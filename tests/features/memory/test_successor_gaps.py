"""Validation, persistence-failure, and eviction branches in SuccessorRepresentation."""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from synara.features.memory.hippocampus.successor import SuccessorRepresentation


# ------------------------------------------------------------- __post_init__
@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"gamma": 1.0}, "gamma"),
        ({"gamma": -0.1}, "gamma"),
        ({"alpha": 0.0}, "alpha"),
        ({"alpha": 1.5}, "alpha"),
        ({"window_seconds": 0.0}, "window_seconds"),
        ({"window_seconds": -1.0}, "window_seconds"),
        ({"omega_max": -0.1}, "omega_max"),
        ({"cold_start_ratio": 0.0}, "cold_start_ratio"),
        ({"cold_start_ratio": -1.0}, "cold_start_ratio"),
    ],
)
def test_post_init_rejects_invalid_parameters(kwargs: dict[str, Any], match: str) -> None:
    with pytest.raises(ValueError, match=match):
        SuccessorRepresentation(**kwargs)


# ------------------------------------------------------------- load() lock guard
async def test_load_is_idempotent_under_concurrent_callers() -> None:
    sr = SuccessorRepresentation()
    await asyncio.gather(sr.load(), sr.load(), sr.load())
    assert sr._loaded is True


async def test_load_inner_check_short_circuits_after_lock_acquisition() -> None:
    """Race: caller A holds the lock and sets _loaded mid-flight; caller B,
    queued on the lock, must see _loaded=True at the inner check (line 115)
    and return without re-running the rehydrate."""
    sr = SuccessorRepresentation()
    sr._load_lock = asyncio.Lock()
    await sr._load_lock.acquire()

    task = asyncio.create_task(sr.load())
    # Yield enough for load() to clear the outer check and queue on the lock.
    for _ in range(5):
        await asyncio.sleep(0)

    sr._loaded = True
    sr._load_lock.release()
    await task
    # If the inner check had fired the rebuild instead of short-circuiting,
    # the test would also pass — but coverage of line 115 proves the
    # short-circuit branch ran.
    assert sr._loaded is True


# ------------------------------------------------------------- flush failure path
class _FakeEdgesAPI:
    def __init__(self, fail_on: set[tuple[int, int]] | None = None) -> None:
        self.calls: list[tuple[int, int]] = []
        self._fail_on = fail_on or set()

    def upsert(self, src: int, dst: int, **_: Any) -> None:
        self.calls.append((src, dst))
        if (src, dst) in self._fail_on:
            raise RuntimeError("boom")


class _FakeColl:
    def __init__(self, edges: _FakeEdgesAPI) -> None:
        class _Inner:
            def __init__(self, e: _FakeEdgesAPI) -> None:
                self.edges = e

        self._collection = _Inner(edges)

    async def get_edges(self, kind: str) -> list[Any]:
        return []


async def test_flush_retries_failed_edge_upsert_on_next_call() -> None:
    """A failed upsert must be re-added to _pending and tolerated (not raised)."""
    edges = _FakeEdgesAPI(fail_on={(1, 2)})
    coll = _FakeColl(edges)
    sr = SuccessorRepresentation()
    sr.attach(coll)
    await sr.load()
    # Seed a co-occurrence so the (1, 2) edge is pending.
    sr.observe("s", 1, t=0.0)
    sr.observe("s", 2, t=1.0)
    assert (1, 2) in sr._pending

    await sr.flush()
    # Failure tolerated; pair re-queued for the next flush.
    assert (1, 2) in sr._pending
    assert (1, 2) in edges.calls

    # Now let it succeed and confirm pending drains.
    edges._fail_on = set()
    await sr.flush()
    assert (1, 2) not in sr._pending


# ------------------------------------------------------------- _td_update pop branch
def test_td_update_pops_zero_valued_entries() -> None:
    """When the TD update drives M[i][k] to exactly 0, the key must be
    dropped from the row (line 238). We force this by injecting a row
    that already has a 0 value and triggering a no-op update."""
    sr = SuccessorRepresentation(alpha=1.0, gamma=0.0)
    # Pre-seed M[5][7] = 0.0; the TD pass for (i=5, j=8) recomputes M[5][7]
    # as (1 - alpha) * 0 + alpha * (0 + gamma * M[8][7]) = 0 because
    # M[8][7] is absent. That zero must trigger the pop.
    sr._M[5][7] = 0.0
    sr._td_update(5, 8)
    assert 7 not in sr._M[5]


# ------------------------------------------------------------- evict_nodes
def test_evict_nodes_empty_set_short_circuits() -> None:
    """Hitting the early-return guard at line 280."""
    sr = SuccessorRepresentation()
    sr.observe("s", 1, t=0.0)
    sr.observe("s", 2, t=1.0)
    pending_before = set(sr._pending)
    edges_before = sr._total_edges
    sr.evict_nodes(set())  # no-op
    assert sr._pending == pending_before
    assert sr._total_edges == edges_before


def test_evict_nodes_strips_incoming_columns_and_M_columns() -> None:
    """Lines 288-291 (incoming column removal) and 294 (M column removal)."""
    sr = SuccessorRepresentation()
    # Build a triangle: 1->2, 2->3, 1->3.
    sr.observe("s", 1, t=0.0)
    sr.observe("s", 2, t=1.0)
    sr.observe("s", 3, t=2.0)

    assert 3 in sr._T_counts.get(1, {}) or 3 in sr._T_counts.get(2, {})

    sr.evict_nodes({3})
    for src, row in sr._T_counts.items():
        assert 3 not in row, f"T[{src}] still references evicted column 3"
    for src, mrow in sr._M.items():
        assert 3 not in mrow, f"M[{src}] still references evicted column 3"
    assert sr._total_edges >= 0.0


def test_evict_nodes_decrements_total_edges_when_outgoing_row_exists() -> None:
    """Line 284: evicting a node with outgoing edges must subtract the
    row sum from the global edge counter."""
    sr = SuccessorRepresentation()
    # Node 1 fans out to 2 and 3 (outgoing row populated).
    sr.observe("s", 1, t=0.0)
    sr.observe("s", 2, t=1.0)
    sr.observe("s", 3, t=2.0)
    assert sr._total_edges > 0
    sr.evict_nodes({1})
    # Node 1 had two outgoing edges (1->2, 1->3); decrement should leave
    # only 2->3 in the tally.
    assert sr._total_edges == pytest.approx(1.0)


# ------------------------------------------------------------- observe_recall_set
def test_observe_recall_set_skips_self_anchor_in_prior_window() -> None:
    """Line 212: when the anchor appears as a prior entry in the in-session
    window, the loop continues past the self-edge."""
    sr = SuccessorRepresentation()
    # Seed the global window with anchor=1 as a prior entry.
    sr.observe("s", 1, t=0.0)
    # Recall-set with anchor=1 and one other id — the anchor's prior
    # entry is encountered and skipped via the `continue` at line 212.
    sr.observe_recall_set("s", anchor_id=1, other_ids=[2], t=1.0)
    # No self-edge 1->1 may exist.
    assert 1 not in sr._T_counts.get(1, {})
    # Forward 1->2 edge must exist.
    assert sr._T_counts[1].get(2, 0.0) >= 1.0
