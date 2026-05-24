"""Deterministic concurrency + correctness tests for the dashboard graph
route helpers refactored to use :func:`asyncio.gather`.

Two kinds of proof are required and produced here:

1. **Concurrency proof** — a stub ``coll`` that gates every
   ``get_edges`` call on a barrier event. The barrier is set only
   after N calls have entered concurrently. If the code under test
   sequentialises its awaits the barrier never fires and
   :func:`asyncio.wait_for` times out — the test fails loudly instead
   of silently passing. With true ``gather``-style fan-out all N calls
   arrive in the same event-loop tick, the barrier fires, and the
   coroutines complete. The timeout is generous (2 s) so it cannot
   pass by accident under any plausible scheduler jitter.

2. **Correctness proof** — for the same fixture graph, the refactored
   implementation produces the same node set, the same SR edge set
   (modulo dedup), and the same plasticity edge list (modulo
   reproducible ordering) as a literal sequential baseline inlined in
   this file. The baseline is a verbatim copy of the pre-refactor
   logic; any future divergence would be caught here, not in
   production.
"""

from __future__ import annotations

import asyncio
from typing import Any

import synara.features.dashboard.routes.graph as graph_mod


# ============================================================ test fixtures
class _FakeEdge:
    __slots__ = ("bonus", "dst_id", "hits", "src_id", "weight")

    def __init__(
        self,
        src: int,
        dst: int,
        *,
        hits: int = 1,
        weight: float = 0.0,
        bonus: float = 0.0,
    ) -> None:
        self.src_id = src
        self.dst_id = dst
        self.hits = hits
        self.weight = weight
        self.bonus = bonus


class _FakeColl:
    """Minimal sync-completing stand-in. Identical query semantics to
    the one in :mod:`tests.features.dashboard.test_graph_internals`."""

    def __init__(
        self,
        sr_edges: list[_FakeEdge] | None = None,
        plast_edges: list[_FakeEdge] | None = None,
    ) -> None:
        self._sr = sr_edges or []
        self._plast = plast_edges or []

    async def get_edges(
        self,
        *,
        kind: str,
        src: int | None = None,
        dst: int | None = None,
        limit: int | None = None,
    ) -> list[_FakeEdge]:
        pool = self._sr if kind == "sr" else self._plast
        out = [
            e for e in pool if (src is None or e.src_id == src) and (dst is None or e.dst_id == dst)
        ]
        if limit is not None:
            out = out[:limit]
        return out


class _GatedColl:
    """Wraps a :class:`_FakeColl` and forces every ``get_edges`` call
    to await a per-stage barrier. The barrier for stage ``i`` fires
    only after ``schedule[i]`` calls have arrived simultaneously,
    which proves the caller scheduled them concurrently rather than
    sequentially. After firing, the gate advances to the next stage
    with a fresh barrier — supporting multi-layer BFS where each layer
    has its own expected fan-out.

    A ``schedule`` of ``[4]`` means: deadlock unless 4 calls arrive
    concurrently, then all subsequent calls pass through immediately.
    A ``schedule`` of ``[4, 12]`` means: first 4 calls form layer 1,
    next 12 calls form layer 2, after which the gate is open.

    Tracks ``peak_in_flight`` so an assertion can verify the observed
    parallelism after the calls drain.
    """

    def __init__(self, inner: _FakeColl, schedule: list[int]) -> None:
        if not schedule:
            raise ValueError("schedule must contain at least one stage")
        self._inner = inner
        self._schedule = list(schedule)
        self._stage = 0
        self._stage_entered = 0
        self.in_flight = 0
        self.peak_in_flight = 0
        self.total_calls = 0
        self._barrier = asyncio.Event()

    async def get_edges(self, **kw: Any) -> list[_FakeEdge]:
        self.total_calls += 1
        self.in_flight += 1
        self.peak_in_flight = max(self.peak_in_flight, self.in_flight)
        # Capture the active barrier *before* possibly rotating it so
        # the threshold-tripping call also waits on the (now-set) old
        # barrier rather than the fresh one. All N coroutines awaiting
        # the same Event proceed in the same tick once it's set.
        active = self._barrier
        if self._stage < len(self._schedule):
            self._stage_entered += 1
            if self._stage_entered >= self._schedule[self._stage]:
                active.set()
                self._stage += 1
                self._stage_entered = 0
                self._barrier = asyncio.Event()
        else:
            # Schedule exhausted: open gate for all further calls.
            active.set()
        await active.wait()
        self.in_flight -= 1
        return await self._inner.get_edges(**kw)


# ============================================================ sequential baseline
async def _seq_adjacent(
    coll: Any,
    node: int,
    limit: int,
    sr_seen: dict[tuple[int, int], dict[str, Any]],
) -> set[int]:
    """Verbatim copy of the pre-refactor ``_adjacent`` body, used as a
    reference for correctness assertions."""
    out: set[int] = set()
    for e in await coll.get_edges(src=node, kind="sr", limit=limit):
        sr_seen.setdefault((int(e.src_id), int(e.dst_id)), graph_mod._sr_edge(e))
        out.add(int(e.dst_id))
    for e in await coll.get_edges(dst=node, kind="sr", limit=limit):
        sr_seen.setdefault((int(e.src_id), int(e.dst_id)), graph_mod._sr_edge(e))
        out.add(int(e.src_id))
    for e in await coll.get_edges(src=node, kind="plasticity", limit=limit):
        out.add(int(e.dst_id))
    for e in await coll.get_edges(dst=node, kind="plasticity", limit=limit):
        out.add(int(e.src_id))
    return out


async def _seq_focus_neighborhood(
    coll: Any, focus: int, depth: int, max_nodes: int
) -> tuple[set[int], list[dict[str, Any]]]:
    """Sequential, sorted-frontier baseline. We sort the frontier here
    so the comparison with the parallel implementation is on equal
    footing; the pre-refactor production code iterated a set in
    undefined order, so it had no canonical baseline of its own."""
    nodes: set[int] = {focus}
    sr_seen: dict[tuple[int, int], dict[str, Any]] = {}
    frontier = {focus}
    for _ in range(depth):
        if len(nodes) >= max_nodes:
            break
        nxt: set[int] = set()
        for node in sorted(frontier):
            neighbours = await _seq_adjacent(coll, node, max_nodes, sr_seen)
            for nb in sorted(neighbours):
                if len(nodes) >= max_nodes:
                    break
                if nb not in nodes:
                    nodes.add(nb)
                    nxt.add(nb)
        frontier = nxt
        if not frontier:
            break
    edges = [e for (s, d), e in sr_seen.items() if s in nodes and d in nodes]
    return nodes, edges


async def _seq_plasticity_overlay(
    coll: Any, nodes: set[int], habit_threshold: int
) -> list[dict[str, Any]]:
    edges: list[dict[str, Any]] = []
    for node in sorted(nodes):
        for e in await coll.get_edges(src=node, kind="plasticity"):
            if int(e.dst_id) not in nodes:
                continue
            weight, bonus, hits = float(e.weight), float(e.bonus), int(e.hits)
            edges.append(
                {
                    "src": int(e.src_id),
                    "dst": int(e.dst_id),
                    "hits": hits,
                    "weight": weight,
                    "bonus": bonus,
                    "strength": weight + bonus,
                    "is_habit": hits >= habit_threshold,
                }
            )
    return edges


# ============================================================ _adjacent
async def test_adjacent_returns_neighbors_and_sr_edges() -> None:
    """Signature change: returns ``(set[int], list[dict])`` instead of
    mutating a caller-owned dict. SR edges are reported exactly once
    per (src, dst) pair *per direction* of the issuing query — caller
    dedups across nodes."""
    coll = _FakeColl(
        sr_edges=[_FakeEdge(0, 1, hits=2), _FakeEdge(3, 0, hits=4)],
        plast_edges=[_FakeEdge(0, 5), _FakeEdge(6, 0)],
    )
    neighbours, sr_edges = await graph_mod._adjacent(coll, 0, limit=10)
    assert neighbours == {1, 3, 5, 6}
    # Two SR queries (src=0, dst=0) each return one edge → two reports.
    assert sorted((e["src"], e["dst"]) for e in sr_edges) == [(0, 1), (3, 0)]


async def test_adjacent_issues_four_queries_concurrently() -> None:
    """The 4 internal ``get_edges`` calls (sr-out, sr-in, plast-out,
    plast-in) must reach the event loop in the same tick. If any pair
    is awaited sequentially, the stage-1 barrier (size 4) never fires
    and the test times out."""
    gated = _GatedColl(_FakeColl(), schedule=[4])
    # Wrap in wait_for so a regression to sequential awaits surfaces as
    # a hard TimeoutError instead of a hang.
    neighbours, _ = await asyncio.wait_for(graph_mod._adjacent(gated, 0, limit=10), timeout=2.0)
    assert neighbours == set()
    assert gated.total_calls == 4
    assert gated.peak_in_flight == 4


# ============================================================ _focus_neighborhood concurrency
async def test_focus_neighborhood_fans_out_across_frontier() -> None:
    """Layer 1 has frontier {0} → 4 concurrent ``get_edges``. Layer 2
    has frontier {1, 2, 3} → 12 concurrent ``get_edges``. The gated
    coll's per-stage schedule [4, 12] deadlocks if EITHER layer fails
    to fan out."""
    sr_edges = [
        _FakeEdge(0, 1),
        _FakeEdge(0, 2),
        _FakeEdge(0, 3),
        # Layer 2 has no further successors → BFS terminates after
        # layer 2 issues its 12 calls.
    ]
    inner = _FakeColl(sr_edges=sr_edges)

    # Single-layer probe at depth=1.
    g1 = _GatedColl(inner, schedule=[4])
    nodes_a, _ = await asyncio.wait_for(
        graph_mod._focus_neighborhood(g1, focus=0, depth=1, max_nodes=100),
        timeout=2.0,
    )
    assert nodes_a == {0, 1, 2, 3}
    assert g1.peak_in_flight >= 4

    # Two-layer probe: both layers must fan out concurrently.
    g2 = _GatedColl(inner, schedule=[4, 12])
    nodes_b, _ = await asyncio.wait_for(
        graph_mod._focus_neighborhood(g2, focus=0, depth=2, max_nodes=100),
        timeout=2.0,
    )
    assert nodes_b == {0, 1, 2, 3}
    # peak_in_flight reflects the busiest layer; depth=2 should hit 12.
    assert g2.peak_in_flight >= 12


# ============================================================ _focus_neighborhood correctness
async def test_focus_neighborhood_matches_sequential_baseline() -> None:
    """Same fixture, same query → same nodes and same SR edges as the
    sorted-frontier sequential baseline."""
    sr_edges = [
        _FakeEdge(0, 1, hits=5),
        _FakeEdge(0, 2, hits=3),
        _FakeEdge(1, 4, hits=2),
        _FakeEdge(2, 4, hits=1),
        _FakeEdge(4, 5, hits=1),
        _FakeEdge(5, 0, hits=1),  # back-edge to anchor
        _FakeEdge(7, 0, hits=1),  # incoming to anchor (dst=0)
    ]
    plast_edges = [
        _FakeEdge(1, 10, hits=4, weight=0.8),
        _FakeEdge(2, 11, hits=2, weight=0.4),
    ]

    coll_a = _FakeColl(sr_edges, plast_edges)
    coll_b = _FakeColl(sr_edges, plast_edges)

    par_nodes, par_edges = await graph_mod._focus_neighborhood(
        coll_a, focus=0, depth=3, max_nodes=50
    )
    seq_nodes, seq_edges = await _seq_focus_neighborhood(coll_b, focus=0, depth=3, max_nodes=50)

    assert par_nodes == seq_nodes
    # Compare edges as sets of (src, dst, hits) — both sides may emit
    # the same edges in different order, but the *content* must match.
    par_keys = {(e["src"], e["dst"], e["hits"]) for e in par_edges}
    seq_keys = {(e["src"], e["dst"], e["hits"]) for e in seq_edges}
    assert par_keys == seq_keys


async def test_focus_neighborhood_deterministic_across_runs() -> None:
    """Repeated runs of the same query against equivalent fixtures
    produce identical results. Targets the cap-boundary nondeterminism
    that the sorted-frontier merge is designed to eliminate."""
    sr_edges = [_FakeEdge(0, k) for k in range(1, 21)]
    runs: list[tuple[list[int], list[tuple[int, int, int]]]] = []
    for _ in range(5):
        coll = _FakeColl(sr_edges=sr_edges)
        nodes, edges = await graph_mod._focus_neighborhood(coll, focus=0, depth=2, max_nodes=5)
        runs.append((sorted(nodes), sorted((e["src"], e["dst"], e["hits"]) for e in edges)))
    first = runs[0]
    for r in runs[1:]:
        assert r == first
    # The cap is 5 nodes. With focus + sorted-merge, we should keep
    # {0, 1, 2, 3, 4} — the lowest-id surviving set.
    assert first[0] == [0, 1, 2, 3, 4]


# ============================================================ _plasticity_overlay
async def test_plasticity_overlay_fans_out_across_nodes() -> None:
    """One ``get_edges`` per node, all concurrent. With 4 nodes the
    stage-1 barrier (size 4) fires when all 4 calls arrive."""
    plast_edges = [
        _FakeEdge(1, 2, hits=1, weight=0.1),
        _FakeEdge(2, 3, hits=1, weight=0.1),
        _FakeEdge(3, 4, hits=1, weight=0.1),
        _FakeEdge(4, 1, hits=1, weight=0.1),
    ]
    inner = _FakeColl(plast_edges=plast_edges)
    gated = _GatedColl(inner, schedule=[4])
    edges = await asyncio.wait_for(
        graph_mod._plasticity_overlay(gated, {1, 2, 3, 4}, habit_threshold=10),
        timeout=2.0,
    )
    assert len(edges) == 4
    assert gated.total_calls == 4
    assert gated.peak_in_flight == 4


async def test_plasticity_overlay_matches_sequential_baseline() -> None:
    plast_edges = [
        _FakeEdge(1, 2, hits=10, weight=0.7, bonus=0.2),
        _FakeEdge(1, 9, hits=10, weight=0.7, bonus=0.2),  # dst off-graph
        _FakeEdge(2, 3, hits=1, weight=0.1, bonus=0.0),
        _FakeEdge(3, 1, hits=20, weight=0.9, bonus=0.5),
    ]
    nodes = {1, 2, 3}
    par = await graph_mod._plasticity_overlay(
        _FakeColl(plast_edges=plast_edges), nodes, habit_threshold=5
    )
    seq = await _seq_plasticity_overlay(
        _FakeColl(plast_edges=plast_edges), nodes, habit_threshold=5
    )
    # Sorted-node iteration on both sides → element-wise equality.
    assert par == seq


async def test_plasticity_overlay_deterministic_ordering() -> None:
    """Repeated calls produce the same edge list in the same order
    regardless of ``gather`` completion order."""
    plast_edges = [
        _FakeEdge(s, d, hits=1, weight=float(s + d) / 10.0)
        for s in range(1, 6)
        for d in range(1, 6)
        if s != d
    ]
    nodes = {1, 2, 3, 4, 5}
    runs = []
    for _ in range(5):
        edges = await graph_mod._plasticity_overlay(
            _FakeColl(plast_edges=plast_edges), nodes, habit_threshold=10
        )
        runs.append([(e["src"], e["dst"]) for e in edges])
    first = runs[0]
    for r in runs[1:]:
        assert r == first
    # Sorted iteration: src in 1..5, dst in sequence of arrival.
    srcs = [s for s, _ in first]
    assert srcs == sorted(srcs)


async def test_plasticity_overlay_empty_nodes_short_circuits() -> None:
    """Pre-refactor: implicit (no edges). Post-refactor: explicit early
    return so we never call ``asyncio.gather()`` with an empty arg
    list. Test pins the behavior."""
    out = await graph_mod._plasticity_overlay(_FakeColl(), set(), habit_threshold=5)
    assert out == []


# ============================================================ guard against
#                                                              regression to seq
async def test_focus_neighborhood_does_not_serialise_adjacent_calls() -> None:
    """If a future change reverts ``_focus_neighborhood`` to call
    ``_adjacent`` sequentially per node, this stage-2 barrier (size 12)
    deadlocks. Targets the most likely regression vector: dropping the
    outer ``asyncio.gather`` while keeping the inner one."""
    sr_edges = [_FakeEdge(0, k) for k in (1, 2, 3)]
    inner = _FakeColl(sr_edges=sr_edges)
    # Layer 1 frontier {0}: 4 inner calls. Layer 2 frontier {1,2,3}: 12
    # concurrent calls. A regression that sequentialises per-frontier
    # fan-out would cap layer 2 at 4 and fail to fire the stage-2
    # barrier.
    gated = _GatedColl(inner, schedule=[4, 12])
    await asyncio.wait_for(
        graph_mod._focus_neighborhood(gated, focus=0, depth=2, max_nodes=100),
        timeout=2.0,
    )
    assert gated.peak_in_flight >= 12
