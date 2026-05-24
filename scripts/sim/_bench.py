"""Wall-clock benchmark of the two optimized full-function paths.

Measures the gain from:

1. :meth:`MemoryService.vectorise` batch path vs the per-text
   fallback. Each ``vectorise(texts)`` call goes from N backend
   round-trips to 1 — meaningful when ``texts`` is theta-segmented
   (encode) or a batch of summaries (consolidate).
2. ``dashboard.routes.graph._focus_neighborhood`` and
   ``_plasticity_overlay``: per-frontier-layer / per-node
   :func:`asyncio.gather` vs an inlined sequential baseline.

Both per-call overheads are *simulated* via a configurable
:func:`asyncio.sleep` inside the embed and edge stubs. That mirrors the
fixed per-call latency (GPU forward pass, HTTP RTT, SQLite query) that
batching and gather are meant to collapse. With ``latency=0`` the bench
degenerates to a pure-CPU comparison; the default 500 us is a
representative remote-embedding / shared-DB cost.

Output: a stdout table and a JSON snapshot under
``scripts/sim/bench_snapshot.json`` (or a path passed via
``--bench-out``).
"""

from __future__ import annotations

import asyncio
import json
import math
import statistics
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from simplevecdb import AsyncVectorDB

# Bench requires the dashboard extra (fastapi) because it benches the
# graph route helpers. Hard-import here: this module is only imported
# when ``--bench`` is requested, so the dep cost is paid only by users
# who explicitly opt in. A missing fastapi installation surfaces as a
# clear ImportError pointing at the bench, not the bare runtime.
import synara.features.dashboard.routes.graph as graph_mod
from synara.features.memory import MemoryConfig, MemoryService


# ============================================================ embed stubs
async def _sleep_latency(seconds: float) -> None:
    """One ``await`` even when ``seconds == 0`` so the per-call
    scheduler overhead is included in either path. This keeps the
    latency=0 case from collapsing to zero time (which would hide the
    CPU portion of the win)."""
    if seconds > 0.0:
        await asyncio.sleep(seconds)
    else:
        await asyncio.sleep(0)


def _single_embed_factory(dim: int, latency_seconds: float) -> Any:
    """Returns an async per-text embed fn that incurs ``latency_seconds``
    per call (modelling fixed backend overhead)."""

    async def embed(text: str) -> list[float]:
        await _sleep_latency(latency_seconds)
        # Deterministic content: hash-derived to mimic a real embedder
        # without numpy. Width must match the configured dim.
        h = hash(text) & 0xFFFFFFFF
        base = (h % 1000) / 1000.0
        return [base + (i * 1e-4) for i in range(dim)]

    return embed


def _batch_embed_factory(dim: int, latency_seconds: float) -> Any:
    """Async batch embed fn: ONE ``latency_seconds`` per call regardless
    of ``len(texts)``. That asymmetry vs the single fn is precisely the
    win we are measuring."""

    async def embed_batch(texts: Any) -> list[list[float]]:
        await _sleep_latency(latency_seconds)
        out: list[list[float]] = []
        for t in texts:
            h = hash(t) & 0xFFFFFFFF
            base = (h % 1000) / 1000.0
            out.append([base + (i * 1e-4) for i in range(dim)])
        return out

    return embed_batch


# ============================================================ workload
async def _vectorise_workload(svc: MemoryService, *, batches: int, texts_per_batch: int) -> None:
    """Drive ``svc.vectorise`` directly so we measure only the embed
    plumbing, not encode/recall overhead. encode/recall vary by reactor
    state and would muddy the signal."""
    payload = [f"sample-text-{i}" for i in range(texts_per_batch)]
    for _ in range(batches):
        await svc.vectorise(payload)


# ============================================================ graph stubs
class _BenchEdge:
    __slots__ = ("bonus", "dst_id", "hits", "src_id", "weight")

    def __init__(
        self, src: int, dst: int, *, hits: int = 1, weight: float = 0.0, bonus: float = 0.0
    ) -> None:
        self.src_id = src
        self.dst_id = dst
        self.hits = hits
        self.weight = weight
        self.bonus = bonus


class _LatentColl:
    """Stub coll with per-call ``asyncio.sleep`` latency. Mirrors a
    real SQLite/AsyncVectorDB round-trip cost so that the
    parallel-gather refactor produces a measurable wall-clock delta."""

    def __init__(self, edges_by_kind: dict[str, list[_BenchEdge]], latency_seconds: float) -> None:
        self._edges = edges_by_kind
        self._lat = latency_seconds

    async def get_edges(
        self,
        *,
        kind: str,
        src: int | None = None,
        dst: int | None = None,
        limit: int | None = None,
    ) -> list[_BenchEdge]:
        await _sleep_latency(self._lat)
        pool = self._edges.get(kind, [])
        out = [
            e for e in pool if (src is None or e.src_id == src) and (dst is None or e.dst_id == dst)
        ]
        return out if limit is None else out[:limit]


def _build_graph(*, nodes: int, branching: int) -> dict[str, list[_BenchEdge]]:
    """Star-of-stars: node 0 at the centre, ``branching`` direct
    successors, each with ``branching`` further successors. Total nodes
    = 1 + branching + branching^2. Picked so ``_focus_neighborhood``
    with depth=2 has a fan-out worth measuring."""
    sr: list[_BenchEdge] = []
    plast: list[_BenchEdge] = []
    next_id = 1
    layer1 = list(range(next_id, next_id + branching))
    next_id += branching
    for n in layer1:
        sr.append(_BenchEdge(0, n, hits=2))
    layer2_total = 0
    for parent in layer1:
        for _ in range(branching):
            if next_id > nodes:
                break
            sr.append(_BenchEdge(parent, next_id, hits=1))
            plast.append(_BenchEdge(parent, next_id, hits=1, weight=0.5))
            next_id += 1
            layer2_total += 1
    return {"sr": sr, "plasticity": plast}


# ============================================================ sequential baselines
async def _seq_adjacent(
    coll: Any,
    node: int,
    limit: int,
    sr_seen: dict[tuple[int, int], dict[str, Any]],
) -> set[int]:
    """Verbatim pre-refactor body — used as the comparison baseline."""
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


# ============================================================ timing
@dataclass
class _Trial:
    name: str
    samples_seconds: list[float] = field(default_factory=list)

    def add(self, t: float) -> None:
        self.samples_seconds.append(t)

    def summary(self) -> dict[str, float]:
        s = self.samples_seconds
        if not s:
            return {"min": math.nan, "median": math.nan, "max": math.nan, "n": 0}
        return {
            "min": min(s),
            "median": statistics.median(s),
            "max": max(s),
            "n": len(s),
        }


async def _time(coro_factory: Any, *, repeats: int) -> _Trial:
    """Run ``coro_factory()`` ``repeats`` times, report per-run elapsed."""
    trial = _Trial(name=coro_factory.__name__)
    for _ in range(repeats):
        t0 = time.perf_counter()
        await coro_factory()
        trial.add(time.perf_counter() - t0)
    return trial


# ============================================================ harness
@dataclass
class BenchConfig:
    dim: int = 32
    embed_latency_ms: float = 0.5
    coll_latency_ms: float = 0.5
    vectorise_batches: int = 20
    vectorise_texts_per_batch: int = 8
    graph_nodes: int = 31  # 1 + 5 + 25
    graph_branching: int = 5
    graph_depth: int = 2
    graph_max_nodes: int = 200
    repeats: int = 5


async def _bench_vectorise(cfg: BenchConfig) -> dict[str, _Trial]:
    embed_latency_s = cfg.embed_latency_ms / 1000.0

    async def _trial_single() -> None:
        db = AsyncVectorDB(":memory:")
        try:
            svc = MemoryService(
                db,
                config=MemoryConfig(),
                embed_fn=_single_embed_factory(cfg.dim, embed_latency_s),
            )
            await _vectorise_workload(
                svc,
                batches=cfg.vectorise_batches,
                texts_per_batch=cfg.vectorise_texts_per_batch,
            )
        finally:
            await db.close()

    async def _trial_batched() -> None:
        db = AsyncVectorDB(":memory:")
        try:
            svc = MemoryService(
                db,
                config=MemoryConfig(),
                embed_fn=_single_embed_factory(cfg.dim, embed_latency_s),
                embed_batch_fn=_batch_embed_factory(cfg.dim, embed_latency_s),
            )
            await _vectorise_workload(
                svc,
                batches=cfg.vectorise_batches,
                texts_per_batch=cfg.vectorise_texts_per_batch,
            )
        finally:
            await db.close()

    single = await _time(_trial_single, repeats=cfg.repeats)
    batched = await _time(_trial_batched, repeats=cfg.repeats)
    return {"single": single, "batched": batched}


async def _bench_graph(cfg: BenchConfig) -> dict[str, _Trial]:
    """Bench BOTH graph helpers. Sequential vs gather, against an
    in-memory star-of-stars graph with synthetic per-query latency."""
    edges = _build_graph(nodes=cfg.graph_nodes, branching=cfg.graph_branching)
    lat = cfg.coll_latency_ms / 1000.0

    async def _trial_seq_focus() -> None:
        coll = _LatentColl(edges, lat)
        await _seq_focus_neighborhood(coll, 0, cfg.graph_depth, cfg.graph_max_nodes)

    async def _trial_par_focus() -> None:
        coll = _LatentColl(edges, lat)
        await graph_mod._focus_neighborhood(coll, 0, cfg.graph_depth, cfg.graph_max_nodes)

    # Pre-resolve the node set the overlay benches operate on so both
    # sequential and parallel overlays see the same input.
    nodes_for_overlay, _ = await graph_mod._focus_neighborhood(
        _LatentColl(edges, 0.0), 0, cfg.graph_depth, cfg.graph_max_nodes
    )

    async def _trial_seq_overlay() -> None:
        coll = _LatentColl(edges, lat)
        await _seq_plasticity_overlay(coll, nodes_for_overlay, habit_threshold=5)

    async def _trial_par_overlay() -> None:
        coll = _LatentColl(edges, lat)
        await graph_mod._plasticity_overlay(coll, nodes_for_overlay, habit_threshold=5)

    return {
        "focus_sequential": await _time(_trial_seq_focus, repeats=cfg.repeats),
        "focus_gather": await _time(_trial_par_focus, repeats=cfg.repeats),
        "overlay_sequential": await _time(_trial_seq_overlay, repeats=cfg.repeats),
        "overlay_gather": await _time(_trial_par_overlay, repeats=cfg.repeats),
    }


# ============================================================ reporting
def _row(name: str, t: _Trial, baseline: _Trial | None) -> str:
    s = t.summary()
    median_ms = s["median"] * 1000.0
    min_ms = s["min"] * 1000.0
    max_ms = s["max"] * 1000.0
    if baseline is None:
        speedup = "—"
    else:
        b = baseline.summary()["median"]
        speedup = f"{b / s['median']:.2f}x" if s["median"] > 0 else "inf"
    return (
        f"  {name:<28} median={median_ms:8.3f} ms  min={min_ms:8.3f}  "
        f"max={max_ms:8.3f}  speedup={speedup}"
    )


def _print_table(vect: dict[str, _Trial], graph: dict[str, _Trial], cfg: BenchConfig) -> None:
    print(
        f"\n# wall-clock bench (embed_latency={cfg.embed_latency_ms}ms, "
        f"coll_latency={cfg.coll_latency_ms}ms, repeats={cfg.repeats})"
    )
    print(
        f"# vectorise: {cfg.vectorise_batches} batches x "
        f"{cfg.vectorise_texts_per_batch} texts/batch"
    )
    print(_row("vectorise single (fallback)", vect["single"], None))
    print(_row("vectorise batched", vect["batched"], vect["single"]))
    print(
        f"\n# graph: nodes={cfg.graph_nodes}, branching={cfg.graph_branching}, "
        f"depth={cfg.graph_depth}"
    )
    print(_row("_focus_neighborhood seq", graph["focus_sequential"], None))
    print(_row("_focus_neighborhood gather", graph["focus_gather"], graph["focus_sequential"]))
    print(_row("_plasticity_overlay seq", graph["overlay_sequential"], None))
    print(_row("_plasticity_overlay gather", graph["overlay_gather"], graph["overlay_sequential"]))


def _snapshot_dict(
    vect: dict[str, _Trial],
    graph: dict[str, _Trial],
    cfg: BenchConfig,
) -> dict[str, Any]:
    def _serialize(d: dict[str, _Trial]) -> dict[str, dict[str, float]]:
        return {name: t.summary() for name, t in d.items()}

    vect_s = _serialize(vect)
    graph_s = _serialize(graph)

    # Speedup ratios computed off median; documented so a future reader
    # doesn't have to re-derive them from the raw samples.
    def _ratio(num: float, denom: float) -> float:
        return float(num / denom) if denom > 0 else float("inf")

    speedups = {
        "vectorise_batched_vs_single": _ratio(
            vect_s["single"]["median"], vect_s["batched"]["median"]
        ),
        "focus_neighborhood_gather_vs_seq": _ratio(
            graph_s["focus_sequential"]["median"], graph_s["focus_gather"]["median"]
        ),
        "plasticity_overlay_gather_vs_seq": _ratio(
            graph_s["overlay_sequential"]["median"], graph_s["overlay_gather"]["median"]
        ),
    }
    return {
        "schema_version": 1,
        "generated_at_unix": time.time(),
        "config": asdict(cfg),
        "vectorise": vect_s,
        "graph": graph_s,
        "speedup_medians": speedups,
    }


# ============================================================ entrypoint
async def run_bench(out_path: Path, *, cfg: BenchConfig | None = None) -> dict[str, Any]:
    cfg = cfg or BenchConfig()
    vect = await _bench_vectorise(cfg)
    graph = await _bench_graph(cfg)
    _print_table(vect, graph, cfg)
    snap = _snapshot_dict(vect, graph, cfg)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(snap, indent=2), encoding="utf-8")
    print(f"\n# JSON snapshot written: {out_path}")
    return snap
