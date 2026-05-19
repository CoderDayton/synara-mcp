"""Bounded SR/plasticity subgraph for visualisation.

Reads the durable edge tables via the public collection API
(``get_edges`` with ``kind``), never SR private state. Hard-bounded
(depth <= 3, node cap <= 1000, per-frontier fan-out) so a large episode
population cannot turn this into an unbounded CPU/memory sink.
"""

from __future__ import annotations

from typing import Annotated, Any

from fastapi import APIRouter, Depends, Query

from synara.features.memory import MemoryService

from ..deps import get_service

router = APIRouter(tags=["graph"])

_Service = Annotated[MemoryService, Depends(get_service)]

_MAX_NODES_CAP = 1000


def _edge_dict(e: Any, *, weighted: bool) -> dict[str, Any]:
    out: dict[str, Any] = {
        "src": int(e.src_id),
        "dst": int(e.dst_id),
        "hits": int(e.hits),
    }
    if weighted:
        out["weight"] = float(e.weight)
        out["bonus"] = float(e.bonus)
    return out


async def _global_sr(coll: Any, max_nodes: int) -> tuple[set[int], list[dict[str, Any]]]:
    nodes: set[int] = set()
    edges: list[dict[str, Any]] = []
    for e in await coll.get_edges(kind="sr", limit=max_nodes * 4):
        if len(nodes) >= max_nodes and int(e.src_id) not in nodes:
            continue
        nodes.add(int(e.src_id))
        nodes.add(int(e.dst_id))
        edges.append(_edge_dict(e, weighted=False))
    return nodes, edges


async def _bfs_sr(
    coll: Any, focus: int, depth: int, max_nodes: int
) -> tuple[set[int], list[dict[str, Any]]]:
    nodes: set[int] = {focus}
    edges: list[dict[str, Any]] = []
    frontier = {focus}
    for _ in range(depth):
        nxt: set[int] = set()
        for node in frontier:
            if len(nodes) >= max_nodes:
                break
            for e in await coll.get_edges(src=node, kind="sr", limit=max_nodes):
                edges.append(_edge_dict(e, weighted=False))
                d = int(e.dst_id)
                if d not in nodes and len(nodes) < max_nodes:
                    nodes.add(d)
                    nxt.add(d)
        frontier = nxt
        if not frontier:
            break
    return nodes, edges


@router.get("/graph")
async def sr_graph(
    service: _Service,
    focus: int | None = None,
    depth: Annotated[int, Query(ge=1, le=3)] = 1,
    max_nodes: Annotated[int, Query(ge=1, le=_MAX_NODES_CAP)] = 200,
) -> dict[str, Any]:
    coll = service.episodic
    if focus is None:
        nodes, sr_edges = await _global_sr(coll, max_nodes)
    else:
        nodes, sr_edges = await _bfs_sr(coll, int(focus), depth, max_nodes)

    plasticity_edges: list[dict[str, Any]] = []
    for node in list(nodes):
        for e in await coll.get_edges(src=node, kind="plasticity"):
            if int(e.dst_id) in nodes:
                plasticity_edges.append(_edge_dict(e, weighted=True))

    return {
        "nodes": sorted(nodes),
        "sr_edges": sr_edges,
        "plasticity_edges": plasticity_edges,
        "truncated": len(nodes) >= max_nodes,
    }
