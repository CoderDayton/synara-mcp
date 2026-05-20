"""Bounded SR/plasticity subgraph for visualisation.

Reads the durable edge tables via the public collection API
(``get_edges`` with ``kind``) and the SR closure via the public
:meth:`SuccessorRepresentation.boost` reader — never SR private
``_T``/``_M`` state. Hard-bounded (depth <= 3, node cap <= 1000,
per-frontier fan-out) so a large episode population cannot turn this
into an unbounded CPU/memory sink.

The payload mirrors the real memory model so the dashboard can render
it faithfully: episodic nodes carry salience / retrieval / session /
consolidation lineage; semantic *schema* nodes are overlaid for any
episode that has been consolidated; SR edges carry the discounted
closure ``M`` (the actual recall-ranking prior); plasticity edges carry
the combined strength and the habit flag (``hits >= habit threshold``).
"""

from __future__ import annotations

from collections import defaultdict
from typing import Annotated, Any

from fastapi import APIRouter, Depends, Query

from synara.features.memory import MemoryService

from ..deps import get_service

router = APIRouter(tags=["graph"])

_Service = Annotated[MemoryService, Depends(get_service)]

_MAX_NODES_CAP = 1000
_PREVIEW_CHARS = 140


def _sr_edge(e: Any) -> dict[str, Any]:
    return {"src": int(e.src_id), "dst": int(e.dst_id), "hits": int(e.hits)}


async def _global_sr(coll: Any, max_nodes: int) -> tuple[set[int], list[dict[str, Any]]]:
    nodes: set[int] = set()
    edges: list[dict[str, Any]] = []
    for e in await coll.get_edges(kind="sr", limit=max_nodes * 4):
        if len(nodes) >= max_nodes and int(e.src_id) not in nodes:
            continue
        nodes.add(int(e.src_id))
        nodes.add(int(e.dst_id))
        edges.append(_sr_edge(e))
    return nodes, edges


async def _adjacent(
    coll: Any,
    node: int,
    limit: int,
    sr_seen: dict[tuple[int, int], dict[str, Any]],
) -> set[int]:
    """Bidirectional SR + plasticity neighbours of ``node``.

    Mirrors the recall neighbourhood: an anchor spreads over successor
    transitions *and* plasticity associations in *both* directions. A
    forward-only walk made every pure-successor (incoming-only) episode
    its own island — the source of the "each episode is its own graph"
    behaviour. SR edges touched here are accumulated in ``sr_seen``
    (deduped) for the caller to filter to the final node set.
    """
    out: set[int] = set()
    for e in await coll.get_edges(src=node, kind="sr", limit=limit):
        sr_seen.setdefault((int(e.src_id), int(e.dst_id)), _sr_edge(e))
        out.add(int(e.dst_id))
    for e in await coll.get_edges(dst=node, kind="sr", limit=limit):
        sr_seen.setdefault((int(e.src_id), int(e.dst_id)), _sr_edge(e))
        out.add(int(e.src_id))
    for e in await coll.get_edges(src=node, kind="plasticity", limit=limit):
        out.add(int(e.dst_id))
    for e in await coll.get_edges(dst=node, kind="plasticity", limit=limit):
        out.add(int(e.src_id))
    return out


async def _focus_neighborhood(
    coll: Any, focus: int, depth: int, max_nodes: int
) -> tuple[set[int], list[dict[str, Any]]]:
    nodes: set[int] = {focus}
    sr_seen: dict[tuple[int, int], dict[str, Any]] = {}
    frontier = {focus}
    for _ in range(depth):
        nxt: set[int] = set()
        for node in frontier:
            if len(nodes) >= max_nodes:
                break
            for nb in await _adjacent(coll, node, max_nodes, sr_seen):
                if nb not in nodes and len(nodes) < max_nodes:
                    nodes.add(nb)
                    nxt.add(nb)
        frontier = nxt
        if not frontier:
            break
    edges = [e for (s, d), e in sr_seen.items() if s in nodes and d in nodes]
    return nodes, edges


def _preview(text: str | None) -> str:
    if not text:
        return ""
    flat = " ".join(text.split())
    return flat[:_PREVIEW_CHARS] + ("…" if len(flat) > _PREVIEW_CHARS else "")


async def _docs_by_id(coll: Any, ids: list[int]) -> dict[int, tuple[str, dict[str, Any]]]:
    """Batch-fetch documents for a bounded id set in one query.

    Episodic and semantic docs both mirror their row id into
    ``metadata.id`` (see hippocampus/encode + consolidate), so the
    metadata ``{"id": [...]}`` IN-filter resolves the whole node set
    without a per-node round trip.
    """
    if not ids:
        return {}
    out: dict[int, tuple[str, dict[str, Any]]] = {}
    for doc_id, text, md in await coll.get_documents({"id": ids}):
        out[int(doc_id)] = (text, md)
    return out


async def _plasticity_overlay(
    coll: Any, nodes: set[int], habit_threshold: int
) -> list[dict[str, Any]]:
    """SR-bounded plasticity edges (both endpoints survived the node cap)."""
    edges: list[dict[str, Any]] = []
    for node in list(nodes):
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


def _attach_closure(sr: Any, sr_edges: list[dict[str, Any]]) -> None:
    """Annotate each SR edge with the discounted closure ``M[src, dst]``."""
    if sr is None:
        for e in sr_edges:
            e["m"] = 0.0
        return
    by_src: dict[int, list[int]] = defaultdict(list)
    for e in sr_edges:
        by_src[e["src"]].append(e["dst"])
    m_by_src = {s: sr.boost(s, dsts) for s, dsts in by_src.items()}
    for e in sr_edges:
        e["m"] = float(m_by_src.get(e["src"], {}).get(e["dst"], 0.0))


def _episodic_node(nid: int, text: str, md: dict[str, Any], focus: int | None) -> dict[str, Any]:
    return {
        "id": nid,
        "key": f"ep:{nid}",
        "kind": "episodic",
        "label": f"#{nid}",
        "salience": float(md.get("salience", 0.0)),
        "retrieval_count": int(md.get("retrieval_count", 0)),
        "session_id": md.get("session_id"),
        "encoded_at": float(md.get("encoded_at", 0.0)),
        "last_accessed": float(md.get("last_accessed", 0.0)),
        "consolidated_into": int(md.get("consolidated_into", 0)),
        "group_id": int(md.get("episode_group_id", nid)),
        "segment_count": int(md.get("segment_count", 1)),
        "preview": _preview(text),
        "is_focus": focus is not None and nid == int(focus),
    }


async def _semantic_overlay(
    service: MemoryService,
    docs: dict[int, tuple[str, dict[str, Any]]],
    schema_ids: set[int],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Schema nodes + episode→schema consolidation edges."""
    if not schema_ids:
        return [], []
    sdocs = await _docs_by_id(service.semantic, sorted(schema_ids))
    full_at = max(1, int(service.config.consolidate_confidence_full_at))
    nodes: list[dict[str, Any]] = []
    for sid in sorted(schema_ids):
        text, md = sdocs.get(sid, ("", {}))
        sources = [int(x) for x in md.get("source_episode_ids", [])]
        nodes.append(
            {
                "id": sid,
                "key": f"sem:{sid}",
                "kind": "semantic",
                "label": f"schema #{sid}",
                "confidence": float(md.get("confidence", min(1.0, len(sources) / full_at))),
                "source_count": len(sources),
                "user_asserted": bool(md.get("user_asserted", False)),
                "preview": _preview(text),
            }
        )
    edges = [
        {"src": nid, "dst": f"sem:{c}"}
        for nid, (_t, md) in docs.items()
        if (c := int(md.get("consolidated_into", 0))) in schema_ids
    ]
    return nodes, edges


@router.get("/graph")
async def sr_graph(
    service: _Service,
    focus: Annotated[int | None, Query(ge=0)] = None,
    depth: Annotated[int, Query(ge=1, le=3)] = 1,
    max_nodes: Annotated[int, Query(ge=1, le=_MAX_NODES_CAP)] = 200,
) -> dict[str, Any]:
    coll = service.episodic

    # SR transition skeleton (global view, or bounded BFS around a focus).
    if focus is None:
        nodes, sr_edges = await _global_sr(coll, max_nodes)
    else:
        nodes, sr_edges = await _focus_neighborhood(coll, int(focus), depth, max_nodes)

    plasticity_edges = await _plasticity_overlay(
        coll, nodes, int(service.config.habit_threshold_hits)
    )

    # Discounted closure M (the real recall-ranking prior) per SR edge.
    await service._ensure_sr_loaded()
    sr = service._sr
    episode_count = await coll.count()
    omega = float(sr.omega(episode_count)) if sr is not None else 0.0
    _attach_closure(sr, sr_edges)

    # Episodic enrichment + semantic schema overlay.
    docs = await _docs_by_id(coll, sorted(nodes))
    node_objs = [_episodic_node(nid, *docs.get(nid, ("", {})), focus) for nid in sorted(nodes)]
    schema_ids = {c for _t, md in docs.values() if (c := int(md.get("consolidated_into", 0))) > 0}
    schema_nodes, consolidation_edges = await _semantic_overlay(service, docs, schema_ids)
    node_objs.extend(schema_nodes)

    return {
        "nodes": node_objs,
        "sr_edges": sr_edges,
        "plasticity_edges": plasticity_edges,
        "consolidation_edges": consolidation_edges,
        "omega": omega,
        "episode_count": episode_count,
        "focus": focus,
        "truncated": len(nodes) >= max_nodes,
    }
