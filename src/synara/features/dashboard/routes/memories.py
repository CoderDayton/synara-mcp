"""Memory listing, detail, and delete.

Every handler delegates to ``MemoryService``; no recall/SR/forget logic
is reimplemented here. Listing/search and the SR/plasticity views are
read paths; delete routes through ``MemoryService.delete_episode`` (the
FK-safe, forget-consistent + SR-evicting core method).
"""

from __future__ import annotations

import asyncio
from typing import Annotated, Any, Literal

from fastapi import APIRouter, Depends, HTTPException, Path, Query, status

from synara.core.errors import ValidationError
from synara.features.memory import MemoryService

from ..deps import get_service

router = APIRouter(tags=["memories"])

_Service = Annotated[MemoryService, Depends(get_service)]

_MAX_LIMIT = 200
# Offset-paginate by fetching offset+limit rows then slicing; cap the
# offset so a single query param cannot trigger a full-table memory load.
_MAX_OFFSET = 100_000
# Mirrors MemoryConfig.max_content_chars default; the service rejects past
# this anyway, but bouncing it at the route saves a round trip and turns a
# 500 (downstream ValidationError) into a clean 422 (FastAPI/Pydantic).
_MAX_Q_CHARS = 8_000


@router.get("/memories")
async def list_memories(
    service: _Service,
    kind: Literal["episodic", "semantic"] = "episodic",
    q: Annotated[str | None, Query(max_length=_MAX_Q_CHARS)] = None,
    limit: Annotated[int, Query(ge=1, le=_MAX_LIMIT)] = 50,
    offset: Annotated[int, Query(ge=0, le=_MAX_OFFSET)] = 0,
) -> dict[str, Any]:
    if q:
        return await _search(service, kind=kind, q=q, limit=limit)

    coll = service.semantic if kind == "semantic" else service.episodic
    # Push the offset down into the store so it doesn't materialise the
    # full ``offset + limit`` prefix (up to _MAX_LIMIT + _MAX_OFFSET wide
    # on cold paths) just to slice the tail off in Python.
    window = await coll.get_documents(filter_dict=None, limit=limit, offset=offset)
    items = [{"id": int(doc_id), "content": text, "metadata": md} for doc_id, text, md in window]
    return {"kind": kind, "items": items, "count": len(items), "offset": offset}


# ---------------------------------------------------------- search ---


async def _search(
    service: MemoryService,
    *,
    kind: Literal["episodic", "semantic"],
    q: str,
    limit: int,
) -> dict[str, Any]:
    """Hybrid semantic-recall + substring fallback for the map search.

    Two failure modes motivate the substring leg:
      1. Cold-store recall: the user has stored a handful of episodes
         but recall returns nothing because the query is semantically
         distant from any stored vector (small store + short query =
         high cosine distance everywhere).
      2. Mode-mismatch: the user typed a literal identifier ("auth",
         "JWT", a session id) and expects exact-substring behaviour, not
         a semantic neighbourhood.

    We run both legs concurrently, merge by id with semantic ranks
    taking precedence over substring matches, and report the regime in
    ``recall_mode`` so the UI can flag what actually drove the result.
    """
    recall_co = (
        service.recall_semantic_memory(q, k=limit)
        if kind == "semantic"
        else service.recall(q, k=limit)
    )
    substring_co = _substring_scan(service, kind=kind, q=q, limit=limit)
    recall_hits, substring_hits = await asyncio.gather(recall_co, substring_co)

    merged, mode = _merge_search_hits(
        recall_hits=recall_hits,
        substring_hits=substring_hits,
        limit=limit,
    )
    return {
        "kind": kind,
        "query": q,
        "items": merged,
        "count": len(merged),
        "recall_mode": mode,
    }


# Substring scan cap: how many rows we pull from get_documents before
# filtering. Large enough to cover the working-set of any plausible
# dashboard store; well under _MAX_OFFSET + _MAX_LIMIT so the substring
# leg is never the bottleneck on the route's overall budget.
_SUBSTRING_SCAN_CAP = 5_000


async def _substring_scan(
    service: MemoryService,
    *,
    kind: Literal["episodic", "semantic"],
    q: str,
    limit: int,
) -> list[dict[str, Any]]:
    """Case-insensitive substring scan over ``get_documents``.

    Returns at most ``limit`` rows in the same dict shape as
    ``service.recall`` so the merge step doesn't have to special-case
    them. ``distance`` is ``None`` to mark substring-origin hits.
    """
    needle = q.casefold().strip()
    if not needle:
        return []
    coll = service.semantic if kind == "semantic" else service.episodic
    rows = await coll.get_documents(filter_dict=None, limit=_SUBSTRING_SCAN_CAP)
    out: list[tuple[int, dict[str, Any]]] = []
    for doc_id, text, md in rows:
        if not isinstance(text, str):
            continue
        idx = text.casefold().find(needle)
        if idx < 0:
            continue
        out.append(
            (
                idx,  # rank key: earlier match wins
                {
                    "id": int(doc_id),
                    "content": text,
                    "distance": None,
                    "source": kind,
                    "metadata": md,
                    "substring_offset": idx,
                },
            )
        )
    # Earliest occurrence first; tie-break by id for stable order.
    out.sort(key=lambda r: (r[0], r[1]["id"]))
    return [item for _, item in out[:limit]]


def _merge_search_hits(
    *,
    recall_hits: list[dict[str, Any]],
    substring_hits: list[dict[str, Any]],
    limit: int,
) -> tuple[list[dict[str, Any]], str]:
    """Merge recall + substring hits and report which leg(s) contributed.

    Semantic hits keep their position (they're already cosine-ranked +
    SR-reweighted); substring-only hits append behind. De-dup is by
    ``id`` so an episode that matches semantically AND by substring
    appears once with the richer recall metadata, with a
    ``substring_offset`` annotation merged in.
    """
    by_id: dict[int, dict[str, Any]] = {}
    ordered: list[int] = []
    for hit in recall_hits:
        hid = int(hit.get("id", -1))
        if hid < 0 or hid in by_id:
            continue
        by_id[hid] = dict(hit)
        ordered.append(hid)
    sub_lookup = {int(h.get("id", -1)): h for h in substring_hits if int(h.get("id", -1)) >= 0}
    # Annotate semantic hits that also matched by substring.
    for hid in ordered:
        sub = sub_lookup.get(hid)
        if sub is not None and "substring_offset" in sub:
            by_id[hid]["substring_offset"] = sub["substring_offset"]
    # Append substring-only hits (preserve their substring ordering).
    for hit in substring_hits:
        hid = int(hit.get("id", -1))
        if hid < 0 or hid in by_id:
            continue
        by_id[hid] = dict(hit)
        ordered.append(hid)

    merged = [by_id[hid] for hid in ordered[:limit]]
    recall_ids = {int(h.get("id", -1)) for h in recall_hits if int(h.get("id", -1)) >= 0}
    substring_ids = set(sub_lookup)
    if recall_ids and substring_ids:
        mode = "hybrid"
    elif recall_ids:
        mode = "semantic"
    elif substring_ids:
        mode = "substring"
    else:
        mode = "empty"
    return merged, mode


@router.get("/memories/{episode_id}")
async def memory_detail(
    service: _Service,
    episode_id: Annotated[int, Path(ge=0)],
) -> dict[str, Any]:
    target = await service.episodic.get_documents({"id": episode_id})
    if not target:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"episode {episode_id} not found",
        )
    _doc_id, _text, md = target[0]
    group_id = int(md.get("episode_group_id", episode_id))
    group = await service.fetch_episode_group(group_id)

    # SR transitions are directed; show both directions so the detail
    # panel isn't silently outgoing-only.
    sr = await service.episodic.get_edges(src=episode_id, kind="sr")
    sr_out = [{"dst": int(e.dst_id), "count": int(e.hits)} for e in sr]
    sr_incoming = await service.episodic.get_edges(dst=episode_id, kind="sr")
    sr_in = [{"src": int(e.src_id), "count": int(e.hits)} for e in sr_incoming]

    plastic = await service.episodic.get_edges(src=episode_id, kind="plasticity")
    plastic_edges = [
        {
            "src": int(e.src_id),
            "dst": int(e.dst_id),
            "weight": float(e.weight),
            "bonus": float(e.bonus),
            "hits": int(e.hits),
        }
        for e in plastic
    ]
    return {
        "id": episode_id,
        "group_id": group_id,
        "segments": group,
        "sr_transitions": sr_out,
        "sr_transitions_in": sr_in,
        "plasticity_edges": plastic_edges,
    }


@router.get("/semantic/{semantic_id}")
async def semantic_detail(
    service: _Service,
    semantic_id: Annotated[int, Path(ge=0)],
) -> dict[str, Any]:
    target = await service.semantic.get_documents({"id": semantic_id})
    if not target:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"semantic {semantic_id} not found",
        )
    _doc_id, text, md = target[0]
    sources = [int(x) for x in md.get("source_episode_ids", [])]
    return {
        "id": semantic_id,
        "content": text,
        "kind": str(md.get("kind", "fact")),
        "tags": [str(t) for t in md.get("tags", [])],
        "confidence": float(md.get("confidence", 0.0)),
        "user_asserted": bool(md.get("authored", False)),
        "source_episode_ids": sources,
        "created_at": float(md.get("created_at", 0.0)),
        "updated_at": float(md.get("updated_at", md.get("created_at", 0.0))),
    }


@router.delete("/memories/{episode_id}")
async def delete_memory(
    service: _Service,
    episode_id: Annotated[int, Path(ge=0)],
) -> dict[str, Any]:
    try:
        return await service.delete_episode(episode_id)
    except ValidationError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
