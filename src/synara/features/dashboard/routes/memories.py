"""Memory listing, detail, and delete.

Every handler delegates to ``MemoryService``; no recall/SR/forget logic
is reimplemented here. Listing/search and the SR/plasticity views are
read paths; delete routes through ``MemoryService.delete_episode`` (the
FK-safe, forget-consistent + SR-evicting core method).
"""

from __future__ import annotations

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
        if kind == "semantic":
            hits = await service.recall_semantic_memory(q, k=limit)
        else:
            hits = await service.recall(q, k=limit)
        return {"kind": kind, "query": q, "items": hits, "count": len(hits)}

    coll = service.semantic if kind == "semantic" else service.episodic
    # simplevecdb's get_documents already returns a list; slicing it
    # directly skips a redundant full-prefix copy (offset+limit can be
    # up to _MAX_LIMIT + _MAX_OFFSET wide on cold paths).
    rows = await coll.get_documents(filter_dict=None, limit=offset + limit)
    window = rows[offset : offset + limit]
    items = [{"id": int(doc_id), "content": text, "metadata": md} for doc_id, text, md in window]
    return {"kind": kind, "items": items, "count": len(items), "offset": offset}


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

    sr = await service.episodic.get_edges(src=episode_id, kind="sr")
    sr_out = [{"dst": int(e.dst_id), "count": int(e.hits)} for e in sr]

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
        "plasticity_edges": plastic_edges,
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
