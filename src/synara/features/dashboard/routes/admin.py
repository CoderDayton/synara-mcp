"""Sanctioned maintenance operations.

Thin pass-throughs to the existing ``MemoryService`` operation
delegates. ``dream`` is intentionally absent: ``_reactor_dream`` is a
private reactor callback with no public seam (deferred — adding one is a
separate gated core change, mirroring ``delete_episode``).
"""

from __future__ import annotations

from typing import Annotated, Any

from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field

from synara.features.memory import MemoryService

from ..deps import get_service

router = APIRouter(tags=["admin"], prefix="/admin")

_Service = Annotated[MemoryService, Depends(get_service)]


class ConsolidateBody(BaseModel):
    session_id: str | None = None
    n_clusters: int | None = Field(default=None, ge=1)
    min_cluster_size: int | None = Field(default=None, ge=1)


class ForgetBody(BaseModel):
    strength_floor: float = Field(default=0.05, ge=0.0, le=1.0)
    decay_tau_seconds: float | None = Field(default=None, gt=0.0)
    dry_run: bool = True
    max_scan: int = Field(default=1000, ge=1, le=100_000)


class ReflectBody(BaseModel):
    session_id: str
    query: str | None = None
    k: int = Field(default=5, ge=1, le=100)


@router.post("/consolidate")
async def admin_consolidate(service: _Service, body: ConsolidateBody) -> dict[str, Any]:
    formed = await service.consolidate(
        session_id=body.session_id,
        n_clusters=body.n_clusters,
        min_cluster_size=body.min_cluster_size,
    )
    return {"schemas_formed": len(formed), "schemas": formed}


@router.post("/forget")
async def admin_forget(service: _Service, body: ForgetBody) -> dict[str, Any]:
    return await service.forget(
        strength_floor=body.strength_floor,
        decay_tau_seconds=body.decay_tau_seconds,
        dry_run=body.dry_run,
        max_scan=body.max_scan,
    )


@router.post("/reflect")
async def admin_reflect(service: _Service, body: ReflectBody) -> dict[str, Any]:
    return await service.reflect(session_id=body.session_id, query=body.query, k=body.k)
