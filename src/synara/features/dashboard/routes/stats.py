"""Aggregate stats + effective tunables (read-only)."""

from __future__ import annotations

import dataclasses
from typing import Annotated, Any

from fastapi import APIRouter, Depends

from synara.features.memory import MemoryService

from ..deps import get_service
from .schemas import StatsResponse

router = APIRouter(tags=["stats"])

_Service = Annotated[MemoryService, Depends(get_service)]


@router.get("/stats", response_model=StatsResponse)
async def stats(service: _Service) -> dict[str, int]:
    return await service.stats()


@router.get("/params")
async def params(service: _Service) -> dict[str, Any]:
    """Effective MemoryConfig snapshot.

    Read-only: ``MemoryConfig`` is frozen; live mutation would require
    rebuilding SR/plasticity and is deliberately out of scope here.
    """
    return dataclasses.asdict(service.config)
