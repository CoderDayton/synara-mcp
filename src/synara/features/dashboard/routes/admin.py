"""Sanctioned maintenance operations.

Thin pass-throughs to the existing ``MemoryService`` operation
delegates. ``dream`` is intentionally absent: ``_reactor_dream`` is a
private reactor callback with no public seam (deferred — adding one is a
separate gated core change, mirroring ``delete_episode``).
"""

from __future__ import annotations

import asyncio
import os
import sys
from typing import Annotated, Any

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field

from synara.core.errors import ValidationError
from synara.features.memory import MemoryService

from ..deps import get_service
from .schemas import ConsolidateResult, ForgetResult, RestartResult

router = APIRouter(tags=["admin"], prefix="/admin")

# Delay before the in-place re-exec so the HTTP reply (and any buffered
# log line) flushes to the client before this process image is replaced.
_RESTART_DELAY_SECONDS = 0.25

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


@router.post("/consolidate", response_model=ConsolidateResult)
async def admin_consolidate(service: _Service, body: ConsolidateBody) -> dict[str, Any]:
    try:
        formed = await service.consolidate(
            session_id=body.session_id,
            n_clusters=body.n_clusters,
            min_cluster_size=body.min_cluster_size,
        )
    except ValidationError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT, detail=str(exc)
        ) from exc
    return {"schemas_formed": len(formed), "schemas": formed}


@router.post("/forget", response_model=ForgetResult)
async def admin_forget(service: _Service, body: ForgetBody) -> dict[str, Any]:
    try:
        return await service.forget(
            strength_floor=body.strength_floor,
            decay_tau_seconds=body.decay_tau_seconds,
            dry_run=body.dry_run,
            max_scan=body.max_scan,
        )
    except ValidationError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT, detail=str(exc)
        ) from exc


@router.post("/reflect")
async def admin_reflect(service: _Service, body: ReflectBody) -> dict[str, Any]:
    try:
        return await service.reflect(session_id=body.session_id, query=body.query, k=body.k)
    except ValidationError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT, detail=str(exc)
        ) from exc


def _reexec() -> None:  # pragma: no cover - replaces this process image
    """Replace this process with a fresh copy of itself.

    Closes every descriptor above stdio first: the inherited dashboard /
    MCP listening sockets would otherwise keep their ports bound and the
    new image would fail to re-bind. fds 0/1/2 carry the stdio MCP pipe
    to the client and must survive into the new process. The argv is
    fixed (``sys.executable`` + ``sys.argv``), with no shell and no
    user input, so the bandit no-shell finding is a false positive.
    """
    try:
        max_fd = int(os.sysconf("SC_OPEN_MAX"))
    except (ValueError, OSError):
        max_fd = 4096
    os.closerange(3, min(max_fd, 65536))
    os.execv(sys.executable, [sys.executable, *sys.argv])  # nosec B606


def request_restart() -> None:
    """Schedule an in-place re-exec on the running event loop.

    Isolated from the route so the test suite can patch it and never
    re-exec the test runner.
    """
    asyncio.get_running_loop().call_later(_RESTART_DELAY_SECONDS, _reexec)


@router.post("/restart", response_model=RestartResult)
async def admin_restart() -> dict[str, str]:
    """Restart the server by re-executing the process in place.

    Reloads on-disk code without losing the stdio MCP pipe, so connected
    clients re-handshake transparently. ``SYNARA_*`` is re-read from the
    current process environment, which ``execv`` preserves unchanged — so
    editing a ``SYNARA_*`` value takes effect only after restarting the
    MCP client (which re-spawns this process), not via this route.
    Disruptive: every
    connected session (and this dashboard) briefly drops while the new
    image comes up. Guarded by the dashboard bearer token like every
    other ``/api`` route. The re-exec is scheduled a beat later so this
    reply reaches the client first.
    """
    request_restart()
    return {"status": "restarting", "detail": "Server re-executing; reconnect in a moment."}
