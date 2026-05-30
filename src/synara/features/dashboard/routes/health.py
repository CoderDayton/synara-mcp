"""Server health + identity."""

from __future__ import annotations

import time
from pathlib import PurePath
from typing import Any

from fastapi import APIRouter, Request

from synara import __version__

from .schemas import HealthResponse

router = APIRouter(tags=["health"])


def _redact_db_path(db_path: str) -> str:
    """Hide the install-tree prefix from the health response.

    The full filesystem path leaks user/install layout to anyone who can
    reach the dashboard (loopback, but also a reverse proxy or shared
    host). The UI only uses this string as a "where am I writing"
    sentinel, so the basename — or the literal ``:memory:`` for
    ephemeral DBs — preserves operator value without the leak.
    """
    if db_path == ":memory:" or not db_path:
        return db_path or ":memory:"
    return PurePath(db_path).name or db_path


@router.get("/health", response_model=HealthResponse)
async def health(request: Request) -> dict[str, Any]:
    state = request.app.state
    settings = state.settings
    backend = "remote" if settings.embedding.url else "local"
    return {
        "status": "ok",
        "version": __version__,
        "transport": settings.transport,
        "db_path": _redact_db_path(settings.db_path),
        "embedding_backend": backend,
        "embedding_model": settings.embedding.model or "default",
        "uptime_seconds": round(time.monotonic() - state.started_at, 3),
    }
