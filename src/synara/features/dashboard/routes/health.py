"""Server health + identity."""

from __future__ import annotations

import time
from typing import Any

from fastapi import APIRouter, Request

from synara import __version__

router = APIRouter(tags=["health"])


@router.get("/health")
async def health(request: Request) -> dict[str, Any]:
    state = request.app.state
    settings = state.settings
    backend = "remote" if settings.embedding.url else "local"
    return {
        "status": "ok",
        "version": __version__,
        "transport": settings.transport,
        "db_path": settings.db_path,
        "embedding_backend": backend,
        "embedding_model": settings.embedding.model or "default",
        "uptime_seconds": round(time.monotonic() - state.started_at, 3),
    }
