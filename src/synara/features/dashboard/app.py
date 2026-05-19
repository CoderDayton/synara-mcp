"""FastAPI assembly for the parallel admin console.

Imported lazily (see package ``__getattr__``) so a default,
non-dashboard install never loads FastAPI/uvicorn.

The app is wired with the *live* objects the MCP tools use — the same
``MemoryService`` instance, ``db``, and ``embedder`` — so the dashboard
observes and mutates exactly one shared state (no SR/plasticity
divergence). Routers delegate to the service; no memory logic here.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

from fastapi import Depends, FastAPI

from synara import __version__

from .auth import make_auth_dependency
from .routes import admin, graph, health, memories, stats

if TYPE_CHECKING:  # pragma: no cover - typing only
    from simplevecdb import AsyncVectorDB

    from synara.config import Settings
    from synara.features.embedding import Embedder
    from synara.features.memory import MemoryService


def build_dashboard_app(
    *,
    settings: Settings,
    db: AsyncVectorDB,
    embedder: Embedder | None,
    service: MemoryService,
) -> FastAPI:
    """Build the dashboard FastAPI app bound to live server objects."""
    app = FastAPI(
        title="Synara Dashboard",
        version=__version__,
        # This is an internal admin API, not a public contract: no
        # interactive docs / OpenAPI surface.
        docs_url=None,
        redoc_url=None,
        openapi_url=None,
    )
    app.state.settings = settings
    app.state.db = db
    app.state.embedder = embedder
    app.state.service = service
    app.state.started_at = time.monotonic()

    auth = make_auth_dependency(settings.dashboard)
    guarded = [Depends(auth)]
    for module in (health, stats, memories, graph, admin):
        app.include_router(module.router, prefix="/api", dependencies=guarded)
    return app
