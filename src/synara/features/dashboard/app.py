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
from pathlib import Path
from typing import TYPE_CHECKING

from fastapi import Depends, FastAPI, HTTPException, status
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from synara import __version__

from .auth import make_auth_dependency
from .routes import admin, graph, health, memories, stats

_STATIC_DIR = Path(__file__).parent / "static"


def _mount_spa(app: FastAPI, static_dir: Path) -> None:
    """Serve the committed Vite build with SPA-history fallback.

    The static shell is intentionally *unauthenticated*: it carries no
    data (every datum is behind the guarded ``/api`` routers) and the
    bearer token is entered through the UI itself, so gating the shell
    would be a chicken-and-egg lockout. Hashed ``/assets`` are immutable;
    any other non-API path returns ``index.html`` for client routing.
    """
    index = static_dir / "index.html"
    app.mount(
        "/assets",
        StaticFiles(directory=static_dir / "assets"),
        name="assets",
    )

    @app.get("/{full_path:path}", include_in_schema=False)
    async def spa(full_path: str) -> FileResponse:
        # /api/* matched the routers above; an unmatched /api path is a
        # real 404, not a client route — don't mask it with the shell.
        # The disabled doc endpoints stay hard 404s too: the SPA catch-all
        # must not resurrect an OpenAPI/Swagger surface as the shell.
        if full_path in {"api", "openapi.json", "docs", "redoc"} or full_path.startswith("api/"):
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND)
        root = static_dir.resolve()
        candidate = (static_dir / full_path).resolve()
        if full_path and candidate.is_relative_to(root) and candidate.is_file():
            return FileResponse(candidate)
        return FileResponse(index)


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

    # Serve the bundled SPA when it has been built/committed. A source
    # checkout without a build still runs the API; the shell 404s until
    # `bun run build` populates the committed static dir.
    if (_STATIC_DIR / "index.html").is_file():
        _mount_spa(app, _STATIC_DIR)
    return app
