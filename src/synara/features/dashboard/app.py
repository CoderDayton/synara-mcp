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
from fastapi.responses import FileResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles
from starlette.datastructures import Headers
from starlette.types import ASGIApp, Receive, Scope, Send

from synara import __version__

from .auth import make_auth_dependency
from .config import DashboardConfig
from .routes import admin, graph, health, memories, stats

_STATIC_DIR = Path(__file__).parent / "static"


def _strip_port(host_header: str) -> str:
    """Extract the host portion from a ``Host`` header value.

    Handles both forms:
    - IPv4 / hostname: ``127.0.0.1:8765`` -> ``127.0.0.1``
    - IPv6 bracketed:  ``[::1]:8765``     -> ``::1``

    ``starlette.middleware.trustedhost`` does a naive ``split(":")[0]``
    which mangles IPv6 to just ``"["`` — we need a correct strip so
    operators can bind to ``::1`` without the middleware rejecting
    their own dashboard.
    """
    if host_header.startswith("["):
        close = host_header.find("]")
        if close != -1:
            return host_header[1:close]
        return host_header
    return host_header.split(":", 1)[0]


def _allowed_hosts(dashboard: DashboardConfig) -> frozenset[str]:
    """Host-header allowlist for :class:`HostAllowlistMiddleware`.

    Blocks DNS-rebinding attacks against the loopback default: an
    attacker-controlled origin bound (via DNS) to 127.0.0.1 still
    presents its own hostname in the ``Host`` header, which is rejected
    before any route or auth dependency runs. Reverse proxies that
    rewrite Host must point ``SYNARA_DASHBOARD_HOST`` at the externally
    visible name (and supply a token).

    Entries are bare hosts only (no ``:port`` suffix) — the middleware
    strips the port before matching.
    """
    hosts: set[str] = {dashboard.host}
    if dashboard.is_loopback:
        # Cover every loopback alias a client could legitimately send,
        # regardless of which loopback variant ``SYNARA_DASHBOARD_HOST``
        # was set to. Bracketed forms are unnecessary because
        # ``_strip_port`` returns the inner address for IPv6.
        hosts.update({"127.0.0.1", "localhost", "::1"})
    return frozenset(hosts)


class HostAllowlistMiddleware:
    """Reject requests whose ``Host`` header is not in the allowlist.

    Stricter, port-aware replacement for Starlette's
    :class:`TrustedHostMiddleware`, which strips on the first ``:`` and
    therefore cannot match IPv6 hosts like ``[::1]:8765`` correctly.
    Matching is exact against the host portion only; the port is
    stripped first.
    """

    __slots__ = ("_allow_any", "_allowed", "_app")

    def __init__(self, app: ASGIApp, *, allowed_hosts: frozenset[str]) -> None:
        self._app = app
        self._allowed = allowed_hosts
        self._allow_any = "*" in allowed_hosts

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] not in ("http", "websocket") or self._allow_any:
            await self._app(scope, receive, send)
            return
        headers = Headers(scope=scope)
        host = _strip_port(headers.get("host", ""))
        if host in self._allowed:
            await self._app(scope, receive, send)
            return
        response = PlainTextResponse("Invalid host header", status_code=400)
        await response(scope, receive, send)


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

    # Host allowlist runs *before* routing/auth so a hostile ``Host``
    # header (DNS rebinding, naked-IP probes against an externally-named
    # bind) is rejected with 400 without ever touching the bearer-token
    # dependency.
    app.add_middleware(
        HostAllowlistMiddleware,
        allowed_hosts=_allowed_hosts(settings.dashboard),
    )

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
