"""FastAPI assembly for the parallel admin console.

Imported lazily (see package ``__getattr__``) so a default,
non-dashboard install never loads FastAPI/uvicorn.

The app is wired with the *live* objects the MCP tools use — the same
``MemoryService`` instance, ``db``, and ``embedder`` — so the dashboard
observes and mutates exactly one shared state (no SR/plasticity
divergence). Routers delegate to the service; no memory logic here.
"""

from __future__ import annotations

import base64
import hashlib
import re
import time
from pathlib import Path
from typing import TYPE_CHECKING

from fastapi import Depends, FastAPI, HTTPException, status
from fastapi.responses import FileResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles
from starlette.datastructures import Headers, MutableHeaders
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from synara import __version__

from .auth import make_auth_dependency
from .config import DashboardConfig
from .routes import admin, graph, health, memories, stats
from .routes import tools as tools_route

_STATIC_DIR = Path(__file__).parent / "static"

# Bind addresses that mean "all interfaces" rather than a real Host a
# client would send. A literal allowlist of these would reject everything.
# These are Host-header *match* sentinels, not a socket bind (hence nosec).
_WILDCARD_BIND_HOSTS = frozenset({"0.0.0.0", "::", "[::]"})  # nosec B104


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
    # Wildcard bind (``0.0.0.0`` / ``::``) is not a real Host header any
    # client would send, so a literal allowlist of it would 400 every
    # request. Binding to all interfaces is an explicit "accept from
    # anywhere" choice (already gated on a token for non-loopback), so
    # disable Host matching with the allow-any sentinel.
    if dashboard.host in _WILDCARD_BIND_HOSTS:
        return frozenset({"*"})
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


# NOTE: the non-greedy body match terminates at the first ``</script>``.
# A script body that itself contains the literal string ``</script>``
# (e.g. inside a JS string) would be truncated, yielding a CSP hash that
# does not cover the full body and so blocks the script. The committed
# shell contains only a small theme-bootstrap script with no such
# literal; if that ever changes, switch to an HTML parser here.
_INLINE_SCRIPT_RE = re.compile(
    rb"<script(?![^>]*\bsrc=)[^>]*>(.*?)</script>",
    re.IGNORECASE | re.DOTALL,
)


def _csp_hashes_for(index_html: Path) -> tuple[str, ...]:
    """Compute CSP `sha256-...` source values for inline ``<script>`` bodies.

    The committed SPA shell contains a small theme-bootstrap inline
    script. A strict CSP without ``'unsafe-inline'`` requires either a
    per-request nonce (we serve the static file unchanged) or a hash of
    each inline script body. We bake the hash at mount time so any
    change to the shell forces a regenerated CSP header automatically.
    """
    if not index_html.is_file():
        return ()
    body = index_html.read_bytes()
    out: list[str] = []
    for match in _INLINE_SCRIPT_RE.finditer(body):
        digest = hashlib.sha256(match.group(1)).digest()
        out.append(f"'sha256-{base64.b64encode(digest).decode('ascii')}'")
    return tuple(out)


class SecurityHeadersMiddleware:
    """Set production-grade security response headers on every response.

    CSP is the most consequential header: it is applied only to HTML
    responses (where script/style/frame loading is meaningful) so that
    hashed ``/assets`` and JSON API replies aren't burdened with policy
    they can't violate. The other headers (X-Content-Type-Options,
    X-Frame-Options, Referrer-Policy, Permissions-Policy) are cheap and
    apply universally.

    The CSP intentionally allows ``'unsafe-inline'`` for ``style-src``
    only — React/shadcn emit inline ``style=`` attributes for
    runtime-computed values, and the equivalent ``style-src-attr``
    directive has incomplete browser coverage. Inline styles don't
    grant script execution, so the residual XSS surface is bounded to
    layout/visual mischief.
    """

    __slots__ = ("_app", "_csp_html")

    def __init__(self, app: ASGIApp, *, csp_html: str) -> None:
        self._app = app
        self._csp_html = csp_html

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self._app(scope, receive, send)
            return

        async def send_wrapper(message: Message) -> None:
            if message["type"] == "http.response.start":
                headers = MutableHeaders(scope=message)
                headers["X-Content-Type-Options"] = "nosniff"
                headers["X-Frame-Options"] = "DENY"
                headers["Referrer-Policy"] = "no-referrer"
                headers["Permissions-Policy"] = (
                    "accelerometer=(), camera=(), geolocation=(), gyroscope=(), "
                    "magnetometer=(), microphone=(), payment=(), usb=()"
                )
                content_type = headers.get("content-type", "")
                if content_type.startswith("text/html"):
                    headers["Content-Security-Policy"] = self._csp_html
            await send(message)

        await self._app(scope, receive, send_wrapper)


def _build_csp(script_hashes: tuple[str, ...]) -> str:
    """Build the HTML Content-Security-Policy string.

    ``script-src`` admits ``'self'`` (the hashed Vite bundle under
    ``/assets``) plus the precomputed inline-script hashes. No remote
    origins are allowed — the dashboard is fully self-hosted.
    """
    script_src = " ".join(("'self'", *script_hashes)) if script_hashes else "'self'"
    return "; ".join(
        (
            "default-src 'self'",
            f"script-src {script_src}",
            "style-src 'self' 'unsafe-inline'",
            "img-src 'self' data:",
            "font-src 'self' data:",
            "connect-src 'self'",
            "frame-ancestors 'none'",
            "base-uri 'self'",
            "form-action 'self'",
            "object-src 'none'",
        )
    )


def _mount_spa(app: FastAPI, static_dir: Path) -> None:
    """Serve the committed Vite build with SPA-history fallback.

    The static shell is intentionally *unauthenticated*: it carries no
    data (every datum is behind the guarded ``/api`` routers) and the
    bearer token is entered through the UI itself, so gating the shell
    would be a chicken-and-egg lockout. Hashed ``/assets`` are immutable;
    any other non-API path returns ``index.html`` for client routing.
    """
    index = static_dir / "index.html"
    # Resolve once at mount: the static directory is fixed for the
    # lifetime of the process, so re-resolving it on every catch-all hit
    # was a needless per-request syscall.
    root = static_dir.resolve()
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
        candidate = (static_dir / full_path).resolve()
        if full_path and candidate.is_relative_to(root) and candidate.is_file():
            return FileResponse(candidate)
        return FileResponse(index)


if TYPE_CHECKING:  # pragma: no cover - typing only
    from simplevecdb import AsyncVectorDB

    from synara.config import Settings
    from synara.features.embedding import Embedder
    from synara.features.memory import MemoryService, ToolMetrics


def build_dashboard_app(
    *,
    settings: Settings,
    db: AsyncVectorDB,
    embedder: Embedder | None,
    service: MemoryService,
    tool_metrics: ToolMetrics | None = None,
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
    # Default to a fresh, empty collector so dashboard-only tests
    # don't have to wire one. NOTE: the fallback is *observably empty*
    # — `register_tools` is never called against it, so the
    # ``/api/tool-metrics`` route will return ``{"tools": []}`` until
    # something records into this instance. The frontend renders that
    # as "no tools registered." In production, server.py passes the
    # same instance the MCP tool wrappers write into.
    from synara.features.memory import ToolMetrics as _ToolMetrics  # noqa: PLC0415

    app.state.tool_metrics = tool_metrics if tool_metrics is not None else _ToolMetrics()
    app.state.started_at = time.monotonic()

    # Host allowlist runs *before* routing/auth so a hostile ``Host``
    # header (DNS rebinding, naked-IP probes against an externally-named
    # bind) is rejected with 400 without ever touching the bearer-token
    # dependency.
    app.add_middleware(
        HostAllowlistMiddleware,
        allowed_hosts=_allowed_hosts(settings.dashboard),
    )

    # Security headers run after the host allowlist (added later wraps
    # outer): hashes are derived from the *committed* shell, so a fresh
    # build that changes the inline bootstrap will regenerate them on
    # the next process restart.
    app.add_middleware(
        SecurityHeadersMiddleware,
        csp_html=_build_csp(_csp_hashes_for(_STATIC_DIR / "index.html")),
    )

    auth = make_auth_dependency(settings.dashboard)
    guarded = [Depends(auth)]
    for module in (health, stats, memories, graph, admin, tools_route):
        app.include_router(module.router, prefix="/api", dependencies=guarded)

    # Serve the bundled SPA when it has been built/committed. A source
    # checkout without a build still runs the API; the shell 404s until
    # `bun run build` populates the committed static dir.
    if (_STATIC_DIR / "index.html").is_file():
        _mount_spa(app, _STATIC_DIR)
    return app
