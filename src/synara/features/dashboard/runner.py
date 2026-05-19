"""Run the dashboard as a background task on the FastMCP event loop.

Phase 0.1 established that FastMCP enters its ``@lifespan`` inside the
single anyio loop that then serves stdio, so ``asyncio.create_task``
here attaches to that same loop — the shared ``AsyncVectorDB`` is safe
(one connection, one loop). Teardown drains the server *before* the
caller closes db/embedder.

stdio transport multiplexes MCP JSON-RPC on stdout, so uvicorn must
never write there: ``log_config=None`` leaves logging unconfigured (no
stdout handler is installed) and access logging is disabled.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
from collections.abc import AsyncIterator
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - typing only
    from fastapi import FastAPI

    from .config import DashboardConfig

_logger = logging.getLogger(__name__)
_DRAIN_TIMEOUT_SECONDS = 5.0


def _make_server(app: FastAPI, cfg: DashboardConfig) -> object:
    import uvicorn  # noqa: PLC0415 - lazy: optional [dashboard] dependency

    config = uvicorn.Config(
        app,
        host=cfg.host,
        port=cfg.port,
        log_config=None,
        access_log=False,
        lifespan="off",
    )
    return uvicorn.Server(config)


@contextlib.asynccontextmanager
async def run_dashboard(app: FastAPI, cfg: DashboardConfig) -> AsyncIterator[None]:
    """Serve the dashboard for the lifetime of the context.

    Spawns ``uvicorn.Server.serve()`` as a task on the current loop;
    on exit signals ``should_exit`` and awaits a bounded drain, then
    cancels if it overran. FastMCP shields lifespan teardown from
    cancellation (Phase 0.1), so this drain survives Ctrl-C.
    """
    server = _make_server(app, cfg)
    task = asyncio.create_task(server.serve(), name="synara-dashboard")  # type: ignore[attr-defined]
    _logger.info("dashboard listening on http://%s:%d", cfg.host, cfg.port)
    try:
        yield
    finally:
        server.should_exit = True  # type: ignore[attr-defined]
        try:
            await asyncio.wait_for(task, _DRAIN_TIMEOUT_SECONDS)
        except (TimeoutError, asyncio.CancelledError):
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task
