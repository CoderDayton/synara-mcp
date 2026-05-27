"""Unified entry point: every synara-mcp subprocess runs the same thing.

A single ``FastMCPProxy`` over stdio whose ``client_factory`` is the
:class:`LeaderRouter`. The router transparently routes each tool call
to whoever is currently leader and promotes this process when the
leader dies — without restarting the subprocess.

This collapses what was originally split as "leader role" vs
"follower role" into one path:

* if we hold leadership, the router returns a Client to our own
  localhost HTTP endpoint (tiny in-process roundtrip);
* if a discoverable leader is alive, the router returns a Client to
  it;
* if no leader is discoverable, the next call promotes us.

Failover is invisible to the Claude session: the
:class:`RetryMiddleware` calls ``router.invalidate()`` between
attempts, so a tool call that hits a dead leader is retried with a
freshly-resolved URL — which, on the second attempt, points at the
newly-promoted leader (which may be this process).
"""

from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable
from typing import Any

from fastmcp.client.client import Client
from fastmcp.server.providers.proxy import FastMCPProxy

from synara.coordination.follower import (
    _DEFAULT_BACKOFF_SECONDS,
    _DEFAULT_MAX_RETRIES,
    _DEFAULT_TIMEOUT_SECONDS,
    RetryMiddleware,
)
from synara.coordination.router import LeaderRouter

_logger = logging.getLogger(__name__)


def _make_router(**kwargs: Any) -> LeaderRouter:
    return LeaderRouter(**kwargs)


def _make_proxy(*, client_factory: Any, name: str) -> FastMCPProxy:
    return FastMCPProxy(client_factory=client_factory, name=name)


async def _default_stdio_runner(proxy: Any) -> None:
    await proxy.run_stdio_async()


async def run_unified_async(
    *,
    settings: Any,
    build_server: Callable[[], Any],
    timeout: float = _DEFAULT_TIMEOUT_SECONDS,
    max_retries: int = _DEFAULT_MAX_RETRIES,
    backoff: float = _DEFAULT_BACKOFF_SECONDS,
    stdio_runner: Callable[[Any], Awaitable[None]] | None = None,
) -> None:
    """Run the stdio MCP server with automatic leadership routing.

    Args:
        settings: synara Settings (used for the dashboard host/port
            recorded in ``leader.json`` after promotion).
        build_server: Factory returning a configured FastMCP server.
            Only invoked when this process is promoted to leader; the
            DB connection lives there.
        timeout: HTTP read timeout for proxied calls.
        max_retries: Retry budget per proxied call (covers leader
            failover windows).
        backoff: Base delay between retries; doubles each attempt.
        stdio_runner: Injected for tests.
    """
    router = _make_router(settings=settings, build_server=build_server)

    async def client_factory() -> Client[Any]:
        url = await router.resolve_url()
        return Client(url, timeout=timeout)

    proxy = _make_proxy(client_factory=client_factory, name="synara-mcp")
    proxy.add_middleware(
        RetryMiddleware(
            max_retries=max_retries,
            backoff=backoff,
            on_retry=router.invalidate,
        )
    )

    runner = stdio_runner or _default_stdio_runner
    try:
        # Eager resolve so the dashboard is bound before the first tool
        # call arrives — visible UX win versus lazy promotion.
        await router.resolve_url()
        if router.is_leader:
            _logger.info("starting stdio proxy (this process is leader)")
        else:
            _logger.info("starting stdio proxy (this process is follower)")
        await runner(proxy)
    finally:
        await router.aclose()


__all__ = ["run_unified_async"]
