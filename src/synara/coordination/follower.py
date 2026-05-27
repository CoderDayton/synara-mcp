"""Retry middleware for proxied tool calls.

When a follower (or the in-process leader's localhost roundtrip)
forwards a call to the current leader and the HTTP transport fails
transiently — typical during a leader-promotion window of a few
hundred ms — the retry middleware re-attempts the call. Between
attempts it invokes a caller-supplied ``on_retry`` hook (the router
uses this to invalidate its cached leader URL and re-resolve).

A ``ToolError`` from the leader means the leader ran the tool and
reported failure — that's the leader's answer, not a liveness issue,
so it propagates unchanged. Only network-level errors are retried.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable
from typing import Any

import httpx
from fastmcp.server.middleware.middleware import CallNext, Middleware, MiddlewareContext

_logger = logging.getLogger(__name__)

_DEFAULT_TIMEOUT_SECONDS = 30.0
_DEFAULT_MAX_RETRIES = 3
_DEFAULT_BACKOFF_SECONDS = 0.2

# Errors that mean "couldn't reach the leader" — safe to retry.
_RETRYABLE_EXCEPTIONS: tuple[type[BaseException], ...] = (
    httpx.ConnectError,
    httpx.ConnectTimeout,
    httpx.ReadError,
    httpx.ReadTimeout,
    httpx.RemoteProtocolError,
    httpx.WriteError,
    ConnectionError,
    ConnectionRefusedError,
    ConnectionResetError,
    TimeoutError,
)


class RetryMiddleware(Middleware):
    """Retry leader-side calls on network-level failures.

    Tool calls, resource reads, and prompt reads are all proxied
    through the same HTTP transport, so they share the retry policy.
    Non-network errors (``ToolError``, ``McpError``, etc.) propagate
    unchanged.
    """

    def __init__(
        self,
        *,
        max_retries: int = _DEFAULT_MAX_RETRIES,
        backoff: float = _DEFAULT_BACKOFF_SECONDS,
        on_retry: Callable[[], None] | None = None,
    ) -> None:
        if max_retries < 1:
            raise ValueError(f"max_retries must be >= 1, got {max_retries}")
        self.max_retries = max_retries
        self.backoff = backoff
        # Invoked between attempts to invalidate any cached leader URL,
        # so the next attempt re-resolves discovery (and may promote
        # this process if the lock is unheld).
        self.on_retry = on_retry

    async def _retry(
        self,
        context: MiddlewareContext[Any],
        call_next: CallNext[Any, Any],
    ) -> Any:
        last_exc: BaseException | None = None
        for attempt in range(self.max_retries):
            try:
                return await call_next(context)
            except _RETRYABLE_EXCEPTIONS as exc:
                last_exc = exc
                if attempt < self.max_retries - 1:
                    if self.on_retry is not None:
                        self.on_retry()
                    delay = self.backoff * (2**attempt)
                    _logger.warning(
                        "leader RPC failed (%s); retry %d/%d in %.2fs",
                        type(exc).__name__,
                        attempt + 1,
                        self.max_retries,
                        delay,
                    )
                    await asyncio.sleep(delay)
        if last_exc is None:  # pragma: no cover - unreachable
            raise RuntimeError("retry loop exited without recording an exception")
        raise last_exc

    async def on_call_tool(
        self,
        context: MiddlewareContext[Any],
        call_next: CallNext[Any, Any],
    ) -> Any:
        return await self._retry(context, call_next)

    async def on_read_resource(
        self,
        context: MiddlewareContext[Any],
        call_next: CallNext[Any, Any],
    ) -> Any:
        return await self._retry(context, call_next)

    async def on_get_prompt(
        self,
        context: MiddlewareContext[Any],
        call_next: CallNext[Any, Any],
    ) -> Any:
        return await self._retry(context, call_next)

    async def on_list_tools(
        self,
        context: MiddlewareContext[Any],
        call_next: CallNext[Any, Any],
    ) -> Any:
        return await self._retry(context, call_next)
