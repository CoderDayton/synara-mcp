"""Tests for synara.coordination.follower.RetryMiddleware.

The middleware re-attempts network-level failures (httpx transport,
``ConnectionError``, ``TimeoutError``) and propagates everything else
unchanged. Tested directly against a fake ``call_next`` — no proxy,
no HTTP.
"""

from __future__ import annotations

import time
from typing import Any

import httpx
import pytest
from fastmcp.server.middleware.middleware import MiddlewareContext

from synara.coordination import follower


async def test_retry_middleware_returns_value_on_first_success() -> None:
    mw = follower.RetryMiddleware(max_retries=3, backoff=0.0)
    ctx = MiddlewareContext(message=object())
    calls = 0

    async def call_next(_ctx: MiddlewareContext[Any]) -> str:
        nonlocal calls
        calls += 1
        return "ok"

    result = await mw.on_call_tool(ctx, call_next)  # type: ignore[arg-type]
    assert result == "ok"
    assert calls == 1


async def test_retry_middleware_retries_until_success() -> None:
    mw = follower.RetryMiddleware(max_retries=3, backoff=0.0)
    ctx = MiddlewareContext(message=object())
    calls = 0

    async def call_next(_ctx: MiddlewareContext[Any]) -> str:
        nonlocal calls
        calls += 1
        if calls < 3:
            raise httpx.ConnectError("leader gone")
        return "ok"

    result = await mw.on_call_tool(ctx, call_next)  # type: ignore[arg-type]
    assert result == "ok"
    assert calls == 3


async def test_retry_middleware_gives_up_after_max_retries() -> None:
    mw = follower.RetryMiddleware(max_retries=2, backoff=0.0)
    ctx = MiddlewareContext(message=object())
    calls = 0

    async def call_next(_ctx: MiddlewareContext[Any]) -> str:
        nonlocal calls
        calls += 1
        raise httpx.ConnectError("leader gone")

    with pytest.raises(httpx.ConnectError):
        await mw.on_call_tool(ctx, call_next)  # type: ignore[arg-type]
    assert calls == 2


async def test_retry_middleware_does_not_retry_non_transient_errors() -> None:
    """A ToolError from the leader (e.g., invalid args) must not be retried."""
    from fastmcp.exceptions import ToolError  # noqa: PLC0415

    mw = follower.RetryMiddleware(max_retries=5, backoff=0.0)
    ctx = MiddlewareContext(message=object())
    calls = 0

    async def call_next(_ctx: MiddlewareContext[Any]) -> str:
        nonlocal calls
        calls += 1
        raise ToolError("bad arguments")

    with pytest.raises(ToolError):
        await mw.on_call_tool(ctx, call_next)  # type: ignore[arg-type]
    assert calls == 1


async def test_retry_middleware_sleeps_between_attempts() -> None:
    mw = follower.RetryMiddleware(max_retries=3, backoff=0.05)
    ctx = MiddlewareContext(message=object())
    calls = 0

    async def call_next(_ctx: MiddlewareContext[Any]) -> str:
        nonlocal calls
        calls += 1
        if calls < 3:
            raise httpx.ConnectError("leader gone")
        return "ok"

    t0 = time.monotonic()
    await mw.on_call_tool(ctx, call_next)  # type: ignore[arg-type]
    elapsed = time.monotonic() - t0
    assert elapsed >= 0.05


def test_retry_middleware_rejects_max_retries_below_one() -> None:
    """max_retries=0 used to silently fall through to a RuntimeError; reject up front."""
    with pytest.raises(ValueError, match="max_retries"):
        follower.RetryMiddleware(max_retries=0)


async def test_retry_middleware_also_retries_resource_reads() -> None:
    """on_read_resource gets the same retry behavior."""
    mw = follower.RetryMiddleware(max_retries=3, backoff=0.0)
    ctx = MiddlewareContext(message=object())
    calls = 0

    async def call_next(_ctx: MiddlewareContext[Any]) -> str:
        nonlocal calls
        calls += 1
        if calls < 2:
            raise httpx.ReadTimeout("timeout")
        return "ok"

    result = await mw.on_read_resource(ctx, call_next)  # type: ignore[arg-type]
    assert result == "ok"
    assert calls == 2
