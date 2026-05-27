"""Tests for synara.coordination.unified: the production entry point.

run_unified_async() ties together:
- LeaderRouter (election + promotion + URL routing)
- FastMCPProxy with router.get_client as the client_factory
- RetryMiddleware that invalidates the router on retry (so a transient
  failure against a now-dead leader triggers a fresh resolve_url() on
  the next attempt, which may promote this process)
- stdio transport
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import httpx
import pytest
from fastmcp.server.middleware.middleware import MiddlewareContext

from synara.coordination import follower, unified


@pytest.fixture(autouse=True)
def _xdg(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("XDG_RUNTIME_DIR", str(tmp_path))


def _fake_settings() -> Any:
    class _Dashboard:
        host = "127.0.0.1"
        port = 8765

    class _Settings:
        dashboard = _Dashboard()

    return _Settings()


class _RecordingProxy:
    """Stand-in for FastMCPProxy that captures wiring + records add_middleware."""

    def __init__(self, *, client_factory: Any, name: str) -> None:
        self.client_factory = client_factory
        self.name = name
        self.middlewares: list[Any] = []

    def add_middleware(self, mw: Any) -> None:
        self.middlewares.append(mw)

    async def run_stdio_async(self) -> None:  # pragma: no cover - injected
        return None


def _record_proxy_factory() -> tuple[list[_RecordingProxy], Any]:
    created: list[_RecordingProxy] = []

    def factory(*, client_factory: Any, name: str) -> _RecordingProxy:
        p = _RecordingProxy(client_factory=client_factory, name=name)
        created.append(p)
        return p

    return created, factory


# ----- wiring -----


async def test_unified_builds_proxy_with_router_client_factory(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    created, factory = _record_proxy_factory()
    monkeypatch.setattr(unified, "_make_proxy", factory)

    # No leader → router will eagerly promote. Provide injected runner.
    runner_calls: list[Any] = []

    async def fake_server_runner(*args: Any, **kwargs: Any) -> None:
        runner_calls.append((args, kwargs))
        # Pretend to bind and serve forever.
        await asyncio.Event().wait()

    monkeypatch.setattr(
        unified, "_make_router", lambda **kw: _make_router_with_runner(kw, runner_calls)
    )

    async def fake_stdio_runner(_proxy: Any) -> None:
        return None

    await unified.run_unified_async(
        settings=_fake_settings(),
        build_server=object,
        stdio_runner=fake_stdio_runner,
    )

    assert len(created) == 1
    assert created[0].name == "synara-mcp"
    assert callable(created[0].client_factory)


async def test_unified_attaches_retry_middleware(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    created, factory = _record_proxy_factory()
    monkeypatch.setattr(unified, "_make_proxy", factory)
    monkeypatch.setattr(unified, "_make_router", lambda **_: _AlwaysLeaderRouter())

    async def fake_stdio_runner(_p: Any) -> None:
        return None

    await unified.run_unified_async(
        settings=_fake_settings(),
        build_server=object,
        stdio_runner=fake_stdio_runner,
    )
    assert len(created[0].middlewares) == 1
    assert isinstance(created[0].middlewares[0], follower.RetryMiddleware)


async def test_unified_eagerly_resolves_url_before_stdio(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Election must complete before stdio handshake begins, so the
    dashboard is up when the user clicks the URL."""
    order: list[str] = []

    class _OrderedRouter:
        is_leader = True

        async def resolve_url(self) -> str:
            order.append("resolve")
            return "http://127.0.0.1:1/mcp/"

        def invalidate(self) -> None:
            pass

        async def aclose(self) -> None:
            order.append("aclose")

    monkeypatch.setattr(unified, "_make_router", lambda **_: _OrderedRouter())
    monkeypatch.setattr(
        unified,
        "_make_proxy",
        lambda *, client_factory, name: _RecordingProxy(client_factory=client_factory, name=name),
    )

    async def fake_stdio_runner(_p: Any) -> None:
        order.append("stdio")

    await unified.run_unified_async(
        settings=_fake_settings(),
        build_server=object,
        stdio_runner=fake_stdio_runner,
    )
    assert order == ["resolve", "stdio", "aclose"]


async def test_unified_closes_router_on_normal_exit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    closed = {"called": False}

    class _R:
        is_leader = False

        async def resolve_url(self) -> str:
            return "http://127.0.0.1:1/mcp/"

        def invalidate(self) -> None:
            pass

        async def aclose(self) -> None:
            closed["called"] = True

    monkeypatch.setattr(unified, "_make_router", lambda **_: _R())
    monkeypatch.setattr(
        unified,
        "_make_proxy",
        lambda *, client_factory, name: _RecordingProxy(client_factory=client_factory, name=name),
    )

    async def fake_stdio_runner(_p: Any) -> None:
        return None

    await unified.run_unified_async(
        settings=_fake_settings(),
        build_server=object,
        stdio_runner=fake_stdio_runner,
    )
    assert closed["called"] is True


async def test_unified_closes_router_on_stdio_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    closed = {"called": False}

    class _R:
        is_leader = False

        async def resolve_url(self) -> str:
            return "http://127.0.0.1:1/mcp/"

        def invalidate(self) -> None:
            pass

        async def aclose(self) -> None:
            closed["called"] = True

    monkeypatch.setattr(unified, "_make_router", lambda **_: _R())
    monkeypatch.setattr(
        unified,
        "_make_proxy",
        lambda *, client_factory, name: _RecordingProxy(client_factory=client_factory, name=name),
    )

    async def explode(_p: Any) -> None:
        raise RuntimeError("stdio went sideways")

    with pytest.raises(RuntimeError, match="stdio went sideways"):
        await unified.run_unified_async(
            settings=_fake_settings(),
            build_server=object,
            stdio_runner=explode,
        )
    assert closed["called"] is True


# ----- retry middleware invalidates the router on retry -----


async def test_retry_middleware_invalidates_router_between_attempts() -> None:
    invalidations: list[int] = []

    mw = follower.RetryMiddleware(
        max_retries=3, backoff=0.0, on_retry=lambda: invalidations.append(1)
    )
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
    # Two failures → two invalidations.
    assert len(invalidations) == 2


# ----- helpers -----


class _AlwaysLeaderRouter:
    is_leader = True

    async def resolve_url(self) -> str:
        return "http://127.0.0.1:1/mcp/"

    def invalidate(self) -> None:
        pass

    async def aclose(self) -> None:
        return None


def _make_router_with_runner(_kw: Any, _runner_calls: list[Any]) -> Any:
    return _AlwaysLeaderRouter()
