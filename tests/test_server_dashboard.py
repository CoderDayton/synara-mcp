"""Phase 3.1 — dashboard lifecycle wiring + stdio-safety.

Drives the real FastMCP lifespan (``mcp._lifespan_manager()`` — the
same path ``run_stdio_async`` uses, per Phase 0.1) with the dashboard
enabled/disabled and asserts: task spawned only when enabled, real HTTP
reachable, nothing written to stdout (stdio transport safety), and clean
drain with no leaked task on exit.
"""

from __future__ import annotations

import asyncio
import contextlib
import io
import socket

import httpx
import pytest

from synara.config import Settings
from synara.server import build_server


def _free_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return int(port)


def _dash_task() -> asyncio.Task[object] | None:
    for t in asyncio.all_tasks():
        if t.get_name() == "synara-dashboard":
            return t
    return None


@pytest.fixture
def _base_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SYNARA_DB_PATH", ":memory:")
    for v in ("SYNARA_DASHBOARD", "SYNARA_DASHBOARD_HOST", "SYNARA_DASHBOARD_PORT"):
        monkeypatch.delenv(v, raising=False)


@pytest.mark.usefixtures("_base_env")
async def test_disabled_spawns_no_task() -> None:
    mcp = build_server(Settings.from_env())
    async with mcp._lifespan_manager():
        assert _dash_task() is None


@pytest.mark.usefixtures("_base_env")
async def test_enabled_serves_and_is_stdout_safe(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    port = _free_port()
    monkeypatch.setenv("SYNARA_DASHBOARD", "true")
    monkeypatch.setenv("SYNARA_DASHBOARD_PORT", str(port))
    mcp = build_server(Settings.from_env())

    captured = io.StringIO()
    with contextlib.redirect_stdout(captured):
        async with mcp._lifespan_manager():
            task = _dash_task()
            assert task is not None
            assert not task.done()

            async with httpx.AsyncClient(base_url=f"http://127.0.0.1:{port}") as client:
                body = None
                for _ in range(50):  # poll until uvicorn is accepting
                    try:
                        r = await client.get("/api/health", timeout=0.5)
                        if r.status_code == 200:
                            body = r.json()
                            break
                    except httpx.TransportError:
                        await asyncio.sleep(0.05)
                assert body is not None
                assert body["status"] == "ok"

        # Lifespan exited: task drained, not leaked.
        drained = _dash_task()
        assert drained is None or drained.done()

    # stdio transport multiplexes MCP JSON on stdout — must stay clean.
    assert captured.getvalue() == ""

    # Server actually stopped (drain happened before db close).
    async with httpx.AsyncClient(base_url=f"http://127.0.0.1:{port}") as client:
        with pytest.raises(httpx.TransportError):
            await client.get("/api/health", timeout=0.5)
