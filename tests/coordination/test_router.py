"""Tests for synara.coordination.router.LeaderRouter.

The router is the brain of automatic failover. It owns:
- the flock (if this process holds leadership),
- the HTTP server asyncio task (if this process is leader),
- a cache of the current leader URL.

Its single public method ``resolve_url()`` returns the URL of whoever
is currently leader, promoting this process if the lock is unheld.
All tests inject fakes for the HTTP server runner and the build_server
factory so they stay fast and deterministic.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest

from synara.coordination import discovery, election, router


@pytest.fixture(autouse=True)
def _xdg(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("XDG_RUNTIME_DIR", str(tmp_path))


@dataclass
class _FakeRunner:
    """Stand-in for FastMCP run_http_async; records calls and signals readiness."""

    calls: list[dict[str, Any]] = field(default_factory=list)
    ready_after_seconds: float = 0.01
    raises: BaseException | None = None
    _servers: list[asyncio.Server] = field(default_factory=list)

    async def __call__(self, mcp: Any, host: str, port: int, *, stateless_http: bool) -> None:
        self.calls.append(
            {"mcp": mcp, "host": host, "port": port, "stateless_http": stateless_http}
        )
        if self.raises is not None:
            raise self.raises
        # Bind a real TCP socket so probes (open_connection) succeed.
        await asyncio.sleep(self.ready_after_seconds)

        async def _handle(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
            writer.close()
            with contextlib.suppress(ConnectionError, OSError):
                await writer.wait_closed()

        server = await asyncio.start_server(_handle, host, port)
        self._servers.append(server)
        try:
            await server.serve_forever()
        except asyncio.CancelledError:
            server.close()
            await server.wait_closed()
            raise


def _fake_settings(
    host: str = "127.0.0.1", port: int = 8765, *, dashboard_enabled: bool = True
) -> Any:
    class _Dashboard:
        def __init__(self) -> None:
            self.host = host
            self.port = port
            self.enabled = dashboard_enabled

    class _Settings:
        def __init__(self) -> None:
            self.dashboard = _Dashboard()

    return _Settings()


# ----- Plain routing -----


async def test_resolves_existing_leader_url_without_promoting() -> None:
    """A live leader's URL is read from discovery; we do NOT promote."""
    discovery.write_leader_info(
        discovery.LeaderInfo(
            pid=1,
            mcp_url="http://127.0.0.1:40100/mcp/",
            dashboard_url="http://127.0.0.1:8765",
            started_at=0.0,
        )
    )
    # Hold the lock from a fake "alive" leader (separate fd via election).
    holder = election.try_acquire_leadership()
    assert holder is not None
    try:
        runner_fake = _FakeRunner()
        r = router.LeaderRouter(
            settings=_fake_settings(),
            build_server=object,
            port_picker=lambda: 40100,
            server_runner=runner_fake,
        )
        url = await r.resolve_url()
        assert url == "http://127.0.0.1:40100/mcp/"
        assert r.is_leader is False
        assert runner_fake.calls == [], "must not start HTTP when not promoting"
    finally:
        holder.close()


# ----- Promotion -----


async def test_promotes_when_no_leader_exists(tmp_path: Path) -> None:
    runner_fake = _FakeRunner()
    sentinel = object()
    picked_port = _free_loopback_port()

    r = router.LeaderRouter(
        settings=_fake_settings(),
        build_server=lambda: sentinel,
        port_picker=lambda: picked_port,
        server_runner=runner_fake,
    )
    try:
        url = await r.resolve_url()
        assert url == f"http://127.0.0.1:{picked_port}/mcp/"
        assert r.is_leader is True
        assert len(runner_fake.calls) == 1
        assert runner_fake.calls[0]["port"] == picked_port
        assert runner_fake.calls[0]["stateless_http"] is True
        assert runner_fake.calls[0]["mcp"] is sentinel
    finally:
        await r.aclose()


async def test_promotion_writes_leader_info_with_pid_and_url() -> None:
    picked_port = _free_loopback_port()
    r = router.LeaderRouter(
        settings=_fake_settings(host="0.0.0.0", port=9000),
        build_server=object,
        port_picker=lambda: picked_port,
        server_runner=_FakeRunner(),
    )
    try:
        await r.resolve_url()
        info = discovery.read_leader_info()
        assert info is not None
        assert info.pid == os.getpid()
        assert info.mcp_url == f"http://127.0.0.1:{picked_port}/mcp/"
        assert info.dashboard_url == "http://0.0.0.0:9000"
    finally:
        await r.aclose()


async def test_does_not_double_promote_when_already_leader() -> None:
    """Repeated resolve_url() calls do not start a second HTTP task."""
    runner_fake = _FakeRunner()
    picked_port = _free_loopback_port()
    r = router.LeaderRouter(
        settings=_fake_settings(),
        build_server=object,
        port_picker=lambda: picked_port,
        server_runner=runner_fake,
    )
    try:
        url1 = await r.resolve_url()
        url2 = await r.resolve_url()
        url3 = await r.resolve_url()
        assert url1 == url2 == url3
        assert len(runner_fake.calls) == 1
    finally:
        await r.aclose()


async def test_concurrent_resolve_serializes_to_single_promotion() -> None:
    """Multiple concurrent resolve_url() in the same router → one promotion."""
    runner_fake = _FakeRunner()
    picked_port = _free_loopback_port()
    r = router.LeaderRouter(
        settings=_fake_settings(),
        build_server=object,
        port_picker=lambda: picked_port,
        server_runner=runner_fake,
    )
    try:
        results = await asyncio.gather(r.resolve_url(), r.resolve_url(), r.resolve_url())
        assert len(set(results)) == 1
        assert len(runner_fake.calls) == 1
    finally:
        await r.aclose()


async def test_waits_for_concurrent_promoter_to_publish_info() -> None:
    """Another process holds the lock but hasn't written leader.json yet.

    The router must poll briefly rather than fail outright.
    """
    holder = election.try_acquire_leadership()
    assert holder is not None
    info = discovery.LeaderInfo(
        pid=999,
        mcp_url="http://127.0.0.1:40200/mcp/",
        dashboard_url="http://127.0.0.1:8765",
        started_at=0.0,
    )

    async def _publish_after_delay() -> None:
        await asyncio.sleep(0.1)
        discovery.write_leader_info(info)

    r = router.LeaderRouter(
        settings=_fake_settings(),
        build_server=object,
        port_picker=lambda: 40200,
        server_runner=_FakeRunner(),
    )
    try:
        publish_task = asyncio.create_task(_publish_after_delay())
        url = await r.resolve_url()
        await publish_task
        assert url == "http://127.0.0.1:40200/mcp/"
        assert r.is_leader is False
    finally:
        holder.close()


async def test_promotion_failure_releases_lock() -> None:
    """If the HTTP runner blows up at startup, the flock must release."""

    async def boom(*_args: Any, **_kwargs: Any) -> None:
        raise RuntimeError("server failed to start")

    r = router.LeaderRouter(
        settings=_fake_settings(),
        build_server=object,
        port_picker=_free_loopback_port,
        server_runner=boom,
    )
    with pytest.raises(RuntimeError, match="server failed"):
        await r.resolve_url()
    # Next process must be able to acquire the lock.
    assert election.probe_leader_dead() is True


async def test_aclose_releases_lock_and_cancels_http_task() -> None:
    runner_fake = _FakeRunner()
    r = router.LeaderRouter(
        settings=_fake_settings(),
        build_server=object,
        port_picker=_free_loopback_port,
        server_runner=runner_fake,
    )
    await r.resolve_url()
    assert r.is_leader is True
    assert election.probe_leader_dead() is False

    await r.aclose()
    assert r.is_leader is False
    assert election.probe_leader_dead() is True


async def test_aclose_logs_when_http_task_ended_with_error(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """If the HTTP runner died with a non-cancellation exception, aclose
    must log it rather than silently swallow — otherwise leader crashes
    leave no trace in the journal."""
    r = router.LeaderRouter(
        settings=_fake_settings(),
        build_server=object,
        port_picker=lambda: 40500,
        server_runner=_FakeRunner(),
    )

    async def crashed() -> None:
        raise RuntimeError("server crashed mid-run")

    task = asyncio.create_task(crashed(), name="synara-leader-http")
    with contextlib.suppress(RuntimeError):
        await task
    r._http_task = task  # isolating aclose contract

    with caplog.at_level(logging.WARNING, logger="synara.coordination.router"):
        await r.aclose()

    messages = " ".join(rec.getMessage() for rec in caplog.records)
    assert "leader" in messages.lower() or "http" in messages.lower(), messages


async def test_promotion_failure_does_not_leak_pending_http_task() -> None:
    """After a failed promotion the cancelled HTTP task must be awaited;
    otherwise asyncio reports "Task was destroyed but it is pending" and
    the orphan socket may outlive the process boot attempt."""

    async def slow_then_boom(*_args: Any, **_kwargs: Any) -> None:
        await asyncio.sleep(0.2)
        raise RuntimeError("delayed crash")

    r = router.LeaderRouter(
        settings=_fake_settings(),
        build_server=object,
        port_picker=_free_loopback_port,
        server_runner=slow_then_boom,
        bind_wait_seconds=0.02,
    )
    with pytest.raises(router.NoLeaderError):
        await r.resolve_url()

    leaked = [
        t for t in asyncio.all_tasks() if t.get_name() == "synara-leader-http" and not t.done()
    ]
    assert leaked == [], f"leaked tasks: {leaked}"


async def test_promotion_omits_dashboard_url_when_dashboard_disabled() -> None:
    """leader.json must not advertise a dashboard URL when nothing is bound there."""
    r = router.LeaderRouter(
        settings=_fake_settings(dashboard_enabled=False),
        build_server=object,
        port_picker=_free_loopback_port,
        server_runner=_FakeRunner(),
    )
    try:
        await r.resolve_url()
        info = discovery.read_leader_info()
        assert info is not None
        assert info.dashboard_url == ""
    finally:
        await r.aclose()


async def test_re_elects_after_leader_http_task_crashes() -> None:
    """A leader whose HTTP task dies mid-life must demote on the next
    resolve_url(); otherwise retries forward to a dead local URL forever."""
    runner_fake = _FakeRunner()
    ports = iter([_free_loopback_port(), _free_loopback_port()])
    r = router.LeaderRouter(
        settings=_fake_settings(),
        build_server=object,
        port_picker=lambda: next(ports),
        server_runner=runner_fake,
    )
    try:
        url1 = await r.resolve_url()
        assert r.is_leader

        # Simulate the leader's HTTP task crashing post-bind: swap in a
        # task that's already finished with an exception, mirroring
        # what a real uvicorn-died scenario leaves behind.
        async def crashed() -> None:
            raise RuntimeError("leader HTTP died mid-life")

        crashed_task = asyncio.create_task(crashed(), name="synara-leader-http")
        with contextlib.suppress(RuntimeError):
            await crashed_task
        # Cancel the original healthy runner before replacement so it
        # doesn't keep the port bound.
        assert r._http_task is not None
        r._http_task.cancel()
        with contextlib.suppress(asyncio.CancelledError, Exception):
            await r._http_task
        r._http_task = crashed_task

        # Next resolve must detect the crash, demote, re-elect.
        url2 = await r.resolve_url()
        assert url1 != url2
        assert r.is_leader
        assert len(runner_fake.calls) == 2
    finally:
        await r.aclose()


async def test_resolve_url_handles_cancelled_http_task() -> None:
    """A *cancelled* leader HTTP task must not crash resolve_url().

    ``Task.exception()`` raises ``CancelledError`` on a cancelled task, so
    the done-task check must skip it via ``Task.cancelled()`` rather than
    treating cancellation as a real crash. Regression for the unguarded
    ``.exception()`` call in ``_resolve_locked``.
    """
    runner_fake = _FakeRunner()
    ports = iter([_free_loopback_port(), _free_loopback_port()])
    r = router.LeaderRouter(
        settings=_fake_settings(),
        build_server=object,
        port_picker=lambda: next(ports),
        server_runner=runner_fake,
    )
    try:
        await r.resolve_url()
        assert r.is_leader

        # Replace the live HTTP task with a cancelled one.
        async def _never() -> None:
            await asyncio.sleep(3600)

        cancelled_task = asyncio.create_task(_never(), name="synara-leader-http")
        assert r._http_task is not None
        r._http_task.cancel()
        with contextlib.suppress(asyncio.CancelledError, Exception):
            await r._http_task
        cancelled_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await cancelled_task
        assert cancelled_task.cancelled()
        r._http_task = cancelled_task

        # Must not raise CancelledError. A cancelled task is not a crash,
        # so leadership is retained and the cached URL is returned.
        url2 = await r.resolve_url()
        assert url2
        assert r.is_leader
    finally:
        await r.aclose()


async def test_resolve_blocks_until_leader_appears_or_times_out() -> None:
    """Lock held but no leader.json and never written → timeout, not hang."""
    holder = election.try_acquire_leadership()
    assert holder is not None
    try:
        r = router.LeaderRouter(
            settings=_fake_settings(),
            build_server=object,
            port_picker=lambda: 40300,
            server_runner=_FakeRunner(),
            promotion_wait_seconds=0.3,
        )
        with pytest.raises(router.NoLeaderError):
            await r.resolve_url()
    finally:
        holder.close()


# ----- helpers -----


def _free_loopback_port() -> int:
    import socket  # noqa: PLC0415

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])
