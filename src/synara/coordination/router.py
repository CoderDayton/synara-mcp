"""LeaderRouter: dynamic routing + lazy failover promotion.

Every synara-mcp subprocess runs a single ``FastMCPProxy`` over stdio.
That proxy's ``client_factory`` is :meth:`LeaderRouter.get_client`,
which on every call returns a ``Client`` pointed at the **current**
leader's HTTP MCP endpoint — even if leadership just changed.

The router is the failover brain:

* If this process already holds leadership, ``resolve_url`` returns
  our own local HTTP URL (the leader serves its own Claude session via
  a localhost roundtrip — same code path as any other follower).
* If a discoverable leader is alive (``probe_leader_dead() is False``),
  we cache its URL and return it.
* If no leader is discoverable (or the lock is unheld → leader is
  dead), we attempt to acquire the flock. Winner promotes (starts the
  HTTP task, writes ``leader.json``, holds the lock for the rest of
  the process lifetime). Loser polls briefly for the winner's
  ``leader.json`` and resolves to it.

All routing decisions go through an ``asyncio.Lock`` so concurrent
tool calls within one process don't double-promote or race the
discovery file.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import os
import socket
import time
from collections.abc import Awaitable, Callable
from typing import Any

from synara.coordination import discovery, election


def pick_free_port() -> int:
    """Ask the kernel for a free loopback TCP port.

    Bind/getsockname/close gives a port that's free *right now*; there
    is a microsecond-scale race where another process could grab it
    before our HTTP server binds. In practice on a single-user
    workstation this never happens; if it does, the caller should
    retry election.
    """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


_logger = logging.getLogger(__name__)


_DEFAULT_PROMOTION_WAIT_SECONDS = 5.0
_DEFAULT_BIND_WAIT_SECONDS = 5.0
_POLL_INTERVAL_SECONDS = 0.05
_PORT_PICK_ATTEMPTS = 3


class NoLeaderError(RuntimeError):
    """Raised when no leader can be discovered within the wait budget."""


ServerRunner = Callable[..., Awaitable[None]]


async def _default_server_runner(mcp: Any, host: str, port: int, *, stateless_http: bool) -> None:
    await mcp.run_http_async(
        transport="streamable-http",
        host=host,
        port=port,
        stateless_http=stateless_http,
    )


class LeaderRouter:
    """Routes proxy client requests to the current leader; promotes on demand."""

    def __init__(
        self,
        *,
        settings: Any,
        build_server: Callable[[], Any],
        port_picker: Callable[[], int] | None = None,
        server_runner: ServerRunner | None = None,
        promotion_wait_seconds: float = _DEFAULT_PROMOTION_WAIT_SECONDS,
        bind_wait_seconds: float = _DEFAULT_BIND_WAIT_SECONDS,
    ) -> None:
        self._settings = settings
        self._build_server = build_server
        self._port_picker = port_picker or pick_free_port
        self._server_runner = server_runner or _default_server_runner
        self._promotion_wait = promotion_wait_seconds
        self._bind_wait = bind_wait_seconds

        self._leadership: election.Leadership | None = None
        self._http_task: asyncio.Task[None] | None = None
        self._current_url: str | None = None
        self._lock = asyncio.Lock()
        self._closed = False

    # ----- public surface -----

    @property
    def is_leader(self) -> bool:
        return self._leadership is not None

    async def resolve_url(self) -> str:
        """Return the URL of whoever is currently leader, promoting if needed."""
        if self._closed:
            raise NoLeaderError("router is closed")
        async with self._lock:
            return await self._resolve_locked()

    def invalidate(self) -> None:
        """Drop the cached URL so the next resolve_url() re-checks discovery.

        Callers (e.g., a retry middleware) use this when a tool call to
        the cached URL fails — the next call then re-probes the lock
        and, if needed, promotes this process.

        Deliberate no-op while *this* process holds leadership: a failed
        call to our own local endpoint is not a stale-discovery problem,
        and dropping ``_current_url`` here would not change re-resolution
        (we'd just return the same local URL). Demotion of a dead leader
        is handled in :meth:`_resolve_locked` via the HTTP-task check.
        """
        if self._leadership is None:
            self._current_url = None

    async def aclose(self) -> None:
        """Release lock + cancel HTTP task. Idempotent."""
        if self._closed:
            return
        self._closed = True
        if self._http_task is not None:
            self._http_task.cancel()
            try:
                await self._http_task
            except asyncio.CancelledError:
                pass
            except Exception:
                # The HTTP task died with a real error before we got
                # here. We're already shutting down, so don't re-raise
                # (would mask the caller's own teardown exception) —
                # but log it so the crash isn't invisible.
                _logger.warning(
                    "leader HTTP task ended with error during aclose",
                    exc_info=True,
                )
            self._http_task = None
        if self._leadership is not None:
            self._leadership.close()
            self._leadership = None

    # ----- internals -----

    async def _demote_locked(self) -> None:
        """Release leadership + tear down the HTTP task. Caller holds the lock."""
        task = self._http_task
        if task is not None:
            if not task.done():
                task.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await task
            self._http_task = None
        if self._leadership is not None:
            self._leadership.close()
            self._leadership = None
        self._current_url = None

    async def _resolve_locked(self) -> str:
        # If we used to lead but our HTTP task died with an exception,
        # the cached URL points at a dead local endpoint. Demote so the
        # election logic below can promote us again on a fresh port (or
        # follow whoever else won the race).
        if (
            self._http_task is not None
            and self._http_task.done()
            and not self._http_task.cancelled()
            and self._http_task.exception() is not None
        ):
            _logger.warning(
                "leader HTTP task died (%s); demoting and re-electing",
                self._http_task.exception(),
            )
            await self._demote_locked()

        if self._leadership is not None and self._current_url is not None:
            return self._current_url

        # TOCTOU note: the leader can die between ``read_leader_info``
        # and ``probe_leader_dead`` (or just after the probe), so a cached
        # URL may already be stale on return. This is inherent to the
        # flock protocol; it is contained by RetryMiddleware calling
        # ``invalidate()`` on a failed call, which forces a fresh probe +
        # promotion on the next ``resolve_url``.
        info = discovery.read_leader_info()
        if info is not None and not election.probe_leader_dead():
            self._current_url = info.mcp_url
            return self._current_url

        # No live leader. Either there never was one, or the previous
        # leader died and the kernel released its flock. Try to promote.
        return await self._promote_or_follow_winner()

    async def _promote_or_follow_winner(self) -> str:
        leadership = election.try_acquire_leadership()
        if leadership is None:
            # Someone else just acquired the lock. Poll for their
            # leader.json to appear (with a bounded budget so we never
            # hang on a wedged promoter).
            return await self._poll_for_published_leader()

        # We won — promote.
        try:
            # Erase any stale leader.json from the previous (dead) leader
            # first: from this instant the flock is held, so a poller's
            # "file present + lock held" liveness check would otherwise
            # validate the dead leader's URL for the entire bind window
            # (seconds). With the file gone, pollers keep polling until
            # we publish our own info at the end of the promotion.
            discovery.info_path().unlink(missing_ok=True)
            url = await self._promote(leadership)
        except BaseException:
            leadership.close()
            if self._http_task is not None:
                self._http_task.cancel()
                with contextlib.suppress(asyncio.CancelledError, Exception):
                    await self._http_task
                self._http_task = None
            raise
        return url

    async def _promote(self, leadership: election.Leadership) -> str:
        # ``pick_free_port`` is racy (TOCTOU): the kernel-chosen port can
        # be grabbed by another process before our HTTP server binds it.
        # Retry a few times with a fresh port instead of failing the whole
        # election attempt on a single unlucky collision.
        mcp = self._build_server()
        last_exc: Exception | None = None
        for _ in range(_PORT_PICK_ATTEMPTS):
            port = self._port_picker()
            url = f"http://127.0.0.1:{port}/mcp/"

            async def _runner_wrapper(port: int = port) -> None:
                await self._server_runner(mcp, "127.0.0.1", port, stateless_http=True)

            self._http_task = asyncio.create_task(_runner_wrapper(), name="synara-leader-http")
            try:
                await self._wait_for_bind(port, self._http_task)
                break
            except (NoLeaderError, OSError) as exc:
                last_exc = exc
                self._http_task.cancel()
                with contextlib.suppress(asyncio.CancelledError, Exception):
                    await self._http_task
                self._http_task = None
        else:
            raise NoLeaderError(
                f"HTTP server failed to bind after {_PORT_PICK_ATTEMPTS} port attempts"
            ) from last_exc

        dashboard = getattr(self._settings, "dashboard", None)
        if getattr(dashboard, "enabled", False):
            dashboard_host = getattr(dashboard, "host", "127.0.0.1")
            dashboard_port = getattr(dashboard, "port", 8765)
            dashboard_url = f"http://{dashboard_host}:{dashboard_port}"
        else:
            dashboard_url = ""
        discovery.write_leader_info(
            discovery.LeaderInfo(
                pid=os.getpid(),
                mcp_url=url,
                dashboard_url=dashboard_url,
                started_at=time.time(),
            )
        )

        self._leadership = leadership
        self._current_url = url
        _logger.info("promoted to leader: serving http MCP on %s (pid=%d)", url, os.getpid())
        return url

    async def _wait_for_bind(self, port: int, task: asyncio.Task[None]) -> None:
        """Wait until ``port`` accepts a TCP connection, or the task errors."""
        deadline = time.monotonic() + self._bind_wait
        while time.monotonic() < deadline:
            if task.done():
                # Server crashed before binding — surface that exception.
                # A cancelled task has no ``.exception()`` (it would raise
                # CancelledError); treat cancellation as a clean stop.
                if task.cancelled():
                    return
                exc = task.exception()
                if exc is not None:
                    raise exc
                return
            try:
                _, writer = await asyncio.open_connection("127.0.0.1", port)
                writer.close()
                with contextlib.suppress(ConnectionError, OSError):
                    await writer.wait_closed()
                return
            except (OSError, ConnectionRefusedError):
                await asyncio.sleep(_POLL_INTERVAL_SECONDS)
        raise NoLeaderError(f"HTTP server failed to bind on :{port} within {self._bind_wait}s")

    async def _poll_for_published_leader(self) -> str:
        deadline = time.monotonic() + self._promotion_wait
        while time.monotonic() < deadline:
            info = discovery.read_leader_info()
            if info is not None and not election.probe_leader_dead():
                self._current_url = info.mcp_url
                return self._current_url
            await asyncio.sleep(_POLL_INTERVAL_SECONDS)
        raise NoLeaderError(f"lock held but no leader.json appeared within {self._promotion_wait}s")
