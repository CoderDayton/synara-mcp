"""Leader election + liveness via BSD ``flock``.

Why ``flock`` and not ``fcntl``:

* ``flock`` ties the lock to the open file description, so duplicate FDs
  via ``dup()``/``fork()`` share the lock and the kernel releases it
  only when every FD to that description closes. ``fcntl`` locks
  release on *any* ``close()`` of *any* FD the process holds to the
  file — a notorious foot-gun (e.g., an unrelated module opening the
  same path drops your lock without warning).
* ``flock`` semantics are simpler: one type per file, no byte ranges,
  no PID-tracking subtleties.
* Death-release behavior is identical, which is all we need for
  liveness.

Probe pattern: open a *separate* FD to the lockfile and try
``LOCK_EX | LOCK_NB``. flock(2) explicitly says distinct open() calls
in the same process produce independent locks, so this works whether
the current process is the leader, a follower, or somewhere in between.
A successful acquire = the previous leader is gone (or never was);
EWOULDBLOCK = leader alive.
"""

from __future__ import annotations

import contextlib
import errno
import fcntl
import logging
import os
from pathlib import Path
from types import TracebackType
from typing import Self

from synara.coordination import discovery

_logger = logging.getLogger(__name__)


class Leadership:
    """Holds the leader flock for its lifetime. Close to release."""

    __slots__ = ("_closed", "_fd", "_path")

    def __init__(self, fd: int, path: Path) -> None:
        self._fd = fd
        self._path = path
        self._closed = False

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        with contextlib.suppress(OSError):
            fcntl.flock(self._fd, fcntl.LOCK_UN)
        with contextlib.suppress(OSError):
            os.close(self._fd)


def _open_lockfile() -> int:
    discovery.ensure_runtime_dir()
    path = discovery.lock_path()
    # O_CLOEXEC: child processes (e.g., uvicorn workers) don't inherit and
    # accidentally extend the lock's lifetime.
    return os.open(str(path), os.O_RDWR | os.O_CREAT | os.O_CLOEXEC, 0o600)


def try_acquire_leadership() -> Leadership | None:
    """Try to become leader. Returns a Leadership token or None if held elsewhere."""
    fd = _open_lockfile()
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError as e:
        os.close(fd)
        if e.errno in (errno.EWOULDBLOCK, errno.EAGAIN):
            return None
        raise
    return Leadership(fd, discovery.lock_path())


def probe_leader_dead() -> bool:
    """Return True if no process currently holds the leader lock.

    Opens a separate FD and attempts a non-blocking exclusive lock —
    success means the lock is unheld (no leader, or it died). The probe
    releases its acquisition immediately; it does **not** become leader.
    """
    try:
        fd = _open_lockfile()
    except OSError:
        # Can't even open the file (no XDG_RUNTIME_DIR, permission, etc.).
        # Treat as "we don't know" → assume alive so we don't stampede a
        # promotion based on filesystem confusion.
        return False
    try:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as e:
            if e.errno in (errno.EWOULDBLOCK, errno.EAGAIN):
                return False
            raise
        # We got the lock — leader is dead. Release immediately.
        with contextlib.suppress(OSError):
            fcntl.flock(fd, fcntl.LOCK_UN)
        return True
    finally:
        with contextlib.suppress(OSError):
            os.close(fd)
