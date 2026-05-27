"""Tests for synara.coordination.election: flock-based leader election + probe.

flock semantics we rely on, per flock(2):
- LOCK_EX | LOCK_NB returns EWOULDBLOCK if the lock is held by anyone.
- The lock is associated with the open file description, so two separate
  open() calls in the same process produce *independent* locks — the
  second flock() will fail while the first is held.
- The kernel releases the lock when the last fd referring to the open
  file description is closed (clean exit, crash, segfault, OOM kill).
"""

from __future__ import annotations

import multiprocessing as mp
import os
import time
from pathlib import Path

import pytest

from synara.coordination import discovery, election


@pytest.fixture(autouse=True)
def _xdg_runtime_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("XDG_RUNTIME_DIR", str(tmp_path))


def test_first_caller_acquires_leadership() -> None:
    leadership = election.try_acquire_leadership()
    assert leadership is not None
    leadership.close()


def test_second_caller_returns_none_while_first_holds() -> None:
    first = election.try_acquire_leadership()
    assert first is not None
    try:
        second = election.try_acquire_leadership()
        assert second is None
    finally:
        first.close()


def test_leadership_releases_on_close() -> None:
    first = election.try_acquire_leadership()
    assert first is not None
    first.close()
    # After release, a fresh acquire must succeed.
    second = election.try_acquire_leadership()
    assert second is not None
    second.close()


def test_leadership_works_as_context_manager() -> None:
    lead = election.try_acquire_leadership()
    assert lead is not None
    with lead:
        assert election.try_acquire_leadership() is None
    # Exited the with-block: lock released.
    again = election.try_acquire_leadership()
    assert again is not None
    again.close()


def test_lockfile_lives_in_runtime_dir() -> None:
    leadership = election.try_acquire_leadership()
    assert leadership is not None
    try:
        assert discovery.lock_path().exists()
    finally:
        leadership.close()


def test_probe_says_dead_when_unheld() -> None:
    assert election.probe_leader_dead() is True


def test_probe_says_alive_while_held() -> None:
    lead = election.try_acquire_leadership()
    assert lead is not None
    with lead:
        assert election.probe_leader_dead() is False


# ----- subprocess-based death detection -----
#
# Runners live at module scope so the ``spawn`` start method can pickle
# them. The child re-sets XDG_RUNTIME_DIR from the passed-in dir so it
# uses the same coordination directory as the parent's test fixture.


def _runner_hold_and_release(
    env_dir: str,
    ready: object,
    release: object,
) -> None:
    os.environ["XDG_RUNTIME_DIR"] = env_dir
    from synara.coordination import election as child_election  # noqa: PLC0415

    lead = child_election.try_acquire_leadership()
    assert lead is not None
    ready.set()  # type: ignore[attr-defined]
    release.wait(timeout=10)  # type: ignore[attr-defined]
    lead.close()


def _runner_hold_forever(env_dir: str, ready: object) -> None:
    os.environ["XDG_RUNTIME_DIR"] = env_dir
    from synara.coordination import election as child_election  # noqa: PLC0415

    lead = child_election.try_acquire_leadership()
    assert lead is not None
    ready.set()  # type: ignore[attr-defined]
    time.sleep(30)  # Parent will SIGKILL.


def _wait_for_probe(expected: bool, timeout: float = 2.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if election.probe_leader_dead() is expected:
            return True
        time.sleep(0.05)
    return election.probe_leader_dead() is expected


def test_probe_detects_dead_after_child_exits(tmp_path: Path) -> None:
    """Spawn a child that holds the lock; verify probe sees alive→dead transition."""
    ctx = mp.get_context("spawn")
    ready = ctx.Event()
    release = ctx.Event()
    proc = ctx.Process(target=_runner_hold_and_release, args=(str(tmp_path), ready, release))
    proc.start()
    try:
        assert ready.wait(timeout=5), "child failed to acquire lock"
        assert election.probe_leader_dead() is False
        release.set()
        proc.join(timeout=5)
        assert proc.exitcode == 0
        assert _wait_for_probe(True)
    finally:
        if proc.is_alive():
            proc.terminate()
            proc.join(timeout=2)


def test_probe_detects_dead_after_child_killed(tmp_path: Path) -> None:
    """Hard kill (SIGKILL) — the kernel must still release the flock."""
    import signal  # noqa: PLC0415

    ctx = mp.get_context("spawn")
    ready = ctx.Event()
    proc = ctx.Process(target=_runner_hold_forever, args=(str(tmp_path), ready))
    proc.start()
    try:
        assert ready.wait(timeout=5)
        assert election.probe_leader_dead() is False
        assert proc.pid is not None
        os.kill(proc.pid, signal.SIGKILL)
        proc.join(timeout=5)
        assert _wait_for_probe(True)
    finally:
        if proc.is_alive():
            proc.terminate()
            proc.join(timeout=2)
