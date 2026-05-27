"""Filesystem layout and atomic discovery file for the elected leader.

The discovery file (``leader.json``) tells followers where the current
leader's HTTP MCP endpoint and dashboard live. It is written via
tmp-file + ``os.replace`` so a follower never observes a half-written
payload — even though writes are rare, atomicity is the only way to
guarantee that during the boot race.

Both the lockfile and the discovery file live under XDG_RUNTIME_DIR,
which is tmpfs-backed, mode 0700, owned by the user, and purged at
logout — exactly the lifecycle ephemeral coordination state wants.
"""

from __future__ import annotations

import contextlib
import json
import os
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path

# Fallback when XDG_RUNTIME_DIR is unset (e.g. some sandboxed environments).
# Per the XDG Base Directory spec the fallback is implementation-defined;
# the system temp dir is the conventional choice. Tests monkeypatch this.
_FALLBACK_BASE = Path(tempfile.gettempdir())


def runtime_dir() -> Path:
    """Resolve the synara coordination directory under XDG_RUNTIME_DIR."""
    raw = os.environ.get("XDG_RUNTIME_DIR")
    base = Path(raw) if raw else _FALLBACK_BASE / f"synara-mcp-{os.getuid()}"
    return base / "synara-mcp"


def _validate_owner_and_perms(path: Path) -> None:
    """Ensure ``path`` is owned by us and not group/world-accessible.

    Required because the fallback root lives under ``/tmp``; a co-tenant
    could pre-create a same-named directory with broader perms and
    substitute ``leader.json`` to redirect the proxy to an attacker URL.
    """
    st = path.stat()
    if st.st_uid != os.geteuid():
        raise PermissionError(f"{path}: owned by uid {st.st_uid}, expected {os.geteuid()}")
    if st.st_mode & 0o077:
        raise PermissionError(f"{path}: mode {oct(st.st_mode & 0o777)} broader than 0700")


def ensure_runtime_dir() -> Path:
    """Create the synara coordination directory if needed; return its path.

    When ``XDG_RUNTIME_DIR`` is set, the host dir is already 0700-managed
    by systemd-logind, so we just create our ``synara-mcp`` subdirectory
    inside it. In the fallback path we create *both* levels with mode
    0700 and refuse to use them if they already exist with broader perms
    or different owner.
    """
    raw = os.environ.get("XDG_RUNTIME_DIR")
    if raw:
        target = Path(raw) / "synara-mcp"
        target.mkdir(mode=0o700, parents=False, exist_ok=True)
        return target
    parent = _FALLBACK_BASE / f"synara-mcp-{os.geteuid()}"
    parent.mkdir(mode=0o700, parents=False, exist_ok=True)
    _validate_owner_and_perms(parent)
    target = parent / "synara-mcp"
    target.mkdir(mode=0o700, parents=False, exist_ok=True)
    _validate_owner_and_perms(target)
    return target


def lock_path() -> Path:
    return runtime_dir() / "leader.lock"


def info_path() -> Path:
    return runtime_dir() / "leader.json"


@dataclass(frozen=True, slots=True)
class LeaderInfo:
    pid: int
    mcp_url: str
    dashboard_url: str
    started_at: float


def write_leader_info(info: LeaderInfo) -> None:
    """Write ``leader.json`` atomically (tmp file + ``os.replace``).

    The tmp file lives in the same directory as the target so
    ``os.replace`` is a rename, not a cross-filesystem copy.
    """
    target = info_path()
    ensure_runtime_dir()
    payload = json.dumps(asdict(info), separators=(",", ":")).encode("utf-8")

    fd, tmp_name = tempfile.mkstemp(prefix="leader.json.", suffix=".tmp", dir=str(target.parent))
    try:
        with os.fdopen(fd, "wb") as fh:
            fh.write(payload)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp_name, target)
    except BaseException:
        # Best-effort cleanup on failure; ignore if rename already removed it.
        with contextlib.suppress(FileNotFoundError):
            os.unlink(tmp_name)
        raise


def read_leader_info() -> LeaderInfo | None:
    """Return the current leader info, or ``None`` if missing/unreadable."""
    path = info_path()
    try:
        raw = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return None
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return None
    try:
        return LeaderInfo(
            pid=int(data["pid"]),
            mcp_url=str(data["mcp_url"]),
            dashboard_url=str(data["dashboard_url"]),
            started_at=float(data["started_at"]),
        )
    except (KeyError, TypeError, ValueError):
        return None
