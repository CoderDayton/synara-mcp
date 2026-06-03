"""Best-effort identification of the MCP client hosting this server.

A stdio MCP server is spawned as a *child* of whatever client launched
it (Claude Code, Claude Desktop, Cursor, Windsurf, ...). We walk the
parent-process chain from this process, skip the launcher shims that sit
between the client and the Python interpreter (``uv``, the interpreter
itself, a wrapping shell), and map the first real ancestor to a friendly
client name.

This is purely cosmetic — it feeds the dashboard header so an operator
can see *which* client is driving the server — so every failure path
(non-Linux, unreadable ``/proc``, a detached/reparented process)
degrades to ``"unknown"`` rather than raising.
"""

from __future__ import annotations

import contextlib
import os
from pathlib import Path

# Process ``comm`` names that merely *launch* the server and sit between
# it and the real client; skipped while walking up the ancestry.
_LAUNCHER_SHIMS = frozenset(
    {
        "uv",
        "uvx",
        "uvenv",
        "python",
        "python3",
        "python3.12",
        "python3.13",
        "synara",
        "synara-mcp",
        "sh",
        "bash",
        "zsh",
        "dash",
        "fish",
        "env",
    }
)

# Substring (tested against ``"<comm> <cmdline>"`` lowercased) -> label.
# Ordered most-specific first; the first match wins. ``cmdline`` is the
# full command, so a client whose ``comm`` was truncated by the 15-char
# kernel limit is still matched via its arguments.
_CLIENT_PATTERNS: tuple[tuple[str, str], ...] = (
    ("claude desktop", "Claude Desktop"),
    ("claude-desktop", "Claude Desktop"),
    ("claude", "Claude Code"),
    ("cursor", "Cursor"),
    ("windsurf", "Windsurf"),
    ("vscodium", "VSCodium"),
    ("code-oss", "VS Code"),
    ("cline", "Cline"),
    ("continue", "Continue"),
    ("zed", "Zed"),
    ("warp", "Warp"),
)


def _proc_entry(pid: int) -> tuple[str, str, int] | None:
    """Return ``(comm, cmdline, ppid)`` for *pid* from ``/proc``, or None.

    Linux-only; any read failure (other platform, race against process
    exit, permission) yields ``None`` so callers fall back gracefully.
    """
    base = Path("/proc") / str(pid)
    try:
        comm = (base / "comm").read_text(encoding="utf-8", errors="replace").strip()
        raw = (base / "cmdline").read_bytes()
        status = (base / "status").read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None
    cmdline = raw.replace(b"\x00", b" ").decode("utf-8", errors="replace").strip()
    ppid = 0
    for line in status.splitlines():
        if line.startswith("PPid:"):
            with contextlib.suppress(ValueError):
                ppid = int(line.split(":", 1)[1].strip())
            break
    return comm, cmdline, ppid


def _label_for(comm: str, cmdline: str) -> str:
    haystack = f"{comm} {cmdline}".lower()
    for needle, label in _CLIENT_PATTERNS:
        if needle in haystack:
            return label
    return comm or "unknown"


def _is_launcher_shim(comm: str) -> bool:
    """True for processes that merely launch the server, so the ancestry
    walk skips them on the way to the real client. Any ``pythonX.Y`` build
    counts, not just the few pinned in ``_LAUNCHER_SHIMS``.
    """
    name = comm.lower()
    return name in _LAUNCHER_SHIMS or name.startswith("python")


def detect_mcp_client(*, max_hops: int = 12) -> str:
    """Best-effort friendly name of the client hosting this MCP server.

    Walks the parent chain from this process and returns the first
    ancestor that is not a launcher shim — for a stdio server that is
    the client that spawned it. ``max_hops`` and a seen-set bound the
    walk against a corrupt chain. Returns ``"unknown"`` when nothing
    identifiable is reachable.
    """
    here = _proc_entry(os.getpid())
    if here is None:
        return "unknown"
    pid = here[2]
    seen: set[int] = set()
    for _ in range(max_hops):
        if pid <= 1 or pid in seen:
            break
        seen.add(pid)
        entry = _proc_entry(pid)
        if entry is None:
            break
        comm, cmdline, ppid = entry
        if not _is_launcher_shim(comm):
            return _label_for(comm, cmdline)
        pid = ppid
    return "unknown"
