"""``detect_mcp_client`` walks the process ancestry to name the hosting
MCP client. It is best-effort and purely cosmetic: every failure path
degrades to ``"unknown"`` rather than raising. These tests drive the
walk over a scripted ``/proc`` so they stay deterministic and
platform-independent.
"""

from __future__ import annotations

import os

import pytest

from synara.features.dashboard import client_info

# pid -> (comm, cmdline, ppid); pid 1000 stands in for "this" process.
_Chain = dict[int, tuple[str, str, int]]


def _install(monkeypatch: pytest.MonkeyPatch, chain: _Chain, *, self_pid: int = 1000) -> None:
    monkeypatch.setattr(os, "getpid", lambda: self_pid)
    monkeypatch.setattr(client_info, "_proc_entry", chain.get)


def test_skips_uv_and_shell_to_reach_claude(monkeypatch: pytest.MonkeyPatch) -> None:
    _install(
        monkeypatch,
        {
            1000: ("synara-mcp", "synara-mcp", 900),
            900: ("uv", "uv run --no-sync synara-mcp", 800),
            800: ("zsh", "-zsh", 700),
            700: ("claude", "claude", 1),
        },
    )
    assert client_info.detect_mcp_client() == "Claude Code"


def test_matches_via_cmdline_when_comm_is_generic(monkeypatch: pytest.MonkeyPatch) -> None:
    # Electron clients show ``comm`` "node"; the client name is only in argv.
    _install(
        monkeypatch,
        {
            1000: ("python3.13", "python3.13 -m synara", 900),
            900: ("node", "/opt/Cursor/cursor --type=utility", 1),
        },
    )
    assert client_info.detect_mcp_client() == "Cursor"


def test_any_python_version_is_treated_as_a_shim(monkeypatch: pytest.MonkeyPatch) -> None:
    # A future ``python3.14`` not pinned in ``_LAUNCHER_SHIMS`` must still be skipped.
    _install(
        monkeypatch,
        {
            1000: ("synara", "synara", 900),
            900: ("python3.14", "python3.14 synara", 800),
            800: ("windsurf", "windsurf", 1),
        },
    )
    assert client_info.detect_mcp_client() == "Windsurf"


def test_cycle_in_chain_degrades_to_unknown(monkeypatch: pytest.MonkeyPatch) -> None:
    _install(
        monkeypatch,
        {
            1000: ("synara", "synara", 900),
            900: ("uv", "uv", 800),
            800: ("bash", "bash", 900),  # points back to 900 -> cycle
        },
    )
    assert client_info.detect_mcp_client() == "unknown"


def test_unreadable_proc_degrades_to_unknown(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(os, "getpid", lambda: 1000)
    monkeypatch.setattr(client_info, "_proc_entry", lambda pid: None)
    assert client_info.detect_mcp_client() == "unknown"


def test_reparented_to_init_degrades_to_unknown(monkeypatch: pytest.MonkeyPatch) -> None:
    # The client already exited; we've been reparented to pid 1.
    _install(monkeypatch, {1000: ("synara", "synara", 1)})
    assert client_info.detect_mcp_client() == "unknown"
