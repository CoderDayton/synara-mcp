"""Tests for synara.coordination.discovery: XDG path + atomic leader.json.

These are intentionally synchronous filesystem tests — no event loop,
no MCP. The discovery layer is just "compute a path; write atomically;
read or return None". Concurrency lives in election (Phase 2).
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from synara.coordination import discovery


def test_runtime_dir_uses_xdg_runtime_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("XDG_RUNTIME_DIR", str(tmp_path))
    assert discovery.runtime_dir() == tmp_path / "synara-mcp"


def test_runtime_dir_falls_back_when_xdg_unset(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("XDG_RUNTIME_DIR", raising=False)
    monkeypatch.setattr(discovery, "_FALLBACK_BASE", tmp_path)
    got = discovery.runtime_dir()
    assert got.is_relative_to(tmp_path)
    assert got.name == "synara-mcp"


def test_lock_path_and_info_path_live_under_runtime_dir(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("XDG_RUNTIME_DIR", str(tmp_path))
    base = discovery.runtime_dir()
    assert discovery.lock_path() == base / "leader.lock"
    assert discovery.info_path() == base / "leader.json"


def test_write_leader_info_creates_parent_dir(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("XDG_RUNTIME_DIR", str(tmp_path))
    info = discovery.LeaderInfo(
        pid=12345,
        mcp_url="http://127.0.0.1:54321/mcp/",
        dashboard_url="http://127.0.0.1:8765",
        started_at=1.0,
    )
    discovery.write_leader_info(info)
    assert discovery.info_path().exists()
    assert discovery.info_path().parent.is_dir()


def test_write_leader_info_is_atomic(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A reader must never see a half-written file: writes go via tmp+rename."""
    monkeypatch.setenv("XDG_RUNTIME_DIR", str(tmp_path))
    info = discovery.LeaderInfo(
        pid=1,
        mcp_url="http://127.0.0.1:1/mcp/",
        dashboard_url="http://127.0.0.1:8765",
        started_at=0.0,
    )

    seen_tmp: list[Path] = []
    real_replace = os.replace

    def spy_replace(src: str | os.PathLike[str], dst: str | os.PathLike[str]) -> None:
        seen_tmp.append(Path(os.fspath(src)))
        real_replace(src, dst)

    monkeypatch.setattr("os.replace", spy_replace)
    discovery.write_leader_info(info)
    assert len(seen_tmp) == 1
    assert seen_tmp[0].name.startswith("leader.json.") or seen_tmp[0].suffix == ".tmp"
    # The temp file is gone after rename; only the final exists.
    assert not seen_tmp[0].exists()
    assert discovery.info_path().exists()


def test_read_leader_info_returns_none_when_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("XDG_RUNTIME_DIR", str(tmp_path))
    assert discovery.read_leader_info() is None


def test_read_leader_info_roundtrip(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("XDG_RUNTIME_DIR", str(tmp_path))
    written = discovery.LeaderInfo(
        pid=4242,
        mcp_url="http://127.0.0.1:11111/mcp/",
        dashboard_url="http://127.0.0.1:8765",
        started_at=42.5,
    )
    discovery.write_leader_info(written)
    read_back = discovery.read_leader_info()
    assert read_back == written


def test_read_leader_info_returns_none_on_malformed_json(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A torn file (shouldn't happen with atomic write, but be defensive) must not crash."""
    monkeypatch.setenv("XDG_RUNTIME_DIR", str(tmp_path))
    discovery.info_path().parent.mkdir(parents=True, exist_ok=True)
    discovery.info_path().write_text("{not json", encoding="utf-8")
    assert discovery.read_leader_info() is None


def test_ensure_runtime_dir_creates_xdg_subdir_with_0700(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """When XDG_RUNTIME_DIR is set, the synara-mcp subdir is created 0700."""
    monkeypatch.setenv("XDG_RUNTIME_DIR", str(tmp_path))
    target = discovery.ensure_runtime_dir()
    assert target == tmp_path / "synara-mcp"
    assert target.is_dir()
    assert (target.stat().st_mode & 0o777) == 0o700


def test_ensure_runtime_dir_fallback_creates_parent_and_child_0700(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """No XDG_RUNTIME_DIR → both /tmp/synara-mcp-<uid> and its child are 0700."""
    monkeypatch.delenv("XDG_RUNTIME_DIR", raising=False)
    monkeypatch.setattr(discovery, "_FALLBACK_BASE", tmp_path)
    target = discovery.ensure_runtime_dir()
    assert target.is_dir()
    assert (target.stat().st_mode & 0o777) == 0o700
    assert (target.parent.stat().st_mode & 0o777) == 0o700


def test_ensure_runtime_dir_fallback_refuses_existing_world_readable_parent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Pre-existing fallback dir with broader perms is rejected, not silently used."""
    monkeypatch.delenv("XDG_RUNTIME_DIR", raising=False)
    monkeypatch.setattr(discovery, "_FALLBACK_BASE", tmp_path)
    hostile_parent = tmp_path / f"synara-mcp-{os.getuid()}"
    hostile_parent.mkdir(mode=0o755)
    with pytest.raises(PermissionError, match="broader than 0700"):
        discovery.ensure_runtime_dir()


def test_leader_info_serializes_known_keys(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("XDG_RUNTIME_DIR", str(tmp_path))
    info = discovery.LeaderInfo(
        pid=7,
        mcp_url="http://127.0.0.1:7/mcp/",
        dashboard_url="http://127.0.0.1:8765",
        started_at=1.0,
    )
    discovery.write_leader_info(info)
    raw = json.loads(discovery.info_path().read_text(encoding="utf-8"))
    assert set(raw.keys()) == {"pid", "mcp_url", "dashboard_url", "started_at"}
    assert raw["pid"] == 7
