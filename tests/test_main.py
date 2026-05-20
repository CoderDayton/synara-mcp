"""Entry-point characterization: main() wires Settings, logging, and server.run."""

from __future__ import annotations

from typing import Any

import pytest

import synara.main as main_module
from synara.config import Settings as _Settings


class _FakeMcp:
    def __init__(self) -> None:
        self.run_calls: list[str] = []

    def run(self, transport: str) -> None:
        self.run_calls.append(transport)


def _patch_main(
    monkeypatch: pytest.MonkeyPatch,
    *,
    transport: str = "stdio",
    log_level: str = "INFO",
) -> dict[str, Any]:
    """Stub Settings.from_env / configure_logging / build_server and capture args."""
    captured: dict[str, Any] = {"argv": None, "level": None, "settings": None}
    fake_mcp = _FakeMcp()

    class _StubSettings:
        def __init__(self) -> None:
            self.log_level = log_level
            self.transport = transport

    def fake_from_env(argv: list[str] | None = None) -> _StubSettings:
        captured["argv"] = argv
        return _StubSettings()

    def fake_configure_logging(level: str) -> None:
        captured["level"] = level

    def fake_build_server(settings: Any) -> _FakeMcp:
        captured["settings"] = settings
        return fake_mcp

    monkeypatch.setattr(_Settings, "from_env", staticmethod(fake_from_env))
    monkeypatch.setattr(main_module, "configure_logging", fake_configure_logging)
    monkeypatch.setattr(main_module, "build_server", fake_build_server)
    captured["mcp"] = fake_mcp
    return captured


def test_main_returns_zero_on_clean_exit(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_main(monkeypatch)
    assert main_module.main([]) == 0


def test_main_passes_explicit_argv_through(monkeypatch: pytest.MonkeyPatch) -> None:
    captured = _patch_main(monkeypatch)
    main_module.main(["--flag", "value"])
    assert captured["argv"] == ["--flag", "value"]


def test_main_falls_back_to_sys_argv(monkeypatch: pytest.MonkeyPatch) -> None:
    captured = _patch_main(monkeypatch)
    monkeypatch.setattr("sys.argv", ["synara-mcp", "--from-sys"])
    main_module.main(None)
    assert captured["argv"] == ["--from-sys"]


def test_main_configures_logging_from_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    captured = _patch_main(monkeypatch, log_level="DEBUG")
    main_module.main([])
    assert captured["level"] == "DEBUG"


def test_main_runs_server_with_configured_transport(monkeypatch: pytest.MonkeyPatch) -> None:
    captured = _patch_main(monkeypatch, transport="http")
    main_module.main([])
    assert captured["mcp"].run_calls == ["http"]


def test_main_builds_server_with_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    captured = _patch_main(monkeypatch)
    main_module.main([])
    assert captured["settings"] is not None
    assert captured["settings"].transport == "stdio"
