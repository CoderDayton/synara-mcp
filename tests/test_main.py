"""Entry-point characterization: main() wires Settings, logging, and dispatch.

For ``stdio`` transport, ``main`` delegates to ``_run_stdio`` (which
runs the multi-process coordinator). For other transports, it builds
the server directly and runs FastMCP with that transport. Tests stub
both paths so they're orthogonal.
"""

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
    transport: str = "http",
    log_level: str = "INFO",
) -> dict[str, Any]:
    """Stub Settings/logging/server.build/coordinator and capture calls.

    Defaults to ``transport=http`` because that path is the simple one
    (direct ``build_server`` + ``mcp.run(transport)``). Stdio tests pass
    ``transport="stdio"`` explicitly and assert on ``stdio_calls``.
    """
    captured: dict[str, Any] = {
        "argv": None,
        "level": None,
        "settings": None,
        "stdio_calls": [],
    }
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

    def fake_run_stdio(settings: Any) -> None:
        captured["stdio_calls"].append(settings)

    monkeypatch.setattr(_Settings, "from_env", staticmethod(fake_from_env))
    monkeypatch.setattr(main_module, "configure_logging", fake_configure_logging)
    monkeypatch.setattr(main_module, "build_server", fake_build_server)
    monkeypatch.setattr(main_module, "_run_stdio", fake_run_stdio)
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


def test_main_builds_server_for_http_transport(monkeypatch: pytest.MonkeyPatch) -> None:
    captured = _patch_main(monkeypatch, transport="http")
    main_module.main([])
    assert captured["settings"] is not None
    assert captured["settings"].transport == "http"
    # Stdio path must not run for http transport.
    assert captured["stdio_calls"] == []


def test_main_dispatches_to_coordinator_for_stdio(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured = _patch_main(monkeypatch, transport="stdio")
    main_module.main([])
    assert len(captured["stdio_calls"]) == 1
    assert captured["stdio_calls"][0].transport == "stdio"
    # In production the coordinator (LeaderRouter) calls build_server
    # only on the process that wins promotion. This test stubs
    # _run_stdio entirely, so neither build_server nor mcp.run fires
    # from main() — they would only fire deeper inside the real stdio
    # path.
    assert captured["mcp"].run_calls == []
    assert captured["settings"] is None
