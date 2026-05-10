from __future__ import annotations

import pytest
from fastmcp import FastMCP

from synara.config import Settings
from synara.server import build_server


def test_build_server_returns_fastmcp_instance() -> None:
    settings = Settings(log_level="INFO", transport="stdio")
    mcp = build_server(settings)
    assert isinstance(mcp, FastMCP)
    assert mcp.name == "synara-mcp"


def test_settings_rejects_unknown_transport(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SYNARA_TRANSPORT", "carrier-pigeon")
    with pytest.raises(ValueError, match="SYNARA_TRANSPORT"):
        Settings.from_env()
