from __future__ import annotations

import pytest
from fastmcp import FastMCP

from synara.config import Settings
from synara.core.argument_normalization import ArgumentNormalizationMiddleware
from synara.server import build_server


def test_build_server_returns_fastmcp_instance() -> None:
    """build_server defers embedder warmup to first tool call, so this is
    a pure construction smoke test — no SentenceTransformer stub needed."""
    settings = Settings(log_level="INFO", transport="stdio", db_path=":memory:")
    mcp = build_server(settings)
    assert isinstance(mcp, FastMCP)
    assert mcp.name == "synara-mcp"


def test_build_server_installs_argument_normalization() -> None:
    """The tolerant-argument layer only works if it is actually wired.

    It sits upstream of pydantic validation, so a missing registration
    would not fail loudly — every near-miss call would just go back to
    being rejected. Pin the wiring rather than the symptom.
    """
    settings = Settings(log_level="INFO", transport="stdio", db_path=":memory:")
    mcp = build_server(settings)
    assert any(isinstance(mw, ArgumentNormalizationMiddleware) for mw in mcp.middleware)


def test_settings_rejects_unknown_transport(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SYNARA_TRANSPORT", "carrier-pigeon")
    with pytest.raises(ValueError, match="SYNARA_TRANSPORT"):
        Settings.from_env()
