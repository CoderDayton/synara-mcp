from __future__ import annotations

import pytest
from fastmcp import FastMCP

from synara.config import Settings
from synara.server import build_server


def test_build_server_returns_fastmcp_instance(monkeypatch: pytest.MonkeyPatch) -> None:
    """build_server warms up the embedder; stub SentenceTransformer to keep
    this unit test offline (the real model load is exercised in the slow
    embedding tests)."""

    class _FakeST:
        def __init__(self, model_id: str, **kwargs: object) -> None: ...

        def encode(self, *args: object, **kwargs: object) -> object:
            raise NotImplementedError

    monkeypatch.setattr("sentence_transformers.SentenceTransformer", _FakeST)
    settings = Settings(log_level="INFO", transport="stdio", db_path=":memory:")
    mcp = build_server(settings)
    assert isinstance(mcp, FastMCP)
    assert mcp.name == "synara-mcp"


def test_settings_rejects_unknown_transport(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SYNARA_TRANSPORT", "carrier-pigeon")
    with pytest.raises(ValueError, match="SYNARA_TRANSPORT"):
        Settings.from_env()
