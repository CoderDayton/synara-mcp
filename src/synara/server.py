"""FastMCP server assembly.

Each feature module exposes a ``register(mcp, db, ...)`` function and is
wired in here. The server itself stays small — features own their
tools/resources/prompts.
"""

from __future__ import annotations

import logging
from pathlib import Path

from fastmcp import FastMCP
from simplevecdb import AsyncVectorDB

from synara import __version__
from synara.config import Settings
from synara.features import hippocampus
from synara.features.embedding import build_embedder

_logger = logging.getLogger(__name__)


def build_server(settings: Settings) -> FastMCP:
    mcp = FastMCP(
        name="synara-mcp",
        instructions=(
            "Synara MCP server. Tools are registered per feature module under "
            "`synara.features.*`. Version: " + __version__
        ),
    )
    if settings.db_path != ":memory:":
        Path(settings.db_path).parent.mkdir(parents=True, exist_ok=True)
    db = AsyncVectorDB(settings.db_path)
    embedder = build_embedder(settings.embedding)
    # Eagerly load the local model (downloads on first run) so the first
    # encode call is not delayed by a cold start. Surface the wait via the
    # log so a startup that appears hung is actually explained.
    backend_kind = "remote" if settings.embedding.url else "local"
    _logger.info(
        "warming up embedder (backend=%s, model=%s)",
        backend_kind,
        settings.embedding.model or "default",
    )
    embedder.warmup()
    hippocampus.register(mcp, db, embed_fn=embedder.embed)
    return mcp
