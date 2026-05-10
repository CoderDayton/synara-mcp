"""FastMCP server assembly.

Each feature module exposes a ``register(mcp, db, ...)`` function and is
wired in here. The server itself stays small — features own their
tools/resources/prompts.

Server-level lifecycle (resource cleanup) goes through a FastMCP
``@lifespan`` so the embedder's HTTP client and the vector DB get torn
down deterministically when the server stops. Embedding model warmup is
deferred to the first tool call so the load can report progress and log
through the MCP context.
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

from fastmcp import FastMCP
from fastmcp.server.lifespan import lifespan
from simplevecdb import AsyncVectorDB

from synara import __version__
from synara.config import Settings
from synara.features import hippocampus
from synara.features.embedding import build_embedder

_logger = logging.getLogger(__name__)


def build_server(settings: Settings) -> FastMCP:
    if settings.db_path != ":memory:":
        Path(settings.db_path).parent.mkdir(parents=True, exist_ok=True)
    db = AsyncVectorDB(settings.db_path)
    embedder = build_embedder(settings.embedding)

    backend_kind = "remote" if settings.embedding.url else "local"
    _logger.info(
        "embedder configured (backend=%s, model=%s); warmup deferred to first tool call",
        backend_kind,
        settings.embedding.model or "default",
    )

    @lifespan
    async def app_lifespan(_server: FastMCP) -> AsyncIterator[dict[str, Any]]:
        try:
            yield {"db": db, "embedder": embedder, "settings": settings}
        finally:
            await embedder.aclose()
            await db.close()

    mcp = FastMCP(
        name="synara-mcp",
        instructions=(
            "Synara MCP server. Tools are registered per feature module under "
            "`synara.features.*`. Version: " + __version__
        ),
        lifespan=app_lifespan,
    )
    hippocampus.register(mcp, db, embedder=embedder)
    return mcp
