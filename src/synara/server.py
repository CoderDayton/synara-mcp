"""FastMCP server assembly.

Each feature module exposes a ``register(mcp, db, ...)`` function and is
wired in here. The server itself stays small — features own their
tools/resources/prompts.
"""

from __future__ import annotations

from fastmcp import FastMCP
from simplevecdb import AsyncVectorDB

from synara import __version__
from synara.config import Settings
from synara.features import hippocampus


def build_server(settings: Settings) -> FastMCP:
    mcp = FastMCP(
        name="synara-mcp",
        instructions=(
            "Synara MCP server. Tools are registered per feature module under "
            "`synara.features.*`. Version: " + __version__
        ),
    )
    db = AsyncVectorDB(settings.db_path)
    hippocampus.register(mcp, db)
    return mcp
