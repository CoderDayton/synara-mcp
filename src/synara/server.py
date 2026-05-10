"""FastMCP server assembly.

Each feature module exposes a `register(mcp)` function and is wired in here.
The server itself stays small — features own their tools/resources/prompts.
"""

from __future__ import annotations

from fastmcp import FastMCP

from synara import __version__
from synara.config import Settings


def build_server(settings: Settings) -> FastMCP:
    del settings  # reserved for per-feature configuration
    mcp = FastMCP(
        name="synara-mcp",
        instructions=(
            "Synara MCP server. Tools are registered per feature module under "
            "`synara.features.*`. Version: " + __version__
        ),
    )
    return mcp
