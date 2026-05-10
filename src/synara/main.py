"""Process entry point. Builds the FastMCP server and runs the chosen transport."""

from __future__ import annotations

import sys

from synara.config import Settings
from synara.logging import configure_logging
from synara.server import build_server


def main(argv: list[str] | None = None) -> int:
    args = sys.argv[1:] if argv is None else argv
    settings = Settings.from_env(args)
    configure_logging(settings.log_level)
    mcp = build_server(settings)
    mcp.run(transport=settings.transport)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
