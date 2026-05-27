"""Process entry point.

Two execution shapes:

* **stdio transport** — every Claude session spawns its own subprocess,
  so we route through ``synara.coordination`` to elect a single leader
  that owns the DB + dashboard + HTTP MCP endpoint, with the other
  subprocesses running as followers that proxy stdio onto the leader.
* **other transports (http / sse / streamable-http)** — the user is
  hosting one server for external clients. No coordination needed; run
  FastMCP directly.
"""

from __future__ import annotations

import sys
from functools import partial

import anyio

from synara.config import Settings
from synara.logging import configure_logging
from synara.server import build_server


def _run_stdio(settings: Settings) -> None:
    """Stdio-mode entry point: run the unified proxy + leader router."""
    from synara.coordination.unified import run_unified_async  # noqa: PLC0415

    anyio.run(
        partial(
            run_unified_async,
            settings=settings,
            build_server=lambda: build_server(settings),
        )
    )


def main(argv: list[str] | None = None) -> int:
    args = sys.argv[1:] if argv is None else argv
    settings = Settings.from_env(args)
    configure_logging(settings.log_level)
    if settings.transport == "stdio":
        _run_stdio(settings)
    else:
        mcp = build_server(settings)
        mcp.run(transport=settings.transport)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
