"""Feature packages.

Each feature is its own subpackage. Convention:

    synara/features/<feature>/
        __init__.py
        service.py    # pure logic; no MCP types
        tools.py      # exposes `register(mcp: FastMCP) -> None`

`server.build_server` wires each feature's `register` function into the
shared FastMCP instance.
"""
