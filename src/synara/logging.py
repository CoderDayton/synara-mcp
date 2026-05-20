"""Logging bootstrap. MCP stdio transports must keep stdout clean — log to stderr."""

from __future__ import annotations

import logging
import sys

# Marker attribute set on the stderr handler this module installs, so
# repeated ``configure_logging`` calls replace *our* handler without
# clobbering handlers installed by the host process or a test fixture.
_OWN_HANDLER_MARK = "_synara_log_handler"


def configure_logging(level: str = "INFO") -> None:
    """Install a stderr log handler; idempotent across reconfigure.

    Replaces any previously-installed Synara handler and strips any
    handler writing to ``sys.stdout`` (a hard requirement under the MCP
    stdio transport — log output on stdout corrupts the JSON-RPC
    framing). Handlers installed by other code on stderr or a file are
    left alone so they can coexist with ours.
    """
    handler = logging.StreamHandler(stream=sys.stderr)
    handler.setFormatter(
        logging.Formatter(
            fmt="%(asctime)s %(levelname)s %(name)s: %(message)s",
            datefmt="%Y-%m-%dT%H:%M:%S%z",
        )
    )
    setattr(handler, _OWN_HANDLER_MARK, True)
    root = logging.getLogger()
    for existing in list(root.handlers):
        owned = getattr(existing, _OWN_HANDLER_MARK, False)
        targets_stdout = getattr(existing, "stream", None) is sys.stdout
        if owned or targets_stdout:
            root.removeHandler(existing)
    root.addHandler(handler)
    root.setLevel(level)
