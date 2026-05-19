"""Dashboard feature.

A standalone FastAPI admin/observability console run *in parallel* with
the FastMCP server, on the same event loop and lifecycle, gated by the
``SYNARA_DASHBOARD`` env var.

Import-safety contract
----------------------
This package ``__init__`` is imported transitively by
:mod:`synara.config` (``Settings`` holds a :class:`DashboardConfig`).
It must therefore stay **stdlib-only** — FastAPI/uvicorn are an optional
``[dashboard]`` extra and must never be imported at module load.

The FastAPI assembly (``build_dashboard_app``) is added in Phase 2.1 and
will be exposed lazily via :pep:`562` ``__getattr__`` so the heavy
import only happens when the dashboard is actually built.
"""

from __future__ import annotations

from .config import DashboardConfig

__all__ = ["DashboardConfig"]
