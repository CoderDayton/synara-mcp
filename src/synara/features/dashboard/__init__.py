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
``build_dashboard_app`` is exposed lazily via :pep:`562` ``__getattr__``
so the heavy import only happens when the dashboard is actually built.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from .config import DashboardConfig

if TYPE_CHECKING:  # pragma: no cover - typing only
    from .app import build_dashboard_app as build_dashboard_app

__all__ = ["DashboardConfig", "build_dashboard_app"]


def __getattr__(name: str) -> Any:
    """Lazily expose ``build_dashboard_app`` without eagerly importing FastAPI."""
    if name == "build_dashboard_app":
        from .app import build_dashboard_app  # noqa: PLC0415

        return build_dashboard_app
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
