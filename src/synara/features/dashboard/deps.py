"""Request-scoped accessors for the live server objects.

``build_dashboard_app`` stashes the shared ``MemoryService`` /
``Settings`` on ``app.state``; routes pull them through these typed
dependencies rather than reaching into ``request.app.state`` directly.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

from fastapi import Request

if TYPE_CHECKING:  # pragma: no cover - typing only
    from synara.config import Settings
    from synara.features.memory import MemoryService, ToolMetrics


def get_service(request: Request) -> MemoryService:
    return cast("MemoryService", request.app.state.service)


def get_settings(request: Request) -> Settings:
    return cast("Settings", request.app.state.settings)


def get_tool_metrics(request: Request) -> ToolMetrics:
    return cast("ToolMetrics", request.app.state.tool_metrics)
