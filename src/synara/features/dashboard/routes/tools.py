"""Per-tool runtime telemetry (read-only)."""

from __future__ import annotations

from typing import Annotated, Any

from fastapi import APIRouter, Depends

from synara.features.memory import ToolMetrics

from ..deps import get_tool_metrics

router = APIRouter(tags=["tools"])

_Metrics = Annotated[ToolMetrics, Depends(get_tool_metrics)]


@router.get("/tool-metrics")
async def tool_metrics(metrics: _Metrics) -> dict[str, Any]:
    """Snapshot the per-tool call count, last-called time, and p50/p95 latency.

    Tool list is server-authoritative: every tool declared on the
    collector at server start appears here even if it hasn't been
    called yet (``count=0``, latency fields ``null``), so the dashboard
    can render the full surface on the first poll.
    """
    return {"tools": [row.as_dict() for row in metrics.snapshot()]}
