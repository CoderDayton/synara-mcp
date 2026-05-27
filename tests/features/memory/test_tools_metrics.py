"""End-to-end: ToolMetrics is updated when a wrapped tool is invoked."""

from __future__ import annotations

import contextlib
from collections.abc import AsyncIterator

import pytest_asyncio
from fastmcp import Client, FastMCP
from simplevecdb import AsyncVectorDB

from synara.features.memory.config import MemoryConfig
from synara.features.memory.metrics import ToolMetrics
from synara.features.memory.service import MemoryService
from synara.features.memory.tools import _TOOL_HEADLINES, register_tools

from .test_tools import _FakeEmbedder, hash_embed


@pytest_asyncio.fixture
async def wired_with_metrics() -> AsyncIterator[tuple[FastMCP, ToolMetrics]]:
    db = AsyncVectorDB(":memory:")
    metrics = ToolMetrics()
    try:
        service = MemoryService(db, config=MemoryConfig(), embed_fn=hash_embed)
        mcp: FastMCP = FastMCP("synara-test-metrics")
        register_tools(
            mcp,
            service,
            embedder=_FakeEmbedder(),  # type: ignore[arg-type]
            metrics=metrics,
        )
        yield mcp, metrics
    finally:
        await db.close()


async def test_declares_every_known_tool_at_registration(
    wired_with_metrics: tuple[FastMCP, ToolMetrics],
) -> None:
    mcp, metrics = wired_with_metrics
    declared = {row.name for row in metrics.snapshot()}
    assert declared == set(_TOOL_HEADLINES)

    # Drift guard: every tool actually registered on the MCP server must
    # have a headline entry. Adding a new tool to ``register_tools``
    # without updating ``_TOOL_HEADLINES`` would otherwise only surface
    # after the first call (late-declare uses the bare name as headline).
    registered = {t.name for t in await mcp.list_tools()}
    assert registered == set(_TOOL_HEADLINES), (
        f"missing headlines for: {registered - set(_TOOL_HEADLINES)}; "
        f"stale headlines for: {set(_TOOL_HEADLINES) - registered}"
    )


async def test_calling_tool_increments_metrics(
    wired_with_metrics: tuple[FastMCP, ToolMetrics],
) -> None:
    mcp, metrics = wired_with_metrics
    async with Client(mcp) as client:
        await client.call_tool("memory_stats", {})
        await client.call_tool("memory_stats", {})

    row = next(r for r in metrics.snapshot() if r.name == "memory_stats")
    assert row.count == 2
    assert row.error_count == 0
    assert row.last_called_at is not None
    assert row.last_duration_seconds is not None
    assert row.p50_ms is not None
    assert row.p95_ms is not None


async def test_tool_error_increments_error_count(
    wired_with_metrics: tuple[FastMCP, ToolMetrics],
) -> None:
    mcp, metrics = wired_with_metrics
    async with Client(mcp) as client:
        # Empty content fails service-side validation — the wrapper
        # must still observe the failure and bump error_count.
        with contextlib.suppress(Exception):
            await client.call_tool(
                "store_episode",
                {"content": "", "session_id": "s1"},
            )

    row = next(r for r in metrics.snapshot() if r.name == "store_episode")
    assert row.count >= 1
    assert row.error_count >= 1
