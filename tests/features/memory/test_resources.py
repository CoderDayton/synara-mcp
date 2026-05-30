"""MCP resource-surface tests (Lever 1 — ambient recall).

Exercises the ``memory://recall/{query}`` resource template through an
in-memory FastMCP client: it must reuse the same recall pipeline as the
``recall_episodes`` tool, be discoverable as a template, and honour the
optional ``k``/``session_id``/``mode`` query parameters.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import AsyncIterator

import numpy as np
import pytest
import pytest_asyncio
from fastmcp import Client, FastMCP
from mcp.shared.exceptions import McpError
from mcp.types import TextResourceContents
from simplevecdb import AsyncVectorDB

from synara.features.memory.config import MemoryConfig
from synara.features.memory.metrics import ToolMetrics
from synara.features.memory.resources import register_resources
from synara.features.memory.service import MemoryService
from synara.features.memory.tools import register_tools


def hash_embed(text: str, dim: int = 32) -> list[float]:
    seed_bytes = hashlib.sha256(text.encode("utf-8")).digest()[:8]
    seed = int.from_bytes(seed_bytes, "big", signed=False)
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(dim).astype(np.float32)
    n = float(np.linalg.norm(v))
    out = (v / n) if n > 0 else v
    return [float(x) for x in out.tolist()]


@pytest_asyncio.fixture
async def wired() -> AsyncIterator[FastMCP]:
    db = AsyncVectorDB(":memory:")
    try:
        service = MemoryService(db, config=MemoryConfig(), embed_fn=hash_embed)
        mcp: FastMCP = FastMCP("synara-resource-test")
        register_tools(mcp, service)
        register_resources(mcp, service)
        yield mcp
    finally:
        await db.close()


async def test_recall_resource_returns_ranked_hits(wired: FastMCP) -> None:
    mcp = wired
    async with Client(mcp) as client:
        await client.call_tool("store_episode", {"content": "alphabeta gamma", "session_id": "s1"})
        contents = await client.read_resource("memory://recall/alphabeta%20gamma")
        assert contents
        body = contents[0]
        assert isinstance(body, TextResourceContents)
        payload = json.loads(body.text)
        assert isinstance(payload, list)
        assert any("alphabeta gamma" in (h.get("content") or "") for h in payload)


async def test_recall_resource_template_is_discoverable(wired: FastMCP) -> None:
    mcp = wired
    async with Client(mcp) as client:
        templates = await client.list_resource_templates()
        match = [t for t in templates if "memory://recall/" in t.uriTemplate]
        assert match, "ambient recall resource template not advertised"
        # The advertised contract is JSON, even though FastMCP stamps the
        # per-read content block of a ``str`` return as text/plain.
        assert match[0].mimeType == "application/json"


async def test_recall_resource_honours_k_query_param(wired: FastMCP) -> None:
    mcp = wired
    async with Client(mcp) as client:
        for i in range(4):
            await client.call_tool(
                "store_episode", {"content": f"vector item {i}", "session_id": "s1"}
            )
        contents = await client.read_resource("memory://recall/vector%20item?k=2&session_id=s1")
        body = contents[0]
        assert isinstance(body, TextResourceContents)
        payload = json.loads(body.text)
        assert isinstance(payload, list)
        assert len(payload) <= 2


async def test_recall_resource_validation_error_surfaces(wired: FastMCP) -> None:
    """A bad ``mode`` must reach the client as the service's actionable
    reason (ValidationError -> ResourceError), not a generic failure."""
    mcp = wired
    async with Client(mcp) as client:
        with pytest.raises(McpError) as excinfo:
            await client.read_resource("memory://recall/hello?mode=bogus")
        assert "unknown recall mode" in str(excinfo.value)


async def test_recall_resource_records_metrics() -> None:
    """Resource reads land in the same ToolMetrics the dashboard reads,
    counting successes and errors under the 'relevant_memories' row."""
    db = AsyncVectorDB(":memory:")
    try:
        service = MemoryService(db, config=MemoryConfig(), embed_fn=hash_embed)
        metrics = ToolMetrics()
        mcp: FastMCP = FastMCP("synara-resource-metrics")
        register_tools(mcp, service, metrics=metrics)
        register_resources(mcp, service, metrics=metrics)
        async with Client(mcp) as client:
            await client.call_tool("store_episode", {"content": "metric trace", "session_id": "s1"})
            await client.read_resource("memory://recall/metric%20trace")
            with pytest.raises(McpError):
                await client.read_resource("memory://recall/metric%20trace?mode=bogus")
        rows = {r.name: r for r in metrics.snapshot()}
        assert "relevant_memories" in rows
        row = rows["relevant_memories"]
        assert row.count == 2
        assert row.error_count == 1
        assert row.headline == "ambient recall (resource)"
    finally:
        await db.close()


async def test_recall_resource_is_non_reinforcing() -> None:
    """Ambient resource reads are GET-like: they must not mutate durable
    memory. A read surfaces the episode but must NOT bump retrieval_count
    or emit an interaction event (the recall_episodes tool does both),
    since a host may prefetch/poll the resource."""
    db = AsyncVectorDB(":memory:")
    try:
        service = MemoryService(db, config=MemoryConfig(), embed_fn=hash_embed)
        mcp: FastMCP = FastMCP("synara-resource-noreinforce")
        register_tools(mcp, service)
        register_resources(mcp, service)
        async with Client(mcp) as client:
            await client.call_tool(
                "store_episode", {"content": "reinforce probe", "session_id": "s1"}
            )
            events_after_store = service._bus.state.total_events
            contents = await client.read_resource("memory://recall/reinforce%20probe")
            body = contents[0]
            assert isinstance(body, TextResourceContents)
            payload = json.loads(body.text)
            assert any("reinforce probe" in (h.get("content") or "") for h in payload)
            # The GET-like read emits no interaction event: no durable log
            # row, no dream pressure (events_since_dream unchanged).
            assert service._bus.state.total_events == events_after_store
        rows = await service.episodic.get_documents({"session_id": "s1"})
        assert all(int(md.get("retrieval_count", 0)) == 0 for _, _, md in rows)

        # Contrast: the tool path on the same query bumps retrieval_count
        # AND emits an interaction event, proving the read above was inert.
        await service.recall("reinforce probe", session_id="s1", k=5)
        rows = await service.episodic.get_documents({"session_id": "s1"})
        assert sum(int(md.get("retrieval_count", 0)) for _, _, md in rows) >= 1
        assert service._bus.state.total_events > events_after_store
    finally:
        await db.close()
