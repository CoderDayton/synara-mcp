"""MCP tool-surface tests.

Exercises every tool registered by ``register_tools`` through an
in-memory FastMCP client so the thin wrapper layer (ctx logging,
result shaping, embedder warmup) is covered end to end.
"""

from __future__ import annotations

import hashlib
from collections.abc import AsyncIterator
from typing import Any

import numpy as np
import pytest_asyncio
from fastmcp import Client, FastMCP
from simplevecdb import AsyncVectorDB

from synara.features.memory.config import MemoryConfig
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


class _FakeEmbedder:
    """Minimal stand-in to exercise the warmup branch in _ensure_warmed."""

    def __init__(self) -> None:
        self.warmups = 0

    async def warmup_async(self, _ctx: object) -> None:
        self.warmups += 1


@pytest_asyncio.fixture
async def wired() -> AsyncIterator[tuple[FastMCP, _FakeEmbedder]]:
    db = AsyncVectorDB(":memory:")
    embedder = _FakeEmbedder()
    try:
        service = MemoryService(db, config=MemoryConfig(), embed_fn=hash_embed)
        mcp: FastMCP = FastMCP("synara-test")
        register_tools(mcp, service, embedder=embedder)  # type: ignore[arg-type]
        yield mcp, embedder
    finally:
        await db.close()


async def test_store_episode_tool_encodes_and_dedups(
    wired: tuple[FastMCP, _FakeEmbedder],
) -> None:
    mcp, embedder = wired
    async with Client(mcp) as client:
        r1 = await client.call_tool(
            "store_episode",
            {"content": "the cat sat on the mat", "session_id": "s1", "tags": ["t"]},
        )
        assert r1.data["deduped"] is False
        assert r1.data["id"] >= 0
        # Identical content in same session -> dedup branch (info path).
        r2 = await client.call_tool(
            "store_episode",
            {"content": "the cat sat on the mat", "session_id": "s1"},
        )
        assert r2.data["deduped"] is True
        assert r2.data["id"] == r1.data["id"]
    # warmup_async invoked on every warmed tool call.
    assert embedder.warmups >= 2


async def test_recall_episodes_tool_returns_hits(
    wired: tuple[FastMCP, _FakeEmbedder],
) -> None:
    mcp, _ = wired
    async with Client(mcp) as client:
        await client.call_tool("store_episode", {"content": "alpha beta gamma", "session_id": "s1"})
        res = await client.call_tool(
            "recall_episodes",
            {"query": "alpha beta gamma", "session_id": "s1", "k": 4, "mode": "auto"},
        )
        hits = res.data
        assert isinstance(hits, list)
        assert any("alpha beta gamma" in (h.get("content") or "") for h in hits)


async def test_consolidate_episodes_tool(
    wired: tuple[FastMCP, _FakeEmbedder],
) -> None:
    mcp, _ = wired
    async with Client(mcp) as client:
        for i in range(6):
            await client.call_tool(
                "store_episode",
                {"content": f"deploy step number {i}", "session_id": "s1"},
            )
        res = await client.call_tool(
            "consolidate_episodes",
            {"session_id": "s1", "min_cluster_size": 2},
        )
        assert isinstance(res.data, list)


async def test_forget_episodes_tool_dry_run_and_delete(
    wired: tuple[FastMCP, _FakeEmbedder],
) -> None:
    mcp, _ = wired
    async with Client(mcp) as client:
        await client.call_tool(
            "store_episode",
            {"content": "ephemeral note", "session_id": "s1", "salience": 0.01},
        )
        dry = await client.call_tool(
            "forget_episodes",
            {"strength_floor": 0.99, "decay_tau_seconds": 1.0, "dry_run": True},
        )
        assert "candidate_ids" in dry.data
        wet = await client.call_tool(
            "forget_episodes",
            {"strength_floor": 0.99, "decay_tau_seconds": 1.0, "dry_run": False},
        )
        assert wet.data["dry_run"] is False
        assert "removed" in wet.data


async def test_forget_episodes_log_reports_real_removed_count(
    wired: tuple[FastMCP, _FakeEmbedder],
) -> None:
    """Regression: the tool logged result['deleted'] (always 0) instead
    of result['removed'], so a real delete reported 'pruned 0'."""
    mcp, _ = wired
    messages: list[str] = []

    async def log_handler(message: Any) -> None:
        data = message.data
        msg = data.get("msg") if isinstance(data, dict) else str(data)
        if msg:
            messages.append(str(msg))

    async with Client(mcp, log_handler=log_handler) as client:
        await client.call_tool(
            "store_episode",
            {"content": "doomed trace", "session_id": "s1", "salience": 0.01},
        )
        res = await client.call_tool(
            "forget_episodes",
            {"strength_floor": 0.99, "decay_tau_seconds": 1.0, "dry_run": False},
        )
    assert res.data["removed"] == 1
    assert "pruned 1 episode(s)" in messages
    assert "pruned 0 episode(s)" not in messages


async def test_reflect_session_tool(
    wired: tuple[FastMCP, _FakeEmbedder],
) -> None:
    mcp, _ = wired
    async with Client(mcp) as client:
        await client.call_tool(
            "store_episode",
            {"content": "reflectable trace", "session_id": "s1", "tags": ["topic"]},
        )
        res = await client.call_tool("reflect_session", {"session_id": "s1", "k": 3})
        assert "schemas" in res.data
        assert "recent_episodes" in res.data
        # explicit query path
        res2 = await client.call_tool(
            "reflect_session", {"session_id": "s1", "query": "reflectable", "k": 3}
        )
        assert "recent_episodes" in res2.data


async def test_semantic_store_and_recall_tools(
    wired: tuple[FastMCP, _FakeEmbedder],
) -> None:
    mcp, _ = wired
    async with Client(mcp) as client:
        stored = await client.call_tool(
            "store_semantic_memory",
            {
                "content": "prefer ruff over flake8",
                "kind": "preference",
                "tags": ["tooling"],
                "confidence": 0.9,
            },
        )
        assert stored.data["id"] >= 0
        hit = await client.call_tool(
            "recall_semantic_memory",
            {"query": "prefer ruff over flake8", "k": 5, "kind": "preference"},
        )
        assert isinstance(hit.data, list)
        assert any("ruff" in (h.get("content") or "") for h in hit.data)
        # kind filter that excludes the only row -> empty
        miss = await client.call_tool(
            "recall_semantic_memory",
            {"query": "prefer ruff over flake8", "kind": "fact"},
        )
        assert miss.data == []


async def test_memory_stats_tool(
    wired: tuple[FastMCP, _FakeEmbedder],
) -> None:
    mcp, _ = wired
    async with Client(mcp) as client:
        empty = await client.call_tool("memory_stats", {})
        assert empty.data == {"episodic_count": 0, "semantic_count": 0}
        await client.call_tool("store_episode", {"content": "one episode", "session_id": "s1"})
        after = await client.call_tool("memory_stats", {})
        assert after.data["episodic_count"] == 1


async def test_ensure_warmed_noop_when_embedder_none() -> None:
    """embedder=None short-circuits _ensure_warmed (line 44-45)."""
    db = AsyncVectorDB(":memory:")
    try:
        service = MemoryService(db, config=MemoryConfig(), embed_fn=hash_embed)
        mcp: FastMCP = FastMCP("synara-test-noembed")
        register_tools(mcp, service, embedder=None)
        async with Client(mcp) as client:
            res = await client.call_tool(
                "store_episode", {"content": "no embedder path", "session_id": "s1"}
            )
            assert res.data["deduped"] is False
    finally:
        await db.close()


async def test_embedding_failure_surfaces_to_caller() -> None:
    """An embed_fn that raises must propagate through the tool layer.

    Silent swallowing of embedding errors would let a broken backend
    return wrong-but-syntactically-valid results to MCP callers; the
    tool surface must fail loudly instead.
    """
    import pytest  # noqa: PLC0415
    from fastmcp.exceptions import ToolError  # noqa: PLC0415

    class _BoomEmbedder:
        async def warmup_async(self, _ctx: object) -> None:
            return None

    def boom(_text: str) -> list[float]:
        raise RuntimeError("embedding backend unreachable")

    db = AsyncVectorDB(":memory:")
    try:
        service = MemoryService(db, config=MemoryConfig(), embed_fn=boom)
        mcp: FastMCP = FastMCP("synara-test-boom")
        register_tools(mcp, service, embedder=_BoomEmbedder())  # type: ignore[arg-type]
        async with Client(mcp) as client:
            with pytest.raises(ToolError):
                await client.call_tool(
                    "store_episode",
                    {"content": "will not embed", "session_id": "s1"},
                )
    finally:
        await db.close()
