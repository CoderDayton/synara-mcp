"""Recall miss reports and the semantic->episodic fallback.

Measured over three days of real transcripts, 29% of recalls returned an
empty list (``recall_semantic_memory`` worst, 3 of 8). An empty array
cannot distinguish an empty store from session scoping, a tag filter, or
a query with no near neighbours — four states with four different fixes.
These tests pin each state to the cause the report names.

The relevance gate is disabled in the scenarios that isolate scope and
tag filtering, per its own config note: it cuts on raw cosine distance
and would otherwise mask which filter emptied the result.
"""

from __future__ import annotations

import hashlib
from collections.abc import AsyncIterator
from typing import Any

import numpy as np
import pytest
import pytest_asyncio
from fastmcp import Client, FastMCP
from fastmcp.exceptions import ToolError
from simplevecdb import AsyncVectorDB

from synara.core.argument_normalization import ArgumentNormalizationMiddleware
from synara.features.memory.config import MemoryConfig
from synara.features.memory.recall_report import (
    RecallDiagnostics,
    RecallRequest,
    build_miss_report,
)
from synara.features.memory.service import MemoryService
from synara.features.memory.tools import register_tools


def hash_embed(text: str, dim: int = 32) -> list[float]:
    seed = int.from_bytes(hashlib.sha256(text.encode("utf-8")).digest()[:8], "big")
    v = np.random.default_rng(seed).standard_normal(dim).astype(np.float32)
    n = float(np.linalg.norm(v))
    return [float(x) for x in ((v / n) if n > 0 else v).tolist()]


def _build(config: MemoryConfig) -> tuple[FastMCP, MemoryService, AsyncVectorDB]:
    db = AsyncVectorDB(":memory:")
    service = MemoryService(db, config=config, embed_fn=hash_embed)
    mcp: FastMCP = FastMCP("miss-test")
    # Mirrors build_server: normalization sits ahead of validation.
    mcp.add_middleware(ArgumentNormalizationMiddleware())
    register_tools(mcp, service)
    return mcp, service, db


@pytest_asyncio.fixture
async def ungated() -> AsyncIterator[tuple[FastMCP, MemoryService]]:
    mcp, service, db = _build(MemoryConfig(recall_relevance_gate=False))
    try:
        yield mcp, service
    finally:
        await db.close()


# --------------------------------------------------------------------
# miss report: dominant cause
# --------------------------------------------------------------------


async def test_empty_store_miss_names_the_empty_store(
    ungated: tuple[FastMCP, MemoryService],
) -> None:
    mcp, _ = ungated
    async with Client(mcp) as client:
        res = await client.call_tool("recall_episodes", {"query": "anything", "k": 4})
    assert res.data["results"] == []
    miss = res.data["miss"]
    assert miss["searched"]["stored"] == 0
    assert "empty" in miss["reason"].lower()
    # With nothing stored there is no retry worth suggesting.
    assert miss["suggestions"] == []


async def test_scope_miss_names_scoping_and_offers_the_opt_out(
    ungated: tuple[FastMCP, MemoryService],
) -> None:
    mcp, _ = ungated
    async with Client(mcp) as client:
        await client.call_tool(
            "store_episode", {"content": "alpha beta gamma delta", "session_id": "owner"}
        )
        res = await client.call_tool(
            "recall_episodes",
            {"query": "alpha beta gamma delta", "session_id": "stranger", "k": 4},
        )
    miss = res.data["miss"]
    assert miss["searched"]["dropped_by_scope"] >= 1
    assert miss["searched"]["stored"] >= 1
    assert "session scoping" in miss["reason"]
    assert any("scope_session=false" in s for s in miss["suggestions"])
    # The echoed scope shows the constraint the caller may not have realised.
    assert miss["scope"]["session_id"] == "stranger"


async def test_tag_miss_names_tags_not_scope(
    ungated: tuple[FastMCP, MemoryService],
) -> None:
    mcp, _ = ungated
    async with Client(mcp) as client:
        await client.call_tool(
            "store_episode",
            {"content": "alpha beta gamma delta", "session_id": "s1", "tags": ["kept"]},
        )
        res = await client.call_tool(
            "recall_episodes",
            {
                "query": "alpha beta gamma delta",
                "session_id": "s1",
                "tags": ["absent"],
                "k": 4,
            },
        )
    miss = res.data["miss"]
    assert miss["searched"]["dropped_by_tags"] >= 1
    assert miss["searched"]["dropped_by_scope"] == 0
    assert "tags filter" in miss["reason"]
    assert any("without tags" in s for s in miss["suggestions"])


async def test_relevance_gate_miss_says_raising_k_will_not_help() -> None:
    # Deterministic geometry rather than a lucky embedding draw: every
    # document is a distinct one-hot vector orthogonal to the query's, so
    # each sits at cosine distance exactly 1.0 while remaining distinct
    # enough from its neighbours to survive dedup. The dynamic ceiling
    # then derives d_ref = 1.0 and keeps only distance <= alpha (0.8), so
    # the whole candidate cloud is cut as off-topic — the exact state the
    # gate exists to produce.
    dim = 8
    docs = [f"orthogonal subject matter {i}" for i in range(6)]
    basis = {text: i + 1 for i, text in enumerate(docs)}

    def one_hot(text: str) -> list[float]:
        v = [0.0] * dim
        v[basis.get(text, 0)] = 1.0
        return v

    db = AsyncVectorDB(":memory:")
    service = MemoryService(db, config=MemoryConfig(), embed_fn=one_hot)
    mcp: FastMCP = FastMCP("gate-test")
    register_tools(mcp, service)
    try:
        async with Client(mcp) as client:
            for text in docs:
                await client.call_tool("store_episode", {"content": text, "session_id": "s1"})
            res = await client.call_tool(
                "recall_episodes",
                {"query": "query on its own axis", "session_id": "s1", "k": 4},
            )
    finally:
        await db.close()
    miss = res.data["miss"]
    assert miss["searched"]["dropped_by_gate"] == len(docs)
    assert miss["searched"]["nearest_distance"] == pytest.approx(1.0)
    assert "relevance gate" in miss["reason"]
    assert any("raising k will not" in s for s in miss["suggestions"])


async def test_nearest_distance_is_reported_and_rounded(
    ungated: tuple[FastMCP, MemoryService],
) -> None:
    mcp, _ = ungated
    async with Client(mcp) as client:
        await client.call_tool("store_episode", {"content": "alpha beta", "session_id": "owner"})
        res = await client.call_tool(
            "recall_episodes", {"query": "alpha beta", "session_id": "stranger"}
        )
    nearest = res.data["miss"]["searched"]["nearest_distance"]
    assert isinstance(nearest, float)
    # Four decimals: the raw float's trailing digits are unactionable noise.
    assert nearest == round(nearest, 4)


# --------------------------------------------------------------------
# hit path stays a plain array
# --------------------------------------------------------------------


async def test_hits_still_return_a_bare_array(
    ungated: tuple[FastMCP, MemoryService],
) -> None:
    mcp, _ = ungated
    async with Client(mcp) as client:
        await client.call_tool("store_episode", {"content": "alpha beta gamma", "session_id": "s1"})
        res = await client.call_tool(
            "recall_episodes", {"query": "alpha beta gamma", "session_id": "s1"}
        )
    assert isinstance(res.data, list)
    assert res.data
    assert "content" in res.data[0]


async def test_miss_report_can_be_disabled() -> None:
    mcp, _service, db = _build(MemoryConfig(recall_miss_report=False))
    try:
        async with Client(mcp) as client:
            res = await client.call_tool("recall_episodes", {"query": "nothing", "k": 4})
            sem = await client.call_tool("recall_semantic_memory", {"query": "nothing"})
    finally:
        await db.close()
    assert res.data == []
    assert sem.data == []


# --------------------------------------------------------------------
# semantic -> episodic fallback
# --------------------------------------------------------------------


async def test_semantic_miss_surfaces_episodic_traces_as_leads(
    ungated: tuple[FastMCP, MemoryService],
) -> None:
    mcp, _ = ungated
    async with Client(mcp) as client:
        await client.call_tool(
            "store_episode", {"content": "the deploy script needs sudo", "session_id": "s1"}
        )
        res = await client.call_tool(
            "recall_semantic_memory", {"query": "the deploy script needs sudo"}
        )
    assert res.data["results"] == []
    fallback = res.data["episodic_fallback"]
    assert fallback
    assert fallback[0]["content"].startswith("the deploy script")
    # The lead is announced first, ahead of any generic retry advice.
    assert "episodic_fallback" in res.data["miss"]["suggestions"][0]
    assert "semantic store is empty" in res.data["miss"]["reason"]


async def test_fallback_is_capped_by_config() -> None:
    mcp, _service, db = _build(
        MemoryConfig(recall_relevance_gate=False, recall_semantic_fallback_k=1)
    )
    try:
        async with Client(mcp) as client:
            for i in range(4):
                await client.call_tool(
                    "store_episode", {"content": f"deploy note number {i}", "session_id": "s1"}
                )
            res = await client.call_tool("recall_semantic_memory", {"query": "deploy note"})
    finally:
        await db.close()
    assert len(res.data["episodic_fallback"]) == 1


async def test_fallback_can_be_disabled() -> None:
    mcp, _service, db = _build(
        MemoryConfig(recall_relevance_gate=False, recall_semantic_episodic_fallback=False)
    )
    try:
        async with Client(mcp) as client:
            await client.call_tool("store_episode", {"content": "deploy note", "session_id": "s1"})
            res = await client.call_tool("recall_semantic_memory", {"query": "deploy note"})
    finally:
        await db.close()
    assert "episodic_fallback" not in res.data
    assert res.data["results"] == []


async def test_fallback_does_not_reinforce_the_episodes_it_surfaces(
    ungated: tuple[FastMCP, MemoryService],
) -> None:
    # The caller asked for distilled facts. Raw traces offered as a
    # courtesy must not bump retrieval counts or rewrite the SR graph as
    # though episodes had been requested.
    mcp, service = ungated
    async with Client(mcp) as client:
        await client.call_tool("store_episode", {"content": "deploy note", "session_id": "s1"})
        before = await service.episodic.get_documents()
        for _ in range(3):
            res = await client.call_tool("recall_semantic_memory", {"query": "deploy note"})
        after = await service.episodic.get_documents()
    assert res.data["episodic_fallback"]

    def counts(docs: list[tuple[int, str, dict[str, Any]]]) -> list[int]:
        return [int(md.get("retrieval_count", 0)) for _id, _text, md in docs]

    assert counts(after) == counts(before)


# --------------------------------------------------------------------
# argument normalization, end to end through the MCP surface
# --------------------------------------------------------------------


async def test_transcript_fault_tags_as_comma_string(
    ungated: tuple[FastMCP, MemoryService],
) -> None:
    # Verbatim from a failing transcript call.
    mcp, service = ungated
    async with Client(mcp) as client:
        res = await client.call_tool(
            "store_episode",
            {
                "content": "lunify model loading regression",
                "session_id": "s1",
                "tags": "lunify,model-loading,bug",
            },
        )
    assert res.data["id"] >= 0
    docs = await service.episodic.get_documents()
    stored = {t for _id, _text, md in docs for t in (md.get("tags") or [])}
    assert {"lunify", "model-loading", "bug"} <= stored


async def test_transcript_fault_semantic_tags_as_comma_string(
    ungated: tuple[FastMCP, MemoryService],
) -> None:
    mcp, _ = ungated
    async with Client(mcp) as client:
        res = await client.call_tool(
            "store_semantic_memory",
            {
                "content": "PRISM keeps the KV cache bounded during training",
                "session_id": "s1",
                "tags": "prism,attention,kv-cache",
            },
        )
    assert set(res.data["tags"]) == {"prism", "attention", "kv-cache"}


async def test_transcript_fault_limit_instead_of_k(
    ungated: tuple[FastMCP, MemoryService],
) -> None:
    mcp, _ = ungated
    async with Client(mcp) as client:
        for i in range(5):
            await client.call_tool(
                "store_episode", {"content": f"alpha beta gamma {i}", "session_id": "s1"}
            )
        res = await client.call_tool(
            "recall_episodes",
            {"query": "alpha beta gamma", "session_id": "s1", "limit": "2"},
        )
    assert isinstance(res.data, list)
    assert len(res.data) == 2


async def test_conflicting_alias_and_canonical_is_rejected_at_the_surface(
    ungated: tuple[FastMCP, MemoryService],
) -> None:
    mcp, _ = ungated
    async with Client(mcp) as client:
        with pytest.raises(ToolError, match="alias for 'k'"):
            await client.call_tool("recall_episodes", {"query": "alpha", "limit": 8, "k": 2})


# --------------------------------------------------------------------
# report construction units
# --------------------------------------------------------------------


def test_nan_distance_does_not_poison_the_nearest() -> None:
    diag = RecallDiagnostics()
    diag.note_candidates([float("nan"), 0.4, 1.2])
    assert diag.candidates_scanned == 3
    assert diag.nearest_distance == pytest.approx(0.4)


def test_all_nan_distances_report_no_nearest() -> None:
    diag = RecallDiagnostics()
    diag.note_candidates([float("nan")])
    assert diag.nearest_distance is None


def test_dominant_cause_wins_when_several_filters_dropped() -> None:
    diag = RecallDiagnostics(stored=50, candidates_scanned=30)
    diag.dropped_by_scope = 2
    diag.dropped_by_tags = 20
    diag.dropped_by_gate = 8
    report = build_miss_report(diag, RecallRequest(query="q", k=4, tags=["x"]))
    assert "tags filter" in report["miss"]["reason"]
    # Every applicable retry is still listed, not just the dominant one.
    assert len(report["miss"]["suggestions"]) == 3


def test_no_filter_activity_reports_no_near_neighbours() -> None:
    diag = RecallDiagnostics(stored=10, candidates_scanned=0)
    report = build_miss_report(diag, RecallRequest(query="q", k=4))
    assert "no neighbours" in report["miss"]["reason"]
    assert any("distinctive keywords" in s for s in report["miss"]["suggestions"])


async def test_zero_k_does_not_read_as_an_empty_store(
    ungated: tuple[FastMCP, MemoryService],
) -> None:
    # k <= 0 short-circuits before the pipeline searches anything, so
    # every counter is zero — the same all-zero state an empty store
    # produces. The report must not conflate them, and the semantic
    # fallback must not volunteer traces the caller asked not to get.
    mcp, _ = ungated
    async with Client(mcp) as client:
        await client.call_tool("store_episode", {"content": "alpha beta", "session_id": "s1"})
        res = await client.call_tool(
            "recall_episodes", {"query": "alpha beta", "session_id": "s1", "k": 0}
        )
        sem = await client.call_tool("recall_semantic_memory", {"query": "alpha beta", "k": 0})
    assert "no search was performed" in res.data["miss"]["reason"]
    assert res.data["miss"]["suggestions"] == []
    assert "empty" not in res.data["miss"]["reason"].lower()
    assert "episodic_fallback" not in sem.data


def test_scope_echo_omits_unset_filters() -> None:
    report = build_miss_report(RecallDiagnostics(stored=1), RecallRequest(query="q", k=4))
    scope = report["miss"]["scope"]
    assert "tags" not in scope
    assert "kind" not in scope
    assert scope["scope_session"] is None
