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
import pytest
import pytest_asyncio
from fastmcp import Client, FastMCP
from fastmcp.exceptions import ToolError
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


async def test_recall_content_only_projects_id_kind_content(
    wired: tuple[FastMCP, _FakeEmbedder],
) -> None:
    """content_only=True reduces each hit to {id, kind, content}, dropping
    distance/source/metadata/recency for both recall tools."""
    mcp, _ = wired
    async with Client(mcp) as client:
        await client.call_tool("store_episode", {"content": "alpha beta gamma", "session_id": "s1"})
        await client.call_tool(
            "store_semantic_memory",
            {"content": "prefer ruff over flake8", "kind": "preference", "scope": "global"},
        )

        ep = await client.call_tool(
            "recall_episodes",
            {"query": "alpha beta gamma", "session_id": "s1", "content_only": True},
        )
        assert ep.data, "expected an episodic hit"
        for h in ep.data:
            assert set(h) == {"id", "kind", "content"}
        assert any("alpha beta gamma" in (h["content"] or "") for h in ep.data)
        # Raw episodic traces carry no kind.
        assert all(h["kind"] is None for h in ep.data)

        sem = await client.call_tool(
            "recall_semantic_memory",
            {"query": "prefer ruff over flake8", "content_only": True},
        )
        assert sem.data, "expected a semantic hit"
        for h in sem.data:
            assert set(h) == {"id", "kind", "content"}
        assert any(h["kind"] == "preference" for h in sem.data)

        # Default (content_only omitted) keeps the rich shape.
        rich = await client.call_tool(
            "recall_episodes", {"query": "alpha beta gamma", "session_id": "s1"}
        )
        assert all("metadata" in h and "distance" in h for h in rich.data)


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
            {"strength_floor": 0.99, "dry_run": True},
        )
        assert "candidate_ids" in dry.data
        wet = await client.call_tool(
            "forget_episodes",
            {"strength_floor": 0.99, "dry_run": False},
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
            {"strength_floor": 0.99, "dry_run": False},
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
                "scope": "global",
            },
        )
        assert stored.data["id"] >= 0
        hit = await client.call_tool(
            "recall_semantic_memory",
            {"query": "prefer ruff over flake8", "k": 5, "kind": "preference"},
        )
        assert isinstance(hit.data, list)
        assert any("ruff" in (h.get("content") or "") for h in hit.data)
        # kind filter that excludes the only row -> miss report naming
        # the kind filter as the cause (full envelope is pinned in
        # test_recall_miss_report.py).
        miss = await client.call_tool(
            "recall_semantic_memory",
            {"query": "prefer ruff over flake8", "kind": "fact"},
        )
        assert miss.data["results"] == []
        assert miss.data["miss"]["searched"]["dropped_by_kind"] == 1
        assert "kind filter" in miss.data["miss"]["reason"]


async def test_memory_stats_tool(
    wired: tuple[FastMCP, _FakeEmbedder],
) -> None:
    mcp, _ = wired
    async with Client(mcp) as client:
        empty = await client.call_tool("memory_stats", {})
        assert empty.data["episodic_count"] == 0
        assert empty.data["semantic_count"] == 0
        assert empty.data["schema_candidate_count"] == 0
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


async def test_validation_error_reaches_client_under_masking() -> None:
    """Bad input must surface to the agent as an actionable ToolError
    message even with ``mask_error_details=True`` (production hardening).

    Without the explicit ValidationError->ToolError mapping, the service's
    rejection is an ordinary Exception, which FastMCP masks to a generic
    "Error calling tool" string under masking — hiding the reason from the
    caller (and logging it as an internal error with a traceback).
    """
    import pytest  # noqa: PLC0415
    from fastmcp.exceptions import ToolError  # noqa: PLC0415

    db = AsyncVectorDB(":memory:")
    try:
        service = MemoryService(db, config=MemoryConfig(), embed_fn=hash_embed)
        mcp: FastMCP = FastMCP("synara-test-validation", mask_error_details=True)
        register_tools(mcp, service)
        async with Client(mcp) as client:
            with pytest.raises(ToolError) as excinfo:
                await client.call_tool("recall_semantic_memory", {"query": "   "})
            assert "non-empty" in str(excinfo.value)
    finally:
        await db.close()


async def test_recall_episodes_truncates_content_with_full_escape_hatch(
    wired: tuple[FastMCP, _FakeEmbedder],
) -> None:
    """recall_episodes must bound per-hit content so a recall can't blow
    the caller's tool-result token budget; full=true restores the text."""
    mcp, _ = wired
    long_content = "alpha beta gamma " * 30  # ~510 chars, single segment (<1024)
    async with Client(mcp) as client:
        await client.call_tool("store_episode", {"content": long_content, "session_id": "s1"})
        # Default snippet path: content bounded, annotation present.
        res = await client.call_tool(
            "recall_episodes",
            {"query": "alpha beta gamma", "session_id": "s1", "max_chars": 50},
        )
        hit = next(h for h in res.data if h["content_chars"] >= 500)
        assert hit["truncated"] is True
        assert len(hit["content"]) == 50
        assert hit["content_chars"] >= 500
        # full=true escape hatch returns the untruncated content.
        full = await client.call_tool(
            "recall_episodes",
            {"query": "alpha beta gamma", "session_id": "s1", "full": True},
        )
        fhit = next(h for h in full.data if h["content_chars"] >= 500)
        assert fhit["truncated"] is False
        assert len(fhit["content"]) == fhit["content_chars"]


async def test_store_episode_session_id_optional(
    wired: tuple[FastMCP, _FakeEmbedder],
) -> None:
    """session_id is optional on store (no more raw pydantic missing-arg
    dump); omitting it stores under the default namespace and the episode
    is still recallable cross-session."""
    mcp, _ = wired
    async with Client(mcp) as client:
        r = await client.call_tool("store_episode", {"content": "unscoped trace zeta"})
        assert r.data["deduped"] is False
        assert r.data["id"] >= 0
        res = await client.call_tool("recall_episodes", {"query": "unscoped trace zeta"})
        assert any("unscoped trace zeta" in (h.get("content") or "") for h in res.data)


async def test_recall_episodes_scope_session_and_tags_filter(
    wired: tuple[FastMCP, _FakeEmbedder],
) -> None:
    """Opt-in scoping: scope_session hard-filters to the caller's session;
    tags keep only episodic hits carrying every requested tag. Without
    them recall stays cross-session (the documented default)."""
    mcp, _ = wired
    async with Client(mcp) as client:
        await client.call_tool(
            "store_episode",
            {"content": "shared topic alpha", "session_id": "proj-a", "tags": ["glf"]},
        )
        await client.call_tool(
            "store_episode",
            {"content": "shared topic alpha", "session_id": "proj-b", "tags": ["other"]},
        )
        # Cross-session default: both sessions can surface.
        unscoped = await client.call_tool(
            "recall_episodes", {"query": "shared topic alpha", "k": 8}
        )
        sids = {h["metadata"].get("session_id") for h in unscoped.data}
        assert {"proj-a", "proj-b"} <= sids
        # scope_session: only the caller's session survives.
        scoped = await client.call_tool(
            "recall_episodes",
            {
                "query": "shared topic alpha",
                "session_id": "proj-a",
                "scope_session": True,
                "k": 8,
            },
        )
        assert scoped.data
        assert all(h["metadata"].get("session_id") == "proj-a" for h in scoped.data)
        # tags: only hits carrying the requested tag.
        tagged = await client.call_tool(
            "recall_episodes",
            {"query": "shared topic alpha", "tags": ["glf"], "k": 8},
        )
        assert tagged.data
        assert all("glf" in (h["metadata"].get("tags") or []) for h in tagged.data)


async def test_store_after_first_recall_is_immediately_recallable(
    wired: tuple[FastMCP, _FakeEmbedder],
) -> None:
    """Read-after-write: an episode stored *after* the one-shot
    index-ready guard has fired must still be recallable without an
    intervening consolidate. Reproduces the simplevecdb pending-buffer
    hole described at service.py:_ensure_index_ready."""
    mcp, _ = wired
    async with Client(mcp) as client:
        await client.call_tool(
            "store_episode", {"content": "first trace omega", "session_id": "s1"}
        )
        # First recall trips the one-shot _ensure_index_ready guard.
        await client.call_tool("recall_episodes", {"query": "first trace omega"})
        # Store again *after* the guard already fired.
        await client.call_tool("store_episode", {"content": "second trace psi", "session_id": "s1"})
        res = await client.call_tool("recall_episodes", {"query": "second trace psi"})
        assert any("second trace psi" in (h.get("content") or "") for h in res.data)


async def test_store_semantic_memory_supersedes_retires_stale_entry(
    wired: tuple[FastMCP, _FakeEmbedder],
) -> None:
    """supersedes retires the prior semantic entry so a correction
    replaces it instead of layering; a bad id is an actionable error."""
    import pytest  # noqa: PLC0415
    from fastmcp.exceptions import ToolError  # noqa: PLC0415

    mcp, _ = wired
    async with Client(mcp) as client:
        old = await client.call_tool(
            "store_semantic_memory",
            {"content": "use flake8 for linting", "kind": "preference", "scope": "global"},
        )
        old_id = old.data["id"]
        new = await client.call_tool(
            "store_semantic_memory",
            {
                "content": "use ruff for linting",
                "kind": "preference",
                "supersedes": old_id,
                "scope": "global",
            },
        )
        assert new.data["superseded"] == old_id
        # The stale entry no longer surfaces; only the correction does.
        hits = await client.call_tool("recall_semantic_memory", {"query": "linting tool", "k": 8})
        ids = {h["id"] for h in hits.data}
        assert old_id not in ids
        assert new.data["id"] in ids
        # Superseding a non-existent id is an actionable error, not a no-op.
        with pytest.raises(ToolError) as excinfo:
            await client.call_tool(
                "store_semantic_memory",
                {"content": "orphan correction", "supersedes": 999_999},
            )
        assert "not found" in str(excinfo.value)


async def test_get_episode_returns_full_content_by_id(
    wired: tuple[FastMCP, _FakeEmbedder],
) -> None:
    """get_episode returns an episode's full untruncated text by id — the
    companion to recall's bounded snippets; a missing id is an error."""
    import pytest  # noqa: PLC0415
    from fastmcp.exceptions import ToolError  # noqa: PLC0415

    mcp, _ = wired
    long_content = "delta epsilon zeta " * 30  # ~570 chars, single segment
    async with Client(mcp) as client:
        stored = await client.call_tool(
            "store_episode", {"content": long_content, "session_id": "s1"}
        )
        ep_id = stored.data["id"]
        # Recall truncates to a snippet...
        rec = await client.call_tool(
            "recall_episodes", {"query": "delta epsilon zeta", "max_chars": 40}
        )
        assert any(h["truncated"] for h in rec.data if h["content_chars"] >= 500)
        # ...get_episode returns the whole thing.
        full = await client.call_tool("get_episode", {"episode_id": ep_id})
        assert full.data["id"] == ep_id
        assert long_content.strip() in full.data["content"]
        assert full.data["content_chars"] >= 500
        assert len(full.data["content"]) == full.data["content_chars"]
        # Missing id is an actionable error, not an empty result.
        with pytest.raises(ToolError) as excinfo:
            await client.call_tool("get_episode", {"episode_id": 999_999})
        assert "not found" in str(excinfo.value)


async def test_get_episode_reassembles_segmented_group(
    wired: tuple[FastMCP, _FakeEmbedder],
) -> None:
    """A theta-segmented episode is reassembled into its whole text rather
    than returning a single 1024-char segment. A mismatched session_id
    raises instead of returning one unreassembled segment."""
    import pytest  # noqa: PLC0415
    from fastmcp.exceptions import ToolError  # noqa: PLC0415

    mcp, _ = wired
    seg_content = "lambda mu nu " * 120  # ~1560 chars > theta_segment_max_chars
    async with Client(mcp) as client:
        stored = await client.call_tool(
            "store_episode", {"content": seg_content, "session_id": "s1"}
        )
        assert "segment_ids" in stored.data  # confirm it segmented
        full = await client.call_tool("get_episode", {"episode_id": stored.data["id"]})
        assert full.data["group_id"] is not None
        assert len(full.data["segment_ids"]) >= 2
        assert full.data["content_chars"] >= 1500
        # A segmented episode not visible in the requested session raises,
        # rather than silently returning a single unreassembled segment.
        with pytest.raises(ToolError) as excinfo:
            await client.call_tool(
                "get_episode", {"episode_id": stored.data["id"], "session_id": "other"}
            )
        assert "not visible" in str(excinfo.value)


async def test_store_semantic_supersedes_rolls_back_new_on_retire_failure() -> None:
    """If retiring the superseded entry fails *after* the correction is
    inserted, the new entry is rolled back and the stale one is left
    intact — never the both-layered state supersedes exists to prevent.

    Drives the partial-failure window directly: ``delete_by_ids`` is made
    to fail for the retire (old id) but succeed for the new-insert
    rollback (fresh id), so the store ends with exactly the original entry.
    """
    import pytest  # noqa: PLC0415

    db = AsyncVectorDB(":memory:")
    try:
        service = MemoryService(db, config=MemoryConfig(), embed_fn=hash_embed)
        old = await service.store_semantic_memory(
            content="use flake8 for linting", kind="preference", scope="global"
        )
        old_id = old["id"]
        assert await service.semantic.count() == 1

        orig_delete = service.semantic.delete_by_ids

        async def selective_delete(ids: list[int]) -> Any:
            # Reject the retire (old id); allow the rollback (fresh id).
            if old_id in ids:
                raise RuntimeError("retire failed")
            return await orig_delete(ids)

        service.semantic.delete_by_ids = selective_delete
        with pytest.raises(RuntimeError, match="retire failed"):
            await service.store_semantic_memory(
                content="use ruff for linting", kind="preference", supersedes=old_id, scope="global"
            )
        service.semantic.delete_by_ids = orig_delete

        # Exactly one entry remains: the correction was rolled back and the
        # stale entry survived — no both-layered residue.
        assert await service.semantic.count() == 1
        hits = await service.recall_semantic_memory(query="linting tool", k=8)
        assert any(h["id"] == old_id for h in hits)
    finally:
        await db.close()


def test_log_scope_cap_flags_saturated_under_return(caplog: Any) -> None:
    """A saturated scoped fetch (episodic candidates reached fetch_k) that
    leaves fewer than k survivors logs at INFO so the under-return is
    visible; an un-saturated fetch emits no such cap warning."""
    import logging  # noqa: PLC0415

    from synara.features.memory.hippocampus.recall import _log_scope_cap  # noqa: PLC0415

    name = "synara.features.memory.hippocampus.recall"
    with caplog.at_level(logging.INFO, logger=name):
        _log_scope_cap(fetch_k=32, ep_before=32, ep_after=1, k=4)
    assert any(r.levelno == logging.INFO and "capped" in r.getMessage() for r in caplog.records)

    caplog.clear()
    with caplog.at_level(logging.INFO, logger=name):
        # Not saturated (ep_before < fetch_k): no INFO cap warning.
        _log_scope_cap(fetch_k=32, ep_before=5, ep_after=5, k=4)
    assert not any("capped" in r.getMessage() for r in caplog.records)


async def test_recall_scope_session_without_session_id_rejected(
    wired: tuple[FastMCP, _FakeEmbedder],
) -> None:
    """scope_session=true with no session_id can't filter — it used to be
    a warn-and-proceed no-op; now the tool rejects it so the caller can't
    be misled into thinking scoping took effect."""
    mcp, _ = wired
    async with Client(mcp) as client:
        await client.call_tool("store_episode", {"content": "scope probe", "session_id": "s1"})
        with pytest.raises(ToolError, match="scope_session"):
            await client.call_tool(
                "recall_episodes", {"query": "scope probe", "scope_session": True}
            )
