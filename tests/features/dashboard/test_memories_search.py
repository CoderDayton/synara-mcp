"""Search route: semantic recall + substring fallback merge.

These tests cover the three regimes the dashboard map relies on:

  * ``semantic`` — recall returns hits; no substring leg needed.
  * ``substring`` — recall is empty (cold store / mode-mismatch) but a
    literal substring scan still lights up the map.
  * ``hybrid`` — both legs contribute; semantic hits keep their position
    and substring-only ids append behind, with overlap deduped.

We also assert that the substring scan is case-insensitive and that
``recall_mode`` reports ``empty`` when neither leg finds anything.
"""

from __future__ import annotations

import hashlib
from collections.abc import AsyncIterator

import httpx
import numpy as np
import pytest_asyncio
from simplevecdb import AsyncVectorDB

from synara.config import Settings
from synara.features.dashboard import build_dashboard_app
from synara.features.dashboard.routes.memories import (
    _merge_search_hits,
    _substring_scan,
)
from synara.features.memory import MemoryService
from synara.features.memory.service import MemoryConfig


def _hash_embed(text: str, dim: int = 32) -> list[float]:
    """Deterministic embedder so tests don't pull a real model.

    The hash is salted so the embedder isn't constant — a recall query
    that's lexically near a stored text still produces a different
    vector, exercising the semantic-vs-substring split.
    """
    seed = int.from_bytes(hashlib.sha256(text.encode("utf-8")).digest()[:8], "big", signed=False)
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(dim).astype(np.float32)
    n = float(np.linalg.norm(v))
    out = (v / n) if n > 0 else v
    return [float(x) for x in out.tolist()]


@pytest_asyncio.fixture
async def ctx() -> AsyncIterator[tuple[httpx.AsyncClient, MemoryService]]:
    db = AsyncVectorDB(":memory:")
    service = MemoryService(db, config=MemoryConfig(), embed_fn=_hash_embed)
    app = build_dashboard_app(
        settings=Settings.from_env(),
        db=db,
        embedder=None,
        service=service,
    )
    transport = httpx.ASGITransport(app=app)
    try:
        async with httpx.AsyncClient(
            transport=transport, base_url="http://127.0.0.1:8765"
        ) as client:
            yield client, service
    finally:
        await db.close()


# ----------------------------------------------------------- route ---


async def test_search_returns_recall_mode_field(
    ctx: tuple[httpx.AsyncClient, MemoryService],
) -> None:
    client, service = ctx
    await service.encode_episode("the auth migration is rolling out", "s1")
    r = await client.get("/api/memories", params={"kind": "episodic", "q": "auth"})
    assert r.status_code == 200
    body = r.json()
    assert body["query"] == "auth"
    assert body["kind"] == "episodic"
    assert body["recall_mode"] in {"semantic", "substring", "hybrid", "empty"}


async def test_search_substring_lights_up_when_recall_misses(
    ctx: tuple[httpx.AsyncClient, MemoryService],
) -> None:
    """Hash-based embedder makes semantic recall miss almost everything;
    the substring leg must still surface the matching episode."""
    client, service = ctx
    await service.encode_episode("zzzz unique sentinel word cachexyz", "s1")
    await service.encode_episode("unrelated content about birds", "s1")
    r = await client.get("/api/memories", params={"kind": "episodic", "q": "cachexyz"})
    assert r.status_code == 200
    body = r.json()
    assert body["count"] >= 1
    ids = [item["id"] for item in body["items"]]
    # Sentinel episode must be present even if recall ranked it nowhere.
    contents = [item["content"] for item in body["items"]]
    assert any("cachexyz" in c for c in contents), (ids, contents, body["recall_mode"])
    assert body["recall_mode"] in {"substring", "hybrid"}


async def test_search_substring_is_case_insensitive(
    ctx: tuple[httpx.AsyncClient, MemoryService],
) -> None:
    client, service = ctx
    await service.encode_episode("CAPSLOCK token rotation", "s1")
    r = await client.get("/api/memories", params={"kind": "episodic", "q": "capslock"})
    assert r.status_code == 200
    body = r.json()
    assert any("CAPSLOCK" in item["content"] for item in body["items"])


async def test_search_empty_query_uses_list_path(
    ctx: tuple[httpx.AsyncClient, MemoryService],
) -> None:
    """No ``q`` → bare listing endpoint; no recall_mode field."""
    client, service = ctx
    await service.encode_episode("anything", "s1")
    r = await client.get("/api/memories", params={"kind": "episodic"})
    body = r.json()
    assert "recall_mode" not in body
    assert "offset" in body


async def test_search_empty_recall_mode_when_store_is_empty(
    ctx: tuple[httpx.AsyncClient, MemoryService],
) -> None:
    """Recall against an empty episodic collection returns []; the
    substring leg also returns []. Mode must be 'empty'."""
    client, _ = ctx
    r = await client.get(
        "/api/memories",
        params={"kind": "episodic", "q": "anything"},
    )
    body = r.json()
    assert body["count"] == 0
    assert body["recall_mode"] == "empty"


async def test_search_semantic_branch_for_semantic_kind(
    ctx: tuple[httpx.AsyncClient, MemoryService],
) -> None:
    """Semantic search routes through ``recall_semantic_memory`` and
    still supports substring fallback."""
    client, service = ctx
    await service.store_semantic_memory("Users authenticate with JWTs", kind="schema")
    r = await client.get("/api/memories", params={"kind": "semantic", "q": "JWTs"})
    assert r.status_code == 200
    body = r.json()
    assert body["kind"] == "semantic"
    assert body["count"] >= 1
    assert body["recall_mode"] in {"substring", "hybrid", "semantic"}


# ----------------------------------------------------- pure helpers ---


def test_merge_search_hits_semantic_only_keeps_order() -> None:
    recall = [
        {"id": 1, "distance": 0.1, "content": "a"},
        {"id": 2, "distance": 0.2, "content": "b"},
    ]
    merged, mode = _merge_search_hits(recall_hits=recall, substring_hits=[], limit=10)
    assert [h["id"] for h in merged] == [1, 2]
    assert mode == "semantic"


def test_merge_search_hits_substring_only_keeps_substring_order() -> None:
    substring = [
        {"id": 9, "distance": None, "substring_offset": 2, "content": "x"},
        {"id": 7, "distance": None, "substring_offset": 4, "content": "y"},
    ]
    merged, mode = _merge_search_hits(recall_hits=[], substring_hits=substring, limit=10)
    assert [h["id"] for h in merged] == [9, 7]
    assert mode == "substring"


def test_merge_search_hits_hybrid_dedups_with_substring_annotation() -> None:
    """A doc that matches both legs should appear once, with the
    substring offset annotated onto the recall hit so the UI can
    highlight the match in-place."""
    recall = [{"id": 1, "distance": 0.1, "content": "abc"}]
    substring = [
        {"id": 1, "distance": None, "substring_offset": 0, "content": "abc"},
        {"id": 2, "distance": None, "substring_offset": 5, "content": "xabc"},
    ]
    merged, mode = _merge_search_hits(recall_hits=recall, substring_hits=substring, limit=10)
    assert mode == "hybrid"
    assert [h["id"] for h in merged] == [1, 2]
    assert merged[0]["substring_offset"] == 0  # annotated onto recall hit
    assert merged[0]["distance"] == 0.1  # recall metadata preserved


def test_merge_search_hits_respects_limit() -> None:
    recall = [{"id": i, "distance": 0.1 * i, "content": str(i)} for i in range(5)]
    substring = [
        {"id": 100 + i, "distance": None, "substring_offset": i, "content": "x"} for i in range(5)
    ]
    merged, _ = _merge_search_hits(recall_hits=recall, substring_hits=substring, limit=4)
    assert len(merged) == 4
    # Semantic hits always come first; substring only fills the tail.
    assert [h["id"] for h in merged] == [0, 1, 2, 3]


async def test_substring_scan_empty_query_returns_nothing(
    ctx: tuple[httpx.AsyncClient, MemoryService],
) -> None:
    _, service = ctx
    await service.encode_episode("hello", "s1")
    assert await _substring_scan(service, kind="episodic", q="   ", limit=10) == []
    assert await _substring_scan(service, kind="episodic", q="", limit=10) == []


async def test_substring_scan_orders_by_offset(
    ctx: tuple[httpx.AsyncClient, MemoryService],
) -> None:
    _, service = ctx
    a = await service.encode_episode("the WORD shows up here", "s1")
    b = await service.encode_episode("WORD at start", "s1")
    c = await service.encode_episode("middle middle WORD tail", "s1")
    hits = await _substring_scan(service, kind="episodic", q="WORD", limit=10)
    by_id = {h["id"]: h for h in hits}
    assert by_id[b["id"]]["substring_offset"] == 0
    assert by_id[a["id"]]["substring_offset"] > by_id[b["id"]]["substring_offset"]
    assert by_id[c["id"]]["substring_offset"] > by_id[a["id"]]["substring_offset"]
    # Returned in offset order.
    assert [h["id"] for h in hits] == [b["id"], a["id"], c["id"]]
