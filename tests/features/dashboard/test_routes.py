"""Phase 2.2 — routes delegate to MemoryService (no inlined memory logic).

Driven against a real :memory: MemoryService with a deterministic
embedder, so correct delegation is proven by behaviour; a spy test
additionally asserts the delete route calls the service method.
"""

from __future__ import annotations

import hashlib
from collections.abc import AsyncIterator

import httpx
import numpy as np
import pytest
import pytest_asyncio
from simplevecdb import AsyncVectorDB

from synara.config import Settings
from synara.features.dashboard import build_dashboard_app
from synara.features.memory import MemoryService
from synara.features.memory.service import MemoryConfig


def hash_embed(text: str, dim: int = 32) -> list[float]:
    seed = int.from_bytes(hashlib.sha256(text.encode("utf-8")).digest()[:8], "big", signed=False)
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(dim).astype(np.float32)
    n = float(np.linalg.norm(v))
    out = (v / n) if n > 0 else v
    return [float(x) for x in out.tolist()]


@pytest_asyncio.fixture
async def ctx() -> AsyncIterator[tuple[httpx.AsyncClient, MemoryService]]:
    db = AsyncVectorDB(":memory:")
    service = MemoryService(db, config=MemoryConfig(), embed_fn=hash_embed)
    app = build_dashboard_app(
        settings=Settings.from_env(),
        db=db,
        embedder=None,
        service=service,
    )
    transport = httpx.ASGITransport(app=app)
    try:
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            yield client, service
    finally:
        await db.close()


async def test_stats_and_params(
    ctx: tuple[httpx.AsyncClient, MemoryService],
) -> None:
    client, _ = ctx
    s = await client.get("/api/stats")
    assert s.status_code == 200
    assert set(s.json()) >= {"episodic_count", "semantic_count"}

    p = await client.get("/api/params")
    assert p.status_code == 200
    assert p.json()["sr_gamma"] == 0.7  # MemoryConfig default


async def test_list_detail_delete_roundtrip(
    ctx: tuple[httpx.AsyncClient, MemoryService],
) -> None:
    client, service = ctx
    enc = await service.encode_episode("an episode about comets", "s1")
    ep_id = enc["id"]

    lst = await client.get("/api/memories", params={"kind": "episodic"})
    assert lst.status_code == 200
    assert any(i["id"] == ep_id for i in lst.json()["items"])

    det = await client.get(f"/api/memories/{ep_id}")
    assert det.status_code == 200
    assert det.json()["id"] == ep_id
    assert "plasticity_edges" in det.json()

    missing = await client.get("/api/memories/999999")
    assert missing.status_code == 404

    dele = await client.request("DELETE", f"/api/memories/{ep_id}")
    assert dele.status_code == 200
    assert ep_id in dele.json()["deleted_ids"]
    assert (await service.episodic.get_documents({"id": ep_id})) == []

    assert (await client.request("DELETE", "/api/memories/999999")).status_code == 404


async def test_search_and_graph(
    ctx: tuple[httpx.AsyncClient, MemoryService],
) -> None:
    client, service = ctx
    await service.encode_episode("graphs and edges and nodes", "s1")
    await service.encode_episode("comets and orbits", "s1")
    await service.recall("graphs", session_id="s1", k=5)

    srch = await client.get("/api/memories", params={"kind": "episodic", "q": "graphs"})
    assert srch.status_code == 200
    assert srch.json()["query"] == "graphs"

    g = await client.get("/api/graph", params={"max_nodes": 50})
    assert g.status_code == 200
    body = g.json()
    assert {
        "nodes",
        "sr_edges",
        "plasticity_edges",
        "consolidation_edges",
        "omega",
        "episode_count",
        "focus",
        "truncated",
    } <= set(body)
    # Nodes are enriched objects, not bare ids, and carry the kind +
    # ranking signals the map renders.
    for n in body["nodes"]:
        assert {"id", "key", "kind", "label", "preview"} <= set(n)
        assert n["kind"] in ("episodic", "semantic")
        if n["kind"] == "episodic":
            assert {"salience", "retrieval_count", "session_id"} <= set(n)
    # SR edges expose the discounted closure M used as the recall prior.
    for e in body["sr_edges"]:
        assert {"src", "dst", "hits", "m"} <= set(e)
    assert isinstance(body["omega"], (int, float))

    # Focusing resolves the *bidirectional* associative neighbourhood,
    # not a forward-only SR walk: a pure-successor episode must not be
    # an island. The second episode co-occurred with the first in s1,
    # so focusing it pulls in its predecessor.
    ep_ids = sorted(int(n["id"]) for n in body["nodes"] if n["kind"] == "episodic")
    assert len(ep_ids) >= 2
    f = await client.get("/api/graph", params={"focus": ep_ids[-1], "depth": 2, "max_nodes": 50})
    assert f.status_code == 200
    fbody = f.json()
    assert fbody["focus"] == ep_ids[-1]
    assert len({int(n["id"]) for n in fbody["nodes"]}) > 1


async def test_admin_forget_dry_run_delegates(
    ctx: tuple[httpx.AsyncClient, MemoryService],
) -> None:
    client, service = ctx
    await service.encode_episode("a trace to maybe forget", "s1")
    r = await client.post("/api/admin/forget", json={"dry_run": True})
    assert r.status_code == 200
    assert "candidate_ids" in r.json()


async def test_delete_route_calls_service_method(
    ctx: tuple[httpx.AsyncClient, MemoryService],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Guard against inlined logic: the route must call delete_episode."""
    client, service = ctx
    enc = await service.encode_episode("spy target", "s1")
    seen: dict[str, int] = {}
    real = service.delete_episode

    async def spy(episode_id: int, **kw: object) -> dict[str, object]:
        seen["id"] = episode_id
        return await real(episode_id, **kw)  # type: ignore[arg-type]

    monkeypatch.setattr(service, "delete_episode", spy)
    resp = await client.request("DELETE", f"/api/memories/{enc['id']}")
    assert resp.status_code == 200
    assert seen["id"] == enc["id"]
