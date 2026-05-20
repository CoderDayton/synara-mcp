"""Phase 4.3 — committed SPA build is served with history fallback.

The static shell is unauthenticated by design (carries no data; the
bearer token is entered in the UI). The guarded ``/api`` surface must
keep precedence over the catch-all. These run against the real
committed ``static/`` produced by ``bun run build``.
"""

from __future__ import annotations

import hashlib
from collections.abc import AsyncIterator
from pathlib import Path

import httpx
import pytest
import pytest_asyncio
from simplevecdb import AsyncVectorDB

from synara.config import Settings
from synara.features.dashboard import build_dashboard_app
from synara.features.dashboard.app import _STATIC_DIR
from synara.features.memory import MemoryService
from synara.features.memory.service import MemoryConfig

_HAS_BUILD = (_STATIC_DIR / "index.html").is_file()
pytestmark = pytest.mark.skipif(
    not _HAS_BUILD, reason="SPA not built (run `bun run build` in dashboard/)"
)


def hash_embed(text: str, dim: int = 32) -> list[float]:
    seed = int.from_bytes(hashlib.sha256(text.encode("utf-8")).digest()[:8], "big", signed=False)
    return [float((seed >> (i % 60)) & 1) for i in range(dim)]


@pytest_asyncio.fixture
async def client() -> AsyncIterator[httpx.AsyncClient]:
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
        async with httpx.AsyncClient(transport=transport, base_url="http://127.0.0.1:8765") as c:
            yield c
    finally:
        await db.close()


async def test_root_serves_index(client: httpx.AsyncClient) -> None:
    r = await client.get("/")
    assert r.status_code == 200
    assert "text/html" in r.headers["content-type"]
    assert '<div id="root">' in r.text


async def test_client_route_falls_back_to_index(client: httpx.AsyncClient) -> None:
    """Deep links (client-side routes) return the shell, not 404."""
    r = await client.get("/memories")
    assert r.status_code == 200
    assert "text/html" in r.headers["content-type"]


async def test_api_precedence_over_spa(client: httpx.AsyncClient) -> None:
    r = await client.get("/api/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


async def test_unknown_api_path_is_404_not_shell(client: httpx.AsyncClient) -> None:
    r = await client.get("/api/does-not-exist")
    assert r.status_code == 404
    assert "text/html" not in r.headers.get("content-type", "")


async def test_no_path_traversal(client: httpx.AsyncClient) -> None:
    """A traversal attempt resolves outside static/ → shell, never a leak."""
    r = await client.get("/../../pyproject.toml")
    assert r.status_code == 200
    assert "[project]" not in r.text
    assert '<div id="root">' in r.text


def test_static_dir_is_the_package_dir() -> None:
    assert Path(__file__).parents[3] / "src/synara/features/dashboard/static" == _STATIC_DIR
