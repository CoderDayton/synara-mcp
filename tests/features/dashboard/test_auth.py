"""Phase 2.3 — bearer auth matrix + non-loopback startup posture."""

from __future__ import annotations

import dataclasses
from collections.abc import AsyncIterator

import httpx
import pytest
import pytest_asyncio
from simplevecdb import AsyncVectorDB

from synara.config import Settings
from synara.features.dashboard import build_dashboard_app
from synara.features.dashboard.config import DashboardConfig
from synara.features.memory import MemoryService
from synara.features.memory.service import MemoryConfig

_TOKEN = "s3cr3t-token"


def _settings_with(dash: DashboardConfig) -> Settings:
    return dataclasses.replace(Settings.from_env(), dashboard=dash)


@pytest_asyncio.fixture
async def db() -> AsyncIterator[AsyncVectorDB]:
    d = AsyncVectorDB(":memory:")
    try:
        yield d
    finally:
        await d.close()


def _client(db: AsyncVectorDB, settings: Settings) -> httpx.AsyncClient:
    app = build_dashboard_app(
        settings=settings,
        db=db,
        embedder=None,
        service=MemoryService(db, config=MemoryConfig(), embed_fn=lambda _t: [0.0]),
    )
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test")


async def test_loopback_no_token_allows(db: AsyncVectorDB) -> None:
    settings = _settings_with(DashboardConfig(enabled=True))  # 127.0.0.1, no token
    async with _client(db, settings) as c:
        assert (await c.get("/api/health")).status_code == 200


async def test_token_required_when_set(db: AsyncVectorDB) -> None:
    settings = _settings_with(DashboardConfig(enabled=True, host="127.0.0.1", token=_TOKEN))
    async with _client(db, settings) as c:
        assert (await c.get("/api/health")).status_code == 401
        bad = await c.get("/api/health", headers={"Authorization": "Bearer wrong"})
        assert bad.status_code == 401
        wrong_scheme = await c.get("/api/health", headers={"Authorization": f"Basic {_TOKEN}"})
        assert wrong_scheme.status_code == 401
        ok = await c.get("/api/health", headers={"Authorization": f"Bearer {_TOKEN}"})
        assert ok.status_code == 200


async def test_token_guards_mutating_routes(db: AsyncVectorDB) -> None:
    settings = _settings_with(DashboardConfig(enabled=True, token=_TOKEN))
    async with _client(db, settings) as c:
        assert (await c.request("DELETE", "/api/memories/1")).status_code == 401
        ok = await c.request(
            "DELETE",
            "/api/memories/999999",
            headers={"Authorization": f"Bearer {_TOKEN}"},
        )
        # Authn passes; episode genuinely absent -> 404 (not 401).
        assert ok.status_code == 404


def test_non_loopback_without_token_refuses_startup(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("SYNARA_DASHBOARD", "true")
    monkeypatch.setenv("SYNARA_DASHBOARD_HOST", "0.0.0.0")
    monkeypatch.delenv("SYNARA_DASHBOARD_TOKEN", raising=False)
    with pytest.raises(ValueError, match="without a token"):
        Settings.from_env()
