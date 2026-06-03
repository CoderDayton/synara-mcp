"""``POST /api/admin/restart`` schedules an in-place re-exec and replies.

The route must reply *before* the process is replaced, and the actual
``os.execv`` is isolated behind ``request_restart`` so the suite can
patch it and never re-exec the test runner.
"""

from __future__ import annotations

from typing import Any, cast

import httpx
import pytest

from synara.config import Settings
from synara.features.dashboard import build_dashboard_app
from synara.features.dashboard.routes import admin


def _build_app() -> Any:
    return build_dashboard_app(
        settings=Settings.from_env(),
        db=cast(Any, object()),  # restart touches neither db
        embedder=None,
        service=cast(Any, object()),  # nor service
    )


async def test_restart_schedules_and_replies(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[bool] = []
    monkeypatch.setattr(admin, "request_restart", lambda: calls.append(True))

    app = _build_app()
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://127.0.0.1:8765") as client:
        resp = await client.post("/api/admin/restart")

    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "restarting"
    assert isinstance(body["detail"], str)
    assert body["detail"]
    # The re-exec was scheduled exactly once, via the patched seam.
    assert calls == [True]
