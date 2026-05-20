"""Phase 2.1 — app builds, /api/health works, FastAPI stays optional."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import Any, cast

import httpx
import pytest

import synara.features.dashboard as dashboard_pkg
from synara.config import Settings
from synara.features.dashboard import build_dashboard_app

_REPO_ROOT = Path(__file__).resolve().parents[3]
_SRC = _REPO_ROOT / "src"


def _build_app() -> Any:
    settings = Settings.from_env()
    return build_dashboard_app(
        settings=settings,
        db=cast(Any, object()),  # health does not touch db
        embedder=None,
        service=cast(Any, object()),  # health does not touch service
    )


async def test_health_returns_identity() -> None:
    app = _build_app()
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://127.0.0.1:8765") as client:
        resp = await client.get("/api/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["transport"] == "stdio"
    assert body["embedding_backend"] == "local"
    assert "version" in body
    assert body["uptime_seconds"] >= 0.0


async def test_no_openapi_surface() -> None:
    app = _build_app()
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://127.0.0.1:8765") as client:
        assert (await client.get("/openapi.json")).status_code == 404
        assert (await client.get("/docs")).status_code == 404


def test_default_import_does_not_load_fastapi() -> None:
    """Importing the default config path must not import FastAPI.

    Run in a clean subprocess: the pytest process itself has FastAPI
    loaded (this test imports it), so isolation must be checked fresh.
    """
    code = (
        "import sys\n"
        "import synara.config\n"
        "synara.config.Settings.from_env()\n"
        "assert 'fastapi' not in sys.modules, 'FastAPI eagerly imported'\n"
        "import synara.features.dashboard as d\n"
        "_ = d.build_dashboard_app\n"
        "assert 'fastapi' in sys.modules, 'lazy export did not load app'\n"
        "print('ISOLATION_OK')\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=_REPO_ROOT,
        env={"PYTHONPATH": str(_SRC), "PATH": sys.exec_prefix + "/bin"},
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "ISOLATION_OK" in result.stdout


@pytest.mark.parametrize("attr", ["does_not_exist", "_private"])
def test_package_getattr_rejects_unknown(attr: str) -> None:
    with pytest.raises(AttributeError):
        getattr(dashboard_pkg, attr)
