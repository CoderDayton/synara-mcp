"""Offline OpenAPI schema export for TypeScript client codegen.

The live dashboard app sets ``openapi_url=None`` (no served schema — it is an
internal admin API, not a public contract). Client codegen still needs the
schema, so this builds a bare FastAPI with the *same* API routers via
:func:`synara.features.dashboard.app.include_api_routers` (the single source of
truth for the route surface) and returns ``app.openapi()``.

No live state, auth, or static mount is required: schema generation reads only
the route signatures and their ``response_model``. Keeping this in the package
(rather than in ``scripts/``) makes it import-testable — see
``tests/features/dashboard/test_openapi_contract.py``.
"""

from __future__ import annotations

from typing import Any

from fastapi import FastAPI

from synara import __version__

from .app import include_api_routers


def build_schema_app() -> FastAPI:
    """A routers-only FastAPI used solely to emit the OpenAPI schema."""
    app = FastAPI(title="Synara Dashboard", version=__version__)
    include_api_routers(app)
    return app


def export_schema() -> dict[str, Any]:
    """Return the dashboard OpenAPI schema as a plain dict."""
    return build_schema_app().openapi()
