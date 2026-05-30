"""Contract guard: the dashboard OpenAPI schema is the TS-codegen source.

These lock the invariants the generated client (``dashboard/src/lib/api-types.ts``)
depends on, so a future route change cannot silently desync the client types —
the exact drift this anchoring was built to prevent (commit ecddf5e added
``sr_transitions_in`` server-side but the hand-mirrored client never grew it).
"""

from __future__ import annotations

from typing import Any

from synara.features.dashboard.openapi_export import export_schema

# Routes whose response is deliberately open-shaped (bare dict, no model) and
# therefore intentionally absent from the generated component types.
_OPEN_PATHS = {"/api/params", "/api/admin/reflect"}


def test_schema_builds_with_expected_paths() -> None:
    schema = export_schema()
    paths = set(schema["paths"])
    assert {
        "/api/health",
        "/api/stats",
        "/api/memories",
        "/api/memories/{episode_id}",
        "/api/semantic/{semantic_id}",
        "/api/graph",
        "/api/tool-metrics",
        "/api/admin/consolidate",
        "/api/admin/forget",
    } <= paths


def test_memory_detail_models_incoming_transitions() -> None:
    """The L1 fix: ``sr_transitions_in`` is a required field of the model.

    It was served by the route but missing from the client contract; the
    response model must now carry it so codegen propagates it to the UI.
    """
    detail = export_schema()["components"]["schemas"]["MemoryDetailResponse"]
    assert "sr_transitions_in" in detail["properties"]
    assert "sr_transitions_in" in detail["required"]


def test_typed_routes_have_component_response() -> None:
    """Every closed-shape route resolves its 200 body to a component schema
    (``$ref``) or a union (``anyOf``), never a bare object — otherwise the
    generated TS types would not be anchored to a named model."""
    schema = export_schema()
    for path, methods in schema["paths"].items():
        if path in _OPEN_PATHS:
            continue
        for method, op in _iter_ops(methods):
            body = op["responses"]["200"]["content"]["application/json"]["schema"]
            assert "$ref" in body or "anyOf" in body, (
                f"{method.upper()} {path} has no component-typed 200 response"
            )


def _iter_ops(methods: dict[str, Any]) -> list[tuple[str, dict[str, Any]]]:
    return [(m, op) for m, op in methods.items() if m in {"get", "post", "put", "delete"}]
