"""Pydantic response models — single source of truth for the API contract.

Every *closed* response shape served under ``/api`` is modelled here. The
TypeScript client types in ``dashboard/src/lib/api-types.ts`` are generated
from the OpenAPI schema these models produce (see
``synara.features.dashboard.openapi_export`` and ``scripts/export_openapi.py``),
so adding or removing a field here is the only way the client surface changes —
the server route and its TS twin cannot drift apart.

Deliberately *open* shapes are intentionally NOT modelled and keep returning
bare ``dict[str, Any]`` on their routes; a strict model would silently drop
their open keys:

* ``/api/params`` — full frozen ``MemoryConfig`` snapshot.
* ``/api/admin/reflect`` — free-form ``{summary, schemas, recent_episodes}``.
* the recall/substring hit items in ``/api/memories?q=`` search.
* episode ``segments`` inside :class:`MemoryDetailResponse`.
"""

from __future__ import annotations

from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field

# --------------------------------------------------------------- health/stats


class HealthResponse(BaseModel):
    status: str
    version: str
    transport: str
    db_path: str
    embedding_backend: Literal["local", "remote"]
    embedding_model: str
    uptime_seconds: float
    # Best-effort name of the client hosting this server (Claude Code,
    # Cursor, ...), resolved from the process ancestry at startup;
    # "unknown" when it can't be determined.
    mcp_client: str


class StatsResponse(BaseModel):
    # MemoryService.stats() appends dynamic hygiene counters past the four
    # declared fields; allow + serialise them rather than truncating the tail.
    model_config = ConfigDict(extra="allow")

    episodic_count: int
    semantic_count: int
    schema_candidate_count: int
    consolidate_epoch: int


# ----------------------------------------------------------- memories list


class MemoryListItem(BaseModel):
    id: int
    content: str
    metadata: dict[str, Any]


class MemoryBrowse(BaseModel):
    """``GET /memories`` without ``q`` — paginated raw listing."""

    kind: str
    items: list[MemoryListItem]
    count: int
    offset: int


class MemorySearch(BaseModel):
    """``GET /memories?q=`` — hybrid recall + substring search.

    Hit items carry open keys (distance, score, source, metadata,
    substring_offset) so they stay ``dict[str, Any]`` — modelling them
    strictly would drop fields the UI relies on.
    """

    kind: str
    query: str
    items: list[dict[str, Any]]
    count: int
    recall_mode: Literal["semantic", "substring", "hybrid", "empty"]


# --------------------------------------------------------- memory/semantic detail


class SrTransition(BaseModel):
    dst: int
    count: int


class SrTransitionIn(BaseModel):
    src: int
    count: int


class PlasticityEdgeDetail(BaseModel):
    src: int
    dst: int
    weight: float
    bonus: float
    hits: int


class MemoryDetailResponse(BaseModel):
    id: int
    group_id: int
    # Episode-group rows are open metadata dicts.
    segments: list[dict[str, Any]]
    sr_transitions: list[SrTransition]
    sr_transitions_in: list[SrTransitionIn]
    plasticity_edges: list[PlasticityEdgeDetail]


class SemanticDetailResponse(BaseModel):
    id: int
    content: str
    kind: str
    tags: list[str]
    confidence: float
    user_asserted: bool
    source_episode_ids: list[int]
    created_at: float
    updated_at: float


class DeleteResult(BaseModel):
    deleted_ids: list[int]
    count: int


# ------------------------------------------------------------------- graph


class EpisodicGraphNode(BaseModel):
    id: int
    key: str
    kind: Literal["episodic"]
    label: str
    salience: float
    retrieval_count: int
    session_id: str | None
    created_at: float
    last_accessed: float
    consolidated_into: int
    group_id: int
    segment_count: int
    preview: str
    is_focus: bool
    embedding: list[float] | None


class SemanticGraphNode(BaseModel):
    id: int
    key: str
    kind: Literal["semantic"]
    label: str
    confidence: float
    source_count: int
    user_asserted: bool
    preview: str
    embedding: list[float] | None


class SrEdge(BaseModel):
    src: int
    dst: int
    hits: int
    m: float


class PlasticityEdge(BaseModel):
    src: int
    dst: int
    hits: int
    weight: float
    bonus: float
    strength: float
    is_habit: bool


class ConsolidationEdge(BaseModel):
    # Mixed addressing (intentional): episodic ``src`` by id, semantic ``dst``
    # by ``"sem:<id>"`` key — see graph._semantic_overlay.
    src: int
    dst: str


class GraphResponse(BaseModel):
    # Discriminated on ``kind`` so each node dict validates against exactly
    # the right node model. ``legacy`` is client-derived only and absent here.
    nodes: list[Annotated[EpisodicGraphNode | SemanticGraphNode, Field(discriminator="kind")]]
    sr_edges: list[SrEdge]
    plasticity_edges: list[PlasticityEdge]
    consolidation_edges: list[ConsolidationEdge]
    omega: float
    episode_count: int
    focus: int | None
    truncated: bool
    semantics_truncated: bool


# ------------------------------------------------------------- tool metrics


class ToolMetricRow(BaseModel):
    name: str
    headline: str
    count: int
    error_count: int
    last_called_at: float | None
    last_duration_seconds: float | None
    p50_ms: float | None
    p95_ms: float | None


class ToolMetricsResponse(BaseModel):
    tools: list[ToolMetricRow]


# -------------------------------------------------------------------- admin


class ConsolidateResult(BaseModel):
    schemas_formed: int
    schemas: list[dict[str, Any]]


class ForgetResult(BaseModel):
    candidate_ids: list[int]
    removed: int
    dry_run: bool
    scanned: int
    # Cold-schema eviction pass (off unless forget_schema_unused_seconds > 0);
    # always present, defaulting to 0/[] when disabled. Modelled so the route's
    # full output reaches the client instead of being silently dropped.
    schemas_scanned: int
    cold_schema_candidate_ids: list[int]
    schemas_removed: int


class RestartResult(BaseModel):
    """``POST /admin/restart`` — the server re-execs itself in place."""

    status: str
    detail: str
