"""MCP resource surface for the memory feature.

Lever 1 — ambient recall. ``recall_episodes`` is a *tool*: the agent must
choose to call it, so the most relevant memory stays invisible unless
explicitly queried. This module exposes the same cross-session recall
pipeline as an MCP **resource template** so a host can auto-attach
relevant memories to the current context with zero agent action —
reading ``memory://recall/<query>`` resolves to the ranked hits (cosine +
successor-representation + spreading activation), the read-only twin of
the ``recall_episodes`` tool.

Reads are *non-reinforcing*: unlike the ``recall_episodes`` tool, an
ambient resource read goes through ``MemoryService.recall_readonly``,
so it does not bump ``retrieval_count`` or update the SR/plasticity
graph. MCP resource reads are GET-like and a host may prefetch, cache,
or poll them, so a read must not mutate durable memory state. The read
is instrumented into the same :class:`ToolMetrics` collector as the tools
(so the dashboard surfaces resource traffic) and the service's
``ValidationError`` is mapped to FastMCP's ``ResourceError`` so the
reason reaches the client verbatim — mirroring the tool surface's
``ToolError`` translation.
"""

from __future__ import annotations

import json
import time

from fastmcp import Context, FastMCP
from fastmcp.exceptions import ResourceError

from synara.core.errors import ValidationError
from synara.features.embedding import Embedder

from .metrics import ToolMetrics
from .service import MemoryService

# Metric key for the ambient-recall resource. Distinct from any tool name
# so the dashboard renders it as its own row (declared after the tools, so
# it sorts last in the snapshot's insertion order).
_RESOURCE_METRIC = "relevant_memories"


def register_resources(
    mcp: FastMCP,
    service: MemoryService,
    *,
    embedder: Embedder | None = None,
    metrics: ToolMetrics | None = None,
) -> None:
    """Wire the memory feature's MCP resources onto ``mcp``.

    Companion to :func:`register_tools`; called from the feature's
    ``register``. Adds ambient cross-session recall as a resource
    template so hosts can inject relevant memories without an explicit
    tool call. Pass ``metrics`` to surface resource reads on the
    dashboard alongside the tools.
    """
    if metrics is not None:
        metrics.declare(_RESOURCE_METRIC, headline="ambient recall (resource)")

    @mcp.resource(
        "memory://recall/{query}{?k,session_id,mode}",
        name="relevant_memories",
        description=(
            "Ambient cross-session recall as a resource: read "
            "memory://recall/<query> to surface the most relevant past "
            "memories for the current context with no explicit tool "
            "call. Hits are ranked by cosine + successor-representation + "
            "spreading activation. Optional query params: k (max hits, "
            "default 8); session_id (ranking hint, never a filter); mode "
            "(auto|hybrid|episodic|semantic, default auto)."
        ),
        mime_type="application/json",
        tags={"memory", "recall", "ambient"},
    )
    async def relevant_memories(
        query: str,
        ctx: Context,
        k: int = 8,
        session_id: str | None = None,
        mode: str = "auto",
    ) -> str:
        # Inline metrics + error translation rather than stacking
        # decorators: keeping ``relevant_memories`` the exact function
        # FastMCP introspects avoids any signature rewrapping that could
        # break URI-template parameter mapping. Return a JSON-serialised
        # array as a string — FastMCP's resource path treats a returned
        # ``list`` as a list of content *items*, not a JSON payload.
        t0 = time.perf_counter()
        ok = False
        try:
            if embedder is not None:
                await embedder.warmup_async(ctx)
            await ctx.debug(f"resource recall: session_id={session_id!r} k={k} mode={mode!r}")
            hits = await service.recall_readonly(query=query, session_id=session_id, k=k, mode=mode)
            ok = True
            return json.dumps(hits)
        except ValidationError as exc:
            raise ResourceError(str(exc)) from exc
        finally:
            if metrics is not None:
                metrics.record(_RESOURCE_METRIC, time.perf_counter() - t0, ok=ok)
