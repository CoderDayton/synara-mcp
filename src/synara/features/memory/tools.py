"""MCP tool surface for the memory feature.

Tool descriptions are compressed for LLM consumption: terse parameter
specs, no prose. Validation lives in ``MemoryService``.

Surface
-------
Episodic store (raw event traces, scoped by ``session_id``):
    store_episode, recall_episodes, consolidate_episodes,
    forget_episodes, reflect_session.

Semantic store (distilled facts/procedures/preferences/schemas, global):
    store_semantic_memory, recall_semantic_memory.

The semantic store is also written to by ``consolidate_episodes`` —
the user-authored ``store_semantic_memory`` path is a separate, direct
write that bypasses the episodic→consolidate pipeline.

System:
    memory_stats.
"""

from __future__ import annotations

import functools
import time
from collections.abc import Awaitable, Callable
from typing import Any, TypeVar

from fastmcp import Context, FastMCP
from fastmcp.exceptions import ToolError

from synara.core.errors import ValidationError
from synara.features.embedding import Embedder

from .metrics import ToolMetrics
from .service import MemoryService

_R = TypeVar("_R")


def _instrument(
    metrics: ToolMetrics | None,
    name: str,
) -> Callable[[Callable[..., Awaitable[_R]]], Callable[..., Awaitable[_R]]]:
    """Wrap an async tool handler so each call updates ``metrics``.

    When ``metrics is None`` the original handler is returned unchanged
    so tests that don't wire a collector pay zero overhead. The wrapper
    preserves the handler's signature via :func:`functools.wraps`, which
    FastMCP's ``inspect.signature(..., follow_wrapped=True)`` honours —
    so the registered tool schema is identical to the unwrapped version.
    """
    if metrics is None:
        return lambda fn: fn

    def decorator(fn: Callable[..., Awaitable[_R]]) -> Callable[..., Awaitable[_R]]:
        @functools.wraps(fn)
        async def wrapped(*args: Any, **kwargs: Any) -> _R:
            t0 = time.perf_counter()
            ok = False
            try:
                result = await fn(*args, **kwargs)
                ok = True
                return result
            finally:
                metrics.record(name, time.perf_counter() - t0, ok=ok)

        return wrapped

    return decorator


def _translate_errors[R](
    fn: Callable[..., Awaitable[R]],
) -> Callable[..., Awaitable[R]]:
    """Map the service's ``ValidationError`` to FastMCP's ``ToolError``.

    Bad input is an expected, actionable rejection. Raising ``ToolError``
    routes it through FastMCP's ``FastMCPError`` passthrough so the reason
    reaches the agent verbatim — unprefixed, traceback-free, and still
    visible when ``mask_error_details`` is enabled. Any other exception is
    left untouched for FastMCP's default internal-error handling.
    """

    @functools.wraps(fn)
    async def wrapped(*args: Any, **kwargs: Any) -> R:
        try:
            return await fn(*args, **kwargs)
        except ValidationError as exc:
            raise ToolError(str(exc)) from exc

    return wrapped


# Single source of truth for the dashboard's per-tool headline. The
# server pre-declares these on the metrics collector so the live panel
# can render the full surface (including never-called tools) on the
# first poll instead of waiting for a call to materialise each row.
_TOOL_HEADLINES: dict[str, str] = {
    "store_episode": "encode an episodic trace",
    "recall_episodes": "cross-session episodic recall",
    "consolidate_episodes": "cluster traces → schemas",
    "forget_episodes": "power-law decay prune",
    "reflect_session": "summarise a session",
    "store_semantic_memory": "write a semantic memory",
    "recall_semantic_memory": "semantic memory recall",
    "memory_stats": "store + tunable snapshot",
}

# Shared compact definition. session_id is a caller-defined string
# namespace, not an MCP/process session.
_SID = (
    "session_id: caller-defined namespace str (e.g. conversation id, "
    "project name, persona). Same string = same namespace; not an "
    "MCP/process session. Acts as a ranking hint for episodic recall; "
    "scopes dedup, consolidate, and reflect. Semantic store ignores it."
)


async def _ensure_warmed(embedder: Embedder | None, ctx: Context) -> None:
    """Warmup embedder with progress reporting on first call."""
    if embedder is None:
        return
    await embedder.warmup_async(ctx)


def register_tools(
    mcp: FastMCP,
    service: MemoryService,
    *,
    embedder: Embedder | None = None,
    metrics: ToolMetrics | None = None,
) -> None:
    if metrics is not None:
        for tool_name, headline in _TOOL_HEADLINES.items():
            metrics.declare(tool_name, headline=headline)

    @mcp.tool(
        name="store_episode",
        description=(
            "Use when something happened worth remembering: a decision, "
            "event, user statement, action outcome, or observation. "
            "Embeds and stores the raw trace; near-duplicates within "
            "session_id merge (retrieval_count bumped) instead of "
            "inserting. Very short content (<8 chars stripped) always "
            "inserts.\n"
            "content: non-empty text to embed/store.\n"
            f"{_SID}\n"
            "tags: optional list[str]; used by reflect seeding and "
            "schema headline selection. Not a recall filter.\n"
            "salience: 0..1, default 0.5. Higher = slower power-law "
            "decay + preferred as schema headline. Use >0.7 for "
            "critical traces."
        ),
    )
    @_instrument(metrics, "store_episode")
    @_translate_errors
    async def store_episode(
        content: str,
        session_id: str,
        ctx: Context,
        tags: list[str] | None = None,
        salience: float = 0.5,
    ) -> dict[str, Any]:
        await _ensure_warmed(embedder, ctx)
        await ctx.debug(f"store_episode: session_id={session_id!r} salience={salience}")
        result = await service.encode_episode(
            content=content,
            session_id=session_id,
            tags=tags,
            salience=salience,
        )
        if result.get("deduped"):
            await ctx.info(f"deduped onto existing episode id={result['id']}")
        else:
            await ctx.info(f"encoded episode id={result['id']}")
        return result

    @mcp.tool(
        name="recall_episodes",
        description=(
            "Use to retrieve relevant past episodes before answering, "
            "continuing a task, or grounding a reply in prior context. "
            "Hits are ranked by cosine + successor-representation + "
            "spreading activation; each hit bumps retrieval_count.\n"
            "query: text, cosine-matched.\n"
            f"{_SID} Optional. Ranking hint only — never a hard "
            "filter. In-session episodes get a small bonus; "
            "cross-session episodes are still returned.\n"
            "k: max results, default 8 (ascending distance).\n"
            "mode: 'auto'/'hybrid' = episodic + semantic merged "
            "(default 'auto'); 'episodic' = raw traces only; "
            "'semantic' = schemas only — prefer recall_semantic_memory "
            "for that case."
        ),
    )
    @_instrument(metrics, "recall_episodes")
    @_translate_errors
    async def recall_episodes(
        query: str,
        ctx: Context,
        session_id: str | None = None,
        k: int = 8,
        mode: str = "auto",
    ) -> list[dict[str, Any]]:
        await _ensure_warmed(embedder, ctx)
        await ctx.debug(f"recall_episodes: session_id={session_id!r} k={k} mode={mode!r}")
        results = await service.recall(query=query, session_id=session_id, k=k, mode=mode)
        await ctx.info(f"recall returned {len(results)} hit(s)")
        return results

    @mcp.tool(
        name="consolidate_episodes",
        description=(
            "Cluster raw episodes into semantic schemas. Runs "
            "automatically in the background — call manually only to "
            "force compression now (e.g. before a reflect, after a "
            "large bulk store, or when the user explicitly asks).\n"
            f"{_SID} Optional; omit to consolidate across all "
            "namespaces.\n"
            "n_clusters: target cluster count; default "
            "floor(sqrt(unconsolidated)), capped automatically.\n"
            "min_cluster_size: discard smaller clusters, default 2."
        ),
    )
    @_instrument(metrics, "consolidate_episodes")
    @_translate_errors
    async def consolidate_episodes(
        ctx: Context,
        session_id: str | None = None,
        n_clusters: int | None = None,
        min_cluster_size: int | None = None,
    ) -> list[dict[str, Any]]:
        await _ensure_warmed(embedder, ctx)
        await ctx.info(
            f"consolidate_episodes: session_id={session_id!r} n_clusters={n_clusters} "
            f"min_cluster_size={min_cluster_size}"
        )
        formed = await service.consolidate(
            session_id=session_id,
            n_clusters=n_clusters,
            min_cluster_size=min_cluster_size,
        )
        await ctx.info(f"consolidation produced {len(formed)} schema(s)")
        return formed

    @mcp.tool(
        name="forget_episodes",
        description=(
            "Prune weak episodes from the episodic store. Use when the "
            "user asks to clean up old or low-value memories, or "
            "proactively before a large consolidation run. Always "
            "preview first: dry_run defaults to true (returns "
            "candidate_ids without deleting).\n"
            "strength_floor: 0..1, default 0.05. Episodes whose "
            "power-law strength falls below this become candidates. "
            "Raise to prune more aggressively.\n"
            "decay_tau_seconds: accepted but has no effect on decay "
            "rate (kept for backward compat). Safe to omit.\n"
            "dry_run: default true. Set false to actually delete."
        ),
    )
    @_instrument(metrics, "forget_episodes")
    @_translate_errors
    async def forget_episodes(
        ctx: Context,
        strength_floor: float = 0.05,
        decay_tau_seconds: float | None = None,
        dry_run: bool = True,
    ) -> dict[str, Any]:
        await ctx.debug(
            f"forget_episodes: strength_floor={strength_floor} "
            f"decay_tau_seconds={decay_tau_seconds} dry_run={dry_run}"
        )
        result = await service.forget(
            strength_floor=strength_floor,
            decay_tau_seconds=decay_tau_seconds,
            dry_run=dry_run,
        )
        verb = "would prune" if dry_run else "pruned"
        candidate_count = len(result.get("candidate_ids") or [])
        removed_count = result.get("removed") or 0
        n = candidate_count if dry_run else removed_count
        await ctx.info(f"{verb} {n} episode(s)")
        return result

    @mcp.tool(
        name="reflect_session",
        description=(
            "Use at session start, on a context switch, or to orient "
            "before a new task in a known namespace. Returns the most "
            "relevant semantic schemas and the most recently accessed "
            "episodes for that session_id.\n"
            f"{_SID} Required.\n"
            "query: seed for schema search; if omitted, falls back to "
            "the first tag of the most-recent episode in the "
            "namespace.\n"
            "k: max schemas AND max episodes (each capped "
            "independently), default 5."
        ),
    )
    @_instrument(metrics, "reflect_session")
    @_translate_errors
    async def reflect_session(
        session_id: str,
        ctx: Context,
        query: str | None = None,
        k: int = 5,
    ) -> dict[str, Any]:
        await _ensure_warmed(embedder, ctx)
        await ctx.debug(f"reflect_session: session_id={session_id!r} query={query!r} k={k}")
        result = await service.reflect(session_id=session_id, query=query, k=k)
        schemas = len(result.get("schemas") or [])
        episodes = len(result.get("recent_episodes") or [])
        await ctx.info(f"reflect returned {schemas} schema(s) + {episodes} episode(s)")
        return result

    @mcp.tool(
        name="store_semantic_memory",
        description=(
            "Use this (not store_episode) when you know something is "
            "a stable, durable truth: a user preference, a project "
            "convention, a procedure, an authored rule, a learned "
            "fact. Writes directly to the semantic store, bypassing "
            "the episodic->consolidate pipeline. No session scope.\n"
            "content: non-empty distilled text.\n"
            "kind: free-form label stored as metadata.kind, default "
            "'fact'. Common values: fact | procedure | preference | "
            "schema. Filterable via recall_semantic_memory.\n"
            "tags: optional list[str] for grouping.\n"
            "confidence: 0..1, default 1.0 (author-asserted)."
        ),
    )
    @_instrument(metrics, "store_semantic_memory")
    @_translate_errors
    async def store_semantic_memory(
        content: str,
        ctx: Context,
        kind: str = "fact",
        tags: list[str] | None = None,
        confidence: float = 1.0,
    ) -> dict[str, Any]:
        await _ensure_warmed(embedder, ctx)
        await ctx.debug(
            f"store_semantic_memory: kind={kind!r} confidence={confidence} tags={tags!r}"
        )
        result = await service.store_semantic_memory(
            content=content,
            kind=kind,
            tags=tags,
            confidence=confidence,
        )
        await ctx.info(f"stored semantic memory id={result['id']} kind={kind!r}")
        return result

    @mcp.tool(
        name="recall_semantic_memory",
        description=(
            "Use when you want facts, procedures, preferences, or "
            "conventions without the noise of raw episodic history. "
            "Searches the semantic store only (consolidated schemas + "
            "store_semantic_memory entries).\n"
            "query: text, cosine-matched.\n"
            "k: max results, default 8 (ascending distance).\n"
            "kind: optional filter on metadata.kind (e.g. "
            "'preference'). Omit to search all kinds."
        ),
    )
    @_instrument(metrics, "recall_semantic_memory")
    @_translate_errors
    async def recall_semantic_memory(
        query: str,
        ctx: Context,
        k: int = 8,
        kind: str | None = None,
    ) -> list[dict[str, Any]]:
        await _ensure_warmed(embedder, ctx)
        await ctx.debug(f"recall_semantic_memory: k={k} kind={kind!r}")
        results = await service.recall_semantic_memory(query=query, k=k, kind=kind)
        await ctx.info(f"semantic recall returned {len(results)} hit(s)")
        return results

    @mcp.tool(
        name="memory_stats",
        description=(
            "Use to check store size before a bulk operation, when "
            "diagnosing recall returning nothing, or to surface a "
            "health-check count. Returns {episodic_count, "
            "semantic_count, schema_candidate_count, "
            "consolidate_epoch, ...hygiene counters}. No params."
        ),
    )
    @_instrument(metrics, "memory_stats")
    @_translate_errors
    async def memory_stats(ctx: Context) -> dict[str, int]:
        result = await service.stats()
        await ctx.debug(f"stats: {result}")
        return result
