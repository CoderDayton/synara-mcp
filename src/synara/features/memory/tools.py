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

from typing import Any

from fastmcp import Context, FastMCP

from synara.features.embedding import Embedder

from .service import MemoryService

# Shared compact definition. session_id is a caller-defined string
# namespace, not an MCP/process session.
_SID = (
    "session_id: free-form namespace str (project|conversation|persona|"
    "ruleset|role|anything). Same str=same namespace. Scopes: dedup, "
    "episodic recall, consolidate, reflect. Semantic store is global."
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
) -> None:
    @mcp.tool(
        name="store_episode",
        description=(
            "Store one episode (raw trace) in the episodic store. Dedups "
            "near-dups within session_id (bumps retrieval_count instead "
            "of inserting); very short content skips dedup and always "
            "inserts.\n"
            "content: non-empty text to embed/store.\n"
            f"{_SID}\n"
            "tags: optional list[str] labels; used only by reflect "
            "seeding + schema tag union. Not a filter.\n"
            "salience: 0..1, default 0.5. Higher = slower power-law "
            "decay + bias to be schema headline."
        ),
    )
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
            "Pattern-completion search over raw episodes; episodic hits "
            "bump retrieval_count.\n"
            "query: text, cosine-matched.\n"
            f"{_SID} Used as a context hint, not a filter — recall is "
            "always cross-session. In-session episodes get a small "
            "ranking bonus (state-dependent retrieval); cross-session "
            "episodes are returned but ranked lower at equal cosine.\n"
            "k: max results, default 8 (sorted by ascending distance).\n"
            "mode: auto|episodic|semantic|hybrid. auto/hybrid = both "
            "stores merged. Use recall_semantic_memory for semantic-only."
        ),
    )
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
            "Cluster unconsolidated episodes -> semantic schemas. Marks "
            "sources consolidated_into=<schema_id>.\n"
            f"{_SID} Optional; omit to consolidate across all namespaces.\n"
            "n_clusters: int, default floor(sqrt(remaining after "
            "schema absorption)), capped.\n"
            "min_cluster_size: drop smaller clusters, default 2."
        ),
    )
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
            "Power-law pruning over the episodic store. "
            "strength = salience * sum_k (1 + age_k)^-d over retrieval "
            "times; consolidated pruned at floor, unconsolidated at "
            "floor/2.\n"
            "strength_floor: 0..1, default 0.05.\n"
            "decay_tau_seconds: kept for API compat only; does NOT tune "
            "decay (model uses a fixed exponent d). Must be positive if "
            "set.\n"
            "dry_run: default true (returns candidate_ids, no delete). "
            "false = delete."
        ),
    )
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
            "Per-namespace summary: related semantic schemas + "
            "most-recently-accessed episodes.\n"
            f"{_SID} Required.\n"
            "query: optional schema-search seed; if omitted, uses first "
            "tag of most-recent episode.\n"
            "k: max schemas AND episodes (independent), default 5."
        ),
    )
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
            "Save a distilled, durable abstraction directly to the "
            "semantic store — bypasses the episodic->consolidate "
            "pipeline. Use for facts, procedures, preferences, "
            "conventions, and authored schemas that should persist "
            "without raw-trace baggage. Semantic store is global "
            "(no session_id).\n"
            "content: non-empty distilled text.\n"
            "kind: free-form label (e.g. fact|procedure|preference|"
            "schema), default 'fact'. Stored as metadata.kind.\n"
            "tags: optional list[str] for retrieval grouping.\n"
            "confidence: 0..1, default 1.0 (author-asserted)."
        ),
    )
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
            "Retrieve durable abstractions from the semantic store only "
            "— skips raw episodic traces. Use to look up facts, "
            "procedures, preferences, project conventions, and "
            "consolidated schemas without dragging in full history.\n"
            "query: text, cosine-matched against semantic store.\n"
            "k: max results, default 8 (ascending distance).\n"
            "kind: optional free-form filter on metadata.kind "
            "(e.g. 'preference')."
        ),
    )
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
        description="Return {episodic_count, semantic_count}. No params.",
    )
    async def memory_stats(ctx: Context) -> dict[str, int]:
        result = await service.stats()
        await ctx.debug(f"stats: {result}")
        return result
