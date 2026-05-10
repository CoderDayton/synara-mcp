"""MCP tool surface for the hippocampus feature.

Tool descriptions are compressed for LLM consumption: terse parameter
specs, no prose. Validation lives in ``HippocampusService``.
"""

from __future__ import annotations

from typing import Any

from fastmcp import Context, FastMCP

from synara.features.embedding import Embedder

from .service import HippocampusService

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
    service: HippocampusService,
    *,
    embedder: Embedder | None = None,
) -> None:
    @mcp.tool(
        name="memory_encode",
        description=(
            "Store episode in episodic store. Dedups near-dups within "
            "session_id (bumps retrieval_count instead of inserting).\n"
            "content: non-empty text to embed/store.\n"
            f"{_SID}\n"
            "tags: optional list[str] labels; used only by reflect "
            "seeding + schema tag union. Not a filter.\n"
            "salience: 0..1, default 0.5. Higher = slower Ebbinghaus "
            "decay + bias to be schema headline."
        ),
    )
    async def memory_encode(
        content: str,
        session_id: str,
        ctx: Context,
        tags: list[str] | None = None,
        salience: float = 0.5,
    ) -> dict[str, Any]:
        await _ensure_warmed(embedder, ctx)
        await ctx.debug(f"memory_encode: session_id={session_id!r} salience={salience}")
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
        name="memory_recall",
        description=(
            "Pattern-completion search; episodic hits bump retrieval_count.\n"
            "query: text, cosine-matched.\n"
            f"{_SID} Filters episodic leg only; semantic leg always global.\n"
            "k: max results, default 8 (sorted by ascending distance).\n"
            "mode: auto|episodic|semantic|hybrid. auto/hybrid = both "
            "stores merged."
        ),
    )
    async def memory_recall(
        query: str,
        ctx: Context,
        session_id: str | None = None,
        k: int = 8,
        mode: str = "auto",
    ) -> list[dict[str, Any]]:
        await _ensure_warmed(embedder, ctx)
        await ctx.debug(f"memory_recall: session_id={session_id!r} k={k} mode={mode!r}")
        results = await service.recall(query=query, session_id=session_id, k=k, mode=mode)
        await ctx.info(f"recall returned {len(results)} hit(s)")
        return results

    @mcp.tool(
        name="memory_consolidate",
        description=(
            "Cluster unconsolidated episodes -> semantic schemas. Marks "
            "sources consolidated_into=<schema_id>.\n"
            f"{_SID} Optional; omit to consolidate across all namespaces.\n"
            "n_clusters: int, default ceil(sqrt(num_candidates)).\n"
            "min_cluster_size: drop smaller clusters, default 2."
        ),
    )
    async def memory_consolidate(
        ctx: Context,
        session_id: str | None = None,
        n_clusters: int | None = None,
        min_cluster_size: int | None = None,
    ) -> list[dict[str, Any]]:
        await _ensure_warmed(embedder, ctx)
        await ctx.info(
            f"consolidate: session_id={session_id!r} n_clusters={n_clusters} "
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
        name="memory_forget",
        description=(
            "Ebbinghaus pruning. strength = salience*exp(-age/tau) + "
            "retrievals*boost. Consolidated pruned at floor; "
            "unconsolidated at floor/2.\n"
            "strength_floor: 0..1, default 0.05.\n"
            "decay_tau_seconds: positive float, default config "
            "(~1 week).\n"
            "dry_run: default true (returns candidate_ids, no delete). "
            "false = delete."
        ),
    )
    async def memory_forget(
        ctx: Context,
        strength_floor: float = 0.05,
        decay_tau_seconds: float | None = None,
        dry_run: bool = True,
    ) -> dict[str, Any]:
        await ctx.debug(
            f"memory_forget: strength_floor={strength_floor} "
            f"decay_tau_seconds={decay_tau_seconds} dry_run={dry_run}"
        )
        result = await service.forget(
            strength_floor=strength_floor,
            decay_tau_seconds=decay_tau_seconds,
            dry_run=dry_run,
        )
        verb = "would prune" if dry_run else "pruned"
        candidate_count = len(result.get("candidate_ids") or [])
        deleted_count = result.get("deleted") or 0
        n = candidate_count if dry_run else deleted_count
        await ctx.info(f"{verb} {n} episode(s)")
        return result

    @mcp.tool(
        name="memory_reflect",
        description=(
            "Per-namespace summary: related semantic schemas + "
            "most-recently-accessed episodes.\n"
            f"{_SID} Required.\n"
            "query: optional schema-search seed; if omitted, uses first "
            "tag of most-recent episode.\n"
            "k: max schemas AND episodes (independent), default 5."
        ),
    )
    async def memory_reflect(
        session_id: str,
        ctx: Context,
        query: str | None = None,
        k: int = 5,
    ) -> dict[str, Any]:
        await _ensure_warmed(embedder, ctx)
        await ctx.debug(f"memory_reflect: session_id={session_id!r} query={query!r} k={k}")
        result = await service.reflect(session_id=session_id, query=query, k=k)
        schemas = len(result.get("schemas") or [])
        episodes = len(result.get("episodes") or [])
        await ctx.info(f"reflect returned {schemas} schema(s) + {episodes} episode(s)")
        return result

    @mcp.tool(
        name="memory_stats",
        description="Return {episodic_count, semantic_count}. No params.",
    )
    async def memory_stats(ctx: Context) -> dict[str, int]:
        result = await service.stats()
        await ctx.debug(f"stats: {result}")
        return result
