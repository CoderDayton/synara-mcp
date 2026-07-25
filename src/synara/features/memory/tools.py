"""MCP tool surface for the memory feature.

Tool descriptions are compressed for LLM consumption: terse parameter
specs, no prose. Validation lives in ``MemoryService``.

Surface
-------
Episodic store (raw event traces, scoped by ``session_id``):
    store_episode, recall_episodes, get_episode, consolidate_episodes,
    forget_episodes, reflect_session.

Semantic store (distilled facts/procedures/preferences/schemas,
session-scoped or global via the scope axis):
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
from .recall_report import RecallDiagnostics, RecallRequest, build_miss_report
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
    "recall_episodes": "ranked episodic recall",
    "get_episode": "fetch full episode by id",
    "remove_episode": "targeted episode delete",
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
    "MCP/process session. Scopes recall to that session plus global "
    "records by default and grants in-session hits a ranking bonus; "
    "also scopes dedup, consolidate, and reflect."
)

# Namespace used when a caller omits ``session_id`` on store. A
# session-less recall (or one with scope_session=false) still sees these
# records cross-session; the default just gives dedup/consolidate/
# reflect a stable bucket instead of forcing the caller to invent one.
_DEFAULT_SESSION_ID = "default"


async def _ensure_warmed(embedder: Embedder | None, ctx: Context) -> None:
    """Warmup embedder with progress reporting on first call."""
    if embedder is None:
        return
    await embedder.warmup_async(ctx)


def _apply_snippet(
    hits: list[dict[str, Any]], *, max_chars: int, full: bool
) -> list[dict[str, Any]]:
    """Bound the per-hit ``content`` so a recall can't blow the caller's
    tool-result token budget.

    Each hit gains ``content_chars`` (the full untruncated length) and a
    ``truncated`` flag. When truncated, ``content`` is cut to ``max_chars``
    and the full text remains available by re-calling with ``full=True``.
    ``full`` or ``max_chars <= 0`` returns content unchanged (but still
    annotated) so callers always see ``content_chars``/``truncated``.
    """
    out: list[dict[str, Any]] = []
    for hit in hits:
        content = hit.get("content") or ""
        n = len(content)
        truncate = not full and max_chars > 0 and n > max_chars
        annotated = {
            **hit,
            "content": content[:max_chars] if truncate else content,
            "truncated": truncate,
            "content_chars": n,
        }
        out.append(annotated)
    return out


def _project_content_only(hits: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Reduce each hit to ``{id, kind, content}`` for callers that opt into
    ``content_only`` and want a minimal payload — the identifier (for
    ``get_episode``/``supersedes``), the ``metadata.kind`` label, and the text,
    dropping distance/source/metadata/recency fields. ``kind`` is ``None`` when
    the hit carries none (e.g. raw episodic traces)."""
    return [
        {
            "id": hit.get("id"),
            "kind": (hit.get("metadata") or {}).get("kind"),
            "content": hit.get("content"),
        }
        for hit in hits
    ]


async def _episodic_fallback(
    service: MemoryService,
    *,
    query: str,
    k: int,
    session_id: str | None,
    scope_session: bool | None,
    content_only: bool,
) -> list[dict[str, Any]]:
    """Probe the episodic store after a semantic recall found nothing.

    The semantic store only fills once episodes consolidate, so early in
    a project a semantic miss routinely sits beside a perfectly relevant
    raw trace. Rather than report "nothing found" and let the caller
    conclude memory is empty, surface those traces as leads.

    Uses the non-reinforcing path deliberately: the caller asked for
    distilled facts, so these hits must not bump retrieval counts or
    rewrite the successor graph as though episodes had been requested.
    Snippet truncation always applies — a fallback rides along with a
    result the caller did not ask for and must not dominate its budget.
    """
    cfg = service.config
    if not cfg.recall_semantic_episodic_fallback or cfg.recall_semantic_fallback_k <= 0:
        return []
    if k <= 0:
        # The caller asked for no results. Volunteering fallback traces
        # would override that, not help it.
        return []
    hits = await service.recall_readonly(
        query=query,
        session_id=session_id,
        k=cfg.recall_semantic_fallback_k,
        mode="episodic",
        scope_session=scope_session,
    )
    if not hits:
        return []
    hits = _apply_snippet(hits, max_chars=cfg.recall_snippet_chars, full=False)
    return _project_content_only(hits) if content_only else hits


def register_tools(  # noqa: PLR0915 -- flat aggregator: one nested handler per MCP tool
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
            "DEFAULT memory write — reach for this first. Use for "
            "anything in this session worth remembering: a decision, "
            "event, user statement, action outcome, or observation. This "
            "INCLUDES things that sound like durable facts or "
            "preferences: a single occurrence is an episode, not a "
            "semantic memory. Episodes are session-scoped and safe, and "
            "the background consolidation pass distils repeated episodes "
            "into semantic memories for you — so you do NOT need "
            "store_semantic_memory to make something persist. When unsure "
            "which store to use, use this one.\n"
            "Embeds and stores the raw trace; near-duplicates within "
            "session_id merge (retrieval_count bumped) instead of "
            "inserting. Very short content (<8 chars stripped) always "
            "inserts.\n"
            "content: non-empty text to embed/store.\n"
            f"{_SID} Optional on store; omit to use the shared "
            f"'{_DEFAULT_SESSION_ID}' namespace.\n"
            "tags: optional list[str]; used by reflect seeding and "
            "schema headline selection. Not a recall filter.\n"
            "salience: 0..1, default 0.5. Higher = slower power-law "
            "decay + preferred as schema headline. Use >0.7 for "
            "critical traces.\n"
            "supersedes: optional id of an existing episode this one "
            "corrects/replaces. When set, that episode (and its segment "
            "group) is deleted after the new one is stored, so a "
            "correction retires the stale trace instead of layering. "
            "Errors if the id does not exist; the response echoes "
            "superseded=<id> (null when the new content merged into the "
            "superseded episode itself, which is kept). After a run of "
            "stores, reflect_session distils them into what to carry "
            "forward."
        ),
    )
    @_instrument(metrics, "store_episode")
    @_translate_errors
    async def store_episode(
        content: str,
        ctx: Context,
        session_id: str | None = None,
        tags: list[str] | None = None,
        salience: float = 0.5,
        supersedes: int | None = None,
    ) -> dict[str, Any]:
        await _ensure_warmed(embedder, ctx)
        sid = session_id or _DEFAULT_SESSION_ID
        await ctx.debug(
            f"store_episode: session_id={sid!r} salience={salience} supersedes={supersedes}"
        )
        result = await service.encode_episode(
            content=content,
            session_id=sid,
            tags=tags,
            salience=salience,
            supersedes=supersedes,
        )
        verb = "deduped onto existing" if result.get("deduped") else "encoded"
        retired = result.get("superseded")
        suffix = f"; retired id={retired}" if retired is not None else ""
        await ctx.info(f"{verb} episode id={result['id']}{suffix}")
        return result

    @mcp.tool(
        name="recall_episodes",
        description=(
            "Use to retrieve relevant past episodes before answering, "
            "continuing a task, or grounding a reply in prior context. "
            "Hits are ranked by cosine + successor-representation + "
            "spreading activation; each hit bumps retrieval_count.\n"
            "query: text, cosine-matched.\n"
            f"{_SID} Optional. When set, recall is scoped to that session "
            "by default (plus any global memories): only this session's "
            "episodes and schemas are returned, and in-session episodes "
            "still get a small ranking bonus. Set scope_session=false to "
            "opt back into cross-session recall.\n"
            "scope_session: optional bool. Default (unset) = scope when a "
            "session_id is given; true = force scoping (errors without a "
            "session_id); false = cross-session (the old default).\n"
            "tags: optional list[str]; when set, keep only episodic hits "
            "whose stored tags include every listed tag.\n"
            "k: max results, default 4 (ascending distance). Kept small "
            "so the result fits a tool-result token budget.\n"
            "max_chars: per-hit content is truncated to this many chars "
            "(default from config). Truncated hits carry truncated=true "
            "and content_chars=<full length>; re-call with full=true to "
            "get the untruncated text. 0 disables truncation.\n"
            "full: when true, return untruncated content — may be large, "
            "so pair with a small k or a narrow query.\n"
            "mode: 'auto'/'hybrid' = episodic + semantic merged "
            "(default 'auto'); 'episodic' = raw traces only; "
            "'semantic' = schemas only — prefer recall_semantic_memory "
            "for that case.\n"
            "content_only: opt-in bool, default false. When true, each hit "
            "is reduced to {id, kind, content} — the metadata, distance, "
            "source, and recency fields are dropped for a minimal payload "
            "(snippet truncation still applies to content).\n"
            "Returns a JSON array of hits. On zero hits it returns an "
            "OBJECT instead — {results: [], miss: {reason, scope, "
            "searched, suggestions}} — naming what emptied the result "
            "(empty store / session scope / tags / relevance gate) and "
            "the retry that would fix it. Read miss.reason before "
            "concluding nothing is stored.\n"
            "Every hit carries created_at/updated_at (unix seconds) and "
            "age_days/updated_age_days so you can tell old memories (and "
            "stale facts) from new. Episodic hits additionally carry "
            "group_id/segment_count: a non-null group_id means the hit is "
            "one segment of a larger episode — call get_episode(group_id) "
            "for the reassembled whole. Sibling segments of one episode "
            "collapse to a single best-ranked hit by default."
        ),
    )
    @_instrument(metrics, "recall_episodes")
    @_translate_errors
    async def recall_episodes(
        query: str,
        ctx: Context,
        session_id: str | None = None,
        k: int = 4,
        mode: str = "auto",
        scope_session: bool | None = None,
        tags: list[str] | None = None,
        max_chars: int | None = None,
        full: bool = False,
        content_only: bool = False,
    ) -> list[dict[str, Any]] | dict[str, Any]:
        await _ensure_warmed(embedder, ctx)
        await ctx.debug(
            f"recall_episodes: session_id={session_id!r} k={k} mode={mode!r} "
            f"scope_session={scope_session} tags={tags} max_chars={max_chars} "
            f"full={full} content_only={content_only}"
        )
        report_misses = service.config.recall_miss_report
        diagnostics = RecallDiagnostics() if report_misses else None
        results = await service.recall(
            query=query,
            session_id=session_id,
            k=k,
            mode=mode,
            scope_session=scope_session,
            tags=tags,
            diagnostics=diagnostics,
        )
        if not results and diagnostics is not None:
            await ctx.info("recall returned 0 hit(s); returning miss report")
            return build_miss_report(
                diagnostics,
                RecallRequest(
                    query=query,
                    k=k,
                    mode=mode,
                    session_id=session_id,
                    scope_session=scope_session,
                    tags=tags,
                ),
            )
        limit = service.config.recall_snippet_chars if max_chars is None else max_chars
        results = _apply_snippet(results, max_chars=limit, full=full)
        n_trunc = sum(1 for h in results if h.get("truncated"))
        await ctx.info(f"recall returned {len(results)} hit(s); {n_trunc} truncated")
        if content_only:
            results = _project_content_only(results)
        return results

    @mcp.tool(
        name="get_episode",
        description=(
            "Fetch one episode's full, untruncated content by id — the "
            "companion to recall_episodes, which returns bounded snippets. "
            "Use when a recalled hit came back with truncated=true and you "
            "need its complete text.\n"
            "episode_id: the id from a recall hit.\n"
            f"{_SID} Optional; restricts a theta-segmented walk to that "
            "namespace. Omit for a cross-session fetch.\n"
            "Returns id, full content, content_chars, metadata, and "
            "group_id/segment_ids when the episode was theta-segmented "
            "(content is the reassembled whole, not a single segment)."
        ),
    )
    @_instrument(metrics, "get_episode")
    @_translate_errors
    async def get_episode(
        episode_id: int,
        ctx: Context,
        session_id: str | None = None,
    ) -> dict[str, Any]:
        await ctx.debug(f"get_episode: id={episode_id} session_id={session_id!r}")
        result = await service.get_episode(episode_id, session_id=session_id)
        await ctx.info(f"fetched episode id={episode_id} ({result['content_chars']} chars)")
        return result

    @mcp.tool(
        name="remove_episode",
        description=(
            "Permanently delete one episode (and its whole segment group) "
            "by id — the targeted companion to forget_episodes' bulk decay "
            "pruning. Destructive and irreversible: confirm with the user "
            "before deleting a memory you did not just create. Always "
            "preview first: dry_run defaults to true.\n"
            "episode_id: required; the id from a recall hit or get_episode.\n"
            f"{_SID} Optional; restricts the segment-group walk to that "
            "namespace.\n"
            "dry_run: default true — returns candidate_ids (every id that "
            "would be deleted, including segment siblings) without "
            "deleting. Set false to actually delete; the response then "
            "carries deleted_ids."
        ),
    )
    @_instrument(metrics, "remove_episode")
    @_translate_errors
    async def remove_episode(
        episode_id: int,
        ctx: Context,
        session_id: str | None = None,
        dry_run: bool = True,
    ) -> dict[str, Any]:
        await ctx.debug(
            f"remove_episode: id={episode_id} session_id={session_id!r} dry_run={dry_run}"
        )
        result = await service.delete_episode(episode_id, session_id=session_id, dry_run=dry_run)
        verb = "would delete" if dry_run else "deleted"
        await ctx.info(f"{verb} {result['count']} episode record(s)")
        return result

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
            "dry_run: default true. Set false to actually delete.\n"
            "Each pass scans one bounded window of the store and rotates "
            "forward on the next non-dry-run pass; the response's "
            "scanned/scan_offset fields say what was covered, so repeat "
            "the call to sweep a large store."
        ),
    )
    @_instrument(metrics, "forget_episodes")
    @_translate_errors
    async def forget_episodes(
        ctx: Context,
        strength_floor: float = 0.05,
        dry_run: bool = True,
    ) -> dict[str, Any]:
        await ctx.debug(f"forget_episodes: strength_floor={strength_floor} dry_run={dry_run}")
        result = await service.forget(
            strength_floor=strength_floor,
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
            "Use at session start or END, on a context switch, or to "
            "orient before a new task in a known namespace — the "
            "end-of-session reflection pass. Returns the most relevant "
            "semantic schemas and the most recently accessed episodes "
            "for that session_id.\n"
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
            "Deliberate, infrequent write for a distilled STANDING truth "
            "meant to outlive this session — a user's explicit standing "
            "rule ('always ...', 'never ...'), a confirmed project "
            "convention, or a stable preference the user has stated as a "
            "rule. NOT for one-off events, decisions, or first-time "
            "observations: a single occurrence is an episode even when it "
            "sounds factual — use store_episode, which auto-consolidates "
            "repeated traces into semantic memories for you. Only promote "
            "to semantic once it is a confirmed, general rule. When "
            "unsure, prefer store_episode — a wrong semantic write "
            "pollutes every future recall. Writes directly to the "
            "semantic store, bypassing the episodic->consolidate "
            "pipeline.\n"
            "content: non-empty distilled text.\n"
            "scope: optional 'session' | 'global'. 'session' (the default "
            "when a session_id is given) ties the memory to session_id, "
            "returned only within that namespace; 'global' surfaces in "
            "EVERY session — reserve it for genuinely cross-project truths "
            "and pass it explicitly. Global is opt-in: there is no silent "
            "global default, so a call with neither a session_id nor "
            "scope='global' is REJECTED. Pass session_id unless you truly "
            "intend a global write.\n"
            "session_id: required when scope is 'session', and the safe "
            "default otherwise; the namespace this memory is tied to. "
            "Pass it unless you explicitly want scope='global'.\n"
            "kind: free-form label stored as metadata.kind, default "
            "'fact'. Common values: fact | procedure | preference | "
            "schema. Filterable via recall_semantic_memory.\n"
            "tags: optional list[str] for grouping.\n"
            "confidence: 0..1, default 1.0 (author-asserted).\n"
            "supersedes: optional id of an existing semantic memory this "
            "one corrects/replaces. When set, that entry is deleted after "
            "the new one is stored, so a correction retires the stale fact "
            "instead of layering (both surfacing in recall). Errors if the "
            "id does not exist; the response echoes superseded=<id>."
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
        supersedes: int | None = None,
        scope: str | None = None,
        session_id: str | None = None,
    ) -> dict[str, Any]:
        await _ensure_warmed(embedder, ctx)
        await ctx.debug(
            f"store_semantic_memory: kind={kind!r} confidence={confidence} "
            f"tags={tags!r} supersedes={supersedes} scope={scope!r} session_id={session_id!r}"
        )
        result = await service.store_semantic_memory(
            content=content,
            kind=kind,
            tags=tags,
            confidence=confidence,
            supersedes=supersedes,
            scope=scope,
            session_id=session_id,
        )
        retired = result.get("superseded")
        if retired is not None:
            await ctx.info(f"stored semantic memory id={result['id']}; retired id={retired}")
        else:
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
            "'preference'). Omit to search all kinds.\n"
            "session_id: optional. When set, scopes to that session's "
            "schemas plus global ones by default; scope_session=false opts "
            "back into cross-session results.\n"
            "scope_session: optional bool, same semantics as on "
            "recall_episodes (unset = scope when session_id given; true "
            "errors without a session_id).\n"
            "content_only: opt-in bool, default false. When true, each hit "
            "is reduced to {id, kind, content}, dropping distance/metadata.\n"
            "Each hit carries created_at/updated_at (unix seconds) and "
            "age_days/updated_age_days so stale facts are visible.\n"
            "Returns a JSON array of hits. On zero hits it returns an "
            "OBJECT instead — {results: [], miss: {reason, scope, "
            "searched, suggestions}} — naming what emptied the result. "
            "The semantic store only fills once episodes consolidate, so "
            "a miss also probes the episodic store and attaches any raw "
            "traces under episodic_fallback: those are UNDISTILLED "
            "traces, not confirmed facts — treat them as leads and call "
            "recall_episodes for the full set."
        ),
    )
    @_instrument(metrics, "recall_semantic_memory")
    @_translate_errors
    async def recall_semantic_memory(
        query: str,
        ctx: Context,
        k: int = 8,
        kind: str | None = None,
        session_id: str | None = None,
        scope_session: bool | None = None,
        content_only: bool = False,
    ) -> list[dict[str, Any]] | dict[str, Any]:
        await _ensure_warmed(embedder, ctx)
        await ctx.debug(
            f"recall_semantic_memory: k={k} kind={kind!r} "
            f"session_id={session_id!r} scope_session={scope_session} "
            f"content_only={content_only}"
        )
        diagnostics = RecallDiagnostics() if service.config.recall_miss_report else None
        results = await service.recall_semantic_memory(
            query=query,
            k=k,
            kind=kind,
            session_id=session_id,
            scope_session=scope_session,
            diagnostics=diagnostics,
        )
        if not results and diagnostics is not None:
            fallback = await _episodic_fallback(
                service,
                query=query,
                k=k,
                session_id=session_id,
                scope_session=scope_session,
                content_only=content_only,
            )
            await ctx.info(
                f"semantic recall returned 0 hit(s); miss report with "
                f"{len(fallback)} episodic fallback hit(s)"
            )
            return build_miss_report(
                diagnostics,
                RecallRequest(
                    query=query,
                    k=k,
                    session_id=session_id,
                    scope_session=scope_session,
                    kind=kind,
                ),
                semantic_only=True,
                episodic_fallback=fallback,
            )
        await ctx.info(f"semantic recall returned {len(results)} hit(s)")
        if content_only:
            results = _project_content_only(results)
        return results

    @mcp.tool(
        name="memory_stats",
        description=(
            "Use to check store size before a bulk operation, when "
            "diagnosing recall returning nothing, or to surface a "
            "health-check count. Returns {episodic_count, "
            "semantic_count, schema_candidate_count, "
            "consolidate_epoch, ...hygiene counters} plus live health "
            "fields: sr_edges (successor-graph mass), "
            "novel_encodes_since_consolidate and events_since_dream "
            "(reactor trigger pressure), last_consolidate_at/"
            "last_dream_at (unix seconds, 0 = never), and "
            "reactor_tasks_inflight (background consolidate/dream still "
            "running). No params."
        ),
    )
    @_instrument(metrics, "memory_stats")
    @_translate_errors
    async def memory_stats(ctx: Context) -> dict[str, int]:
        result = await service.stats()
        await ctx.debug(f"stats: {result}")
        return result
