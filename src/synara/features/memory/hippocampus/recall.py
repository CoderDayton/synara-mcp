"""Hybrid episodic/semantic recall with CA3 pattern completion and
successor-representation re-ranking.

The pipeline is:
  1. shape the query for simplevecdb (text vs. precomputed vector)
  2. optionally refine the query by CA3 iterative pattern completion
  3. fetch top-k from the chosen leg(s) and merge
  4. re-rank episodic hits with the successor-representation boost
  5. bump retrieval counts and fold one anchor-style edge into the SR
"""

from __future__ import annotations

import asyncio
import logging
import math
from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING, Any

import numpy as np

from synara.core.errors import ValidationError

from ..config import validate_tags
from ..memory_types import in_session_scope
from ..service import now_seconds
from ..timestamps import created_at as _created_at
from ..timestamps import last_accessed as _last_accessed_ts
from ..tracing import current_context as _trace_current
from ..tracing import record_span as _trace_span
from . import complete as _complete_mod

if TYPE_CHECKING:  # pragma: no cover - typing only
    from ..config import MemoryConfig
    from ..port import MemoryServicePort as MemoryService


# Type alias for a merged hit row.
_Hit = tuple[int, str, dict[str, Any], float, str]

_VALID_MODES = frozenset({"auto", "episodic", "semantic", "hybrid"})

_LOG = logging.getLogger(__name__)

# Cosine distance ceiling (1 - cos for opposed unit vectors is 2.0). Used
# as the published fallback for a non-finite distance so a hit never
# surfaces with an unrankable ``null`` score.
_MAX_COSINE_DISTANCE = 2.0


def _validate_recall_inputs(
    service: MemoryService,
    *,
    query: str,
    session_id: str | None,
    k: int,
    mode: str,
    scope_session: bool | None = None,
    tags: list[str] | None = None,
) -> None:
    if not query.strip():
        raise ValidationError("query must be non-empty")
    cfg = service.config
    if cfg.max_content_chars and len(query) > cfg.max_content_chars:
        raise ValidationError(f"query exceeds max_content_chars ({cfg.max_content_chars})")
    if cfg.max_recall_k and k > cfg.max_recall_k:
        raise ValidationError(f"k exceeds max_recall_k ({cfg.max_recall_k})")
    if (
        session_id is not None
        and cfg.max_session_id_chars
        and len(session_id) > cfg.max_session_id_chars
    ):
        raise ValidationError(
            f"session_id exceeds max_session_id_chars ({cfg.max_session_id_chars})"
        )
    if mode not in _VALID_MODES:
        raise ValidationError(f"unknown recall mode: {mode}")
    # An explicit scope_session=true with nothing to scope to would be a
    # silent no-op (recall stays cross-session) — the exact opposite of
    # what the caller asked for. Reject it rather than warn-and-proceed.
    if scope_session is True and session_id is None:
        raise ValidationError(
            "scope_session=true requires a session_id to scope to; "
            "pass one or leave scope_session unset"
        )
    # Tags are untrusted input on recall exactly as they are on encode;
    # same caps, same single home (config.validate_tags).
    validate_tags(cfg, tags)


def elbow_cutoff(
    distances: Sequence[float],
    *,
    sensitivity: float = 0.12,
    min_candidates: int = 3,
    min_spread: float = 0.05,
) -> float | None:
    """Adaptive relevance gate: the largest cosine distance worth keeping,
    or ``None`` to keep everything.

    Given one source's recall-candidate distances, locate the knee of the
    sorted-ascending curve — the transition from the steep "relevant" ramp
    to the flat "noise" plateau — and return the distance of the last point
    *before* the knee. Hits with ``distance <= cutoff`` are relevant; the
    plateau beyond it is dropped. ``None`` means no defensible knee.

    Method: normalised Kneedle (Satopaa et al., "Finding a 'Kneedle' in a
    Haystack", 2011). Sorted distances are rescaled so rank and distance
    each span ``[0, 1]``; on that unit square the difference ``y - x`` is
    maximised exactly where the local slope crosses 1 — i.e. where the ramp
    gives way to the plateau. That peak's height is the knee's prominence,
    and a knee is accepted only if it clears ``sensitivity``, so a roughly
    linear curve (no real elbow) is left uncut.

    Isotonic regression is deliberately *not* used: the input is sorted
    ascending, hence already monotone, so a PAVA fit returns it unchanged —
    a no-op. The knee lives in the curvature, which the normalised
    difference curve captures directly. (Isotonic smoothing would earn its
    keep only if the gate ran on the *non-monotone* SR-adjusted scores
    rather than on raw cosine distance.)
    """
    finite = [d for d in distances if math.isfinite(d)]
    n = len(finite)
    if n < min_candidates:
        return None
    ds = sorted(finite)
    lo, hi = ds[0], ds[-1]
    spread = hi - lo
    if spread < min_spread:
        return None
    inv = 1.0 / (n - 1)
    best_i = 0
    best_delta = 0.0
    for i in range(n):
        delta = (ds[i] - lo) / spread - i * inv
        if delta > best_delta:
            best_delta = delta
            best_i = i
    if best_delta < sensitivity or best_i < 1:
        return None
    # The knee sits at the first plateau point (the foot of the cliff), so
    # the last distance we keep is its predecessor.
    return ds[best_i - 1]


def _quantile(sorted_vals: Sequence[float], q: float) -> float:
    """Linear-interpolated quantile of an ascending-sorted, non-empty seq."""
    last = len(sorted_vals) - 1
    if last <= 0:
        return sorted_vals[0]
    pos = q * last
    lo = math.floor(pos)
    hi = min(lo + 1, last)
    frac = pos - lo
    return sorted_vals[lo] * (1.0 - frac) + sorted_vals[hi] * frac


def dynamic_ceiling(
    distances: Sequence[float],
    *,
    alpha: float = 0.8,
    quantile: float = 0.9,
    min_candidates: int = 4,
    standout_gap: float = 0.0,
) -> float | None:
    """Adaptive *absolute* relevance ceiling, calibrated to the embedding
    model from the candidate distance distribution — no hardcoded cutoff.

    The far end of a recall's over-fetched candidates approximates this
    model's "unrelated" distance. We take the ``quantile`` (default p90) of
    the per-source distances as that reference ``d_ref`` and keep only hits
    within ``alpha`` of it (``distance <= alpha * d_ref``). Because d_ref
    scales with the embedder, one ``alpha`` yields ~0.7 on a normalised
    sentence-transformer and ~0.8 on a hash stub, with no per-model retuning
    and no model-specific footgun.

    There is no floor by default: when even the nearest hit exceeds
    ``alpha * d_ref`` (``d_min > alpha * d_ref``), the candidate cloud is
    packed at the far end with no near-field structure — the off-topic
    signature — and the query returns empty rather than its least-bad hit.
    The one exception is a *standout*: when ``standout_gap > 0`` and the
    nearest hit is separated from the next by at least that gap, it is a real
    match in an otherwise-far cloud (a relevance cliff right after it), not the
    least-bad of a uniform cloud, so it is floored back in and kept even above
    the ceiling. Returns ``None`` (no gate) below ``min_candidates``, where the
    sample is too small to estimate a reference — which also spares small
    synthetic corpora from emptying.
    """
    finite = sorted(d for d in distances if math.isfinite(d))
    if len(finite) < min_candidates:
        return None
    ceiling = alpha * _quantile(finite, quantile)
    if standout_gap > 0.0 and len(finite) > 1 and finite[1] - finite[0] >= standout_gap:
        return max(ceiling, finite[0])
    return ceiling


def gate_relevance(hits: list[dict[str, Any]], cfg: MemoryConfig) -> list[dict[str, Any]]:
    """Apply the dynamic relevance ceiling, the adaptive elbow gate, and the
    gap cut to a single-source list of recall-hit dicts (each carrying a
    ``distance``). The ceiling drops hits far on the model's own scale; the
    elbow trims the noise plateau; the gap cut ends the run at the first
    large jump. No-op when every knob is off or none finds a defensible cut.
    Used by the semantic-only recall path; the hybrid path uses
    :func:`_gate_merged`.
    """
    if not cfg.recall_relevance_gate:
        return hits
    if cfg.recall_max_distance_alpha > 0:
        cut = dynamic_ceiling(
            [h["distance"] for h in hits],
            alpha=cfg.recall_max_distance_alpha,
            quantile=cfg.recall_max_distance_quantile,
            min_candidates=cfg.recall_max_distance_min_candidates,
            standout_gap=cfg.recall_max_distance_standout_gap,
        )
        if cut is not None:
            hits = [h for h in hits if h["distance"] <= cut + 1e-9]
    if cfg.recall_elbow_cutoff:
        cutoff = elbow_cutoff(
            [h["distance"] for h in hits],
            sensitivity=cfg.recall_elbow_sensitivity,
            min_candidates=cfg.recall_elbow_min_candidates,
            min_spread=cfg.recall_elbow_min_spread,
        )
        if cutoff is not None:
            hits = [h for h in hits if h["distance"] <= cutoff + 1e-9]
    if cfg.recall_gap_cut > 0:
        gap = gap_cutoff([h["distance"] for h in hits], min_gap=cfg.recall_gap_cut)
        if gap is not None:
            hits = [h for h in hits if h["distance"] <= gap + 1e-9]
    return hits


def _apply_per_source(
    merged: list[_Hit], cutoff_fn: Callable[[list[float]], float | None]
) -> list[_Hit]:
    """Filter merged rows by a per-source distance cutoff (``cutoff_fn`` maps
    one source's distances to the largest distance to keep, or ``None`` for
    no cut). Episodic and semantic legs are gated independently — their
    cosine distances live on different scales — and filtering never
    re-ranks, so SR order is preserved.
    """
    by_source: dict[str, list[float]] = {}
    for row in merged:
        if math.isfinite(row[3]):
            by_source.setdefault(row[4], []).append(row[3])
    cutoffs: dict[str, float] = {}
    for src, dists in by_source.items():
        c = cutoff_fn(dists)
        if c is not None:
            cutoffs[src] = c
    if not cutoffs:
        return merged
    return [
        row
        for row in merged
        if row[4] not in cutoffs or (math.isfinite(row[3]) and row[3] <= cutoffs[row[4]] + 1e-9)
    ]


def gap_cutoff(
    distances: Sequence[float], *, min_gap: float, min_candidates: int = 2
) -> float | None:
    """Largest distance to keep before the first relevance cliff: walking the
    ascending distances, the first consecutive jump of at least ``min_gap``
    ends the relevant run, and everything beyond it is dropped. A *gap* is
    scale-relative — a compressed embedder never produces one, so this fails
    safe (no cut) rather than over-trimming. ``None`` when no such jump.
    """
    finite = sorted(d for d in distances if math.isfinite(d))
    if len(finite) < min_candidates:
        return None
    for i in range(len(finite) - 1):
        if finite[i + 1] - finite[i] >= min_gap:
            return finite[i]
    return None


def _apply_gap_cut(merged: list[_Hit], min_gap: float) -> list[_Hit]:
    """Cross-source relevance-cliff cut: a single large jump in the pooled
    distances marks where relevance falls off regardless of source. The
    per-source ceiling/elbow can't see a head-vs-tail cliff that straddles
    the episodic and semantic legs; this can.
    """
    cut = gap_cutoff([row[3] for row in merged if math.isfinite(row[3])], min_gap=min_gap)
    if cut is None:
        return merged
    return [row for row in merged if not math.isfinite(row[3]) or row[3] <= cut + 1e-9]


def _gate_merged(merged: list[_Hit], cfg: MemoryConfig) -> list[_Hit]:
    """Apply the dynamic relevance ceiling, the per-source elbow gate, and a
    final cross-source gap cut to the merged hybrid rows — each a no-op when
    its knob is off. The ceiling drops far hits on the model's own scale, the
    elbow trims each source's noise plateau, and the gap cut ends the pooled
    ranking at the first large head-vs-tail cliff.
    """
    if not cfg.recall_relevance_gate:
        return merged
    if cfg.recall_max_distance_alpha > 0:
        merged = _apply_per_source(
            merged,
            lambda dists: dynamic_ceiling(
                dists,
                alpha=cfg.recall_max_distance_alpha,
                quantile=cfg.recall_max_distance_quantile,
                min_candidates=cfg.recall_max_distance_min_candidates,
                standout_gap=cfg.recall_max_distance_standout_gap,
            ),
        )
    if cfg.recall_elbow_cutoff:
        merged = _apply_per_source(
            merged,
            lambda dists: elbow_cutoff(
                dists,
                sensitivity=cfg.recall_elbow_sensitivity,
                min_candidates=cfg.recall_elbow_min_candidates,
                min_spread=cfg.recall_elbow_min_spread,
            ),
        )
    if cfg.recall_gap_cut > 0:
        merged = _apply_gap_cut(merged, cfg.recall_gap_cut)
    return merged


async def run(
    service: MemoryService,
    *,
    query: str,
    session_id: str | None = None,
    k: int = 8,
    mode: str = "auto",
    scope_session: bool | None = None,
    tags: list[str] | None = None,
    reinforce: bool,
) -> list[dict[str, Any]]:
    # Validate before the ``k <= 0`` short-circuit: an empty/oversized
    # query or unknown mode is always a programmer error, regardless of
    # how many results were requested. Returning ``[]`` for ``k <= 0``
    # is a convenience for callers who compute ``k`` from a budget that
    # may legitimately go to zero.
    _validate_recall_inputs(
        service,
        query=query,
        session_id=session_id,
        k=k,
        mode=mode,
        scope_session=scope_session,
        tags=tags,
    )
    if k <= 0:
        return []

    # The SR window key is still ``session_id`` regardless of scoping, so
    # cross-session bridges keep forming in the caller's context.
    # Scope axis (default-on). A supplied ``session_id`` scopes recall to
    # that session plus any global records; ``scope_session=False`` opts
    # back into cross-session results, ``scope_session=True`` forces
    # scoping, and ``None`` means auto (scope when a session is given).
    # The scope check applies to both episodic and semantic hits via
    # ``in_session_scope`` (global records always pass; legacy records
    # without a ``scope`` field fall back to session_id presence). ``tags``
    # is an additional episodic-only constraint. Both are Python
    # post-filters over an over-fetched candidate set (not a simplevecdb
    # metadata filter), keeping cosine order the final arbiter.
    tagset = frozenset(tags) if tags else None
    scope_active = scope_session is not False and session_id is not None
    filtering = scope_active or bool(tagset)
    collapse = service.config.recall_collapse_groups
    # Over-fetch when filtering, collapsing groups, *or* gating so that,
    # after dropping out-of-scope/same-group fragments, ``k`` hits remain
    # and the relevance gate estimates its knee over a real candidate
    # distribution rather than just ``k`` (mirrors the semantic path).
    cfg = service.config
    gating = cfg.recall_relevance_gate and (
        cfg.recall_elbow_cutoff or cfg.recall_max_distance_alpha > 0 or cfg.recall_gap_cut > 0
    )
    fetch_k = max(k * 4, 32) if (filtering or collapse or gating) else k
    with _trace_span("query_arg"):
        q = await service.query_arg(query)
    # CA3 iterative pattern completion: refine the query by
    # softmax-recombining its near-neighbours before the final search.
    # Skipped when no embed_fn is configured (q is a string) or when
    # iters == 0.
    iters = service.config.recall_completion_iters
    if iters > 0 and isinstance(q, list):
        with _trace_span("ca3_completion", payload={"iters": iters}):
            result = await _complete_mod.run(
                service,
                q,
                k_inner=max(k, 8),
                iters=iters,
                beta=service.config.recall_completion_beta,
                eta0=service.config.recall_completion_anchor,
            )
        # CA3 can collapse to an all-zero query (anchor/pattern
        # cancellation). A zero vector is a degenerate search input
        # (undefined cosine), so fall back to the pre-completion query.
        completed = result.query
        if isinstance(completed, list) and any(v != 0.0 for v in completed):
            q = completed

    with _trace_span("merge_hits"):
        merged = await _merge_hits(service, q, mode=mode, k=fetch_k)
    if filtering:
        ep_before = sum(1 for row in merged if row[4] == "episodic")
        merged = [
            row
            for row in merged
            if (not scope_active or in_session_scope(row[2], session_id=session_id))
            and (row[4] != "episodic" or _passes_tags(row[2], tagset))
        ]
        _log_scope_cap(
            fetch_k=fetch_k,
            ep_before=ep_before,
            ep_after=sum(1 for row in merged if row[4] == "episodic"),
            k=k,
        )
    with _trace_span("sr_rank"):
        rank_keys = await _sr_rank_keys(service, merged, caller_sid=session_id)
    # Key by (doc_id, source) instead of object identity: stable across
    # any future merge-list copying or wrapping.
    merged.sort(key=lambda r: rank_keys.get((r[0], r[4]), r[3]))

    if collapse:
        merged = _collapse_groups(merged)

    # Adaptive relevance gate: drop the low-relevance plateau tail so a
    # sharply-peaked query returns only its real hits instead of padding
    # to ``k`` with noise (per-source, post-rank — see ``elbow_cutoff``).
    with _trace_span("relevance_gate"):
        merged = _gate_merged(merged, service.config)

    t_now = now_seconds()
    # Cold-schema GC liveness: hybrid recall is the primary consumer of
    # schemas, so returned semantic hits must refresh ``last_accessed`` or
    # a schema recalled only through recall_episodes would eventually be
    # evicted as cold. Only when the GC knob is on and the recall
    # reinforces — a read-only recall stays write-free.
    bump_schemas = reinforce and service.config.forget_schema_unused_seconds > 0
    out, bumps, sem_bumps, observed_episodic = _assemble_output(
        merged, k=k, reinforce=reinforce, bump_schemas=bump_schemas, now=t_now
    )
    if bumps:
        # Hits within one recall are distinct doc ids, so their
        # read-append-write sections are independent (per-doc locks never
        # contend) — run them concurrently instead of paying k sequential
        # round-trips on the hot path.
        await asyncio.gather(*(service.bump_retrieval(d, m) for d, m in bumps))
    if sem_bumps:
        # Plain last-write-wins metadata patch (mirrors ``run_semantic``);
        # no per-doc lock needed for a single timestamp field.
        await service.semantic.update_metadata(sem_bumps)

    await _reinforce_recall_set(
        service,
        out=out,
        observed_episodic=observed_episodic,
        session_id=session_id,
        q=q,
        now=t_now,
    )
    return out


def _assemble_output(
    merged: list[_Hit],
    *,
    k: int,
    reinforce: bool,
    bump_schemas: bool,
    now: float,
) -> tuple[
    list[dict[str, Any]],
    list[tuple[int, dict[str, Any]]],
    list[tuple[int, dict[str, Any]]],
    list[tuple[str, int]],
]:
    """Shape the top-``k`` merged rows into hit dicts, collecting the
    write-backs recall owes: episodic retrieval bumps, semantic
    ``last_accessed`` bumps (cold-schema GC liveness), and the
    (session, id) pairs the SR observes.

    Returns ``(out, bumps, sem_bumps, observed_episodic)``.
    """
    out: list[dict[str, Any]] = []
    bumps: list[tuple[int, dict[str, Any]]] = []
    sem_bumps: list[tuple[int, dict[str, Any]]] = []
    observed_episodic: list[tuple[str, int]] = []
    for doc_id, text, md, dist, source in merged[:k]:
        hit: dict[str, Any] = {
            "id": doc_id,
            "content": text,
            # A non-finite cosine distance (NaN from a degenerate /
            # zero-norm stored vector) would serialise to JSON ``null``
            # and leave the hit unrankable against scored ones. Publish
            # the cosine ceiling instead so it stays comparable (and
            # sorts last, which the SR rank already enforces).
            "distance": dist if math.isfinite(dist) else _MAX_COSINE_DISTANCE,
            "source": source,
            "metadata": md,
        }
        hit.update(_recency_fields(md or {}, source=source, now=now))
        out.append(hit)
        # ``reinforce=False`` (ambient resource reads) makes recall a
        # pure read: no retrieval_count bump and — since the SR block
        # in ``run`` guards on ``observed_episodic`` — no SR/plasticity
        # update.
        if reinforce and source == "episodic" and doc_id >= 0:
            bumps.append((doc_id, md))
            sid = str(md.get("session_id", "")) if md else ""
            if sid:
                observed_episodic.append((sid, doc_id))
        elif bump_schemas and source == "semantic" and doc_id >= 0:
            sem_bumps.append((doc_id, {"last_accessed": now}))
    return out, bumps, sem_bumps, observed_episodic


async def run_semantic(
    service: MemoryService,
    *,
    query: str,
    k: int = 8,
    kind: str | None = None,
    session_id: str | None = None,
    scope_session: bool | None = None,
) -> list[dict[str, Any]]:
    """Semantic-only recall: cosine over the semantic store with the same
    validation, scope filter, and relevance gate as the hybrid path.

    Lives here (not on the service) so the semantic and hybrid recall
    paths share one home — ``_validate_recall_inputs``, the over-fetch
    policy, ``in_session_scope``, and ``gate_relevance`` — and cannot
    drift apart. Bumps ``last_accessed`` on returned schemas when the
    cold-schema GC is enabled (``forget_schema_unused_seconds > 0``) so
    an actively-recalled schema is never evicted as cold; recall stays
    write-free otherwise.
    """
    _validate_recall_inputs(
        service,
        query=query,
        session_id=session_id,
        k=k,
        mode="semantic",
        scope_session=scope_session,
    )
    if k <= 0:
        return []
    if await service.semantic.count() == 0:
        return []
    await service._ensure_index_ready()
    q = await service.query_arg(query)
    cfg = service.config
    # Scope filter: a supplied session_id scopes to that session (plus
    # global schemas) by default; scope_session=False opts out.
    scope_active = scope_session is not False and session_id is not None
    # When a kind or scope filter is requested, over-fetch and filter in
    # Python so cosine ranking still drives the final order; the semantic
    # store is small (gist-level) so the over-fetch is cheap. Also
    # over-fetch when the relevance gate is on, so the knee is estimated
    # over a real candidate distribution, not just k.
    gating = cfg.recall_relevance_gate and (
        cfg.recall_elbow_cutoff or cfg.recall_max_distance_alpha > 0 or cfg.recall_gap_cut > 0
    )
    wide = kind is not None or scope_active or gating
    fetch_k = max(k * 4, 32) if wide else k
    hits = await service.semantic.similarity_search(q, k=fetch_k)
    t_now = now_seconds()
    out: list[dict[str, Any]] = []
    for doc, dist in hits:
        md = dict(doc.metadata)
        if (kind is not None and str(md.get("kind", "")) != kind) or (
            scope_active and not in_session_scope(md, session_id=session_id)
        ):
            continue
        out.append(
            {
                "id": int(md.get("id", -1)),
                "content": doc.page_content,
                "distance": float(dist),
                "metadata": md,
                # Same recency surface as the hybrid path: stale facts
                # should be as visible as stale traces.
                **_recency_fields(md, source="semantic", now=t_now),
            }
        )
    # Adaptive relevance gate over the semantic candidates, then cap to
    # k. Drops the low-relevance plateau so a weak query returns few (or
    # no) schemas instead of k tenuous ones (see ``gate_relevance``).
    out = gate_relevance(out, cfg)[:k]
    # Bump ``last_accessed`` on the schemas we actually returned so the
    # cold-schema eviction path (forget.run with
    # forget_schema_unused_seconds > 0) treats a recently-recalled
    # schema as alive.
    if out and cfg.forget_schema_unused_seconds > 0:
        hit_bumps = [(int(r["id"]), {"last_accessed": t_now}) for r in out if int(r["id"]) >= 0]
        if hit_bumps:
            await service.semantic.update_metadata(hit_bumps)
    return out


def _log_scope_cap(*, fetch_k: int, ep_before: int, ep_after: int, k: int) -> None:
    """Make a scoped-recall under-return visible in logs.

    Scoping post-filters a cosine-bounded candidate set (``fetch_k``), not
    the DB query, so a matching trace ranked beyond ``fetch_k`` is silently
    absent. A saturated raw fetch (``ep_before >= fetch_k``) leaving fewer
    survivors than ``k`` is the tell that matches may lie beyond the cap;
    log it at ``info`` so the under-return reads as "capped", not "nothing
    else matched". Otherwise emit a ``debug`` breadcrumb of the counts.
    """
    if ep_before >= fetch_k and ep_after < k:
        _LOG.info(
            "scoped recall capped: fetch_k=%d saturated, kept %d/%d episodic "
            "after filter; matches beyond the cap are not returned",
            fetch_k,
            ep_after,
            ep_before,
        )
    else:
        _LOG.debug(
            "scoped recall: fetch_k=%d episodic_candidates=%d kept=%d k=%d",
            fetch_k,
            ep_before,
            ep_after,
            k,
        )


def _passes_tags(md: dict[str, Any] | None, tags: frozenset[str] | None) -> bool:
    """True if an episodic hit carries every requested tag.

    Session/global scoping is handled separately by ``in_session_scope``;
    this is the additional episodic-only tag constraint, keeping only hits
    whose stored tag list is a superset of every requested tag.
    """
    if not tags:
        return True
    hit_tags = {str(t) for t in ((md or {}).get("tags") or [])}
    return tags.issubset(hit_tags)


def _cosine_score_from_distance(dist: float | None) -> float:
    """Map cosine distance (in [0, 2]) to a similarity-like score in [0, 1]."""
    if dist is None:
        return 0.5
    s = 1.0 - 0.5 * float(dist)
    return max(0.0, min(1.0, s))


def _age_days(ts: float | None, *, now: float) -> float | None:
    return None if ts is None else max(0.0, (now - ts) / 86_400.0)


def _recency_fields(md: dict[str, Any], *, source: str, now: float) -> dict[str, Any]:
    """Promote a hit's recency (and, for episodic hits, group lineage) to
    top-level fields so a caller can tell old memories from new — and
    reassemble a segmented episode (via ``group_id`` -> ``get_episode``)
    — without digging through the raw metadata blob. Semantic gists get
    the recency stamps too (staleness matters as much for a fact as for
    a trace) but carry no group lineage."""
    created_f = _created_at(md)
    if source != "episodic":
        upd = md.get("updated_at")
        stamps = [
            float(v)
            for v in (upd if isinstance(upd, (int, float)) else None, _last_accessed_ts(md))
            if v is not None
        ]
        updated = max(stamps) if stamps else created_f
        return {
            "created_at": created_f,
            "updated_at": updated,
            "age_days": _age_days(created_f, now=now),
            "updated_age_days": _age_days(updated, now=now),
        }
    gid = md.get("episode_group_id")
    seg_count = int(md.get("segment_count", 1))
    stamps = [
        float(v)
        for v in (md.get("last_accessed"), md.get("last_reconsolidated_at"))
        if isinstance(v, (int, float))
    ]
    updated = max(stamps) if stamps else created_f
    return {
        # ``episode_group_id`` is set even for a standalone episode (to its
        # own id); surface it only when actually segmented, so a non-null
        # ``group_id`` means "reassemble the whole via get_episode".
        "group_id": int(gid) if (gid is not None and seg_count > 1) else None,
        "segment_count": seg_count,
        "created_at": created_f,
        "updated_at": updated,
        "age_days": _age_days(created_f, now=now),
        "updated_age_days": _age_days(updated, now=now),
    }


def _collapse_groups(merged: list[_Hit]) -> list[_Hit]:
    """Fold the sibling segments of one theta-segmented episode into its
    single best-ranked member (episodic only — semantic gists have no
    group). Fragments of one memory then don't consume several ``k``
    slots; the surfaced ``group_id`` / ``segment_count`` point the caller
    at ``get_episode`` for the reassembled whole."""
    seen_groups: set[int] = set()
    kept: list[_Hit] = []
    for row in merged:
        if row[4] == "episodic":
            gid = int((row[2] or {}).get("episode_group_id", row[0]))
            if gid in seen_groups:
                continue
            seen_groups.add(gid)
        kept.append(row)
    return kept


async def _reinforce_recall_set(
    service: MemoryService,
    *,
    out: list[dict[str, Any]],
    observed_episodic: list[tuple[str, int]],
    session_id: str | None,
    q: str | list[float],
    now: float,
) -> None:
    """Anchor-model SR + plasticity update for one recall.

    Fold one edge from the best-cosine anchor (= first episodic hit after
    re-ranking) to each other episodic hit. Avoids the pairwise
    n*(n-1)/2 inflation a naive "everything observed at the same t" loop
    would produce — five hits add four edges, not ten.

    The window is keyed on the *caller's* session, not the recalled
    episodes' original sessions, so a recall in session B that pulls
    episodes from session A enters B's window and a later recall in B
    chains across via the still-in-window anchor — bridging A and B into
    one connected graph. Falls back to the anchor's original session only
    when the caller didn't supply one.
    """
    if service._sr is None or not observed_episodic:
        return
    anchor_id = observed_episodic[0][1]
    window_sid = session_id or observed_episodic[0][0]
    others = [j for _, j in observed_episodic[1:]]
    await service._sr.observe_recall_set(window_sid, anchor_id, others, now)
    await service._sr.flush()
    # Plasticity layer: reinforce anchor->each-other edge with the
    # cosine-similarity score (1 - distance, clipped). Co-recall is the
    # brain's bread-and-butter Hebbian event.
    out_lookup = {row["id"]: row for row in out}
    anchor_row = out_lookup.get(anchor_id)
    if others:
        # Each (anchor, j) edge is distinct, so the per-edge locks never
        # contend; gather pipelines the read+upsert round-trips instead
        # of serialising them per co-recalled hit.
        await asyncio.gather(
            *(
                service._plasticity.reinforce(
                    anchor_id,
                    j,
                    score=(
                        _cosine_score_from_distance(jrow["distance"])
                        if (jrow := out_lookup.get(j)) is not None
                        else 0.5
                    ),
                    now=now,
                )
                for j in others
            )
        )
    # Reconsolidation (Nader 2000): when alpha is set, bump drift_total
    # and pull the stored vector toward the cue (buffered; flushed to
    # HNSW by the next consolidate pass).
    if service.config.reconsolidation_alpha > 0.0 and anchor_row is not None:
        cue = q if isinstance(q, list) else None
        await _accrue_drift(service, anchor_row, t=now, cue=cue)


async def _accrue_drift(
    service: MemoryService,
    row: dict[str, Any],
    *,
    t: float,
    cue: list[float] | None = None,
) -> None:
    """Apply one reconsolidation step (Nader 2000) to the episode.

    Drift bound: cosine-distance from the original embedding is bounded
    above by the cumulative blend ``sum(alpha * score)``. Once it
    crosses ``reconsolidation_max_total_drift`` the episode is marked
    ``drift_locked`` and further drift is rejected. ``min_score`` gates
    accumulation to avoid drifting on noisy recalls.

    When a ``cue`` embedding is supplied the stored vector is buffered
    via ``pending.update`` and pulled toward the cue:
    ``v_new = (1 - a*s) v_old + a*s cue`` (re-normalised). The buffered
    update is promoted to HNSW by ``flush_pending()`` during consolidate.
    """
    cfg = service.config
    md = row.get("metadata") or {}
    # Fast-path snapshot check: skip the lock + re-read entirely when
    # the cached metadata already shows we have nothing to do. Worst
    # case the snapshot is stale and we acquire the lock unnecessarily
    # — re-checked inside.
    if md.get("drift_locked"):
        return
    score = _cosine_score_from_distance(row.get("distance"))
    if score < cfg.reconsolidation_min_score:
        return
    doc_id = int(row["id"])
    # Per-doc lock so the read of ``drift_total`` and the cap check
    # see the same value the write commits — without this, two
    # concurrent recalls both compute ``projected < cap``, neither
    # marks the row locked, and the atomic increment lets the actual
    # drift_total exceed the configured cap unboundedly (§3b).
    async with service._doc_lock(doc_id):
        rows = await service.episodic.get_documents({"id": doc_id})
        if not rows:
            return
        _, _, fresh_md = rows[0]
        if fresh_md.get("drift_locked"):
            return
        last = float(fresh_md.get("last_reconsolidated_at", 0.0))
        if last > 0.0 and (t - last) > cfg.reconsolidation_window_seconds:
            # Outside the window for this episode; reset the recall clock.
            await service.episodic.update_metadata([(doc_id, {"last_reconsolidated_at": t})])
            return
        step = cfg.reconsolidation_alpha * score
        current_drift = float(fresh_md.get("drift_total", 0.0))
        new_drift = current_drift + step
        locked = new_drift >= cfg.reconsolidation_max_total_drift
        md_update: dict[str, Any] = {
            "last_reconsolidated_at": t,
            "drift_total": new_drift,
        }
        if locked:
            md_update["drift_locked"] = True
        await service.episodic.update_metadata([(doc_id, md_update)])
    # Vector pull is outside the lock — it touches the HNSW pending
    # buffer, not metadata, and the per-doc lock semantics only need
    # to cover the read-compute-write of drift_total above.
    if cue is not None:
        await _apply_drift_to_vector(service, doc_id, cue=cue, blend=step)


async def _apply_drift_to_vector(
    service: MemoryService,
    doc_id: int,
    *,
    cue: list[float],
    blend: float,
) -> None:
    """Pull the stored vector toward the cue and buffer the update.

    Buffered through ``pending.update`` so HNSW only sees the new vector
    after the next ``flush_pending`` (typically run by consolidate).
    """
    if blend <= 0.0:
        return
    embeds = await service.episodic.get_embeddings_by_ids([doc_id])
    v_old = embeds.get(doc_id)
    if v_old is None:
        return
    v_old_arr = np.asarray(v_old, dtype=np.float64)
    cue_arr = np.asarray(cue, dtype=np.float64)
    if v_old_arr.shape != cue_arr.shape:
        return
    blended = (1.0 - blend) * v_old_arr + blend * cue_arr
    norm = float(np.linalg.norm(blended))
    if norm <= 0.0:
        return
    v_new = (blended / norm).tolist()
    await service.episodic.update_embedding(doc_id, v_new, source="reconsolidation")


async def _merge_hits(
    service: MemoryService,
    q: str | list[float],
    *,
    mode: str,
    k: int,
) -> list[_Hit]:
    want_sem = mode in {"auto", "semantic", "hybrid"}
    want_ep = mode in {"auto", "episodic", "hybrid"}
    # Both legs are independent; run their count + search concurrently.
    sem_count, ep_count = await asyncio.gather(
        service.semantic.count() if want_sem else _zero(),
        service.episodic.count() if want_ep else _zero(),
    )
    sem_hits_co = (
        service.semantic.similarity_search(q, k=k) if want_sem and sem_count > 0 else _empty_hits()
    )
    ep_hits_co = (
        service.episodic.similarity_search(q, k=k) if want_ep and ep_count > 0 else _empty_hits()
    )
    sem_hits, ep_hits = await asyncio.gather(sem_hits_co, ep_hits_co)
    merged: list[_Hit] = []
    for doc, dist in sem_hits:
        merged.append(
            (
                int(doc.metadata.get("id", -1)),
                doc.page_content,
                dict(doc.metadata),
                float(dist),
                "semantic",
            )
        )
    for doc, dist in ep_hits:
        merged.append(
            (
                int(doc.metadata.get("id", -1)),
                doc.page_content,
                dict(doc.metadata),
                float(dist),
                "episodic",
            )
        )
    return merged


async def _zero() -> int:
    return 0


async def _empty_hits() -> list[Any]:
    return []


async def _sr_rank_keys(
    service: MemoryService,
    merged: list[_Hit],
    *,
    caller_sid: str | None = None,
) -> dict[tuple[int, str], float]:
    """Return ``{(doc_id, source): rank_key}`` overrides for SR sort.

    We pick the best-cosine episodic anchor ``i*`` and assign each
    episodic candidate the rank key
    ``(1 - omega) * dist - omega * M[i*, j] - sa_w * spread[j]
      - same_session_bonus * 1[md.session_id == caller_sid]``.
    Rows not in the returned mapping fall back to their raw cosine
    distance, so published ``distance`` values stay unmodified. Keying
    by ``(doc_id, source)`` keeps the lookup stable even if ``merged``
    is rebuilt or copied between this call and the sort, and avoids
    cross-source key collisions.
    """
    if not merged:
        return {}
    episodic_hits = [(doc_id, dist) for doc_id, _, _, dist, src in merged if src == "episodic"]
    if not episodic_hits:
        return {}
    anchor_id = min(episodic_hits, key=lambda r: r[1])[0]
    cfg = service.config
    # Only pay the count() round-trip when the SR can actually use it.
    omega = service._sr.omega(await service.episodic.count()) if service._sr is not None else 0.0
    sr_boost: dict[int, float] = {}
    if service._sr is not None and omega > 0.0:
        # Lock-guarded snapshot so a concurrent ``observe`` running on
        # the same SR cannot mutate ``_M[anchor]`` between this read
        # and the rank computation below (which has further awaits
        # interleaved through the spreading-activation call).
        sr_boost = await service._sr.snapshot_boost(anchor_id, [r[0] for r in episodic_hits])
    spread: dict[int, float] = {}
    if cfg.spreading_activation_hops > 0 and cfg.spreading_activation_weight > 0.0:
        spread = await service._plasticity.spreading(
            anchor_id,
            [r[0] for r in episodic_hits],
            hops=cfg.spreading_activation_hops,
            gamma=cfg.spreading_activation_decay,
            max_fanout=cfg.spreading_activation_max_fanout,
        )
    same_sess_bonus = float(cfg.same_session_bonus)
    use_context = bool(caller_sid) and same_sess_bonus > 0.0
    if omega <= 0.0 and not spread and not use_context:
        return {}
    keys: dict[tuple[int, str], float] = {}
    sa_w = float(cfg.spreading_activation_weight)
    for doc_id, _, md, dist, src in merged:
        if src != "episodic":
            continue
        rank = (1.0 - omega) * dist - omega * sr_boost.get(doc_id, 0.0)
        rank -= sa_w * float(spread.get(doc_id, 0.0))
        if use_context:
            sid = str(md.get("session_id", "")) if md else ""
            if sid == caller_sid:
                rank -= same_sess_bonus
        # Non-finite rank (NaN dist from a broken backend, inf from a
        # corrupt SR row) would make ``sort`` undefined. Push offenders
        # to the end deterministically.
        keys[(doc_id, src)] = rank if math.isfinite(rank) else math.inf
    _ctx = _trace_current()
    if _ctx is not None and _ctx.enabled:
        _ctx.add_event(
            "sr_rank.decomposition",
            anchor_id=int(anchor_id),
            omega=float(omega),
            sr_boost_nonzero=sum(1 for v in sr_boost.values() if v),
            spread_nonzero=sum(1 for v in spread.values() if v),
            same_session_bonus=same_sess_bonus if use_context else 0.0,
            episodic_candidates=len(episodic_hits),
        )
    return keys
