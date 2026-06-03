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
from typing import TYPE_CHECKING, Any

import numpy as np

from synara.core.errors import ValidationError

from ..service import now_seconds
from ..tracing import current_context as _trace_current
from ..tracing import record_span as _trace_span
from . import complete as _complete_mod

if TYPE_CHECKING:  # pragma: no cover - typing only
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
    service: MemoryService, *, query: str, session_id: str | None, k: int, mode: str
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


async def run(
    service: MemoryService,
    *,
    query: str,
    session_id: str | None = None,
    k: int = 8,
    mode: str = "auto",
    scope_session: bool = False,
    tags: list[str] | None = None,
    reinforce: bool,
) -> list[dict[str, Any]]:
    # Validate before the ``k <= 0`` short-circuit: an empty/oversized
    # query or unknown mode is always a programmer error, regardless of
    # how many results were requested. Returning ``[]`` for ``k <= 0``
    # is a convenience for callers who compute ``k`` from a budget that
    # may legitimately go to zero.
    _validate_recall_inputs(service, query=query, session_id=session_id, k=k, mode=mode)
    if k <= 0:
        return []

    # Recall is always cross-session: ``session_id`` is the caller's
    # current-session hint, used as the SR window key (so cross-session
    # bridges form in the caller's context) but never as a hard filter
    # on the simplevecdb search. Callers that want strict per-session
    # results should post-filter the returned list themselves.
    ep_filter: dict[str, Any] | None = None
    # Opt-in hard scoping (P1). ``session_id`` stays a ranking hint by
    # default; ``scope_session`` promotes it to a filter, and ``tags``
    # restricts episodic hits to traces carrying every requested tag.
    # Tags are stored as a list, so this is a Python post-filter over an
    # over-fetched candidate set (mirroring ``recall_semantic_memory``)
    # rather than a simplevecdb metadata filter, keeping cosine order the
    # final arbiter. Semantic (gist) hits are cross-session by design and
    # are never scoped out.
    tagset = frozenset(tags) if tags else None
    filtering = scope_session or bool(tagset)
    collapse = service.config.recall_collapse_groups
    # Over-fetch when filtering *or* collapsing groups so that, after
    # dropping out-of-scope or same-group fragments, ``k`` hits remain.
    fetch_k = max(k * 4, 32) if (filtering or collapse) else k
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
                ep_filter=ep_filter,
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
        merged = await _merge_hits(service, q, mode=mode, k=fetch_k, ep_filter=ep_filter)
    if filtering:
        ep_before = sum(1 for row in merged if row[4] == "episodic")
        merged = [
            row
            for row in merged
            if row[4] != "episodic"
            or _passes_scope(
                row[2], session_id=session_id, scope_session=scope_session, tags=tagset
            )
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

    t_now = now_seconds()
    out: list[dict[str, Any]] = []
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
        hit.update(_recency_fields(md or {}, source=source, now=t_now))
        out.append(hit)
        # ``reinforce=False`` (ambient resource reads) makes recall a
        # pure read: no retrieval_count bump and — since the SR block
        # below guards on ``observed_episodic`` — no SR/plasticity update.
        if reinforce and source == "episodic" and doc_id >= 0:
            await service.bump_retrieval(doc_id, md)
            sid = str(md.get("session_id", "")) if md else ""
            if sid:
                observed_episodic.append((sid, doc_id))

    await _reinforce_recall_set(
        service,
        out=out,
        observed_episodic=observed_episodic,
        session_id=session_id,
        q=q,
        now=t_now,
    )
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


def _passes_scope(
    md: dict[str, Any] | None,
    *,
    session_id: str | None,
    scope_session: bool,
    tags: frozenset[str] | None,
) -> bool:
    """True if an episodic hit satisfies the opt-in recall scope.

    ``scope_session`` keeps only hits whose ``session_id`` matches the
    caller's; ``tags`` keeps only hits whose stored tag list is a
    superset of every requested tag.
    """
    md = md or {}
    if scope_session and session_id is not None and str(md.get("session_id", "")) != session_id:
        return False
    if tags:
        hit_tags = {str(t) for t in (md.get("tags") or [])}
        if not tags.issubset(hit_tags):
            return False
    return True


def _cosine_score_from_distance(dist: float | None) -> float:
    """Map cosine distance (in [0, 2]) to a similarity-like score in [0, 1]."""
    if dist is None:
        return 0.5
    s = 1.0 - 0.5 * float(dist)
    return max(0.0, min(1.0, s))


def _recency_fields(md: dict[str, Any], *, source: str, now: float) -> dict[str, Any]:
    """Promote an episodic hit's group lineage and recency to top-level
    fields so a caller can tell old memories from new and reassemble a
    segmented episode (via ``group_id`` -> ``get_episode``) without
    digging through the raw metadata blob. Semantic gists carry none of
    these, so they pass through unchanged."""
    if source != "episodic":
        return {}
    gid = md.get("episode_group_id")
    seg_count = int(md.get("segment_count", 1))
    created = md.get("encoded_at")
    created_f = float(created) if isinstance(created, (int, float)) else None
    stamps = [
        float(v)
        for v in (md.get("last_accessed"), md.get("last_reconsolidated_at"))
        if isinstance(v, (int, float))
    ]
    updated = max(stamps) if stamps else created_f

    def _age_days(ts: float | None) -> float | None:
        return None if ts is None else max(0.0, (now - ts) / 86_400.0)

    return {
        # ``episode_group_id`` is set even for a standalone episode (to its
        # own id); surface it only when actually segmented, so a non-null
        # ``group_id`` means "reassemble the whole via get_episode".
        "group_id": int(gid) if (gid is not None and seg_count > 1) else None,
        "segment_count": seg_count,
        "created_at": created_f,
        "updated_at": updated,
        "age_days": _age_days(created_f),
        "updated_age_days": _age_days(updated),
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
    for j in others:
        jrow = out_lookup.get(j)
        score = _cosine_score_from_distance(jrow["distance"]) if jrow is not None else 0.5
        await service._plasticity.reinforce(anchor_id, j, score=score, now=now)
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
    ep_filter: dict[str, Any] | None,
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
        service.episodic.similarity_search(q, k=k, filter=ep_filter)
        if want_ep and ep_count > 0
        else _empty_hits()
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
    ep_count = await service.episodic.count()
    cfg = service.config
    omega = service._sr.omega(ep_count) if service._sr is not None else 0.0
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
