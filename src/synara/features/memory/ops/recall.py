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
from typing import TYPE_CHECKING, Any

import numpy as np

from synara.core.errors import ValidationError

from ..primitives import complete as _complete_mod
from ..primitives.tracing import record_span as _trace_span
from ..service import now_seconds

if TYPE_CHECKING:  # pragma: no cover - typing only
    from ..primitives.port import MemoryServicePort as MemoryService


# Type alias for a merged hit row.
_Hit = tuple[int, str, dict[str, Any], float, str]


async def run(
    service: MemoryService,
    *,
    query: str,
    session_id: str | None = None,
    k: int = 8,
    mode: str = "auto",
) -> list[dict[str, Any]]:
    if not query.strip():
        raise ValidationError("query must be non-empty")
    if k <= 0:
        return []
    if mode not in {"auto", "episodic", "semantic", "hybrid"}:
        raise ValidationError(f"unknown recall mode: {mode}")

    ep_filter: dict[str, Any] | None = {"session_id": session_id} if session_id else None
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
        q = result.query

    with _trace_span("merge_hits"):
        merged = await _merge_hits(service, q, mode=mode, k=k, ep_filter=ep_filter)
    with _trace_span("sr_rank"):
        rank_keys = await _sr_rank_keys(service, merged)
    # Key by (doc_id, source) instead of object identity: stable across
    # any future merge-list copying or wrapping.
    merged.sort(key=lambda r: rank_keys.get((r[0], r[4]), r[3]))

    out: list[dict[str, Any]] = []
    observed_episodic: list[tuple[str, int]] = []
    for doc_id, text, md, dist, source in merged[:k]:
        out.append(
            {
                "id": doc_id,
                "content": text,
                "distance": dist,
                "source": source,
                "metadata": md,
            }
        )
        if source == "episodic" and doc_id >= 0:
            await service.bump_retrieval(doc_id, md)
            sid = str(md.get("session_id", "")) if md else ""
            if sid:
                observed_episodic.append((sid, doc_id))

    # Anchor-model SR update: fold one edge from the best-cosine anchor
    # (= first episodic hit after re-ranking) to each other episodic
    # hit. Avoids the pairwise n*(n-1)/2 inflation that a naive
    # "everything observed at the same t" loop would produce — five hits
    # add four edges, not ten.
    if service._sr is not None and observed_episodic:
        t = now_seconds()
        anchor_sid, anchor_id = observed_episodic[0]
        others = [j for _, j in observed_episodic[1:]]
        service._sr.observe_recall_set(anchor_sid, anchor_id, others, t)
        await service._sr.flush()
        # Plasticity layer: reinforce anchor->each-other edge with the
        # cosine-similarity score (1 - distance, clipped). Co-recall is
        # the brain's bread-and-butter Hebbian event.
        out_lookup = {row["id"]: row for row in out}
        anchor_row = out_lookup.get(anchor_id)
        for j in others:
            jrow = out_lookup.get(j)
            score = _cosine_score_from_distance(jrow["distance"]) if jrow is not None else 0.5
            await service._plasticity.reinforce(anchor_id, j, score=score, now=t)
        # Reconsolidation drift accounting: bumps drift_total in the
        # episode's metadata when the alpha is set. Vector rewrite is
        # deferred (see config note); this only records the bound.
        if service.config.reconsolidation_alpha > 0.0 and anchor_row is not None:
            cue = q if isinstance(q, list) else None
            await _accrue_drift(service, anchor_row, t=t, cue=cue)
    return out


def _cosine_score_from_distance(dist: float | None) -> float:
    """Map cosine distance (in [0, 2]) to a similarity-like score in [0, 1]."""
    if dist is None:
        return 0.5
    s = 1.0 - 0.5 * float(dist)
    return max(0.0, min(1.0, s))


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
    if md.get("drift_locked"):
        return
    score = _cosine_score_from_distance(row.get("distance"))
    if score < cfg.reconsolidation_min_score:
        return
    last = float(md.get("last_reconsolidated_at", 0.0))
    if last > 0.0 and (t - last) > cfg.reconsolidation_window_seconds:
        # Outside the window for this episode; reset the recall clock.
        await service.episodic.update_metadata([(int(row["id"]), {"last_reconsolidated_at": t})])
        return
    step = cfg.reconsolidation_alpha * score
    projected = float(md.get("drift_total", 0.0)) + step
    locked = projected >= cfg.reconsolidation_max_total_drift
    # Atomic counter delta — concurrent recalls cannot lose drift.
    # Lock flag is sticky-true so racing on it is safe (worst case the
    # cap is briefly exceeded by one alpha-step, ~0.02).
    await service.episodic.increment_metadata(int(row["id"]), {"drift_total": step})
    md_update: dict[str, Any] = {"last_reconsolidated_at": t}
    if locked:
        md_update["drift_locked"] = True
    await service.episodic.update_metadata([(int(row["id"]), md_update)])
    if cue is not None:
        await _apply_drift_to_vector(service, int(row["id"]), cue=cue, blend=step)


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
) -> dict[tuple[int, str], float]:
    """Return ``{(doc_id, source): rank_key}`` overrides for SR sort.

    We pick the best-cosine episodic anchor ``i*`` and assign each
    episodic candidate the rank key
    ``(1 - omega) * dist - omega * M[i*, j]``. Rows not in the returned
    mapping fall back to their raw cosine distance, so published
    ``distance`` values stay unmodified. Keying by ``(doc_id, source)``
    keeps the lookup stable even if ``merged`` is rebuilt or copied
    between this call and the sort, and avoids cross-source key
    collisions.
    """
    if not merged:
        return {}
    episodic_hits = [(doc_id, dist) for doc_id, _, _, dist, src in merged if src == "episodic"]
    if not episodic_hits:
        return {}
    anchor_id = min(episodic_hits, key=lambda r: r[1])[0]
    ep_count = await service.episodic.count()
    cfg = service.config
    # SR omega-weighted boost (existing behaviour).
    omega = service._sr.omega(ep_count) if service._sr is not None else 0.0
    sr_boost: dict[int, float] = {}
    if service._sr is not None and omega > 0.0:
        sr_boost = service._sr.boost(anchor_id, [r[0] for r in episodic_hits])
    # Spreading activation over the plasticity graph (gated on hops > 0).
    spread: dict[int, float] = {}
    if cfg.spreading_activation_hops > 0 and cfg.spreading_activation_weight > 0.0:
        spread = await service._plasticity.spreading(
            anchor_id,
            [r[0] for r in episodic_hits],
            hops=cfg.spreading_activation_hops,
            gamma=cfg.spreading_activation_decay,
        )
    if omega <= 0.0 and not spread:
        return {}
    keys: dict[tuple[int, str], float] = {}
    sa_w = float(cfg.spreading_activation_weight)
    for doc_id, _, _, dist, src in merged:
        if src != "episodic":
            continue
        rank = (1.0 - omega) * dist - omega * sr_boost.get(doc_id, 0.0)
        rank -= sa_w * float(spread.get(doc_id, 0.0))
        keys[(doc_id, src)] = rank
    return keys
