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

from typing import TYPE_CHECKING, Any

from synara.core.errors import ValidationError

from ..primitives import complete as _complete_mod
from ..service import now_seconds

if TYPE_CHECKING:  # pragma: no cover - typing only
    from ..service import HippocampusService


# Type alias for a merged hit row.
_Hit = tuple[int, str, dict[str, Any], float, str]


async def run(
    service: HippocampusService,
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
    q = await service.query_arg(query)
    # CA3 iterative pattern completion: refine the query by
    # softmax-recombining its near-neighbours before the final search.
    # Skipped when no embed_fn is configured (q is a string) or when
    # iters == 0.
    iters = service.config.recall_completion_iters
    if iters > 0 and isinstance(q, list):
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

    merged = await _merge_hits(service, q, mode=mode, k=k, ep_filter=ep_filter)
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
        service._sr.observe_recall_set(
            anchor_sid, anchor_id, [j for _, j in observed_episodic[1:]], t
        )
    return out


async def _merge_hits(
    service: HippocampusService,
    q: str | list[float],
    *,
    mode: str,
    k: int,
    ep_filter: dict[str, Any] | None,
) -> list[_Hit]:
    merged: list[_Hit] = []
    if mode in {"auto", "semantic", "hybrid"} and await service.semantic.count() > 0:
        for doc, dist in await service.semantic.similarity_search(q, k=k):
            merged.append(
                (
                    int(doc.metadata.get("id", -1)),
                    doc.page_content,
                    dict(doc.metadata),
                    float(dist),
                    "semantic",
                )
            )
    if mode in {"auto", "episodic", "hybrid"} and await service.episodic.count() > 0:
        for doc, dist in await service.episodic.similarity_search(q, k=k, filter=ep_filter):
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


async def _sr_rank_keys(
    service: HippocampusService,
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
    if service._sr is None or not merged:
        return {}
    ep_count = await service.episodic.count()
    omega = service._sr.omega(ep_count)
    if omega <= 0.0:
        return {}
    episodic_hits = [(doc_id, dist) for doc_id, _, _, dist, src in merged if src == "episodic"]
    if not episodic_hits:
        return {}
    anchor_id = min(episodic_hits, key=lambda r: r[1])[0]
    boost = service._sr.boost(anchor_id, [r[0] for r in episodic_hits])
    keys: dict[tuple[int, str], float] = {}
    for doc_id, _, _, dist, src in merged:
        if src != "episodic":
            continue
        keys[(doc_id, src)] = (1.0 - omega) * dist - omega * boost.get(doc_id, 0.0)
    return keys
