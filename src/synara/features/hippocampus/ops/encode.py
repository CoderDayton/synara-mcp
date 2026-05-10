"""Episode encoding: pattern-separation gate, theta segmentation, insert.

Pipeline:
  1. embed once
  2. dedup gate — DG-Jaccard sparse-code overlap by default; cosine
     distance threshold as a fallback when pattern separation is off
  3. theta-segment long content into ``<= theta_segment_max_items``
     ordered sub-records sharing an ``episode_group_id``
  4. write one or many records, then patch each row with its own id so
     downstream consolidate/forget/reflect can address it directly
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

from synara.core.errors import ValidationError

from ..primitives.segment import split_into_segments
from ..primitives.separate import jaccard as _dg_jaccard
from ..service import UNCONSOLIDATED, now_seconds

if TYPE_CHECKING:  # pragma: no cover - typing only
    from ..service import HippocampusService


async def run(
    service: HippocampusService,
    *,
    content: str,
    session_id: str,
    tags: Sequence[str] | None = None,
    salience: float = 0.5,
) -> dict[str, Any]:
    if not content.strip():
        raise ValidationError("content must be non-empty")
    if not session_id:
        raise ValidationError("session_id must be non-empty")
    if not 0.0 <= salience <= 1.0:
        raise ValidationError("salience must be in [0, 1]")

    # Pattern separation: cosine threshold OR DG-Jaccard, depending on
    # config. The DG path embeds once, computes the sparse code, and
    # compares against the stored ``dg_support`` of the top cosine
    # candidates within the same session.
    new_embs = await service.vectorise([content])
    new_emb = new_embs[0] if new_embs else None
    new_support: tuple[int, ...] = ()
    use_dg = service.config.dg_pattern_separation and new_emb is not None
    dedup_hit: dict[str, Any] | None = None
    if use_dg and new_emb is not None:
        new_support = service._ensure_projector(len(new_emb)).support(new_emb)
        dedup_hit = await _dedup_jaccard(service, new_emb, new_support, session_id)
    else:
        q_arg: str | list[float] = (
            new_emb if new_emb is not None else await service.query_arg(content)
        )
        existing = await service.episodic.similarity_search(
            q_arg, k=1, filter={"session_id": session_id}
        )
        if existing and existing[0][1] <= service.config.dedup_distance:
            doc, dist = existing[0]
            doc_id = int(doc.metadata.get("id", -1))
            if doc_id >= 0:
                await service.bump_retrieval(doc_id, doc.metadata)
            dedup_hit = {
                "id": doc_id,
                "deduped": True,
                "distance": float(dist),
                "session_id": session_id,
            }
    if dedup_hit is not None:
        return dedup_hit

    encoded_at = now_seconds()
    segments = split_into_segments(
        content,
        max_chars=service.config.theta_segment_max_chars,
        max_items=service.config.theta_segment_max_items,
    )
    if len(segments) == 1:
        return await _insert_single(
            service,
            content=content,
            session_id=session_id,
            tags=tags,
            salience=salience,
            encoded_at=encoded_at,
            new_embs=new_embs,
            dg_support=new_support if (use_dg and new_support) else None,
        )
    return await _insert_segmented(
        service,
        segments=segments,
        session_id=session_id,
        tags=tags,
        salience=salience,
        encoded_at=encoded_at,
    )


async def _dedup_jaccard(
    service: HippocampusService,
    new_emb: list[float],
    new_support: tuple[int, ...],
    session_id: str,
) -> dict[str, Any] | None:
    """Return a dedup result if any candidate's stored support has
    Jaccard overlap >= ``dg_jaccard_threshold`` with ``new_support``."""
    cands = await service.episodic.similarity_search(
        new_emb,
        k=service.config.dg_dedup_candidates,
        filter={"session_id": session_id},
    )
    best_j = 0.0
    best_doc: Any = None
    best_dist = 0.0
    for doc, dist in cands:
        cand_support = doc.metadata.get("dg_support") or []
        if not cand_support:
            continue
        j = _dg_jaccard(new_support, cand_support)
        if j > best_j:
            best_j = j
            best_doc = doc
            best_dist = float(dist)
    if best_doc is None or best_j < service.config.dg_jaccard_threshold:
        return None
    doc_id = int(best_doc.metadata.get("id", -1))
    if doc_id >= 0:
        await service.bump_retrieval(doc_id, best_doc.metadata)
    return {
        "id": doc_id,
        "deduped": True,
        "distance": best_dist,
        "jaccard": float(best_j),
        "session_id": session_id,
    }


async def _insert_single(
    service: HippocampusService,
    *,
    content: str,
    session_id: str,
    tags: Sequence[str] | None,
    salience: float,
    encoded_at: float,
    new_embs: list[list[float]] | None,
    dg_support: tuple[int, ...] | None,
) -> dict[str, Any]:
    meta: dict[str, Any] = {
        "session_id": session_id,
        "tags": list(tags) if tags else [],
        "salience": float(salience),
        "encoded_at": encoded_at,
        "last_accessed": encoded_at,
        "retrieval_count": 0,
        "access_history": [encoded_at],
        "consolidated_into": UNCONSOLIDATED,
    }
    if dg_support:
        meta["dg_support"] = list(dg_support)
    ids = await service.episodic.add_texts([content], metadatas=[meta], embeddings=new_embs)
    new_id = int(ids[0])
    await service.episodic.update_metadata([(new_id, {"id": new_id})])
    return {
        "id": new_id,
        "deduped": False,
        "distance": None,
        "session_id": session_id,
    }


async def _insert_segmented(
    service: HippocampusService,
    *,
    segments: list[str],
    session_id: str,
    tags: Sequence[str] | None,
    salience: float,
    encoded_at: float,
) -> dict[str, Any]:
    """Encode each segment as a sub-record sharing ``episode_group_id``.

    The first segment's auto-assigned id doubles as the group id, so we
    avoid an external id allocator. Subsequent segments are written
    with that group id already set in their metadata.
    """
    segment_embs = await service.vectorise(segments)
    tags_list = list(tags) if tags else []
    seg_count = len(segments)
    seg_ids: list[int] = []
    group_id: int | None = None
    for pos, seg in enumerate(segments):
        seg_meta: dict[str, Any] = {
            "session_id": session_id,
            "tags": tags_list,
            "salience": float(salience),
            "encoded_at": encoded_at,
            "last_accessed": encoded_at,
            "retrieval_count": 0,
            "access_history": [encoded_at],
            "consolidated_into": UNCONSOLIDATED,
            "position_in_episode": pos,
            "segment_count": seg_count,
        }
        if group_id is not None:
            seg_meta["episode_group_id"] = group_id
        seg_emb_arg = [segment_embs[pos]] if segment_embs is not None else None
        ids = await service.episodic.add_texts([seg], metadatas=[seg_meta], embeddings=seg_emb_arg)
        seg_id = int(ids[0])
        if group_id is None:
            group_id = seg_id
            await service.episodic.update_metadata(
                [(seg_id, {"id": seg_id, "episode_group_id": group_id})]
            )
        else:
            await service.episodic.update_metadata([(seg_id, {"id": seg_id})])
        seg_ids.append(seg_id)
    resolved_group_id = group_id if group_id is not None else seg_ids[0]
    return {
        "id": seg_ids[0],
        "deduped": False,
        "distance": None,
        "session_id": session_id,
        "group_id": resolved_group_id,
        "segment_ids": seg_ids,
    }
