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

import contextlib
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any

from synara.core.errors import ValidationError

from ..amygdala.signals import (
    SignalRegistry,
    derive_salience,
    derive_signals,
)
from ..memory_types import SCOPE_SESSION
from ..service import UNCONSOLIDATED, now_seconds
from .segment import split_into_segments
from .separate import jaccard as _dg_jaccard

# Reserved meta keys: signals returned by SignalRegistry.derive() must not
# clobber these. A custom registry that emits e.g. "session_id" or
# "consolidated_into" would otherwise corrupt provenance, retrieval
# counts, or consolidation state.
_RESERVED_META_KEYS: frozenset[str] = frozenset(
    {
        "id",
        "session_id",
        "tags",
        "salience",
        "created_at",
        "encoded_at",
        "last_accessed",
        "retrieval_count",
        "access_history",
        "consolidated_into",
        "dg_support",
        "position_in_episode",
        "segment_count",
        "episode_group_id",
        "drift_total",
        "drift_locked",
        "last_reconsolidated_at",
    }
)


def _safe_signals(signals: Mapping[str, Any] | None) -> dict[str, Any] | None:
    if signals is None:
        return None
    return {k: v for k, v in signals.items() if k not in _RESERVED_META_KEYS}


if TYPE_CHECKING:  # pragma: no cover - typing only
    from ..config import MemoryConfig
    from ..port import MemoryServicePort as MemoryService


def _check_input_caps(
    cfg: MemoryConfig,
    content: str,
    session_id: str,
    tags: Sequence[str] | None,
) -> None:
    """Reject oversized untrusted MCP-tool input before it is stored."""
    if cfg.max_session_id_chars and len(session_id) > cfg.max_session_id_chars:
        raise ValidationError(
            f"session_id exceeds max_session_id_chars ({cfg.max_session_id_chars})"
        )
    if cfg.max_content_chars and len(content) > cfg.max_content_chars:
        raise ValidationError(
            f"content exceeds max_content_chars ({cfg.max_content_chars}); "
            "split it before storing so the embedding represents the full text"
        )
    if cfg.max_tags and tags is not None and len(tags) > cfg.max_tags:
        raise ValidationError(f"too many tags (>{cfg.max_tags})")
    if cfg.max_tag_chars and tags is not None:
        for tag in tags:
            if len(tag) > cfg.max_tag_chars:
                raise ValidationError(f"tag exceeds max_tag_chars ({cfg.max_tag_chars})")


async def run(
    service: MemoryService,
    *,
    content: str,
    session_id: str,
    tags: Sequence[str] | None = None,
    salience: float | None = None,
) -> dict[str, Any]:
    if not content.strip():
        raise ValidationError("content must be non-empty")
    if not session_id:
        raise ValidationError("session_id must be non-empty")
    cfg = service.config
    _check_input_caps(cfg, content, session_id, tags)
    registry = cfg.signal_registry if isinstance(cfg.signal_registry, SignalRegistry) else None
    signals = _derive_signals(content, registry) if cfg.auto_signal_metadata else None
    if salience is None:
        salience = _derive_salience(content, registry, cfg, signals)
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
    # Below the content-length floor, embedding-based dedup is
    # unreliable (distinct short episodes crowd into the same cosine
    # cone, where neither cosine nor DG can separate them) and a false
    # merge is irreversible data loss, so skip dedup and always store.
    skip_dedup = len(content.strip()) < cfg.min_dedup_chars
    if use_dg and new_emb is not None:
        new_support = service._ensure_projector(len(new_emb)).support(new_emb)
        if not skip_dedup:
            dedup_hit = await _dedup_jaccard(service, new_emb, new_support, session_id)
    elif not skip_dedup:
        # Non-DG path: cosine dedup, unless the episode is below the
        # short-content floor (``skip_dedup``) where it is unreliable.
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

    salience = await _maybe_surprise_boost(service, new_emb, session_id, salience)
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
            signals=signals,
        )
    return await _insert_segmented(
        service,
        segments=segments,
        session_id=session_id,
        tags=tags,
        salience=salience,
        encoded_at=encoded_at,
        signals=signals,
    )


def _derive_signals(content: str, registry: SignalRegistry | None) -> dict[str, Any]:
    if registry is not None:
        return registry.derive(content)
    return dict(derive_signals(content))


def _derive_salience(
    content: str,
    registry: SignalRegistry | None,
    cfg: Any,
    signals: dict[str, Any] | None,
) -> float:
    if not cfg.auto_salience:
        return 0.5
    base_signals = signals if signals is not None else _derive_signals(content, registry)
    if registry is not None:
        return registry.salience(base_signals)
    return derive_salience(base_signals, base=cfg.auto_salience_base)


async def _maybe_surprise_boost(
    service: MemoryService,
    new_emb: list[float] | None,
    session_id: str,
    salience: float,
) -> float:
    """Apply prediction-error-style salience boost when the new episode is
    far in cosine distance from anything else in this namespace.

    Defaults to a no-op (``surprise_salience_boost == 0``); the floor and
    boost are both runtime-tunable via ``MemoryConfig``.
    """
    cfg = service.config
    if cfg.surprise_salience_boost <= 0.0 or new_emb is None:
        return salience
    nearest = await service.episodic.similarity_search(
        new_emb, k=1, filter={"session_id": session_id}
    )
    # Surprise = prediction error vs an existing memory; with nothing
    # to predict against, the encode is novel-by-default, not surprising.
    if not nearest:
        return salience
    d_star = float(nearest[0][1])
    if d_star >= cfg.surprise_distance_floor:
        return min(1.0, salience + cfg.surprise_salience_boost)
    return salience


async def _dedup_jaccard(
    service: MemoryService,
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
    service: MemoryService,
    *,
    content: str,
    session_id: str,
    tags: Sequence[str] | None,
    salience: float,
    encoded_at: float,
    new_embs: list[list[float]] | None,
    dg_support: tuple[int, ...] | None,
    signals: Mapping[str, Any] | None,
) -> dict[str, Any]:
    meta: dict[str, Any] = {
        "session_id": session_id,
        # Episodes always carry a (mandatory) session_id, so they are
        # always session-scoped; recall honours this via in_session_scope.
        "scope": SCOPE_SESSION,
        "tags": list(tags) if tags else [],
        "salience": float(salience),
        "created_at": encoded_at,
        "last_accessed": encoded_at,
        "retrieval_count": 0,
        "access_history": [encoded_at],
        "consolidated_into": UNCONSOLIDATED,
    }
    if dg_support:
        meta["dg_support"] = list(dg_support)
    safe = _safe_signals(signals)
    if safe:
        meta.update(safe)
    ids = await service.episodic.add_texts([content], metadatas=[meta], embeddings=new_embs)
    new_id = int(ids[0])
    # Roll back the row if the id-patch fails so an orphan record (one
    # without ``id`` in its metadata, which downstream id-keyed filters
    # silently skip) cannot persist. The crash window is still open —
    # a process kill between ``add_texts`` and ``update_metadata`` will
    # leave an orphan — but that is bounded to true process death and
    # the ``synara doctor`` reaper (follow-up) cleans it up.
    try:
        await service.episodic.update_metadata([(new_id, {"id": new_id})])
    except Exception:
        # Suppress so the original id-patch error is what propagates;
        # a failing cleanup would otherwise mask the root cause.
        with contextlib.suppress(Exception):
            await service.episodic.delete_by_ids([new_id])
        raise
    # ``distance`` is None on a fresh insert: there was no near-duplicate
    # to match against, so there is no distance to report (the dedup-hit
    # path reports the matched cosine distance instead). This is a store
    # confirmation, not a recall hit — recall hits always carry a float.
    return {
        "id": new_id,
        "deduped": False,
        "distance": None,
        "session_id": session_id,
    }


async def _insert_segmented(
    service: MemoryService,
    *,
    segments: list[str],
    session_id: str,
    tags: Sequence[str] | None,
    salience: float,
    encoded_at: float,
    signals: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Encode each segment as a sub-record sharing ``episode_group_id``.

    The first segment's auto-assigned id doubles as the group id, so we
    avoid an external id allocator. Subsequent segments are written
    with that group id already set in their metadata.
    """
    segment_embs = await service.vectorise(segments)
    tags_list = list(tags) if tags else []
    seg_count = len(segments)
    safe = _safe_signals(signals)
    # One bulk ``add_texts`` for the whole episode so a crash mid-loop
    # cannot leave an incomplete ``episode_group_id`` chain. The group
    # id (the first row's auto-assigned id) is patched into every
    # segment's metadata in one bulk ``update_metadata`` — still two
    # writes, but each one is atomic across all segments, so any
    # interruption either lands the whole group or none of it. Failure
    # rolls back the entire group rather than leaving partial segments.
    seg_metas: list[dict[str, Any]] = []
    for pos, _seg in enumerate(segments):
        m: dict[str, Any] = {
            "session_id": session_id,
            # Segmented episodes are session-scoped like any other episode.
            "scope": SCOPE_SESSION,
            "tags": tags_list,
            "salience": float(salience),
            "created_at": encoded_at,
            "last_accessed": encoded_at,
            "retrieval_count": 0,
            "access_history": [encoded_at],
            "consolidated_into": UNCONSOLIDATED,
            "position_in_episode": pos,
            "segment_count": seg_count,
        }
        if safe:
            m.update(safe)
        seg_metas.append(m)
    seg_ids_raw = await service.episodic.add_texts(
        list(segments),
        metadatas=seg_metas,
        embeddings=segment_embs,
    )
    seg_ids = [int(s) for s in seg_ids_raw]
    group_id = seg_ids[0]
    try:
        await service.episodic.update_metadata(
            [(sid, {"id": sid, "episode_group_id": group_id}) for sid in seg_ids]
        )
    except Exception:
        with contextlib.suppress(Exception):
            await service.episodic.delete_by_ids(seg_ids)
        raise
    # Hebbian sibling chain: bond consecutive segments so spreading
    # activation can resurface the whole episode from any fragment.
    # Best-effort: the episode is already durably stored, so a bond
    # failure must not turn a successful store into a reported error.
    with contextlib.suppress(Exception):
        await service._reinforce_segment_chain(seg_ids, encoded_at)
    return {
        "id": seg_ids[0],
        "deduped": False,
        "distance": None,
        "session_id": session_id,
        "group_id": group_id,
        "segment_ids": seg_ids,
    }
