"""Episodic -> semantic consolidation.

Two-stage replay-driven consolidation, modelled on hippocampal
sharp-wave ripples (SWR) followed by neocortical schema integration:

* **Stage 1 — replay-biased absorption.** Score each unconsolidated
  episode by ``strength * novelty`` (strength uses the same power-law
  activation that ``forget`` uses; novelty is the cosine distance to
  the nearest existing schema). Process in descending-score order: if
  an episode's nearest schema is within ``consolidate_absorb_distance``,
  *absorb* the episode by appending it to the schema's source list and
  bumping confidence — no new schema is created. This implements the
  Tse et al. (2007) fast-track for schema-fitting episodes and prevents
  representation drift across consolidation passes.

* **Stage 2 — clustering of the residual.** Whatever was *not* absorbed
  is clustered (MiniBatch K-Means) into new semantic schemas — the
  classical CLS slow-pathway. Source episodes are marked
  ``consolidated_into=<schema_id>`` so ``forget`` knows their gist is
  preserved upstream.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

from ..service import UNCONSOLIDATED, now_seconds
from .forget import memory_strength

if TYPE_CHECKING:  # pragma: no cover - typing only
    from ..service import HippocampusService


def _replay_score(
    md: dict[str, Any],
    *,
    novelty: float,
    now: float,
    d: float,
) -> float:
    """Replay-sampling weight: strength * novelty (floored at epsilon).

    novelty: cosine distance to nearest schema (1.0 = novel, 0.0 = identical).
    Biases replay toward salient, recent, not-yet-covered episodes.
    """
    history = md.get("access_history")
    if isinstance(history, list) and history:
        access_times: list[float] = [float(t) for t in history]
    else:
        enc = float(md.get("encoded_at", now))
        last = float(md.get("last_accessed", enc))
        rc = int(md.get("retrieval_count", 0))
        access_times = [enc] + [last] * max(rc, 0)
    strength = memory_strength(
        salience=float(md.get("salience", 0.0)),
        access_times=access_times,
        now=now,
        d=d,
    )
    return max(strength * float(novelty), 1e-9)


async def _nearest_schema(service: HippocampusService, text: str) -> tuple[int, float] | None:
    if await service.semantic.count() == 0:
        return None
    # Route through ``query_arg`` so the configured embed_fn produces the
    # query vector — using raw text would invoke simplevecdb's bundled
    # embedder, which can mismatch the stored vector dimension.
    q = await service.query_arg(text)
    try:
        hits = await service.semantic.similarity_search(q, k=1)
    except (ValueError, RuntimeError):
        return None
    if not hits:
        return None
    sch_doc, dist = hits[0]
    sch_id = int(sch_doc.metadata.get("id", -1))
    if sch_id < 0:
        return None
    return sch_id, float(dist)


async def _absorb(
    service: HippocampusService,
    candidates: Sequence[tuple[int, str, dict[str, Any]]],
    *,
    now: float,
) -> tuple[set[int], list[dict[str, Any]]]:
    """Run the replay/absorption stage. Returns absorbed episode IDs and
    the list of schema records (one per schema that grew this pass)."""
    if await service.semantic.count() == 0:
        return set(), []

    d = service.config.forget_d
    absorb_dist = service.config.consolidate_absorb_distance

    scored: list[tuple[float, int, str, int, float]] = []
    for ep_id, text, md in candidates:
        nearest = await _nearest_schema(service, text)
        if nearest is None:
            continue
        sch_id, dist = nearest
        novelty = max(0.0, dist)  # cosine distance in [0, 2]; treat as novelty
        score = _replay_score(md, novelty=novelty, now=now, d=d)
        scored.append((score, int(ep_id), text, sch_id, dist))

    # Process in descending replay score — high-strength, high-novelty
    # episodes get the absorption attempt first.
    scored.sort(key=lambda r: -r[0])

    by_schema: dict[int, list[int]] = {}
    for _score, ep_id, _text, sch_id, dist in scored:
        if dist > absorb_dist:
            continue
        by_schema.setdefault(sch_id, []).append(ep_id)

    absorbed_ids: set[int] = set()
    formed: list[dict[str, Any]] = []
    for sch_id, ep_ids in by_schema.items():
        existing = await service.semantic.get_documents({"id": sch_id}, limit=1)
        if not existing:
            continue
        _, sch_text, sch_md = existing[0]
        prior = list(sch_md.get("source_episode_ids") or [])
        new_sources = sorted(set(prior) | {int(e) for e in ep_ids})
        if len(new_sources) == len(prior):
            continue
        # Confidence ~ fraction of total candidate evidence this schema
        # has now absorbed; bounded in [0, 1].
        denom = max(1, len(candidates) + len(prior))
        new_conf = min(1.0, len(new_sources) / denom)
        await service.semantic.update_metadata(
            [
                (
                    sch_id,
                    {
                        "source_episode_ids": new_sources,
                        "confidence": float(new_conf),
                        "updated_at": now,
                    },
                )
            ]
        )
        await service.episodic.update_metadata(
            [(eid, {"consolidated_into": sch_id}) for eid in ep_ids]
        )
        absorbed_ids.update(ep_ids)
        formed.append(
            {
                "id": sch_id,
                "summary": sch_text,
                "source_episode_ids": new_sources,
                "confidence": float(new_conf),
                "tags": list(sch_md.get("tags") or []),
                "absorbed": True,
            }
        )
    return absorbed_ids, formed


async def run(
    service: HippocampusService,
    *,
    session_id: str | None = None,
    n_clusters: int | None = None,
    min_cluster_size: int | None = None,
) -> list[dict[str, Any]]:
    flt: dict[str, Any] = {"consolidated_into": UNCONSOLIDATED}
    if session_id:
        flt["session_id"] = session_id

    candidates = await service.episodic.get_documents(flt)
    # Schema-eligibility gates: skip episodes too young or too rarely
    # retrieved. Mirrors the day-to-week ramp of HC->cortex handover.
    # Defaults are 0/0 (no gate) so existing call patterns work.
    cfg = service.config
    now_for_gate = now_seconds()
    if cfg.consolidate_min_age_seconds > 0 or cfg.consolidate_min_retrievals > 0:
        eligible: list[tuple[int, str, dict[str, Any]]] = []
        for ep_id, text, md in candidates:
            age = now_for_gate - float(md.get("encoded_at", now_for_gate))
            rc = int(md.get("retrieval_count", 0))
            if age < cfg.consolidate_min_age_seconds:
                continue
            if rc < cfg.consolidate_min_retrievals:
                continue
            eligible.append((ep_id, text, md))
        candidates = eligible
    floor = min_cluster_size or cfg.consolidate_min_cluster
    if len(candidates) < floor:
        return []

    now = now_seconds()
    formed: list[dict[str, Any]] = []

    # ---- Stage 1: replay-driven absorption into existing schemas ----
    absorbed_ids, absorbed_schemas = await _absorb(service, candidates, now=now)
    formed.extend(absorbed_schemas)

    # ---- Stage 2: K-means on the residual ----
    remaining = [c for c in candidates if int(c[0]) not in absorbed_ids]
    if len(remaining) < floor:
        return formed

    n = n_clusters or max(1, int(math.sqrt(len(remaining))))
    n = max(1, min(n, len(remaining)))
    try:
        result = await service.episodic.cluster(
            n_clusters=n,
            algorithm="minibatch_kmeans",
            filter=flt,  # absorbed episodes are no longer UNCONSOLIDATED
            random_state=0,
        )
    except (ValueError, ImportError):
        return formed

    groups: dict[int, list[int]] = {}
    for label, ep_id in zip(result.labels, result.doc_ids, strict=True):
        groups.setdefault(int(label), []).append(int(ep_id))

    ep_lookup: dict[int, tuple[str, dict[str, Any]]] = {
        int(ep_id): (text, dict(md)) for ep_id, text, md in remaining
    }
    total = max(1, len(remaining))

    for ep_ids in groups.values():
        members = [ep_lookup[eid] for eid in ep_ids if eid in ep_lookup]
        if len(members) < floor:
            continue

        head_text, _head_md = max(members, key=lambda m: float(m[1].get("salience", 0.0)))
        tag_union = sorted(
            {t for _, md in members for t in (md.get("tags") or []) if isinstance(t, str)}
        )
        summary = build_gist(head_text, [m[0] for m in members])
        confidence = min(1.0, len(ep_ids) / total)
        sem_meta: dict[str, Any] = {
            "source_episode_ids": list(ep_ids),
            "tags": tag_union,
            "confidence": float(confidence),
            "created_at": now,
            "updated_at": now,
        }
        sem_ids = await service.semantic.add_texts(
            [summary],
            metadatas=[sem_meta],
            embeddings=await service.vectorise([summary]),
        )
        sem_id = int(sem_ids[0])
        await service.semantic.update_metadata([(sem_id, {"id": sem_id})])

        await service.episodic.update_metadata(
            [(eid, {"consolidated_into": sem_id}) for eid in ep_ids]
        )
        formed.append(
            {
                "id": sem_id,
                "summary": summary,
                "source_episode_ids": list(ep_ids),
                "confidence": float(confidence),
                "tags": tag_union,
                "absorbed": False,
            }
        )
    return formed


def build_gist(headline: str, texts: Sequence[str]) -> str:
    """Heuristic semantic gist (no LLM).

    First non-empty line of the highest-salience episode becomes the
    headline; deduplicated leads from sibling episodes follow as bullets.
    """

    def _clip(t: str) -> str:
        line = t.strip().splitlines()[0] if t.strip() else ""
        return line[:200]

    head = _clip(headline)
    leads = sorted({_clip(t) for t in texts if t.strip()} - {head})
    if not leads:
        return head
    bullets = "\n".join(f"- {line}" for line in leads if line)
    return f"{head}\n\nRelated:\n{bullets}" if bullets else head
