"""Episodic -> semantic consolidation.

Two-stage replay-driven consolidation, modelled on hippocampal
sharp-wave ripples (SWR) followed by neocortical schema integration:

* **Stage 1 — replay-biased absorption.** Score each unconsolidated
  episode by ``strength * error``, where ``error = d1 / (1 + beta * (d2 - d1))``
  is a top-2 schema-margin error signal (strength uses the same
  power-law activation that ``forget`` uses; ``d1`` and ``d2`` are
  cosine distances to the nearest and second-nearest schemas). The
  margin term distinguishes three regimes: *stable* (one schema clearly
  owns the episode, large margin -> low score), *perturbed* (top-2
  schemas disagree, small margin -> high score), and *novel + isolated*
  (no schema fits, full d1 weight survives). Process in descending
  score order: if an episode's nearest schema is within
  ``consolidate_absorb_distance``, *absorb* the episode by appending it
  to the schema's source list and bumping confidence — no new schema
  is created. This implements the Tse et al. (2007) fast-track for
  schema-fitting episodes and prevents representation drift across
  consolidation passes. Margin scoring is the algorithmic stand-in for
  the McClelland et al. error-driven replay prescription: replay budget
  is spent first on episodes the cortical model is most confused about.

* **Stage 2 — clustering of the residual.** Whatever was *not* absorbed
  is clustered (MiniBatch K-Means) into new semantic schemas — the
  classical CLS slow-pathway. Source episodes are marked
  ``consolidated_into=<schema_id>`` so ``forget`` knows their gist is
  preserved upstream.
"""

from __future__ import annotations

import asyncio
import math
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

from synara.core.errors import ValidationError

from ..service import UNCONSOLIDATED, now_seconds
from .forget import memory_strength

if TYPE_CHECKING:  # pragma: no cover - typing only
    from ..config import MemoryConfig
    from ..port import MemoryServicePort as MemoryService


def _replay_score(
    md: dict[str, Any],
    *,
    d_near: float,
    margin: float,
    beta: float,
    now: float,
    d: float,
) -> float:
    """Replay-sampling weight: strength * schema-margin error (floored).

    d_near: cosine distance to nearest schema (1.0 = novel, 0.0 = identical).
    margin: ``d2 - d1`` gap to second-nearest schema; 0 when only one
        schema exists, in which case the score reduces to the legacy
        ``strength * d_near``.
    beta:   non-negative margin sensitivity. 0 disables the perturbation
        amplification; larger values suppress stable (high-margin)
        episodes more aggressively relative to perturbed ones.

    Biases replay toward salient, recent, schema-boundary-confused
    episodes — episodes the cortex would mispredict most.
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
    error = float(d_near) / (1.0 + max(0.0, float(beta)) * max(0.0, float(margin)))
    return max(strength * error, 1e-9)


def _schema_confidence(n_sources: int, full_at: int) -> float:
    """Saturating confidence from a schema's source-episode count.

    Single source of truth so the absorption and clustering paths cannot
    drift apart: identical ``len(source_episode_ids)`` => identical
    confidence, independent of the consolidation pass shape.
    """
    return min(1.0, max(0, n_sources) / max(1, full_at))


async def _nearest_schema(service: MemoryService, text: str) -> tuple[int, float, float] | None:
    # Caller is responsible for the upfront ``service.semantic.count()``
    # check; skipping it here saves an O(N) DB roundtrip across the
    # _absorb candidate loop.
    # Route through ``query_arg`` so the configured embed_fn produces the
    # query vector — using raw text would invoke simplevecdb's bundled
    # embedder, which can mismatch the stored vector dimension.
    # k=2 so the caller can compute the schema margin (d2 - d1) used by
    # the perturbation-amplified replay score. With only one schema we
    # fall back to d2 = d1 -> margin = 0, which makes the score reduce
    # to the legacy ``strength * d1``.
    q = await service.query_arg(text)
    try:
        hits = await service.semantic.similarity_search(q, k=2)
    except (ValueError, RuntimeError):
        return None
    if not hits:
        return None
    sch_doc, d1 = hits[0]
    sch_id = int(sch_doc.metadata.get("id", -1))
    if sch_id < 0:
        return None
    d2 = float(hits[1][1]) if len(hits) > 1 else float(d1)
    return sch_id, float(d1), d2


async def _absorb(
    service: MemoryService,
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
    beta = service.config.schema_margin_beta

    # Each _nearest_schema call (embed + vector search) is independent,
    # so fan them out concurrently instead of blocking the event loop on
    # up to ``max_scan`` sequential round-trips inside the reactor.
    nearests = await asyncio.gather(*(_nearest_schema(service, text) for _, text, _ in candidates))
    scored: list[tuple[float, int, str, int, float]] = []
    for (ep_id, text, md), nearest in zip(candidates, nearests, strict=True):
        if nearest is None:
            continue
        sch_id, d1, d2 = nearest
        d_near = max(0.0, d1)  # cosine distance in [0, 2]
        margin = max(0.0, d2 - d1)  # 0 when only one schema exists
        score = _replay_score(md, d_near=d_near, margin=margin, beta=beta, now=now, d=d)
        scored.append((score, int(ep_id), text, sch_id, d1))

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
        new_conf = _schema_confidence(
            len(new_sources), service.config.consolidate_confidence_full_at
        )
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


def _cap_candidates(
    cfg: MemoryConfig,
    candidates: list[tuple[int, str, dict[str, Any]]],
) -> list[tuple[int, str, dict[str, Any]]]:
    """Bound peak memory + clustering cost on unbounded episode growth.

    Keeps the most-retrieved episodes — they carry the strongest
    consolidation signal. Cap of 0 disables the limit.
    """
    cap = cfg.consolidate_max_candidates
    if not cap or len(candidates) <= cap:
        return candidates
    return sorted(
        candidates,
        key=lambda c: int(c[2].get("retrieval_count", 0)),
        reverse=True,
    )[:cap]


def _validate_run_inputs(
    service: MemoryService,
    *,
    session_id: str | None,
    n_clusters: int | None,
    min_cluster_size: int | None,
) -> None:
    if n_clusters is not None and n_clusters <= 0:
        raise ValidationError("n_clusters must be positive when provided")
    if min_cluster_size is not None and min_cluster_size <= 0:
        raise ValidationError("min_cluster_size must be positive when provided")
    cap = service.config.max_session_id_chars
    if session_id is not None and cap and len(session_id) > cap:
        raise ValidationError(f"session_id exceeds max_session_id_chars ({cap})")


async def run(
    service: MemoryService,
    *,
    session_id: str | None = None,
    n_clusters: int | None = None,
    min_cluster_size: int | None = None,
) -> list[dict[str, Any]]:
    _validate_run_inputs(
        service,
        session_id=session_id,
        n_clusters=n_clusters,
        min_cluster_size=min_cluster_size,
    )
    flt: dict[str, Any] = {"consolidated_into": UNCONSOLIDATED}
    if session_id:
        flt["session_id"] = session_id

    # Promote any reconsolidation-buffered vector updates into HNSW so
    # subsequent searches see the drifted embeddings. Cheap no-op when
    # the pending buffer is empty.
    flushed = await service.episodic.flush_pending()
    if flushed > 0:
        await service.episodic.rebuild_if_needed()

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
    candidates = _cap_candidates(cfg, candidates)
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
    n = max(1, min(n, len(remaining), cfg.consolidate_max_n_clusters or n))
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

    for ep_ids in groups.values():
        members = [ep_lookup[eid] for eid in ep_ids if eid in ep_lookup]
        if len(members) < floor:
            continue

        head_text, _head_md = max(members, key=lambda m: float(m[1].get("salience", 0.0)))
        tag_union = sorted(
            {t for _, md in members for t in (md.get("tags") or []) if isinstance(t, str)}
        )
        summary = build_gist(head_text, [m[0] for m in members])
        confidence = _schema_confidence(len(ep_ids), service.config.consolidate_confidence_full_at)
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
