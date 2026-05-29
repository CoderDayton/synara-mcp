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

* **Stage 2 — clustering + schema-level dedup.** Whatever was *not*
  absorbed is clustered (MiniBatch K-Means) into candidate new schemas
  — the classical CLS slow-pathway. Before each candidate is written,
  its gist is checked against existing schema centroids: if the nearest
  is within ``consolidate_schema_merge_distance``, the cluster's
  episodes are merged into that existing schema instead of forking a
  duplicate. This mirrors vmPFC's schema-comparator role (Gilboa &
  Marlatte 2017, TiCS) and prevents the parallel-duplicate failure mode
  that emerges when the reactor fires consolidate repeatedly under
  realistic cadence. Only genuinely novel clusters (sim < ~0.75 to any
  existing schema) form fresh entries. Source episodes are marked
  ``consolidated_into=<schema_id>`` so ``forget`` knows their gist is
  preserved upstream.

The merge threshold is intentionally one-sided: above it, cluster
merges; below, fresh schema. The neuroscience literature (Sinclair &
Barense 2019, Yassa & Stark 2011) supports a middle "reconsolidate"
zone with prediction-error-gated forking after repeated mismatches;
add that once we have evidence the simple two-zone version under- or
over-absorbs at our embedding density.

Ontology hygiene knobs (all optional, all default off / back-compat):

* ``consolidate_min_recurrence`` (effective floor 1) — the schema-
  candidate buffer is the SOLE promotion path. A K-Means cluster's
  gist is parked and only promoted to a durable schema once it
  recurs ``consolidate_min_recurrence`` times. Setting it to 1
  fast-promotes on first park (still through the buffer).
* ``consolidate_min_hit_sessions`` — require hits to come from N
  distinct ``session_id`` values before promoting. Defends against an
  in-session loop minting schema from same-context repetition.
* ``consolidate_min_hit_epochs`` — require hits to span N distinct
  consolidate-pass epochs. Defends against same-pass / dream-replay
  inflation of the hit count.
* ``consolidate_min_schema_size`` — minimum source-episode count at
  promotion. Stops 2-episode microtrends from calcifying.
* ``consolidate_candidate_hit_decay`` — per-tick power-law decay on
  parked-candidate hits. Mirrors episodic forgetting so stale
  candidates cannot snipe promotion on a single re-occurrence.
* ``consolidate_min_promotion_confidence`` — reject promotion when
  derived confidence is below the floor.
* ``forget_schema_unused_seconds`` — cold-schema eviction in
  ``forget.run``: semantic schemas whose ``last_hit_at`` is older than
  this are deleted. ``last_hit_at`` is set at schema creation, bumped
  on every absorb-merge, and bumped on every ``recall_semantic_memory``
  hit (only when the knob is enabled, to keep recall write-free
  otherwise).
"""

from __future__ import annotations

import asyncio
import contextlib
import math
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

from synara.core.errors import ValidationError

from ..service import UNCONSOLIDATED, now_seconds
from .forget import _DEFAULT_SALIENCE, memory_strength

if TYPE_CHECKING:  # pragma: no cover - typing only
    from ..config import MemoryConfig
    from ..port import MemoryServicePort as MemoryService

# Cap on concurrent embed+search round-trips during the absorption stage
# so a large candidate batch cannot swamp the embedding backend / DB pool.
_ABSORB_MAX_CONCURRENCY = 16


async def _age_schema_candidates(service: MemoryService) -> None:
    """Tick every parked candidate's age forward, decay hits, and evict expired rows.

    Called once per ``run()`` invocation so each consolidate pass is
    one tick. Candidates whose gist has not recurred within
    ``consolidate_candidate_max_age`` ticks are dropped; their episodes
    remain UNCONSOLIDATED and re-enter the next pass through K-Means.
    When ``consolidate_candidate_hit_decay > 0`` the hit count also
    decays each tick (floor of 1) so stale candidates cannot snipe
    promotion on one re-occurrence. Inert only when
    ``consolidate_candidate_max_age <= 0`` AND decay is off.
    """
    cfg = service.config
    if cfg.consolidate_candidate_max_age <= 0 and cfg.consolidate_candidate_hit_decay <= 0.0:
        return
    if await service.schema_candidates.count() == 0:
        return
    max_age = cfg.consolidate_candidate_max_age
    decay = max(0.0, min(1.0, cfg.consolidate_candidate_hit_decay))
    rows = await service.schema_candidates.get_documents(filter_dict=None)
    updates: list[tuple[int, dict[str, Any]]] = []
    evict: list[int] = []
    for doc_id, _text, md in rows:
        new_age = int(md.get("age", 0)) + 1
        if max_age > 0 and new_age > max_age:
            evict.append(int(doc_id))
            continue
        patch: dict[str, Any] = {"age": new_age}
        if decay > 0.0:
            cur_hits = int(md.get("hits", 0))
            patch["hits"] = max(1, int(cur_hits * (1.0 - decay)))
        updates.append((int(doc_id), patch))
    if updates:
        await service.schema_candidates.update_metadata(updates)
    if evict:
        await service.schema_candidates.delete_by_ids(evict)


async def _v2_promote_or_park(  # noqa: PLR0912, PLR0915 -- branch-per-gate aids audit
    service: MemoryService,
    *,
    summary: str,
    summary_emb: list[float],
    merge_dist: float,
    min_recurrence: int,
    session_id: str | None,
    epoch: int,
) -> bool:
    """Candidate-to-promotion gate (the sole promotion path).

    Returns ``True`` when the caller should fall through to fresh-schema
    creation (the candidate just crossed all diversity gates and has
    been removed from the buffer). Returns ``False`` when the cluster
    was instead parked as a new candidate or had its hit count bumped
    short of promotion -- the caller skips schema creation.

    Hits, observed sessions, and observed epochs are tracked per
    candidate so optional ``consolidate_min_hit_sessions`` /
    ``consolidate_min_hit_epochs`` gates can require diversity before
    promotion. Operates on ``service.schema_candidates`` so all state
    survives restarts.
    """
    cfg = service.config
    counters = service._hygiene_counters
    nearest_id: int | None = None
    if await service.schema_candidates.count() > 0:
        try:
            hits = await service.schema_candidates.similarity_search(summary_emb, k=1)
        except (ValueError, RuntimeError):
            hits = []
        if hits:
            doc, dist = hits[0]
            if float(dist) <= merge_dist:
                candidate_id = int(doc.metadata.get("id", -1))
                if candidate_id >= 0:
                    nearest_id = candidate_id
    if nearest_id is None:
        # New candidate. When the recurrence threshold is met by a
        # single hit AND both diversity knobs are at their off defaults,
        # promote immediately without writing to the buffer -- this
        # preserves legacy "one pass = one schema" behaviour for callers
        # that explicitly opt in via min_recurrence=1, while still
        # routing through the candidate-buffer code path. Diversity is keyed on the
        # *knob*, not on what this single hit happens to carry, so
        # anonymous (session_id=None) callers are not silently denied
        # fast-promote. The caller increments ``schemas_promoted``.
        if (
            min_recurrence <= 1
            and cfg.consolidate_min_hit_sessions <= 1
            and cfg.consolidate_min_hit_epochs <= 1
        ):
            return True
        sess_seed = [session_id] if session_id else []
        new_ids = await service.schema_candidates.add_texts(
            [summary],
            metadatas=[
                {
                    "hits": 1,
                    "age": 0,
                    "sessions": sess_seed,
                    "epochs": [int(epoch)],
                }
            ],
            embeddings=[summary_emb],
        )
        if new_ids:
            new_id = int(new_ids[0])
            # Roll back the parked candidate if its id-patch fails, so the
            # ``candidates_parked`` counter only ever counts a fully-formed,
            # addressable candidate row (no orphan).
            try:
                await service.schema_candidates.update_metadata([(new_id, {"id": new_id})])
            except Exception:
                with contextlib.suppress(Exception):
                    await service.schema_candidates.delete_by_ids([new_id])
                raise
        counters["candidates_parked"] = counters.get("candidates_parked", 0) + 1
        return False
    # Existing candidate matched: bump hits and either promote or wait.
    existing = await service.schema_candidates.get_documents({"id": nearest_id}, limit=1)
    if not existing:
        # Race: row disappeared between search and read. Treat as miss
        # to avoid creating an orphan; the next consolidate pass will
        # re-park if the cluster recurs.
        return False
    _, _, md = existing[0]
    new_hits = int(md.get("hits", 0)) + 1
    prior_sessions = [str(s) for s in (md.get("sessions") or []) if isinstance(s, str)]
    prior_epochs = [int(e) for e in (md.get("epochs") or []) if isinstance(e, int)]
    if session_id and session_id not in prior_sessions:
        prior_sessions = [*prior_sessions, session_id]
    if int(epoch) not in prior_epochs:
        prior_epochs = [*prior_epochs, int(epoch)]
    distinct_sessions = len(prior_sessions)
    distinct_epochs = len(prior_epochs)
    patch: dict[str, Any] = {
        "hits": new_hits,
        "age": 0,  # alive on a hit
        "sessions": prior_sessions,
        "epochs": prior_epochs,
    }
    # Diversity gates are inert at their off defaults (knob <= 1) -- the
    # candidate's observed sessions/epochs are recorded for telemetry
    # but do NOT block promotion. This preserves back-compat for the
    # common case where ``_infer_cluster_session`` returned None
    # (cluster has tied session contributions) and the prior_sessions
    # list is empty.
    promote = new_hits >= min_recurrence
    need_sess = cfg.consolidate_min_hit_sessions
    need_epoch = cfg.consolidate_min_hit_epochs
    if need_sess > 1 and distinct_sessions < need_sess:
        promote = False
    if need_epoch > 1 and distinct_epochs < need_epoch:
        promote = False
    if not promote:
        await service.schema_candidates.update_metadata([(nearest_id, patch)])
        # Telemetry: bucket why we did NOT promote.
        if new_hits >= min_recurrence:
            if (
                cfg.consolidate_min_hit_sessions > 1
                and distinct_sessions < cfg.consolidate_min_hit_sessions
            ):
                counters["candidates_rejected_session_diversity"] = (
                    counters.get("candidates_rejected_session_diversity", 0) + 1
                )
            if (
                cfg.consolidate_min_hit_epochs > 1
                and distinct_epochs < cfg.consolidate_min_hit_epochs
            ):
                counters["candidates_rejected_epoch_diversity"] = (
                    counters.get("candidates_rejected_epoch_diversity", 0) + 1
                )
        return False
    # Crossed every diversity gate: drop the candidate and let the
    # caller create the durable schema row.
    await service.schema_candidates.delete_by_ids([nearest_id])
    return True


def _infer_cluster_session(
    members: Sequence[tuple[str, dict[str, Any]]],
) -> str | None:
    """Plurality session_id among cluster members, or None on a tie / absence.

    Used by the candidate gate to record which session contributed
    this hit, enabling cross-session diversity gating without making
    the caller pass session_id through every cluster. Ties return
    ``None`` so the gate does not credit a contested cluster to an
    arbitrary winner (which would depend on dict insertion order).
    """
    counts: dict[str, int] = {}
    for _text, md in members:
        sid = md.get("session_id")
        if isinstance(sid, str) and sid:
            counts[sid] = counts.get(sid, 0) + 1
    if not counts:
        return None
    top_sid, top_count = max(counts.items(), key=lambda kv: kv[1])
    # Tie => no plurality winner; refuse to credit the cluster.
    if sum(1 for c in counts.values() if c == top_count) > 1:
        return None
    return top_sid


def _cosine_distance(a: Sequence[float], b: Sequence[float]) -> float:
    """Cosine distance for two equal-length, non-zero vectors.

    Used by the candidate gate to compare an in-memory gist embedding
    against buffered candidates without a DB round-trip. Returns 1.0 on
    a length mismatch or all-zero vector so the caller treats them as
    non-matching rather than crashing the consolidate pass.
    """
    if len(a) != len(b):
        return 1.0
    dot = 0.0
    na = 0.0
    nb = 0.0
    for x, y in zip(a, b, strict=True):
        dot += x * y
        na += x * x
        nb += y * y
    if na <= 0.0 or nb <= 0.0:
        return 1.0
    return 1.0 - dot / math.sqrt(na * nb)


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
        salience=float(md.get("salience", _DEFAULT_SALIENCE)),
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
    # up to ``max_scan`` sequential round-trips inside the reactor. Bound
    # the in-flight count with a semaphore: an unbounded gather over
    # ``max_scan`` (~5000) candidates would launch thousands of embed +
    # search tasks at once and can swamp the embedding backend / DB pool.
    sem = asyncio.Semaphore(_ABSORB_MAX_CONCURRENCY)

    async def _bounded_nearest(text: str) -> tuple[int, float, float] | None:
        async with sem:
            return await _nearest_schema(service, text)

    nearests = await asyncio.gather(*(_bounded_nearest(text) for _, text, _ in candidates))
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
        merged = await _merge_into_schema(service, sch_id=sch_id, ep_ids=ep_ids, now=now)
        if merged is None:
            continue
        absorbed_ids.update(ep_ids)
        formed.append(merged)
    return absorbed_ids, formed


async def _merge_into_schema(
    service: MemoryService,
    *,
    sch_id: int,
    ep_ids: Sequence[int],
    now: float,
) -> dict[str, Any] | None:
    """Append ``ep_ids`` to an existing schema's source list.

    Shared by Stage 1's per-episode absorption and Stage 2's per-cluster
    merge — keeps the metadata-update path identical so confidence and
    timestamps cannot drift between the two routes. Returns ``None`` if
    the schema vanished mid-pass or the merge would be a no-op.
    """
    existing = await service.semantic.get_documents({"id": sch_id}, limit=1)
    if not existing:
        return None
    _, sch_text, sch_md = existing[0]
    prior = list(sch_md.get("source_episode_ids") or [])
    new_sources = sorted(set(prior) | {int(e) for e in ep_ids})
    if len(new_sources) == len(prior):
        return None
    new_conf = _schema_confidence(len(new_sources), service.config.consolidate_confidence_full_at)
    await service.semantic.update_metadata(
        [
            (
                sch_id,
                {
                    "source_episode_ids": new_sources,
                    "confidence": float(new_conf),
                    "updated_at": now,
                    "last_hit_at": now,
                },
            )
        ]
    )
    await service.episodic.update_metadata(
        [(int(eid), {"consolidated_into": sch_id}) for eid in ep_ids]
    )
    return {
        "id": sch_id,
        "summary": sch_text,
        "source_episode_ids": new_sources,
        "confidence": float(new_conf),
        "tags": list(sch_md.get("tags") or []),
        "absorbed": True,
    }


def _apply_eligibility_gates(
    cfg: MemoryConfig,
    candidates: list[tuple[int, str, dict[str, Any]]],
) -> list[tuple[int, str, dict[str, Any]]]:
    """Skip episodes too young, too rarely retrieved, or too faint to be schema fodder.

    Mirrors the day-to-week ramp of HC->cortex handover. The salience
    floor prevents low-salience noise from forming durable clusters once
    given enough re-runs; missing-salience defaults to the neutral base
    (matches ``forget._DEFAULT_SALIENCE``) so legacy episodes are not
    prune-on-sight.
    """
    if (
        cfg.consolidate_min_age_seconds <= 0
        and cfg.consolidate_min_retrievals <= 0
        and cfg.consolidate_min_salience <= 0
    ):
        return candidates
    now_for_gate = now_seconds()
    eligible: list[tuple[int, str, dict[str, Any]]] = []
    for ep_id, text, md in candidates:
        age = now_for_gate - float(md.get("encoded_at", now_for_gate))
        rc = int(md.get("retrieval_count", 0))
        sal = float(md.get("salience", _DEFAULT_SALIENCE))
        if age < cfg.consolidate_min_age_seconds:
            continue
        if rc < cfg.consolidate_min_retrievals:
            continue
        if sal < cfg.consolidate_min_salience:
            continue
        eligible.append((ep_id, text, md))
    return eligible


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
    cfg = service.config
    candidates = _apply_eligibility_gates(cfg, candidates)
    candidates = _cap_candidates(cfg, candidates)
    # Advance the consolidate-pass epoch before aging so the candidate
    # gate sees a monotonically increasing tick for each pass. Epoch
    # is the primitive that powers cross-epoch diversity gating.
    service._consolidate_epoch = int(service._consolidate_epoch) + 1
    await _age_schema_candidates(service)
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

    merge_dist = cfg.consolidate_schema_merge_distance
    have_existing_schemas = await service.semantic.count() > 0

    for ep_ids in groups.values():
        result = await _process_stage2_cluster(
            service,
            ep_ids=ep_ids,
            ep_lookup=ep_lookup,
            floor=floor,
            merge_dist=merge_dist,
            have_existing_schemas=have_existing_schemas,
            now=now,
        )
        if result is None:
            continue
        record, created_new = result
        formed.append(record)
        if created_new:
            have_existing_schemas = True
    return formed


async def _process_stage2_cluster(  # noqa: PLR0911 -- explicit branch-per-gate aids audit
    service: MemoryService,
    *,
    ep_ids: Sequence[int],
    ep_lookup: dict[int, tuple[str, dict[str, Any]]],
    floor: int,
    merge_dist: float,
    have_existing_schemas: bool,
    now: float,
) -> tuple[dict[str, Any], bool] | None:
    """One K-means cluster -> either a merge into an existing schema or
    a fresh schema row. Returns ``(record, created_new)`` or ``None`` if
    the cluster was sub-floor or its members are missing.
    """
    members = [ep_lookup[eid] for eid in ep_ids if eid in ep_lookup]
    if len(members) < floor:
        return None

    head_text, _head_md = max(members, key=lambda m: float(m[1].get("salience", _DEFAULT_SALIENCE)))
    tag_union = sorted(
        {t for _, md in members for t in (md.get("tags") or []) if isinstance(t, str)}
    )
    summary = build_gist(head_text, [m[0] for m in members])

    # Schema-level dedup: check the cluster gist against existing
    # schema centroids before forking a fresh one. Mirrors vmPFC's
    # schema-comparator role — if the cluster looks like an instance
    # of an already-known schema, route it there instead of creating
    # a parallel duplicate. Guarded on `have_existing_schemas` so the
    # first-ever consolidate pass skips the lookup cheaply.
    if have_existing_schemas and merge_dist > 0:
        nearest = await _nearest_schema(service, summary)
        if nearest is not None:
            sch_id, d1, _d2 = nearest
            if d1 <= merge_dist:
                merged = await _merge_into_schema(service, sch_id=sch_id, ep_ids=ep_ids, now=now)
                if merged is not None:
                    return merged, False

    # Compute the summary embedding once: needed both for the
    # candidate gate below and (if we promote) for the schema add_texts
    # call. If no embed_fn is configured the gate cannot run, so
    # promotion is rejected outright -- the caller skips the cluster
    # rather than minting an ungated schema.
    summary_emb_batch = await service.vectorise([summary])
    summary_emb: list[float] | None = list(summary_emb_batch[0]) if summary_emb_batch else None
    cfg = service.config
    counters = service._hygiene_counters
    if summary_emb is None or merge_dist <= 0:
        return None

    # Candidate-to-promotion gate (the sole promotion path). Operates
    # on a separate ``schema_candidates`` collection so production
    # recall paths never see pending gists, and so the buffer survives
    # process restarts.
    min_recurrence = max(1, int(cfg.consolidate_min_recurrence))
    cluster_session = _infer_cluster_session(members)
    promoted = await _v2_promote_or_park(
        service,
        summary=summary,
        summary_emb=summary_emb,
        merge_dist=merge_dist,
        min_recurrence=min_recurrence,
        session_id=cluster_session,
        epoch=int(service._consolidate_epoch),
    )
    if not promoted:
        return None

    # Min-source-episode floor: a cluster too small to be ontology gets
    # dropped after clearing the diversity gates.
    if cfg.consolidate_min_schema_size > 0 and len(ep_ids) < cfg.consolidate_min_schema_size:
        counters["candidates_rejected_size"] = counters.get("candidates_rejected_size", 0) + 1
        return None

    confidence = _schema_confidence(len(ep_ids), cfg.consolidate_confidence_full_at)
    if confidence < cfg.consolidate_min_promotion_confidence:
        counters["candidates_rejected_confidence"] = (
            counters.get("candidates_rejected_confidence", 0) + 1
        )
        return None
    sem_meta: dict[str, Any] = {
        "source_episode_ids": list(ep_ids),
        "tags": tag_union,
        "confidence": float(confidence),
        "created_at": now,
        "updated_at": now,
        "last_hit_at": now,
    }
    sem_ids = await service.semantic.add_texts(
        [summary],
        metadatas=[sem_meta],
        embeddings=[summary_emb] if summary_emb is not None else None,
    )
    sem_id = int(sem_ids[0])
    # The row exists but its metadata has no ``id`` yet. If the id-patch
    # fails, roll the insert back so we don't leave an un-addressable
    # orphan schema (callers and recall key on metadata ``id``).
    try:
        await service.semantic.update_metadata([(sem_id, {"id": sem_id})])
    except Exception:
        with contextlib.suppress(Exception):
            await service.semantic.delete_by_ids([sem_id])
        raise
    counters["schemas_promoted"] = counters.get("schemas_promoted", 0) + 1
    await service.episodic.update_metadata(
        [(int(eid), {"consolidated_into": sem_id}) for eid in ep_ids]
    )
    return {
        "id": sem_id,
        "summary": summary,
        "source_episode_ids": list(ep_ids),
        "confidence": float(confidence),
        "tags": tag_union,
        "absorbed": False,
    }, True


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
