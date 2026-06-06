"""Memory tunables.

Kept in its own module so the service core and the operation-specific
siblings (encode/recall/consolidate/forget/...) can each import the
config without pulling each other.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .memory_types import MemoryTypeRegistry

# Neutral salience base shared by the consolidate floor, the dream-replay
# floor, and the auto-salience default. Equal to the ``derive_salience``
# neutral base in ``amygdala/signals.py``: an episode carrying no salient
# signals sits exactly here, so these gates admit a neutrally-salient trace
# and reject only below-neutral noise. Kept as one constant so the invariant
# the three field comments used to assert by hand cannot silently drift.
NEUTRAL_SALIENCE_BASE: float = 0.3


@dataclass(frozen=True, slots=True)
class MemoryConfig:
    """Tunable parameters of the memory framework."""

    episodic_collection: str = "memory_episodic"
    semantic_collection: str = "memory_semantic"
    # Backing collection for the schema-candidate buffer (see
    # consolidate_min_recurrence). Always created; only populated when
    # the recurrence gate is active. Kept separate from the semantic
    # collection so production recall queries don't need a kind-filter.
    schema_candidate_collection: str = "memory_schema_candidates"
    # First-class embedding dimensionality. ``None`` (default) lets the
    # service probe the configured embedder once and cache the observed
    # dim. Explicit values are validated against the probe at first use
    # so a swapped-out embedder cannot silently mismatch downstream
    # shape assumptions (DG projector, reconsolidation drift, ...).
    embedding_dimension: int | None = None
    # Optional override for the memory-type registry. ``None`` (default)
    # builds the standard ``episodic``/``semantic`` registry from the
    # two ``*_collection`` fields above. Custom registries enable
    # additional kinds (procedural, conceptual, ...) without touching
    # service.py / encode.py / consolidate.py.
    memory_types: MemoryTypeRegistry | None = None
    # Per-request trace collection. When True, recall (and the consolidate
    # / forget / reflect paths that opt in) publish a RequestContext
    # spans list on a ContextVar; recall surfaces it back as
    # ``__trace__`` on the result list when callers need observability.
    # Default OFF — measured overhead is two ContextVar reads per op.
    tracing_enabled: bool = False
    # Optional signal-registry override for the encode path. ``None``
    # falls back to the hardcoded :data:`SALIENCE_WEIGHTS` table inside
    # ``primitives/signals.py``. Pass a custom registry to add a signal
    # without touching ``encode.py``.
    signal_registry: object | None = field(default=None, repr=False)
    # Cosine distance below this threshold counts as a duplicate within a
    # session — encode_episode bumps the existing record instead of inserting.
    dedup_distance: float = 0.05
    # Minimum stripped content length for embedding-based dedup to run.
    # Distinct short episodes crowd into the same cosine cone (at the
    # same similarity as true paraphrases), so neither cosine nor DG
    # can separate them; a false merge there is irreversible data loss.
    # Below this floor, dedup is skipped and the episode is always
    # stored. 0 disables the floor (always attempt dedup).
    min_dedup_chars: int = 8
    # Hard caps on untrusted MCP-tool input. Without these a client can
    # push a multi-MB ``content`` (silently truncated by the embedder's
    # max_seq_length, so the stored vector misrepresents the text) or a
    # huge ``tags`` list that bloats every recall/forget metadata scan.
    # 0 disables the respective cap.
    max_content_chars: int = 50_000
    max_tags: int = 64
    # Cap on each individual tag string. Unbounded tag values bloat
    # metadata scans and the durable row payload. 0 disables the cap.
    max_tag_chars: int = 256
    # Cap on caller-requested result count (recall / reflect / semantic
    # recall). Unbounded k lets a client force an arbitrarily large
    # vector-DB fetch + result allocation. 0 disables the cap.
    max_recall_k: int = 1_000
    # Cap on the session_id namespace string. session_id keys dedup, the
    # SR window, and metadata; an unbounded value bloats every scan and
    # the durable edge table. 0 disables the cap.
    max_session_id_chars: int = 1_024
    # Default per-hit content truncation for the ``recall_episodes`` tool.
    # Recall returns one dict per hit carrying the raw ``content``; with a
    # handful of full-length episodes the serialised result can exceed an
    # MCP client's tool-result token budget and be spilled to disk unread
    # (pure cost, zero context gained). The tool truncates each hit's
    # content to this many characters and flags ``truncated`` + the full
    # ``content_chars`` so a caller can re-fetch with ``full=true``. The
    # tool's ``max_chars`` argument overrides this; 0 disables truncation.
    recall_snippet_chars: int = 600
    # Power-law (Wickelgren/Wixted) decay exponent used by ``forget``:
    #   S(t) = salience * sum_k (1 + (t - t_k))^(-d)
    # Wixted & Ebbesen 1991 / 1997 fit individual episodic retention curves
    # with d typically in 0.1-0.5; the modal value for verbal episodic
    # material is ~0.15-0.25. Pooled-subject fits push d higher than the
    # individual-trace exponent (Kahana's caveat), so the prior 0.5 default
    # was on the over-aggressive edge. Larger d = faster decay.
    forget_d: float = 0.2
    # Cap on retained access-history timestamps per episode; older entries
    # are FIFO-evicted. Keeps storage bounded while preserving Anderson's
    # base-level activation shape for typical usage.
    access_history_cap: int = 32
    # Minimum cluster size that yields a semantic schema during consolidation.
    consolidate_min_cluster: int = 2
    # Hard cap on candidate episodes loaded into one consolidation pass.
    # Unbounded accumulation would load every unconsolidated episode into
    # memory and into sklearn k-means at once (OOM risk). When exceeded,
    # the highest-retrieval-count candidates are kept. 0 disables the cap.
    consolidate_max_candidates: int = 5_000
    # Hard cap on the k-means cluster count, bounding clustering cost
    # regardless of candidate volume. 0 disables the cap.
    consolidate_max_n_clusters: int = 256
    # Cosine distance below which an unconsolidated episode is *absorbed*
    # into the nearest existing schema (instead of contributing to a new
    # cluster). Implements schema-fitting fast-track consolidation
    # (Tse et al. 2007). Smaller = stricter fit required.
    consolidate_absorb_distance: float = 0.4
    # Schema-level (cluster-centroid) merge distance for Stage-2 dedup.
    # After K-means clusters the residual unconsolidated tail, each new
    # cluster's gist is checked against existing schema centroids; if the
    # nearest one is within this cosine distance, the cluster's episodes
    # are merged into that schema instead of forming a parallel duplicate.
    # Anchor: human MST pattern-separation knee at ~60-70% feature overlap
    # (Yassa & Stark 2011, Trends Neurosci; Lacy/Yassa/Stark 2011, Learn
    # Mem); biased slightly tighter (0.25 distance ~ 0.75 cosine sim) to
    # compensate for higher density in sentence-transformer space. Without
    # this gate, every consolidate fire produces fresh duplicates of the
    # same underlying topics ~25x bloat under daily reactor cadence.
    consolidate_schema_merge_distance: float = 0.25
    # Candidate-to-promotion gate (the sole promotion path). A stage-2
    # K-Means cluster whose gist does not match an existing schema does
    # NOT immediately become a durable schema; instead its gist embedding
    # enters the persistent ``schema_candidates`` collection. A
    # subsequent consolidate pass whose new cluster gist lands within
    # ``consolidate_schema_merge_distance`` of that candidate bumps its
    # hit count; promotion to a durable schema only happens once
    # hits >= consolidate_min_recurrence. Effective floor is 1 at
    # runtime (``max(1, ...)`` inside the stage-2 path): callers may
    # opt into 1 to get fast-promote-on-park (one pass = one schema,
    # still routed through the candidate buffer).
    consolidate_min_recurrence: int = 2
    # Candidate buffer TTL: a pending candidate that does not recur
    # within this many consolidate passes is dropped (the episodes it
    # would have promoted remain UNCONSOLIDATED and re-enter the next
    # pass). Default 5 gives real-embedder workloads ~5 opportunities
    # for the gist to recur within consolidate_schema_merge_distance,
    # which empirically clears the bar on pass 2-3; the cap stops the
    # buffer from growing unboundedly with one-off K-Means noise. 0
    # disables expiry (candidates live forever).
    consolidate_candidate_max_age: int = 5
    # Cross-session diversity required for promotion. When >= 2, a
    # candidate's accumulated hits must come from at least this many
    # distinct ``session_id`` values before it can be promoted. Defends
    # against same-loop in-session repetition minting durable schema.
    # Default 1 = off (back-compat).
    consolidate_min_hit_sessions: int = 1
    # Cross-epoch diversity required for promotion. When >= 2, hits must
    # span at least this many distinct consolidate-pass epochs. Defends
    # against dream-replay or same-pass inflation. Default 1 = off.
    consolidate_min_hit_epochs: int = 1
    # Minimum source-episode count required at fresh-schema promotion
    # time. A K-Means cluster smaller than this is rejected even if it
    # cleared the candidate gate. 0 = off (back-compat); set to 3-4 to
    # prevent microtrends from calcifying.
    consolidate_min_schema_size: int = 0
    # Per-tick power-law decay applied to a parked candidate's hit count
    # in ``_age_schema_candidates``: ``hits = max(1, floor(hits * (1 -
    # decay)))``. Mirrors episodic forgetting on the candidate buffer so
    # stale candidates cannot snipe promotion on one re-occurrence.
    # 0.0 = off (back-compat).
    consolidate_candidate_hit_decay: float = 0.0
    # Minimum confidence required at fresh-schema promotion. The cluster
    # is rejected if its computed confidence (see
    # ``consolidate_confidence_full_at``) is below this floor. 0.0 = off.
    consolidate_min_promotion_confidence: float = 0.0
    # Cold-schema eviction in the forget pass: semantic schemas whose
    # ``last_accessed`` is older than this many seconds are deleted. A
    # schema's ``last_accessed`` is set at creation and bumped by every
    # absorb-merge and every ``recall_semantic_memory`` hit (legacy schemas
    # predating field unification carry ``last_hit_at``; the timestamps
    # helper reads either). 0.0 = off
    # (no schema lifecycle pruning); set to e.g. 60*60*24*30 (30 days)
    # for a slow ontology garbage-collector.
    forget_schema_unused_seconds: float = 0.0
    # CA3 iterative pattern completion (modern Hopfield, Ramsauer 2020).
    # ``recall_completion_iters > 0`` enables the iteration; each step
    # softmax-recombines stored vectors at inverse-temperature
    # ``recall_completion_beta`` and anchors the new query a fraction
    # ``recall_completion_anchor`` toward the original. 0 = single-pass
    # k-NN (the legacy behaviour).
    recall_completion_iters: int = 0
    recall_completion_beta: float = 8.0
    recall_completion_anchor: float = 0.6
    # DG-style pattern separation (random expansive projection + k-WTA).
    # When enabled, ``encode_episode`` checks Jaccard overlap of sparse
    # codes against the top cosine candidates instead of the brittle
    # cosine-distance threshold. Stored episodes carry their support set
    # in ``dg_support`` metadata.
    dg_pattern_separation: bool = True
    dg_expansion: int = 4
    dg_sparsity: float = 0.05
    dg_jaccard_threshold: float = 0.5
    dg_dedup_candidates: int = 4
    dg_seed: int = 0
    # Theta-segmented intra-episode encoding (Lisman & Jensen 2013).
    # Content longer than ``theta_segment_max_chars`` is split into
    # ``<= theta_segment_max_items`` ordered sub-records that share an
    # ``episode_group_id``. Each sub-record carries ``position_in_episode``
    # and ``segment_count`` metadata so callers can reconstruct the
    # original ordering. Set ``theta_segment_max_chars=0`` to disable.
    theta_segment_max_chars: int = 1024
    theta_segment_max_items: int = 7
    # Recall folds the sibling segments of one theta-segmented episode
    # into a single best-ranked hit (so fragments of one memory don't eat
    # several ``k`` slots); the collapsed hit carries ``group_id`` /
    # ``segment_count`` so the caller can fetch the whole via
    # ``get_episode``. Set false to return each segment as its own hit.
    recall_collapse_groups: bool = True
    # Master switch over the three-stage relevance gate below (dynamic
    # ceiling + elbow + gap cut). False disables all of them in one knob;
    # recall then returns the raw top-k by rank, ungated.
    # Note: the gate cuts on *raw cosine distance*, not on the SR-/
    # spreading-activation-adjusted rank. It never re-orders, but it can
    # drop a relationally-linked neighbour that SR boosted to the top yet
    # whose cosine distance is high — i.e. the gate can suppress an
    # SR-surfaced-but-cosine-far hit. Disable it on paths that test or
    # rely on spreading activation surfacing cosine-distant neighbours.
    recall_relevance_gate: bool = True
    # Adaptive relevance gate (normalised Kneedle knee detection; Satopaa
    # et al., 2011). After ranking, recall fits the sorted cosine distances
    # of each source (episodic / semantic gated independently) and drops the
    # low-relevance "plateau" tail at the knee — so a query with one strong
    # hit returns that hit instead of padding to ``k`` with noise, and an
    # all-weak query can return nothing. ``sensitivity`` is the minimum
    # normalised knee prominence in [0, 1] needed to accept a cut (higher =
    # only sharper elbows); ``min_candidates`` skips the gate below a count
    # too small to estimate a knee; ``min_spread`` skips it when every
    # candidate sits within this cosine-distance band (no structure to cut).
    recall_elbow_cutoff: bool = True
    recall_elbow_sensitivity: float = 0.12
    recall_elbow_min_candidates: int = 3
    recall_elbow_min_spread: float = 0.05
    # Dynamic relevance ceiling: an absolute distance cutoff *derived per
    # recall* from the candidate distribution, so it tracks the embedding
    # model's own distance scale instead of a hardcoded value. The far end
    # of the over-fetched candidates approximates this model's "unrelated"
    # distance; we take its ``quantile`` (p90) as a reference ``d_ref`` and
    # keep hits with ``distance <= alpha * d_ref``. This covers the all-weak /
    # no-knee case the relative elbow can't, without the model-dependence of a
    # fixed threshold. There is no floor: an off-topic query whose nearest hit
    # is itself beyond the ceiling empties rather than returning its least-bad
    # hit. The one exception is ``standout_gap``: when the nearest hit is
    # separated from the next by at least that gap, it is a real match in an
    # otherwise-far cloud (a relevance cliff right after it) and is kept even
    # above the ceiling — so a lone genuine hit is not lost, while a uniform
    # off-topic cloud (no such gap) still empties. A gap is scale-robust like
    # ``recall_gap_cut``, but smaller here since it acts on a single standout.
    # Per-source (episodic / semantic) like the elbow. ``alpha`` <= 0 disables;
    # ``min_candidates`` skips small samples (also sparing tiny synthetic
    # corpora); ``standout_gap`` <= 0 disables the exception (pure
    # empty-on-far). Applies to both episodic and semantic hits.
    recall_max_distance_alpha: float = 0.8
    recall_max_distance_quantile: float = 0.9
    recall_max_distance_min_candidates: int = 4
    recall_max_distance_standout_gap: float = 0.15
    # Cross-source relevance-cliff cut, applied last to the pooled returned
    # rows: walking the distances ascending, the first consecutive jump of at
    # least this size ends the relevant run and the tail is dropped. Catches a
    # head-vs-tail cliff straddling the episodic and semantic legs, which the
    # per-source ceiling/elbow can't see. A gap (a difference) is more
    # scale-robust than an absolute position — a compressed embedder never
    # produces one, so this fails safe. Set <= 0 to disable.
    recall_gap_cut: float = 0.35
    # Hebbian strength of the within-episode sibling chain: consecutive
    # segments are reinforced at encode (and re-bonded on ``get_episode``)
    # so spreading activation can resurface the whole episode. 0 disables.
    # Requires ``sr_enabled``: bonds are written only when the SR/plasticity
    # layer is active, so this is a no-op when SR is off.
    segment_assoc_score: float = 0.8
    # Successor representation (Stachenfeld 2017). When enabled, recall
    # blends a temporal co-occurrence boost ``M[i*, j]`` into the rank
    # score, where ``i*`` is the best-cosine episodic anchor. ``omega``
    # ramps from 0 to ``sr_omega_max`` after enough edges accumulate.
    sr_enabled: bool = True
    sr_gamma: float = 0.7
    sr_alpha: float = 0.1
    sr_window_seconds: float = 60.0
    sr_omega_max: float = 0.3
    # Cold-start gate: omega ramps linearly until the edge population
    # crosses ``sr_cold_start_ratio * episode_count``, then plateaus at
    # ``sr_omega_max``. 1.0 = needs as many edges as episodes before
    # full plateau. Lower = trust SR sooner.
    sr_cold_start_ratio: float = 1.0

    # ----- Neuroplasticity timescales -----
    # Master switch for the event-bus reactor + plasticity tracking.
    # When True the service emits ``InteractionEvent`` records per op
    # and runs the reactor (auto-consolidate / auto-dream) inside the
    # originating call. Plasticity bookkeeping (E-LTP/L-LTP/habits) runs
    # whenever this is True; behavioural effects are individually gated
    # below. Reconsolidation drift and spreading activation are ON by
    # default; surprise salience and the age/retrieval gates default to
    # OFF so they never alter observed behaviour without explicit opt-in.
    self_learning_enabled: bool = True

    # ``time_compression`` shifts the felt pace of SLOW processes
    # (forgetting, schema age, LTD) without retuning each constant.
    # 1.0 = real-time; 24.0 (default) = 1 real day per app hour.
    # FAST-plasticity fields (E-LTP, reconsolidation window) deliberately
    # bypass compression and use real wall-clock so a bursty 100-turn
    # session does not race past transient plasticity in seconds.
    # Habit / L-LTP triggers use interaction count and are
    # compression-invariant.
    time_compression: float = 24.0

    # Early-LTP: a recall briefly amplifies the edge to its co-activated
    # neighbours; the bonus decays with this tau (REAL wall-clock,
    # uncompressed) unless reinforced. E-LTP literature: 1-3h, midpoint ~2h.
    e_ltp_decay_seconds: float = 7200.0
    # Late-LTP: an edge reinforced ``l_ltp_threshold_hits`` times inside
    # the E-LTP window flips to durable (slow-decay) state. Real L-LTP
    # needs protein synthesis and persists for days.
    l_ltp_threshold_hits: int = 3

    # Reconsolidation window (REAL wall-clock, uncompressed): after a
    # successful recall the episode is labile this long; further recalls
    # in-window may nudge the stored embedding toward the query.
    # Real window ~6h (Nader 2000); 6h compressed 24x lands at ~15 min
    # of conversation, which is a sensible "still in the same chat" gate.
    reconsolidation_window_seconds: float = 21600.0
    # Per-recall blend factor for reconsolidation (Nader 2000). When
    # > 0, an in-window recall accumulates per-episode drift in metadata
    # (``drift_total`` / ``drift_locked``) AND pulls the stored vector a
    # fraction ``alpha * score`` toward the cue via
    # ``episodic.update_embedding`` (buffered through pending.update;
    # promoted to HNSW by the next consolidate ``flush_pending``).
    # Default 0.05: a strong recall drifts ~5%, so the 0.15 total cap is
    # reached after ~3 corroborating recalls, then the episode locks.
    # Set 0 to disable (pure read-only recall).
    reconsolidation_alpha: float = 0.05
    # Hard cap on cumulative drift (1 - cosine(v_original, v_current))
    # before reconsolidation is locked out for this episode.
    reconsolidation_max_total_drift: float = 0.15
    # Novelty gate: only nudge when the recall's success score
    # (cosine to query) clears this floor, so noisy recalls don't drift
    # the memory toward unrelated queries.
    reconsolidation_min_score: float = 0.4

    # Habit threshold: once an edge accumulates this many reinforcements
    # it counts as a habit. The unit here is per-event (plasticity.py
    # increments hits on every reinforce call), so the neural analogue
    # is rodent overtraining-trial counts (Smith & Graybiel 2013, CSH
    # Persp Biol; Graybiel 2008), not Lally 2010's per-day automaticity
    # estimate of 66 days. DLS task-bracketing patterns stabilise after
    # hundreds of trials; 250 sits at the conservative end of that band.
    # Habits are NOT immune to LTD - real habits decay with disuse, just
    # much more slowly, and they relearn faster (Ebbinghaus savings).
    habit_threshold_hits: int = 250
    # Per-real-idle-day LTD applied to non-habit edges during
    # ``memory_dream``/consolidate offline passes.
    ltd_decay_per_idle_day: float = 0.02
    # Habit edges decay at this fraction of the normal LTD rate.
    # 0.18 ~= a habit losing strength about 5x slower than a fresh edge,
    # so a habit fades over months rather than weeks of disuse.
    habit_ltd_multiplier: float = 0.18
    # Savings: once an edge has EVER crossed ``habit_threshold_hits``
    # (even if it has since decayed below it), future reinforcements
    # accumulate at this multiplier - a lapsed habit comes back faster
    # than a fresh one is built.
    habit_savings_factor: float = 3.0

    # Spreading activation on recall: BFS this many hops on the
    # plasticity-weight graph from the cosine anchor, attenuating by
    # ``decay`` per hop. 1 hop = co-activated episodes share a small
    # boost; 0 disables. Default 1 (cheap; only fires once durable
    # edges exist via L-LTP, otherwise the spread is empty).
    spreading_activation_hops: int = 1
    spreading_activation_decay: float = 0.5
    # Mixing weight of the spreading-activation contribution into the
    # final rank key (added on top of the SR omega-weighted boost).
    spreading_activation_weight: float = 0.2
    # Hub-degree guard for spreading activation: each frontier node
    # expands only its top-N plasticity edges by weight per hop, so a
    # single high-out-degree episode cannot flood the frontier. Default 8
    # caps only genuine hubs -- the weakest edges drop first, so it is
    # metric-neutral at the default 1 hop and only reshapes ranking once
    # hops >= 2 or a hub accumulates many durable edges. Lower it to
    # tighten the guard; set 0 to disable it (unbounded expansion).
    spreading_activation_max_fanout: int = 8

    # Soft context bonus: episodes whose ``session_id`` matches the
    # caller's current session get this much subtracted from their rank
    # key (lower = better). Models state-dependent retrieval — same
    # context biases recall toward in-context memories without
    # censoring cross-context ones. Set 0 to disable.
    same_session_bonus: float = 0.05

    # Schema-eligibility gates for systems consolidation. An episode is
    # absorbed into a schema only once both are met. ~60 s real-time
    # combined with the default 24x compression maps to ~1 compressed
    # day, mirroring the day-to-week ramp of HC->cortex handover.
    # Tests that need immediate consolidation pass ``=0`` overrides.
    consolidate_min_age_seconds: float = 60.0
    consolidate_min_retrievals: int = 1
    # Salience floor for consolidate eligibility. Episodes whose
    # ``salience`` is strictly below this value are skipped, preventing
    # low-salience noise from accumulating enough co-occurrence over
    # repeated runs to form durable schema clusters. Defaults to the shared
    # ``NEUTRAL_SALIENCE_BASE`` so any episode at or above the neutral
    # baseline still consolidates; set to 0.0 to disable.
    consolidate_min_salience: float = NEUTRAL_SALIENCE_BASE
    # Schema-margin replay weighting. The consolidation pass scores each
    # unconsolidated episode by ``strength * d1 / (1 + beta * margin)``,
    # where ``d1`` is the cosine distance to the nearest schema and
    # ``margin = d2 - d1`` is the gap to the second-nearest. Larger beta
    # damps episodes that one schema clearly owns (stable) relative to
    # episodes whose top-2 schemas disagree (perturbed) - the latter
    # carry the strongest schema-boundary learning signal. ``beta=0``
    # recovers the legacy ``strength * novelty`` ordering.
    schema_margin_beta: float = 2.0
    # Saturation point for consolidation-derived schema confidence:
    # ``confidence = min(1.0, n_source_episodes / consolidate_confidence_full_at)``.
    # Both consolidation paths (absorption into an existing schema and
    # fresh K-means clustering) derive confidence from the schema's
    # ``source_episode_ids`` count, so identical evidence yields
    # identical confidence regardless of which path produced the schema.
    # User-asserted confidence (via ``store_semantic_memory``) is
    # unaffected. Default 5: a 2-episode cluster lands at 0.4, a fully
    # corroborated schema saturates at 5+ supporting episodes.
    consolidate_confidence_full_at: int = 5

    # Surprise-modulated encoding: when a new episode's nearest-neighbour
    # cosine distance exceeds the floor, salience is boosted by
    # ``surprise_salience_boost`` (capped at 1.0). Boost only fires when
    # there IS a neighbour - an empty namespace has nothing to predict
    # against, so by definition cannot surprise.
    surprise_distance_floor: float = 0.6
    surprise_salience_boost: float = 0.1

    # Auto-fill structural signal flags (``has_traceback``,
    # ``has_diff_markers``, ``references``, ...) from a regex pass over
    # the content into every encoded episode's metadata. Pure and
    # constant-cost per encode; the upside is that recall filters can
    # pivot on shape without the caller declaring an event ``kind``.
    # Defaults ON because the cost is negligible.
    auto_signal_metadata: bool = True

    # Substitute a structurally-derived salience when the caller omits
    # one (``encode_episode(..., salience=None)``). On by default — the
    # encoder calls ``derive_salience(derive_signals(content),
    # base=auto_salience_base)`` so failure / diff / decision records
    # float to the top without per-call tuning. Defaults to the shared
    # ``NEUTRAL_SALIENCE_BASE`` (== ``consolidate_min_salience``) so untagged
    # auto-derived episodes sit at the consolidate boundary; signalled
    # content lifts above it.
    auto_salience: bool = True
    auto_salience_base: float = NEUTRAL_SALIENCE_BASE

    # Reactor policy thresholds (only consulted when
    # ``self_learning_enabled``). Defaults are conservative enough that
    # small test runs do not auto-trigger consolidate or dream.
    # consolidate_cooldown: schema consolidation in vivo runs on the
    #   order of hours-to-days even with prior schemas (Tse et al 2007,
    #   Science). 10 min keeps the gate above normal agent-session
    #   length without being so loose that every burst re-fires.
    # dream_after_idle: awake hippocampal replay fires during brief
    #   pauses (Carr, Jadhav & Frank 2011, Nat Neurosci), not only
    #   during sleep. 10 min matches a single user coffee-break,
    #   keeping replay tied to agent idle windows rather than to
    #   biological sleep-cycle gating.
    reactor_consolidate_after_novel: int = 32
    reactor_consolidate_cooldown_seconds: float = 600.0
    reactor_dream_after_events: int = 128
    reactor_dream_after_idle_seconds: float = 600.0
    reactor_event_log_capacity: int = 1024

    # Off-policy replay during the dream pass (SWR rehearsal). Before
    # the LTD pass the reactor reactivates the ``dream_replay_top_k``
    # highest power-law-strength unconsolidated episodes, groups them by
    # originating session, and reinforces the within-session
    # associations with gain ``dream_replay_gain * mean_salience``
    # (McGaugh arousal modulation). ``dream_replay_top_k=0`` disables
    # replay (LTD-only dream, the legacy behaviour).
    dream_replay_top_k: int = 16
    # Salience floor for dream-replay rehearsal. Defaults to the shared
    # ``NEUTRAL_SALIENCE_BASE`` (== ``consolidate_min_salience``) so
    # low-salience noise is not rehearsed into durable associations during
    # SWR replay. Set to 0.0 to disable.
    dream_replay_min_salience: float = NEUTRAL_SALIENCE_BASE
    dream_replay_gain: float = 0.3
    # Per-cycle scan size for dream replay. ``get_documents`` orders by
    # id, so a fixed ``limit`` would always return the oldest episodes
    # and starve newer ones; replay uses a rotating offset cursor
    # (``service._replay_cursor``) so successive cycles sweep through
    # the whole unconsolidated set. Lower values reduce per-cycle
    # memory at the cost of more cycles to cover the table.
    # 0 disables the cursor and falls back to an unbounded fetch.
    dream_replay_max_scan: int = 2_000
