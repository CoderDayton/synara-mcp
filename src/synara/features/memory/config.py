"""Memory tunables.

Kept in its own module so the service core and the operation-specific
siblings (encode/recall/consolidate/forget/...) can each import the
config without pulling each other.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .memory_types import MemoryTypeRegistry


@dataclass(frozen=True, slots=True)
class MemoryConfig:
    """Tunable parameters of the memory framework."""

    episodic_collection: str = "memory_episodic"
    semantic_collection: str = "memory_semantic"
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
    # Power-law (Wickelgren/Wixted) decay exponent used by ``forget``:
    #   S(t) = salience * sum_k (1 + (t - t_k))^(-d)
    # d ~ 0.5 fits behavioural retention curves better than the exponential
    # Ebbinghaus form. Larger d = faster decay.
    forget_d: float = 0.5
    # Cap on retained access-history timestamps per episode; older entries
    # are FIFO-evicted. Keeps storage bounded while preserving Anderson's
    # base-level activation shape for typical usage.
    access_history_cap: int = 32
    # Minimum cluster size that yields a semantic schema during consolidation.
    consolidate_min_cluster: int = 2
    # Cosine distance below which an unconsolidated episode is *absorbed*
    # into the nearest existing schema (instead of contributing to a new
    # cluster). Implements schema-fitting fast-track consolidation
    # (Tse et al. 2007). Smaller = stricter fit required.
    consolidate_absorb_distance: float = 0.4
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
    # (Lally 2010 median = 66 days, here count-based and
    # compression-invariant) it counts as a habit. Habits are NOT immune
    # to LTD - real habits decay with disuse, just much more slowly,
    # and they relearn faster (Ebbinghaus savings).
    habit_threshold_hits: int = 66
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
    # one (``encode_episode(..., salience=None)``). Off by default — when
    # on, the encoder calls ``derive_salience(derive_signals(content),
    # base=auto_salience_base)`` so failure / diff / decision records
    # float to the top without per-call tuning.
    auto_salience: bool = False
    auto_salience_base: float = 0.3

    # Reactor policy thresholds (only consulted when
    # ``self_learning_enabled``). Defaults are conservative enough that
    # small test runs do not auto-trigger consolidate or dream.
    reactor_consolidate_after_novel: int = 32
    reactor_consolidate_cooldown_seconds: float = 60.0
    reactor_dream_after_events: int = 128
    reactor_dream_after_idle_seconds: float = 1800.0
    reactor_event_log_capacity: int = 1024

    # Off-policy replay during the dream pass (SWR rehearsal). Before
    # the LTD pass the reactor reactivates the ``dream_replay_top_k``
    # highest power-law-strength unconsolidated episodes, groups them by
    # originating session, and reinforces the within-session
    # associations with gain ``dream_replay_gain * mean_salience``
    # (McGaugh arousal modulation). ``dream_replay_top_k=0`` disables
    # replay (LTD-only dream, the legacy behaviour).
    dream_replay_top_k: int = 16
    dream_replay_min_salience: float = 0.0
    dream_replay_gain: float = 0.3
