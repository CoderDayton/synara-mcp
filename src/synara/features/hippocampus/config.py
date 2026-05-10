"""Hippocampus tunables.

Kept in its own module so the service core and the operation-specific
siblings (encode/recall/consolidate/forget/...) can each import the
config without pulling each other.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class HippocampusConfig:
    """Tunable parameters of the memory framework."""

    episodic_collection: str = "hippocampus_episodic"
    semantic_collection: str = "hippocampus_semantic"
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
