"""Forgetting pass: power-law decay + selective pruning.

Anderson's ACT-R base-level activation, in the Wickelgren / Wixted
power-law form that empirically beats the exponential Ebbinghaus model
across multiple time scales (Wixted 2004):

    S(t) = salience * sum_k (1 + (t - t_k))^(-d)

The sum aggregates every retrieval event ``t_k`` (encoding included).
Power-law decay obeys Jost's law — older traces decay slower at the
same instantaneous rate, so well-rehearsed memories survive long after
an exponential model would have culled them.

Episodes whose strength has fallen below ``strength_floor`` are flagged
(``dry_run=True``) or deleted (``dry_run=False``). Consolidated episodes
are pruned at the configured threshold; unconsolidated ones at half
that threshold — fresh-but-low traces get a second chance to be
consolidated before they vanish.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

from synara.core.errors import ValidationError

from ..service import UNCONSOLIDATED, now_seconds

if TYPE_CHECKING:  # pragma: no cover - typing only
    from ..port import MemoryServicePort as MemoryService

# Neutral salience for episodes missing the field. Matches the base in
# ``amygdala.signals.derive_salience`` so an un-tagged episode decays
# like a default-tagged one rather than being prune-on-sight.
_DEFAULT_SALIENCE = 0.3


def memory_strength(
    salience: float,
    access_times: Sequence[float],
    *,
    now: float,
    d: float = 0.5,
) -> float:
    """Compute power-law (Wickelgren/Wixted) memory strength.

    access_times: encoding + all retrieval events.
    Returns salience * sum_k (1 + age_k)^(-d). Decays slower than
    exponential for large ages.
    """
    if d <= 0.0:
        raise ValidationError("d must be positive")
    if not access_times:
        return float(salience)
    total = 0.0
    for t_k in access_times:
        delta = max(0.0, now - float(t_k))
        total += (1.0 + delta) ** (-d)
    return float(salience) * total


def access_times_from_meta(md: dict[str, Any], *, fallback_now: float) -> list[float]:
    """Extract access-time list from episode metadata.

    Newer: explicit access_history. Older: [encoded_at] + last_accessed
    repeated retrieval_count times (coarse approximation).
    """
    history = md.get("access_history")
    if isinstance(history, list) and history:
        return [float(t) for t in history]
    enc = float(md.get("encoded_at", fallback_now))
    last = float(md.get("last_accessed", enc))
    rc = int(md.get("retrieval_count", 0))
    return [enc] + [last] * max(rc, 0)


async def run(
    service: MemoryService,
    *,
    strength_floor: float = 0.05,
    decay_tau_seconds: float | None = None,
    dry_run: bool = True,
    max_scan: int = 1000,
) -> dict[str, Any]:
    if not 0.0 <= strength_floor <= 1.0:
        raise ValidationError("strength_floor must be in [0, 1]")
    if max_scan <= 0:
        raise ValidationError("max_scan must be positive")
    # ``decay_tau_seconds`` is preserved for API compatibility but the
    # power-law model is parameterised by the dimensionless exponent
    # ``d`` rather than a time constant. We allow callers to override
    # ``d`` indirectly: tau <= 0 still raises so misconfigured callers
    # get an immediate error instead of silently surviving.
    if decay_tau_seconds is not None and (
        not math.isfinite(decay_tau_seconds) or decay_tau_seconds <= 0.0
    ):
        raise ValidationError("decay_tau_seconds must be a positive, finite number")

    now = now_seconds()
    rows = await service.episodic.get_documents(filter_dict=None, limit=max_scan)
    weak: list[int] = []
    for ep_id, _text, md in rows:
        access_times = access_times_from_meta(md, fallback_now=now)
        # Absent salience must not mean "delete me": salience is a
        # multiplicative factor in ``memory_strength``, so a 0.0 default
        # forces strength to 0.0 regardless of recency/access and makes
        # any episode lacking the field a guaranteed prune candidate
        # (direct DB writes, fixtures, pre-field episodes). Fall back to
        # the same neutral base ``derive_salience`` assigns (0.3).
        strength = memory_strength(
            salience=float(md.get("salience", _DEFAULT_SALIENCE)),
            access_times=access_times,
            now=now,
            d=service.config.forget_d,
        )
        consolidated = int(md.get("consolidated_into", UNCONSOLIDATED))
        if strength < strength_floor and consolidated != UNCONSOLIDATED:
            # Gist preserved upstream — safe to drop the raw episode.
            weak.append(int(ep_id))
        elif strength < strength_floor / 2.0 and consolidated == UNCONSOLIDATED:
            # Very weak and never consolidated — drop, but only at a stricter
            # threshold to avoid amnesia for fresh-but-low traces.
            weak.append(int(ep_id))

    removed = 0
    if weak and not dry_run:
        # coll.edges has ON DELETE CASCADE to documents: deleting these
        # docs vaporises their durable SR edges, but a lingering
        # in-memory _T/_pending/window entry would make the next SR
        # flush upsert a FK-violating edge for a now-deleted id. Evict
        # before the delete — the same invariant
        # MemoryService.delete_episode relies on (SuccessorRepresentation
        # .evict_nodes). Plasticity holds no in-memory state.
        if service._sr is not None:
            service._sr.evict_nodes(set(weak))
        await service.episodic.delete_by_ids(weak)
        removed = len(weak)
    return {
        "candidate_ids": weak,
        "removed": removed,
        "dry_run": dry_run,
        "scanned": len(rows),
    }
