"""Forgetting pass: Ebbinghaus-style decay + selective pruning.

A memory's strength at time ``t`` is:

    strength = salience * exp(-age / tau) + retrieval_count * boost

Episodes whose strength has fallen below ``strength_floor`` are flagged
(``dry_run=True``) or deleted (``dry_run=False``). Consolidated episodes
are pruned at the configured threshold; unconsolidated ones are pruned
only when their strength is below half the floor — to avoid losing
fresh-but-low-salience traces before they've had a chance to be
consolidated.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any

from synara.core.errors import ValidationError

from .service import UNCONSOLIDATED, now_seconds

if TYPE_CHECKING:  # pragma: no cover - typing only
    from .service import HippocampusService


def memory_strength(
    salience: float,
    age_seconds: float,
    retrievals: int,
    *,
    tau_seconds: float,
    retrieval_boost: float,
) -> float:
    """Public so callers can score memories with the same formula."""
    if tau_seconds <= 0.0:
        raise ValidationError("tau_seconds must be positive")
    decay = math.exp(-max(age_seconds, 0.0) / tau_seconds)
    return float(salience) * decay + float(retrievals) * retrieval_boost


async def run(
    service: HippocampusService,
    *,
    strength_floor: float = 0.05,
    decay_tau_seconds: float | None = None,
    dry_run: bool = True,
    max_scan: int = 1000,
) -> dict[str, Any]:
    if not 0.0 <= strength_floor <= 1.0:
        raise ValidationError("strength_floor must be in [0, 1]")
    tau = decay_tau_seconds if decay_tau_seconds is not None else service.config.decay_tau_seconds
    if tau <= 0.0:
        raise ValidationError("decay_tau_seconds must be positive")

    now = now_seconds()
    rows = await service.episodic.get_documents(filter_dict=None, limit=max_scan)
    weak: list[int] = []
    for ep_id, _text, md in rows:
        age = now - float(md.get("encoded_at", now))
        strength = memory_strength(
            salience=float(md.get("salience", 0.0)),
            age_seconds=age,
            retrievals=int(md.get("retrieval_count", 0)),
            tau_seconds=tau,
            retrieval_boost=service.config.retrieval_boost,
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
        await service.episodic.delete_by_ids(weak)
        removed = len(weak)
    return {
        "candidate_ids": weak,
        "removed": removed,
        "dry_run": dry_run,
        "scanned": len(rows),
    }
