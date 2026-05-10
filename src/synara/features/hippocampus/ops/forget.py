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

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

from synara.core.errors import ValidationError

from ..service import UNCONSOLIDATED, now_seconds

if TYPE_CHECKING:  # pragma: no cover - typing only
    from ..service import HippocampusService


def memory_strength(
    salience: float,
    access_times: Sequence[float],
    *,
    now: float,
    d: float = 0.5,
) -> float:
    """Power-law (Wickelgren/Wixted) memory strength.

    ``access_times`` should include the encoding event plus every
    retrieval event. With one access at the present moment, returns the
    salience verbatim; with one access at age ``a`` and exponent ``d``,
    returns ``salience * (1 + a)^(-d)`` — a strictly slower decay than
    the exponential ``salience * exp(-a/tau)`` for ``a >> 1``.
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


def _access_times_from_meta(md: dict[str, Any], *, fallback_now: float) -> list[float]:
    """Build an access-time list from episode metadata.

    Newer episodes carry an explicit ``access_history``; older ones
    (encoded before this field existed) fall back to ``[encoded_at]``
    plus ``last_accessed`` repeated ``retrieval_count`` times — a coarse
    approximation that still gives the activation function the right
    qualitative shape.
    """
    history = md.get("access_history")
    if isinstance(history, list) and history:
        return [float(t) for t in history]
    enc = float(md.get("encoded_at", fallback_now))
    last = float(md.get("last_accessed", enc))
    rc = int(md.get("retrieval_count", 0))
    return [enc] + [last] * max(rc, 0)


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
    # ``decay_tau_seconds`` is preserved for API compatibility but the
    # power-law model is parameterised by the dimensionless exponent
    # ``d`` rather than a time constant. We allow callers to override
    # ``d`` indirectly: tau <= 0 still raises so misconfigured callers
    # get an immediate error instead of silently surviving.
    if decay_tau_seconds is not None and decay_tau_seconds <= 0.0:
        raise ValidationError("decay_tau_seconds must be positive")

    now = now_seconds()
    rows = await service.episodic.get_documents(filter_dict=None, limit=max_scan)
    weak: list[int] = []
    for ep_id, _text, md in rows:
        access_times = _access_times_from_meta(md, fallback_now=now)
        strength = memory_strength(
            salience=float(md.get("salience", 0.0)),
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
        await service.episodic.delete_by_ids(weak)
        removed = len(weak)
    return {
        "candidate_ids": weak,
        "removed": removed,
        "dry_run": dry_run,
        "scanned": len(rows),
    }
