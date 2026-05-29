"""Off-policy hippocampal replay (sharp-wave-ripple rehearsal).

Live recall is the only thing that strengthens the associative graph
during waking: ``recall`` reinforces the anchor->co-recalled plasticity
edges and folds one SR edge per query. Real memory also rehearses
*offline* — during quiet rest / sleep, sharp-wave ripples reactivate
high-priority traces and re-strengthen their associations without any
external cue (Wilson & McNaughton 1994; O'Neill et al. 2010).

This module supplies that missing leg. The dream reactor calls
:func:`run` before the LTD pass: it samples the highest-priority
unconsolidated episodes, groups them by their originating session, and
reinforces the within-session associations off-policy. Replay priority
is the same power-law memory strength ``forget`` uses, so salient,
recent, and frequently-retrieved traces are rehearsed preferentially
(McClelland et al. 1995 error-driven replay budget).

Emotional modulation (McGaugh 2004): the reinforcement gain is scaled
by the group's mean salience, so arousing episodes consolidate their
relational structure faster than routine ones.
"""

from __future__ import annotations

from collections import defaultdict
from typing import TYPE_CHECKING, Any

from ..neocortex.forget import _DEFAULT_SALIENCE, access_times_from_meta, memory_strength
from ..service import UNCONSOLIDATED

if TYPE_CHECKING:  # pragma: no cover - typing only
    from ..port import MemoryServicePort as MemoryService

# Smallest group that can form an association (anchor + >=1 target).
_MIN_GROUP = 2


async def _fetch_candidates(
    service: MemoryService, *, scan_cap: int
) -> list[tuple[int, str, dict[str, Any]]]:
    """Page through unconsolidated episodes with a rotating offset cursor.

    ``get_documents`` orders by id, so a plain ``limit`` without offset
    would always return the lowest-IDed (oldest) rows and starve newer
    episodes of replay budget. The cursor advances by ``scan_cap`` per
    pass and wraps on partial pages, so successive cycles sweep through
    the whole table. ``scan_cap=0`` falls back to an unbounded fetch.
    """
    flt = {"consolidated_into": UNCONSOLIDATED}
    if scan_cap <= 0:
        result: list[tuple[int, str, dict[str, Any]]] = await service.episodic.get_documents(flt)
        return result
    candidates: list[tuple[int, str, dict[str, Any]]] = await service.episodic.get_documents(
        flt, limit=scan_cap, offset=service._replay_cursor
    )
    if len(candidates) < scan_cap:
        service._replay_cursor = 0
    else:
        service._replay_cursor += scan_cap
    return candidates


async def run(service: MemoryService, *, now: float) -> int:
    """Replay top-priority unconsolidated episodes off-policy.

    Returns the number of reinforced (anchor -> j) associations. A
    no-op (returns 0) when replay is disabled (``dream_replay_top_k`` or
    ``dream_replay_gain`` non-positive), there are no eligible
    candidates, or no session has >= 2 eligible co-encoded episodes.

    Plasticity is always reinforced (``_plasticity`` is non-optional).
    The SR update is skipped when ``_sr`` is absent, but that alone does
    not make the pass a no-op.
    """
    cfg = service.config
    top_k = cfg.dream_replay_top_k
    base_gain = cfg.dream_replay_gain
    if top_k <= 0 or base_gain <= 0.0:
        return 0
    candidates = await _fetch_candidates(service, scan_cap=cfg.dream_replay_max_scan)
    if not candidates:
        return 0

    # Rank by the same power-law strength forget/consolidate use, so
    # replay budget flows to salient, recent, oft-retrieved traces.
    scored: list[tuple[float, int, str, float]] = []
    for ep_id, _text, md in candidates:
        sal = float(md.get("salience", _DEFAULT_SALIENCE))
        if sal < cfg.dream_replay_min_salience:
            continue
        sid = md.get("session_id")
        if not sid:
            # No originating context -> no associative neighbourhood to
            # rehearse; an isolated trace cannot drive co-activation.
            continue
        strength = memory_strength(
            salience=sal,
            access_times=access_times_from_meta(md, fallback_now=now),
            now=now,
            d=cfg.forget_d,
        )
        scored.append((strength, int(ep_id), str(sid), sal))

    if not scored:
        return 0
    scored.sort(key=lambda r: -r[0])
    selected = scored[:top_k]

    # Group the replay set by originating session. The strongest trace
    # in each group is the reactivation anchor; co-encoded episodes are
    # the rehearsed targets (anchor-style, matching recall + SR and
    # avoiding the n*(n-1)/2 inflation of a naive pairwise loop).
    groups: dict[str, list[tuple[float, int, float]]] = defaultdict(list)
    for strength, ep_id, sid, sal in selected:
        groups[sid].append((strength, ep_id, sal))

    reinforced = 0
    for sid, members in groups.items():
        if len(members) < _MIN_GROUP:
            continue
        members.sort(key=lambda r: -r[0])
        anchor_id = members[0][1]
        others = [eid for _s, eid, _sal in members[1:]]
        mean_sal = sum(sal for _s, _e, sal in members) / len(members)
        # McGaugh arousal modulation: salient memories rehearse harder.
        gain = max(0.0, min(1.0, base_gain * mean_sal))
        if gain <= 0.0 or not others:
            continue
        for j in others:
            await service._plasticity.reinforce(anchor_id, j, score=gain, now=now)
        if service._sr is not None:
            await service._sr.observe_recall_set(sid, anchor_id, others, now)
            await service._sr.flush()
        reinforced += len(others)
    return reinforced
