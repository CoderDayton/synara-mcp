"""Interaction events and the self-triggering reactor.

Every public op on ``MemoryService`` emits an
:class:`InteractionEvent` after its main work completes. Events are
appended to the persistent ``coll.events`` change feed so cross-process
subscribers and post-mortem inspection both see the same log; the
in-memory :class:`ReactorState` is hot-path bookkeeping for the
:class:`TriggerPolicy`; it is rebuilt from that persistent log on restart
via :meth:`EventBus.ensure_state_loaded`, so the consolidate/dream
triggers survive the stdio-per-session relaunch cycle.

Reactor invariants
------------------
* Reactor callbacks for ``consolidate`` / ``dream`` are themselves
  service ops that emit their own events. ``react()`` skips trigger
  evaluation when the originating event is itself a reactor output, so
  the call graph is a tree of depth at most 2:

      user_call (encode|recall|forget|reflect)
          -> emit -> react -> [consolidate? dream?]
                              -> emit (no further react)

* Triggers are deterministic functions of public state
  (``ReactorState``) and contain no learning — the "self-learning"
  emerges from the *side effects* the triggers schedule.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    from simplevecdb import AsyncVectorCollection

__all__ = [
    "EventBus",
    "EventKind",
    "InteractionEvent",
    "ReactorCallback",
    "ReactorState",
    "TriggerPolicy",
    "now_seconds",
]

EventKind = Literal[
    "encode",
    "recall",
    "consolidate",
    "forget",
    "reflect",
    "dream",
]


@dataclass(frozen=True, slots=True)
class InteractionEvent:
    """One observable interaction with the memory service."""

    kind: EventKind
    timestamp: float
    session_id: str | None
    payload: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def create(
        cls,
        kind: EventKind,
        *,
        session_id: str | None = None,
        payload: dict[str, Any] | None = None,
        timestamp: float | None = None,
    ) -> InteractionEvent:
        """Build an event for the common case: stamp ``now`` and copy payload.

        ``timestamp`` defaults to :func:`now_seconds`; ``payload`` is copied
        defensively so a caller's dict can't later mutate the (frozen)
        event's contents.
        """
        return cls(
            kind=kind,
            timestamp=now_seconds() if timestamp is None else timestamp,
            session_id=session_id,
            payload=dict(payload) if payload else {},
        )


@dataclass(slots=True)
class ReactorState:
    """Mutable bookkeeping the policy reads to decide on triggers.

    :class:`TriggerPolicy` reads ``last_consolidate_at`` / ``last_dream_at``
    / ``novel_encodes_since_consolidate`` / ``events_since_dream`` /
    ``prev_event_at`` (the timestamp of the event *before* the current
    one — the only way an event-driven system can observe an idle gap,
    since ``record`` has already stamped ``last_event_at`` with the
    current event by the time ``react`` runs). ``last_event_at`` and
    ``total_events`` are inspection-only lifetime tallies that no trigger
    consults; like the rest of the state they reflect only the
    ``log_capacity`` rehydration window after a restart, so treat them
    as approximate.
    """

    last_event_at: float = 0.0  # inspection-only; not read by the policy
    prev_event_at: float = 0.0  # event before the current one; feeds the idle gate
    last_consolidate_at: float = 0.0
    last_dream_at: float = 0.0
    novel_encodes_since_consolidate: int = 0
    events_since_dream: int = 0
    total_events: int = 0  # inspection-only; not read by the policy


@dataclass(slots=True)
class TriggerPolicy:
    """When to schedule consolidate / dream as side effects."""

    consolidate_after_novel_encodes: int = 32
    consolidate_cooldown_seconds: float = 600.0  # 10 min; Tse et al 2007 cadence
    dream_after_events: int = 128
    dream_after_idle_seconds: float = 600.0  # 10 min; Carr et al 2011 awake replay

    def __post_init__(self) -> None:
        # Count thresholds use the ``<= 0 disables`` convention, so any int
        # is meaningful. The time fields have no such escape hatch: a
        # negative (or NaN) cooldown/idle would silently make the time gate
        # always-true. ``not (x >= 0)`` rejects both negatives and NaN.
        for name in ("consolidate_cooldown_seconds", "dream_after_idle_seconds"):
            value: float = getattr(self, name)
            if not (value >= 0):
                raise ValueError(f"{name} must be >= 0, got {value!r}")

    def consolidate_due(self, state: ReactorState, now: float) -> bool:
        if self.consolidate_after_novel_encodes <= 0:
            return False
        if state.novel_encodes_since_consolidate < self.consolidate_after_novel_encodes:
            return False
        return (now - state.last_consolidate_at) >= self.consolidate_cooldown_seconds

    def dream_due(self, state: ReactorState, now: float) -> bool:
        # Disable convention: a config threshold of ``<= 0`` disables that
        # gate (mirrors ``consolidate_after_novel_encodes <= 0`` above).
        # The leading ``events_since_dream <= 0`` test is a *state* guard
        # (nothing happened since the last dream), not a config disable.
        if state.events_since_dream <= 0:
            return False
        if self.dream_after_events > 0 and state.events_since_dream >= self.dream_after_events:
            return True
        # Idle gate: awake replay fires in the pauses *between* activity
        # (Carr et al 2011), so "idle" is the gap between this event and
        # the previous one — the dream runs on the first event after a
        # quiet window, not periodically during continuous activity
        # (which the old ``now - last_dream_at`` form amounted to). The
        # ``prev_event_at > 0`` guard keeps a fresh store (no prior
        # event on record) from dreaming on its very first interaction.
        return (
            self.dream_after_idle_seconds > 0
            and state.prev_event_at > 0.0
            and (now - state.prev_event_at) >= self.dream_after_idle_seconds
        )


_REACTOR_KINDS: frozenset[EventKind] = frozenset({"consolidate", "dream"})

# Persistent event-log entries get this kind prefix to avoid colliding
# with simplevecdb's own internal events (edge_upsert, rebuild, etc.).
EVENT_KIND_PREFIX = "memory."


def _event_kind(kind: str) -> str:
    return f"{EVENT_KIND_PREFIX}{kind}"


# Payload sentinel carrying ``InteractionEvent.session_id`` through the
# durable row (which only stores ``kind`` + ``payload``).
_SESSION_ID_KEY = "__session_id"


def _encode_payload(event: InteractionEvent) -> dict[str, Any]:
    """Fold ``session_id`` into the payload written to the durable row."""
    payload = dict(event.payload)
    if event.session_id is not None:
        payload[_SESSION_ID_KEY] = event.session_id
    return payload


def _decode_row(row: Any) -> InteractionEvent | None:
    """Inverse of :func:`_encode_payload`: durable row -> event.

    Returns ``None`` for rows that are not ours (foreign ``kind`` prefix,
    e.g. simplevecdb's own ``edge_upsert`` / ``rebuild`` entries).
    """
    if not row.kind.startswith(EVENT_KIND_PREFIX):
        return None
    payload = dict(row.payload or {})
    session_id = payload.pop(_SESSION_ID_KEY, None)
    return InteractionEvent(
        kind=row.kind[len(EVENT_KIND_PREFIX) :],
        timestamp=float(row.ts),
        session_id=session_id,
        payload=payload,
    )


# Async handler the reactor schedules when a consolidate/dream is due.
ReactorCallback = Callable[[InteractionEvent], Awaitable[None]]


class EventBus:
    """Persistent event log + reactor for self-triggered follow-ups.

    ``record()`` is async because it appends to the underlying
    ``coll.events`` table. When no collection is supplied, the bus
    becomes a thin in-memory shim (used by unit tests for the policy).
    """

    def __init__(
        self,
        *,
        collection: AsyncVectorCollection | None = None,
        policy: TriggerPolicy | None = None,
        log_capacity: int = 1024,
        on_consolidate: ReactorCallback | None = None,
        on_dream: ReactorCallback | None = None,
    ) -> None:
        if log_capacity <= 0:
            raise ValueError("log_capacity must be positive")
        self._coll = collection
        self.policy = policy or TriggerPolicy()
        self.state = ReactorState()
        self.log_capacity = int(log_capacity)
        # Reactor handlers. Still settable post-construction, but prefer the
        # constructor params so the bus's reactive contract is explicit at
        # creation instead of wired up in a separate step.
        self.on_consolidate: ReactorCallback | None = on_consolidate
        self.on_dream: ReactorCallback | None = on_dream
        # In-memory fallback when no collection is wired (test paths).
        self._mem_log: list[InteractionEvent] = []
        # Throttle prune calls so we don't hit SQL on every record().
        self._records_since_prune = 0
        self._prune_every = max(64, log_capacity // 4)
        # Reactor counters live in ``self.state`` but are durable in the
        # persistent event log. ``ensure_state_loaded`` rebuilds them from
        # that log once per process so the consolidate/dream triggers
        # survive the stdio-per-session relaunch cycle. Double-checked
        # under a lock so concurrent first ``record`` calls replay once.
        self._state_loaded = False
        self._state_lock = asyncio.Lock()

    async def log(self) -> list[InteractionEvent]:
        """Snapshot of the most recent events (up to ``log_capacity``)."""
        if self._coll is None:
            return list(self._mem_log)
        # Pruning only fires every ``_prune_every`` records, so the table
        # can transiently exceed ``log_capacity``. ``read_events`` is
        # ``seq ASC LIMIT n``, which would then return the *oldest* window
        # and drop the newest events. Anchor the read at
        # ``last_seq - log_capacity`` so we always get the newest slice.
        last_seq = await self._coll.last_event_seq()
        since = max(0, last_seq - self.log_capacity)
        rows = await self._coll.read_events(since=since, limit=self.log_capacity)
        kept: list[InteractionEvent] = []
        for r in rows:
            event = _decode_row(r)
            if event is not None:
                kept.append(event)
        return kept[-self.log_capacity :]

    async def record(self, event: InteractionEvent) -> None:
        """Append the event and update reactor state counters."""
        await self.ensure_state_loaded()
        if self._coll is not None:
            # Coupling note: the async collection exposes ``read_events`` /
            # ``last_event_seq`` but no async ``append`` / ``prune``, so we
            # reach through ``_collection.events`` (the sync engine) under
            # ``to_thread``. If simplevecdb adds async append/prune, switch
            # to those and drop this private access (here and in
            # ``_maybe_prune``).
            await asyncio.to_thread(
                self._coll._collection.events.append,
                _event_kind(event.kind),
                payload=_encode_payload(event),
            )
            self._records_since_prune += 1
            if self._records_since_prune >= self._prune_every:
                await self._maybe_prune()
        else:
            self._mem_log.append(event)
            if len(self._mem_log) > self.log_capacity:
                del self._mem_log[: len(self._mem_log) - self.log_capacity]
        self._apply_event_to_state(event)

    def _apply_event_to_state(self, event: InteractionEvent) -> None:
        """Fold one event into the reactor counters.

        Shared by the live ``record`` path and ``ensure_state_loaded`` so
        the restart-time replay stays consistent with the hot-path
        bookkeeping.
        """
        st = self.state
        st.prev_event_at = st.last_event_at
        st.last_event_at = event.timestamp
        st.total_events += 1
        st.events_since_dream += 1
        if event.kind == "encode" and event.payload.get("deduped") is False:
            st.novel_encodes_since_consolidate += 1
        elif event.kind == "consolidate":
            st.last_consolidate_at = event.timestamp
            st.novel_encodes_since_consolidate = 0
        elif event.kind == "dream":
            st.last_dream_at = event.timestamp
            st.events_since_dream = 0

    async def ensure_state_loaded(self) -> None:
        """Rebuild reactor counters from the durable event log, once.

        ``ReactorState`` is process-local and constructed empty, but the
        triggers it feeds (consolidate after N novel encodes, dream after
        N events) track lifetime activity. Under stdio the server is
        relaunched per session, so without this replay the novel-encode
        counter would reset to 0 every launch and the consolidate
        threshold could never be reached. Idempotent and double-checked so
        concurrent first ``record`` calls replay exactly once.

        The replay window is bounded by ``log_capacity``: if the last
        consolidate event has been pruned, novel encodes preceding it may
        be recounted, at worst scheduling one extra (idempotent)
        consolidate pass.
        """
        if self._state_loaded:
            return
        async with self._state_lock:
            if self._state_loaded:
                return
            # Mark loaded before the replay: ``_apply_event_to_state`` only
            # mutates counters (never calls back into ``record``), so the
            # flag keeps a re-entrant ``record`` from double-counting.
            # Reset on failure so a transient log-read error doesn't wedge
            # the counters at their defaults.
            self._state_loaded = True
            try:
                events = await self.log()
            except Exception:
                self._state_loaded = False
                raise
            for event in events:
                self._apply_event_to_state(event)

    async def _maybe_prune(self) -> None:
        """Keep the persistent log bounded to ``log_capacity`` entries."""
        if self._coll is None:
            return
        last_seq = await self._coll.last_event_seq()
        # ``prune_events`` deletes ``seq < before_seq``; to retain exactly
        # the newest ``log_capacity`` entries (seqs in
        # ``last_seq - log_capacity + 1 .. last_seq``) the cutoff must be
        # ``last_seq - log_capacity + 1``. The previous ``last_seq -
        # log_capacity`` kept one extra row, and since ``read_events`` is
        # ``seq ASC LIMIT n`` that surplus row pushed the newest event out
        # of the ``log()`` window.
        cutoff = last_seq - self.log_capacity + 1
        if cutoff <= 0:
            self._records_since_prune = 0
            return
        await asyncio.to_thread(
            self._coll._collection.events.prune,
            before_seq=cutoff,
        )
        self._records_since_prune = 0

    async def react(self, event: InteractionEvent) -> list[str]:
        """Run any due reactor callbacks. Returns triggered action names."""
        if event.kind in _REACTOR_KINDS:
            return []
        # ``react`` reads ``self.state``; rehydrate first so a standalone
        # caller (one not preceded by ``record``) evaluates triggers
        # against durable counters, not process-start defaults. Idempotent
        # and ~free after the first load.
        await self.ensure_state_loaded()
        triggered: list[str] = []
        if self.on_consolidate is not None and self.policy.consolidate_due(
            self.state, event.timestamp
        ):
            await self.on_consolidate(event)
            triggered.append("consolidate")
        if self.on_dream is not None and self.policy.dream_due(self.state, event.timestamp):
            await self.on_dream(event)
            triggered.append("dream")
        return triggered


def now_seconds() -> float:
    return time.time()
