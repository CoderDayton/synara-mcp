"""Interaction events and the self-triggering reactor.

Every public op on ``MemoryService`` emits an
:class:`InteractionEvent` after its main work completes. Events are
appended to the persistent ``coll.events`` change feed so cross-process
subscribers and post-mortem inspection both see the same log; the
in-memory :class:`ReactorState` is just hot-path bookkeeping for the
:class:`TriggerPolicy` and is rebuilt from defaults on restart.

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


@dataclass(slots=True)
class ReactorState:
    """Mutable bookkeeping the policy reads to decide on triggers."""

    last_event_at: float = 0.0
    last_consolidate_at: float = 0.0
    last_dream_at: float = 0.0
    novel_encodes_since_consolidate: int = 0
    events_since_dream: int = 0
    total_events: int = 0


@dataclass(slots=True)
class TriggerPolicy:
    """When to schedule consolidate / dream as side effects."""

    consolidate_after_novel_encodes: int = 32
    consolidate_cooldown_seconds: float = 600.0  # 10 min; Tse et al 2007 cadence
    dream_after_events: int = 128
    dream_after_idle_seconds: float = 600.0  # 10 min; Carr et al 2011 awake replay

    def consolidate_due(self, state: ReactorState, now: float) -> bool:
        if self.consolidate_after_novel_encodes <= 0:
            return False
        if state.novel_encodes_since_consolidate < self.consolidate_after_novel_encodes:
            return False
        return (now - state.last_consolidate_at) >= self.consolidate_cooldown_seconds

    def dream_due(self, state: ReactorState, now: float) -> bool:
        if state.events_since_dream <= 0:
            return False
        if self.dream_after_events > 0 and state.events_since_dream >= self.dream_after_events:
            return True
        return (
            self.dream_after_idle_seconds > 0
            and (now - state.last_dream_at) >= self.dream_after_idle_seconds
        )


_REACTOR_KINDS: frozenset[EventKind] = frozenset({"consolidate", "dream"})

# Persistent event-log entries get this kind prefix to avoid colliding
# with simplevecdb's own internal events (edge_upsert, rebuild, etc.).
EVENT_KIND_PREFIX = "memory."


def _event_kind(kind: str) -> str:
    return f"{EVENT_KIND_PREFIX}{kind}"


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
    ) -> None:
        if log_capacity <= 0:
            raise ValueError("log_capacity must be positive")
        self._coll = collection
        self.policy = policy or TriggerPolicy()
        self.state = ReactorState()
        self.log_capacity = int(log_capacity)
        self.on_consolidate: Callable[[InteractionEvent], Awaitable[None]] | None = None
        self.on_dream: Callable[[InteractionEvent], Awaitable[None]] | None = None
        # In-memory fallback when no collection is wired (test paths).
        self._mem_log: list[InteractionEvent] = []
        # Throttle prune calls so we don't hit SQL on every record().
        self._records_since_prune = 0
        self._prune_every = max(64, log_capacity // 4)

    async def log(self) -> list[InteractionEvent]:
        """Snapshot of the most recent events (up to ``log_capacity``)."""
        if self._coll is None:
            return list(self._mem_log)
        rows = await self._coll.read_events(limit=self.log_capacity)
        kept: list[InteractionEvent] = []
        for r in rows:
            if not r.kind.startswith(EVENT_KIND_PREFIX):
                continue
            payload = dict(r.payload or {})
            session_id = payload.pop("__session_id", None)
            kept.append(
                InteractionEvent(
                    kind=r.kind[len(EVENT_KIND_PREFIX) :],
                    timestamp=float(r.ts),
                    session_id=session_id,
                    payload=payload,
                )
            )
        return kept[-self.log_capacity :]

    async def record(self, event: InteractionEvent) -> None:
        """Append the event and update reactor state counters."""
        if self._coll is not None:
            payload = dict(event.payload)
            if event.session_id is not None:
                payload["__session_id"] = event.session_id
            await asyncio.to_thread(
                self._coll._collection.events.append,
                _event_kind(event.kind),
                payload=payload,
            )
            self._records_since_prune += 1
            if self._records_since_prune >= self._prune_every:
                await self._maybe_prune()
        else:
            self._mem_log.append(event)
            if len(self._mem_log) > self.log_capacity:
                del self._mem_log[: len(self._mem_log) - self.log_capacity]
        st = self.state
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

    async def _maybe_prune(self) -> None:
        """Keep the persistent log bounded to ``log_capacity`` entries."""
        if self._coll is None:
            return
        last_seq = await self._coll.last_event_seq()
        cutoff = last_seq - self.log_capacity
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
