"""Interaction events and the self-triggering reactor.

Every public op on ``HippocampusService`` emits an
:class:`InteractionEvent` after its main work completes. The bus stores
a bounded log and consults a :class:`TriggerPolicy` on each emit; any
matched trigger runs synchronously, so the user-facing call returns
*after* any auto-consolidate / auto-dream side effects have already
landed. There is no background thread - the system "self-learns"
reactively, on the next inbound op.

Reactor invariants
------------------
* Reactor callbacks for ``consolidate`` / ``dream`` are themselves
  service ops that emit their own events. ``react()`` therefore *skips*
  trigger evaluation when the originating event is itself a reactor
  output. This makes the call graph a tree of depth at most 2:

      user_call (encode|recall|forget|reflect)
          -> emit -> react -> [consolidate? dream?]
                              -> emit (no further react)

  No recursion on the reactor side, by construction.

* Events are append-only. The log is a bounded ring (``log_capacity``);
  callers wanting full history should externalise it.

* Triggers are *deterministic functions of the public state*
  (``ReactorState``). They are independently testable and contain no
  RL / learning - the "self-learning" emerges from the *side effects*
  the triggers schedule, not from the trigger logic itself.
"""

from __future__ import annotations

import time
from collections import deque
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any, Literal

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
    """One observable interaction with the memory service.

    ``kind``       which op produced this event
    ``timestamp``  wall-clock seconds (real)
    ``session_id`` namespace if applicable; None for global ops
    ``payload``    op-specific data: ids, counts, scores - kept small
                   (this object is logged in a bounded ring)
    """

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
    """When to schedule consolidate / dream as side effects.

    Defaults are conservative enough that a small test run (a handful
    of encodes/recalls in fast succession) does not trip them. Real
    deployments tune ``consolidate_after_novel_encodes`` down toward
    ~16 and ``dream_after_idle_seconds`` toward ~600 (10 min).
    """

    consolidate_after_novel_encodes: int = 32
    consolidate_cooldown_seconds: float = 60.0
    dream_after_events: int = 128
    dream_after_idle_seconds: float = 1800.0  # 30 min real

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


class EventBus:
    """Bounded event log + reactor for self-triggered follow-ups."""

    def __init__(
        self,
        *,
        policy: TriggerPolicy | None = None,
        log_capacity: int = 1024,
    ) -> None:
        if log_capacity <= 0:
            raise ValueError("log_capacity must be positive")
        self.policy = policy or TriggerPolicy()
        self.state = ReactorState()
        self._log: deque[InteractionEvent] = deque(maxlen=log_capacity)
        self.on_consolidate: Callable[[InteractionEvent], Awaitable[None]] | None = None
        self.on_dream: Callable[[InteractionEvent], Awaitable[None]] | None = None

    def log(self) -> list[InteractionEvent]:
        return list(self._log)

    def record(self, event: InteractionEvent) -> None:
        """Append the event and update reactor state counters."""
        self._log.append(event)
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

    async def react(self, event: InteractionEvent) -> list[str]:
        """Run any due reactor callbacks. Returns triggered action names.

        No-op when the originating event is itself a reactor output,
        which prevents recursion (see module docstring for the call
        graph guarantee).
        """
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
