"""Request-scoped tracing for the recall path.

A single :class:`RequestContext` is published on a ``ContextVar`` for
the duration of a service op. Any layer inside the op (ops/recall,
primitives/complete, primitives/plasticity, ...) can append a span
without threading a parameter through every signature.

Tracing is *opt-in* via ``MemoryConfig.tracing_enabled``. When
disabled, :func:`start_request` returns a sentinel that no-ops on
:meth:`RequestContext.span`, so the overhead of running with tracing
off is one ContextVar read.
"""

from __future__ import annotations

import contextlib
import time
from collections.abc import Iterator, Mapping
from contextvars import ContextVar, Token
from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class Span:
    """A single timed step within a request."""

    name: str
    started_at: float
    duration_seconds: float
    payload: Mapping[str, Any] | None = None

    def as_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "name": self.name,
            "started_at": self.started_at,
            "duration_seconds": self.duration_seconds,
        }
        if self.payload:
            out["payload"] = dict(self.payload)
        return out


@dataclass(slots=True)
class RequestContext:
    """Container for trace state of one in-flight service op.

    Active context is published via :data:`_CTX`; :meth:`span` records
    a step. ``enabled=False`` makes every method a cheap no-op so the
    same code path works for both modes.
    """

    request_id: str
    started_at: float
    enabled: bool = True
    spans: list[Span] = field(default_factory=list)

    @contextlib.contextmanager
    def span(
        self,
        name: str,
        *,
        payload: Mapping[str, Any] | None = None,
    ) -> Iterator[None]:
        """Record one timed step.

        Used as ``with ctx.span("name"): ...``. Disabled contexts skip
        the timing call entirely.
        """
        if not self.enabled:
            yield
            return
        start = time.perf_counter()
        try:
            yield
        finally:
            self.spans.append(
                Span(
                    name=name,
                    started_at=start,
                    duration_seconds=time.perf_counter() - start,
                    payload=dict(payload) if payload else None,
                ),
            )

    def add_event(self, name: str, **payload: Any) -> None:
        """Record a point-event (zero duration). No-op when disabled."""
        if not self.enabled:
            return
        self.spans.append(
            Span(
                name=name,
                started_at=time.perf_counter(),
                duration_seconds=0.0,
                payload=dict(payload) if payload else None,
            ),
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "request_id": self.request_id,
            "started_at": self.started_at,
            "duration_seconds": time.perf_counter() - self.started_at if self.enabled else 0.0,
            "spans": [s.as_dict() for s in self.spans],
        }


_CTX: ContextVar[RequestContext | None] = ContextVar(
    "synara_memory_request_ctx",
    default=None,
)


def current_context() -> RequestContext | None:
    """Return the active :class:`RequestContext`, or ``None`` outside one."""
    return _CTX.get()


@contextlib.contextmanager
def start_request(
    name: str,
    *,
    enabled: bool,
    request_id: str | None = None,
) -> Iterator[RequestContext]:
    """Publish a :class:`RequestContext` for the lifetime of the block.

    ``enabled=False`` still publishes a context, but spans recorded on
    it are dropped — keeping the call surface symmetric.
    """
    rid = request_id or f"{name}-{int(time.time() * 1000)}"
    ctx = RequestContext(
        request_id=rid,
        started_at=time.perf_counter(),
        enabled=bool(enabled),
    )
    token: Token[RequestContext | None] = _CTX.set(ctx)
    try:
        yield ctx
    finally:
        _CTX.reset(token)


def record_span(name: str, *, payload: Mapping[str, Any] | None = None) -> Any:
    """Convenience: context-manager that ignores no-context state.

    Use inside primitives that may run either standalone (no active
    request) or under a request. Always safe to enter.
    """
    ctx = current_context()
    if ctx is None or not ctx.enabled:
        return contextlib.nullcontext()
    return ctx.span(name, payload=payload)


__all__ = [
    "RequestContext",
    "Span",
    "current_context",
    "record_span",
    "start_request",
]
