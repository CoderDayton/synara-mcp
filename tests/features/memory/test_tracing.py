"""Request-scoped tracing tests (pure, synchronous)."""

from __future__ import annotations

import contextlib

from synara.features.memory.tracing import (
    RequestContext,
    Span,
    current_context,
    record_span,
    start_request,
)


def test_span_as_dict_includes_payload_when_present() -> None:
    s = Span(name="x", started_at=1.0, duration_seconds=0.5, payload={"k": 1})
    d = s.as_dict()
    assert d == {
        "name": "x",
        "started_at": 1.0,
        "duration_seconds": 0.5,
        "payload": {"k": 1},
    }
    bare = Span(name="y", started_at=2.0, duration_seconds=0.0).as_dict()
    assert "payload" not in bare


def test_enabled_context_records_span_and_event() -> None:
    ctx = RequestContext(request_id="r1", started_at=0.0, enabled=True)
    with ctx.span("step", payload={"n": 2}):
        pass
    ctx.add_event("marker", detail="hit")
    names = [s.name for s in ctx.spans]
    assert names == ["step", "marker"]
    assert ctx.spans[0].payload == {"n": 2}
    assert ctx.spans[1].duration_seconds == 0.0
    out = ctx.as_dict()
    assert out["request_id"] == "r1"
    assert len(out["spans"]) == 2
    assert out["duration_seconds"] >= 0.0


def test_disabled_context_is_a_noop() -> None:
    ctx = RequestContext(request_id="r2", started_at=0.0, enabled=False)
    with ctx.span("ignored"):
        pass
    ctx.add_event("ignored2")
    assert ctx.spans == []
    assert ctx.as_dict()["duration_seconds"] == 0.0


def test_span_records_even_when_body_raises() -> None:
    ctx = RequestContext(request_id="r3", started_at=0.0, enabled=True)
    with contextlib.suppress(RuntimeError), ctx.span("boom"):
        raise RuntimeError("fail")
    assert [s.name for s in ctx.spans] == ["boom"]


def test_start_request_publishes_and_resets_contextvar() -> None:
    assert current_context() is None
    with start_request("recall", enabled=True) as ctx:
        assert current_context() is ctx
        assert ctx.request_id.startswith("recall-")
        with record_span("inner", payload={"a": 1}):
            pass
    assert current_context() is None
    assert [s.name for s in ctx.spans] == ["inner"]


def test_start_request_honours_explicit_request_id_and_disabled() -> None:
    with start_request("op", enabled=False, request_id="fixed-id") as ctx:
        assert ctx.request_id == "fixed-id"
        with record_span("dropped"):
            pass
    assert ctx.spans == []


def test_record_span_without_active_context_is_nullcontext() -> None:
    assert current_context() is None
    cm = record_span("orphan")
    with cm:
        pass  # must not raise


def test_record_span_disabled_context_is_nullcontext() -> None:
    with start_request("op", enabled=False):
        with record_span("nope"):
            pass
        assert current_context() is not None
        assert current_context().enabled is False  # type: ignore[union-attr]
