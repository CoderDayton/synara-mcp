"""Extensibility-registry tests: SignalRegistry + MemoryTypeRegistry."""

from __future__ import annotations

import pytest

from synara.core.errors import ValidationError
from synara.features.memory.amygdala.signals import (
    SignalRegistry,
    SignalSpec,
    default_signal_registry,
)
from synara.features.memory.memory_types import (
    SCOPE_GLOBAL,
    SCOPE_SESSION,
    MemoryType,
    MemoryTypeRegistry,
    MemoryTypeSpec,
    default_registry,
    in_session_scope,
    resolve_scope,
)

# ---- SignalRegistry ---------------------------------------------------


def test_default_signal_registry_reproduces_legacy_behaviour() -> None:
    reg = default_signal_registry()
    assert reg.specs == ()
    out = reg.derive("see src/foo.py and `bar`\nValueError: x")
    assert out["has_traceback"] is True
    assert "src/foo.py" in out["references"]
    # legacy salience composition still active
    assert reg.salience(out) > reg.base_salience


def test_signal_registry_custom_bool_and_numeric_specs() -> None:
    reg = SignalRegistry(
        specs=(
            SignalSpec(name="shouty", weight=0.2, compute=lambda c: c.isupper()),
            SignalSpec(name="wordcount", weight=0.1, compute=lambda c: len(c.split())),
        ),
        include_legacy_structural=False,
    )
    out = reg.derive("LOUD TEXT HERE")
    assert out["shouty"] is True
    assert out["wordcount"] == 3
    # no legacy keys when structural pass disabled
    assert "has_traceback" not in out
    s = reg.salience(out)
    # base + bool weight + numeric weight
    assert s == pytest.approx(reg.base_salience + 0.2 + 0.1)


def test_signal_registry_false_bool_spec_recorded_but_unweighted() -> None:
    reg = SignalRegistry(
        specs=(SignalSpec(name="flag", weight=0.5, compute=lambda _c: False),),
        include_legacy_structural=False,
    )
    out = reg.derive("anything")
    assert out["flag"] is False  # booleans stored as-is
    assert reg.salience(out) == pytest.approx(reg.base_salience)


def test_signal_registry_zero_numeric_spec_skipped_from_metadata() -> None:
    reg = SignalRegistry(
        specs=(SignalSpec(name="n", weight=0.3, compute=lambda _c: 0),),
        include_legacy_structural=False,
    )
    out = reg.derive("anything")
    assert "n" not in out  # falsy non-bool dropped
    assert reg.salience({"n": 0}) == pytest.approx(reg.base_salience)


def test_signal_registry_salience_clamped_to_unit_interval() -> None:
    reg = SignalRegistry(
        specs=(SignalSpec(name="big", weight=99.0, compute=lambda _c: True),),
        include_legacy_structural=False,
    )
    assert reg.salience(reg.derive("x")) == 1.0


# ---- MemoryTypeRegistry ----------------------------------------------


def test_default_registry_lookups() -> None:
    reg = default_registry(
        episodic_collection="ep",
        semantic_collection="sem",
        schema_candidate_collection="cand",
    )
    assert reg.collection_name(MemoryType.EPISODIC) == "ep"
    assert reg.collection_name(MemoryType.SEMANTIC) == "sem"
    assert reg.collection_name(MemoryType.SCHEMA_CANDIDATE) == "cand"
    assert reg.consolidation_target(MemoryType.EPISODIC) is MemoryType.SEMANTIC
    assert reg.consolidation_target(MemoryType.SEMANTIC) is None
    assert reg.consolidation_target(MemoryType.SCHEMA_CANDIDATE) is None
    assert reg.has(MemoryType.EPISODIC) is True
    assert {s.type for s in reg} == {
        MemoryType.EPISODIC,
        MemoryType.SEMANTIC,
        MemoryType.SCHEMA_CANDIDATE,
    }


def test_registry_spec_unregistered_raises_keyerror() -> None:
    reg = MemoryTypeRegistry(
        by_type={MemoryType.SEMANTIC: MemoryTypeSpec(type=MemoryType.SEMANTIC, collection="sem")}
    )
    assert reg.has(MemoryType.EPISODIC) is False
    with pytest.raises(KeyError, match="episodic"):
        reg.spec(MemoryType.EPISODIC)


def test_registry_key_must_match_spec_type() -> None:
    with pytest.raises(ValueError, match=r"does not match spec\.type"):
        MemoryTypeRegistry(
            by_type={MemoryType.EPISODIC: MemoryTypeSpec(type=MemoryType.SEMANTIC, collection="x")}
        )


def test_registry_dangling_consolidation_target_rejected() -> None:
    with pytest.raises(ValueError, match="not registered"):
        MemoryTypeRegistry(
            by_type={
                MemoryType.EPISODIC: MemoryTypeSpec(
                    type=MemoryType.EPISODIC,
                    collection="ep",
                    consolidate_into=MemoryType.SEMANTIC,
                )
            }
        )


# ---- Scope axis -------------------------------------------------------


def test_resolve_scope_explicit_and_inferred() -> None:
    assert resolve_scope("global", None) == SCOPE_GLOBAL
    assert resolve_scope("session", "s1") == SCOPE_SESSION
    # Unset: inferred from session_id presence.
    assert resolve_scope(None, "s1") == SCOPE_SESSION
    # Global is opt-in: a bare call (no scope, no session_id) is rejected
    # rather than silently defaulting to a global write.
    with pytest.raises(ValidationError):
        resolve_scope(None, None)


def test_resolve_scope_rejects_unknown() -> None:
    with pytest.raises(ValidationError):
        resolve_scope("bogus", "s1")


def test_in_session_scope_explicit() -> None:
    assert in_session_scope({"scope": SCOPE_GLOBAL}, session_id="s1") is True
    assert in_session_scope({"scope": SCOPE_GLOBAL}, session_id=None) is True
    assert in_session_scope({"scope": SCOPE_SESSION, "session_id": "s1"}, session_id="s1") is True
    assert in_session_scope({"scope": SCOPE_SESSION, "session_id": "s1"}, session_id="s2") is False
    # A session-scoped record has nothing to match against without a caller session.
    assert in_session_scope({"scope": SCOPE_SESSION, "session_id": "s1"}, session_id=None) is False


def test_in_session_scope_legacy_fallback() -> None:
    # No scope field: infer from session_id presence (zero-migration path).
    assert in_session_scope({"session_id": "s1"}, session_id="s1") is True
    assert in_session_scope({"session_id": "s1"}, session_id="s2") is False
    assert in_session_scope({}, session_id="s1") is True  # no session -> global
    assert in_session_scope(None, session_id="s1") is True
