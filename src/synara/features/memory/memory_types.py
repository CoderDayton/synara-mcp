"""Memory-type enum + registry.

A small, explicit replacement for the implicit episodic/semantic split
that used to live as bare collection-name strings on
``MemoryConfig``. New memory kinds (procedural, conceptual, ...) can
be added by registering a :class:`MemoryTypeSpec` instead of touching
``encode``, ``consolidate``, and ``service`` in parallel.

The registry is intentionally a small dataclass — not a global mutable
singleton — so tests can supply their own and so a service can be
constructed with a non-default layout without forking the config.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from synara.core.errors import ValidationError


class MemoryType(StrEnum):
    """Discriminator for the kind of memory a record belongs to.

    Subclassing :class:`StrEnum` keeps the value JSON-serialisable so
    it can land directly in document metadata when callers want a
    stable, human-readable tag.
    """

    EPISODIC = "episodic"
    SEMANTIC = "semantic"
    # Candidate-to-promotion gate: cluster gists wait here until their
    # embedding recurs across enough consolidate passes
    # (``consolidate_min_recurrence``) to earn a SEMANTIC schema row. Filtered out of
    # production recall paths by the service code; isolated in its own
    # collection so similarity_search on the semantic store can stay
    # filter-free.
    SCHEMA_CANDIDATE = "schema_candidate"


@dataclass(frozen=True, slots=True)
class MemoryTypeSpec:
    """How one memory kind is realised on top of the vector store.

    collection: name of the simplevecdb collection backing this kind.
    consolidate_into: optional target kind; episodes of *this* type are
        clustered into schemas of ``consolidate_into`` by the
        consolidation pass. ``None`` means the kind has no consolidation
        target (e.g. semantic itself, or future procedural records that
        are their own terminal store).
    """

    type: MemoryType
    collection: str
    consolidate_into: MemoryType | None = None


@dataclass(frozen=True, slots=True)
class MemoryTypeRegistry:
    """Resolved table of memory-type specs.

    Built from a :class:`MemoryConfig` (see
    :func:`registry_from_config`) or directly by callers that want a
    bespoke layout. Lookup is O(1) and the registry is immutable.
    """

    by_type: Mapping[MemoryType, MemoryTypeSpec] = field(default_factory=dict)

    def __post_init__(self) -> None:
        # Frozen dataclass — coerce input to a plain dict so callers can
        # pass any Mapping and downstream consumers see a stable type.
        object.__setattr__(self, "by_type", dict(self.by_type))
        for k, spec in self.by_type.items():
            if k != spec.type:
                raise ValueError(
                    f"registry key {k!r} does not match spec.type {spec.type!r}",
                )
            target = spec.consolidate_into
            if target is not None and target not in self.by_type:
                raise ValueError(
                    f"{k.value!r} consolidates into {target.value!r} which is not registered",
                )

    def __iter__(self) -> Iterator[MemoryTypeSpec]:
        return iter(self.by_type.values())

    def spec(self, t: MemoryType) -> MemoryTypeSpec:
        try:
            return self.by_type[t]
        except KeyError as exc:
            raise KeyError(f"memory type {t.value!r} is not registered") from exc

    def collection_name(self, t: MemoryType) -> str:
        return self.spec(t).collection

    def consolidation_target(self, t: MemoryType) -> MemoryType | None:
        return self.spec(t).consolidate_into

    def has(self, t: MemoryType) -> bool:
        return t in self.by_type


def default_registry(
    *,
    episodic_collection: str,
    semantic_collection: str,
    # Matches ``MemoryConfig.schema_candidate_collection`` so a caller that
    # omits this arg lands on the same collection the service uses.
    schema_candidate_collection: str = "memory_schema_candidates",
) -> MemoryTypeRegistry:
    """Build the standard registry used by the existing code path.

    Includes :attr:`MemoryType.SCHEMA_CANDIDATE` for the candidate-
    to-promotion gate; the collection exists unconditionally so the
    promotion threshold can be tuned at runtime without a schema migration.
    """
    return MemoryTypeRegistry(
        by_type={
            MemoryType.EPISODIC: MemoryTypeSpec(
                type=MemoryType.EPISODIC,
                collection=episodic_collection,
                consolidate_into=MemoryType.SEMANTIC,
            ),
            MemoryType.SEMANTIC: MemoryTypeSpec(
                type=MemoryType.SEMANTIC,
                collection=semantic_collection,
                consolidate_into=None,
            ),
            MemoryType.SCHEMA_CANDIDATE: MemoryTypeSpec(
                type=MemoryType.SCHEMA_CANDIDATE,
                collection=schema_candidate_collection,
                consolidate_into=None,
            ),
        },
    )


# --- Scope axis (orthogonal to MemoryType) -------------------------------
# A record is either tied to a session (visible only when recalled in that
# session) or global (visible from any session). Scope is stored as a plain
# metadata field, independent of the episodic/semantic type, so it needs no
# schema migration: legacy records that predate the field fall back to
# ``session_id`` presence (see :func:`in_session_scope`).
SCOPE_SESSION = "session"
SCOPE_GLOBAL = "global"


def resolve_scope(scope: str | None, session_id: str | None) -> str:
    """Normalise an explicit ``scope`` against session presence.

    An explicit value wins (and is validated); otherwise a record is
    session-scoped when it carries a ``session_id`` -- the rule applied
    at every write site. Global is opt-in: a global record surfaces in
    every future session, so promoting a write to global is a deliberate
    act, never a silent default. A bare call (no scope, no session_id)
    is therefore rejected rather than falling back to a global write.
    """
    if scope is not None:
        if scope not in (SCOPE_SESSION, SCOPE_GLOBAL):
            raise ValidationError(
                f"unknown scope {scope!r}; expected {SCOPE_SESSION!r} or {SCOPE_GLOBAL!r}"
            )
        if scope == SCOPE_SESSION and not session_id:
            raise ValidationError("session-scoped memory requires a session_id")
        return scope
    if session_id:
        return SCOPE_SESSION
    raise ValidationError(
        "pass a session_id for a session-scoped memory, or scope='global' "
        "to write a global one; refusing to default to a global write"
    )


def in_session_scope(md: Mapping[str, Any] | None, *, session_id: str | None) -> bool:
    """True if a record is visible from the caller's ``session_id``.

    Honours an explicit ``scope`` field; for legacy records without one,
    infers scope from ``session_id`` presence -- so existing episodes
    (always session-stamped) stay session-scoped and existing scope-less
    semantics stay global. A global record is always visible; a session
    record only when its ``session_id`` matches the caller's.
    """
    md = md or {}
    scope = md.get("scope")
    if scope == SCOPE_GLOBAL:
        return True
    if scope == SCOPE_SESSION:
        return session_id is not None and str(md.get("session_id", "")) == session_id
    # Legacy / missing scope: infer from session_id presence.
    rec_sid = md.get("session_id")
    if rec_sid:
        return session_id is not None and str(rec_sid) == session_id
    return True


__all__ = [
    "SCOPE_GLOBAL",
    "SCOPE_SESSION",
    "MemoryType",
    "MemoryTypeRegistry",
    "MemoryTypeSpec",
    "default_registry",
    "in_session_scope",
    "resolve_scope",
]
