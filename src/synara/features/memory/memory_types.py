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


class MemoryType(StrEnum):
    """Discriminator for the kind of memory a record belongs to.

    Subclassing :class:`StrEnum` keeps the value JSON-serialisable so
    it can land directly in document metadata when callers want a
    stable, human-readable tag.
    """

    EPISODIC = "episodic"
    SEMANTIC = "semantic"
    # v2 candidate-to-promotion gate (consolidate_min_recurrence > 1):
    # cluster gists wait here until their embedding recurs across enough
    # consolidate passes to earn a SEMANTIC schema row. Filtered out of
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
    schema_candidate_collection: str = "schema_candidates",
) -> MemoryTypeRegistry:
    """Build the standard registry used by the existing code path.

    Includes :attr:`MemoryType.SCHEMA_CANDIDATE` for the v2 candidate-
    to-promotion gate; the collection exists unconditionally so the
    feature can be toggled at runtime without a schema migration.
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


__all__ = [
    "MemoryType",
    "MemoryTypeRegistry",
    "MemoryTypeSpec",
    "default_registry",
]
