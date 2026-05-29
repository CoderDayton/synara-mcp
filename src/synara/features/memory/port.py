"""Narrow service port for ops/.

The ops/ submodules (``encode``, ``recall``, ``consolidate``, ``forget``,
``reflect``) historically took the whole ``MemoryService`` as a
parameter, which made them un-testable in isolation and tangled every
new op into the full service surface.

:class:`MemoryServicePort` defines the structural subset the ops
actually need. ``MemoryService`` satisfies it by structure, so no
runtime change happens — but tests and alternative implementations can
now provide a stub that implements just this protocol.

Kept deliberately narrow: only the methods that ops/ call on
``self``-typed parameters. Anything an op imports directly from
``primitives/*`` does not appear here.
"""

from __future__ import annotations

import asyncio
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

if TYPE_CHECKING:  # pragma: no cover - typing only
    from .config import MemoryConfig
    from .hippocampus.complete import CompletionResult  # noqa: F401
    from .hippocampus.plasticity import PlasticityGraph
    from .hippocampus.separate import DGProjector
    from .hippocampus.successor import SuccessorRepresentation
    from .memory_types import MemoryType, MemoryTypeRegistry


@runtime_checkable
class _Collection(Protocol):
    """Subset of the simplevecdb collection used by ops/.

    Defined for type-checker hygiene; ops/ call these directly and we
    want the port to advertise what they touch.
    """

    async def count(self) -> int: ...
    async def add_texts(
        self,
        texts: Sequence[str],
        *,
        metadatas: Sequence[dict[str, Any]] | None = ...,
        embeddings: Sequence[Sequence[float]] | None = ...,
    ) -> Sequence[int]: ...
    async def get_documents(
        self,
        filter_dict: dict[str, Any] | None = ...,
        *,
        limit: int | None = ...,
        offset: int | None = ...,
    ) -> Sequence[tuple[int, str, dict[str, Any]]]: ...
    async def similarity_search(
        self,
        query: Any,
        *,
        k: int,
        filter: dict[str, Any] | None = ...,
    ) -> Sequence[Any]: ...
    async def update_metadata(
        self,
        updates: Sequence[tuple[int, dict[str, Any]]],
    ) -> None: ...
    async def increment_metadata(self, doc_id: int, deltas: dict[str, Any]) -> None: ...
    async def delete_by_ids(self, ids: Sequence[int]) -> None: ...
    async def get_embeddings_by_ids(
        self,
        ids: Sequence[int],
    ) -> dict[int, Sequence[float]]: ...
    async def update_embedding(
        self,
        doc_id: int,
        vector: Sequence[float],
        *,
        source: str = ...,
    ) -> None: ...


@runtime_checkable
class MemoryServicePort(Protocol):
    """Structural interface ops/ depend on.

    Implemented by :class:`MemoryService`; tests can stub the
    handful of attributes/methods listed here to drive an op in
    isolation.
    """

    config: MemoryConfig
    episodic: Any
    semantic: Any
    memory_types: MemoryTypeRegistry
    schema_candidates: Any
    # Package-private state that ops/ read or mutate directly on ``self``.
    # They are single-underscore implementation detail of ``MemoryService``
    # but appear here so in-package ops type-check against the abstract
    # port; external callers should not depend on them.
    _sr: SuccessorRepresentation | None
    _plasticity: PlasticityGraph
    _dg: DGProjector | None
    _replay_cursor: int
    _consolidate_epoch: int
    _hygiene_counters: dict[str, int]

    def collection_for(self, kind: MemoryType) -> Any: ...
    def _ensure_projector(self, dim: int) -> DGProjector: ...

    async def vectorise(self, texts: Sequence[str]) -> list[list[float]] | None: ...
    async def query_arg(self, query: str) -> str | list[float]: ...
    async def bump_retrieval(self, doc_id: int, current: dict[str, Any]) -> None: ...
    async def embedding_dimension(self) -> int | None: ...
    # Package-private (single underscore): only callers inside
    # ``features/memory`` hold per-doc locks for read-modify-write
    # metadata sequences. Kept on the Protocol so the in-package
    # callers type-check against the abstract port.
    def _doc_lock(self, doc_id: int) -> asyncio.Lock: ...
    async def _evict_and_delete(self, ids: list[int]) -> None: ...
    async def _ensure_index_ready(self) -> None: ...


__all__ = ["MemoryServicePort"]
