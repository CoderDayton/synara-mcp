"""Hippocampus service.

Pure logic: no MCP types here. ``tools.py`` is the only layer that touches
fastmcp. The service is constructed with a ``simplevecdb.VectorDB`` and an
optional embedder; both are dependency-injected so tests can drive the
service against ``:memory:`` with a deterministic embedder.

Operation surface is split across sibling modules to keep file size
modest:
``service.py``      - encode, recall, stats, shared internals
``consolidate.py``  - episodic -> semantic transformation
``forget.py``       - Ebbinghaus-style decay + pruning
``reflect.py``      - schema/episode summary for a session
"""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

from simplevecdb import AsyncVectorDB

from synara.core.errors import ValidationError

# Sentinel for "this episode has not yet been consolidated into a semantic
# schema". simplevecdb's filter format only supports exact equality and
# IN — not "IS NULL" — so we use 0, with positive ints referencing a real
# semantic doc id.
UNCONSOLIDATED: int = 0


@dataclass(frozen=True, slots=True)
class HippocampusConfig:
    """Tunable parameters of the memory framework."""

    episodic_collection: str = "hippocampus_episodic"
    semantic_collection: str = "hippocampus_semantic"
    # Cosine distance below this threshold counts as a duplicate within a
    # session — encode_episode bumps the existing record instead of inserting.
    dedup_distance: float = 0.05
    # Ebbinghaus-style decay time constant. Strength halves roughly every
    # ``decay_tau_seconds * ln 2`` for an episode that is never retrieved.
    decay_tau_seconds: float = 7.0 * 24.0 * 3600.0
    # Per-retrieval boost added to strength score.
    retrieval_boost: float = 0.05
    # Minimum cluster size that yields a semantic schema during consolidation.
    consolidate_min_cluster: int = 2


def now_seconds() -> float:
    return time.time()


class HippocampusService:
    """Episodic + semantic memory over two simplevecdb collections."""

    def __init__(
        self,
        db: AsyncVectorDB,
        config: HippocampusConfig | None = None,
        *,
        embed_fn: Callable[[str], Sequence[float]] | None = None,
    ) -> None:
        self.config = config or HippocampusConfig()
        self.db = db
        # store_embeddings=True keeps vectors at hand for cluster()/rebuild
        # even after process restarts.
        self.episodic = db.collection(self.config.episodic_collection, store_embeddings=True)
        self.semantic = db.collection(self.config.semantic_collection, store_embeddings=True)
        self._embed = embed_fn

    # ------------------------------------------------------------------ embed
    def vectorise(self, texts: Sequence[str]) -> list[list[float]] | None:
        """Return embeddings for ``texts`` or ``None`` to defer to simplevecdb."""
        if self._embed is None:
            return None
        return [list(self._embed(t)) for t in texts]

    def query_arg(self, query: str) -> str | list[float]:
        """Shape a query for simplevecdb: text (auto-embed) or precomputed vec."""
        if self._embed is None:
            return query
        return list(self._embed(query))

    # ------------------------------------------------------------------ encode
    async def encode_episode(
        self,
        content: str,
        session_id: str,
        *,
        tags: Sequence[str] | None = None,
        salience: float = 0.5,
    ) -> dict[str, Any]:
        if not content.strip():
            raise ValidationError("content must be non-empty")
        if not session_id:
            raise ValidationError("session_id must be non-empty")
        if not 0.0 <= salience <= 1.0:
            raise ValidationError("salience must be in [0, 1]")

        # Pattern separation: refuse near-duplicates within the same session.
        existing = await self.episodic.similarity_search(
            self.query_arg(content), k=1, filter={"session_id": session_id}
        )
        if existing and existing[0][1] <= self.config.dedup_distance:
            doc, dist = existing[0]
            doc_id = int(doc.metadata.get("id", -1))
            if doc_id >= 0:
                await self.bump_retrieval(doc_id, doc.metadata)
            return {
                "id": doc_id,
                "deduped": True,
                "distance": float(dist),
                "session_id": session_id,
            }

        encoded_at = now_seconds()
        meta: dict[str, Any] = {
            "session_id": session_id,
            "tags": list(tags) if tags else [],
            "salience": float(salience),
            "encoded_at": encoded_at,
            "last_accessed": encoded_at,
            "retrieval_count": 0,
            "consolidated_into": UNCONSOLIDATED,
        }
        ids = await self.episodic.add_texts(
            [content], metadatas=[meta], embeddings=self.vectorise([content])
        )
        new_id = int(ids[0])
        # Mirror the auto-assigned id into metadata so similarity_search
        # results carry it without a follow-up join.
        await self.episodic.update_metadata([(new_id, {"id": new_id})])
        return {
            "id": new_id,
            "deduped": False,
            "distance": None,
            "session_id": session_id,
        }

    # ------------------------------------------------------------------ recall
    async def recall(
        self,
        query: str,
        *,
        session_id: str | None = None,
        k: int = 8,
        mode: str = "auto",
    ) -> list[dict[str, Any]]:
        if not query.strip():
            raise ValidationError("query must be non-empty")
        if k <= 0:
            return []
        if mode not in {"auto", "episodic", "semantic", "hybrid"}:
            raise ValidationError(f"unknown recall mode: {mode}")

        ep_filter: dict[str, Any] | None = {"session_id": session_id} if session_id else None
        q = self.query_arg(query)
        merged: list[tuple[int, str, dict[str, Any], float, str]] = []

        if mode in {"auto", "semantic", "hybrid"} and await self.semantic.count() > 0:
            for doc, dist in await self.semantic.similarity_search(q, k=k):
                merged.append(
                    (
                        int(doc.metadata.get("id", -1)),
                        doc.page_content,
                        dict(doc.metadata),
                        float(dist),
                        "semantic",
                    )
                )
        if mode in {"auto", "episodic", "hybrid"} and await self.episodic.count() > 0:
            for doc, dist in await self.episodic.similarity_search(q, k=k, filter=ep_filter):
                merged.append(
                    (
                        int(doc.metadata.get("id", -1)),
                        doc.page_content,
                        dict(doc.metadata),
                        float(dist),
                        "episodic",
                    )
                )

        merged.sort(key=lambda r: r[3])
        out: list[dict[str, Any]] = []
        for doc_id, text, md, dist, source in merged[:k]:
            out.append(
                {
                    "id": doc_id,
                    "content": text,
                    "distance": dist,
                    "source": source,
                    "metadata": md,
                }
            )
            if source == "episodic" and doc_id >= 0:
                await self.bump_retrieval(doc_id, md)
        return out

    async def bump_retrieval(self, doc_id: int, current: dict[str, Any]) -> None:
        rc = int(current.get("retrieval_count", 0)) + 1
        await self.episodic.update_metadata(
            [(doc_id, {"retrieval_count": rc, "last_accessed": now_seconds()})]
        )

    # ----------------------------------------------------------------- stats
    async def stats(self) -> dict[str, int]:
        return {
            "episodic_count": await self.episodic.count(),
            "semantic_count": await self.semantic.count(),
        }

    # Sibling-module operations are bound as thin delegates so callers see
    # one cohesive service surface. The sub-modules import only the
    # already-bound symbols above (UNCONSOLIDATED / now_seconds), avoiding
    # an import cycle.
    async def consolidate(
        self,
        *,
        session_id: str | None = None,
        n_clusters: int | None = None,
        min_cluster_size: int | None = None,
    ) -> list[dict[str, Any]]:
        return await _consolidate_mod.run(
            self,
            session_id=session_id,
            n_clusters=n_clusters,
            min_cluster_size=min_cluster_size,
        )

    async def forget(
        self,
        *,
        strength_floor: float = 0.05,
        decay_tau_seconds: float | None = None,
        dry_run: bool = True,
        max_scan: int = 1000,
    ) -> dict[str, Any]:
        return await _forget_mod.run(
            self,
            strength_floor=strength_floor,
            decay_tau_seconds=decay_tau_seconds,
            dry_run=dry_run,
            max_scan=max_scan,
        )

    async def reflect(
        self,
        *,
        session_id: str,
        query: str | None = None,
        k: int = 5,
    ) -> dict[str, Any]:
        return await _reflect_mod.run(self, session_id=session_id, query=query, k=k)


# Late imports avoid a top-of-file cycle: the sub-modules need
# UNCONSOLIDATED and now_seconds (bound above), and they only reference
# HippocampusService through TYPE_CHECKING.
from . import consolidate as _consolidate_mod  # noqa: E402
from . import forget as _forget_mod  # noqa: E402
from . import reflect as _reflect_mod  # noqa: E402
