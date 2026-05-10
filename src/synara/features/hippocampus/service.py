"""Hippocampus service.

Pure logic: no MCP types here. ``tools.py`` is the only layer that touches
fastmcp. The service is constructed with a ``simplevecdb.VectorDB`` and an
optional embedder; both are dependency-injected so tests can drive the
service against ``:memory:`` with a deterministic embedder.

This module owns only the cross-cutting glue — collection wiring,
embedder normalisation, the DG projector + SR data structures, and the
small handful of helpers that every operation needs (``bump_retrieval``,
``fetch_episode_group``, ``stats``). Each high-level operation lives in
its own sibling module and is exposed here as a thin delegate so callers
see a single cohesive class.

Operation surface
-----------------
``encode.py``       - episode encoding + DG/cosine dedup + theta segments
``recall.py``       - hybrid search + CA3 completion + SR re-rank
``consolidate.py``  - episodic -> semantic transformation
``forget.py``       - power-law decay + pruning
``reflect.py``      - schema/episode summary for a session
``complete.py``     - CA3 iterative pattern completion (used by recall)
``segment.py``      - theta-style intra-episode text segmentation
``separate.py``     - DG sparse projection + Jaccard
``successor.py``    - temporal-context successor representation
``config.py``       - tunable parameters dataclass
"""

from __future__ import annotations

import asyncio
import inspect
import time
from collections.abc import Awaitable, Callable, Sequence
from typing import Any

from simplevecdb import AsyncVectorDB

from .config import HippocampusConfig
from .primitives.separate import DGProjector as _DGProjector
from .primitives.successor import SuccessorRepresentation as _SR

# Either a sync embedder ``f(text) -> vec`` or an async one
# ``async f(text) -> vec``. The service normalises both to async at
# construction so internal call sites only ever ``await``.
EmbedFn = Callable[[str], Sequence[float] | Awaitable[Sequence[float]]]

# Sentinel for "this episode has not yet been consolidated into a semantic
# schema". simplevecdb's filter format only supports exact equality and
# IN — not "IS NULL" — so we use 0, with positive ints referencing a real
# semantic doc id.
UNCONSOLIDATED: int = 0

# ``HippocampusConfig`` is re-exported so existing code (and tests) that
# imports it from ``service`` keeps working after the split-out.
__all__ = [
    "UNCONSOLIDATED",
    "EmbedFn",
    "HippocampusConfig",
    "HippocampusService",
    "now_seconds",
]


def now_seconds() -> float:
    return time.time()


def _normalise_embed_fn(fn: EmbedFn) -> Callable[[str], Awaitable[Sequence[float]]]:
    """Wrap a sync embed_fn so all internal calls can ``await`` uniformly.

    Sync embedders (e.g. the hash-based test embedder) are pushed onto a
    worker thread to keep blocking work off the event loop; async ones
    are returned as-is.
    """
    if inspect.iscoroutinefunction(fn):
        return fn

    async def _wrapped(text: str) -> Sequence[float]:
        result = await asyncio.to_thread(fn, text)
        # An async callable still flagged as not-coroutinefunction
        # (e.g. a lambda returning a coroutine) ends up returning an
        # Awaitable; honour it.
        if inspect.isawaitable(result):
            return await result
        return result

    return _wrapped


class HippocampusService:
    """Episodic + semantic memory over two simplevecdb collections."""

    def __init__(
        self,
        db: AsyncVectorDB,
        config: HippocampusConfig | None = None,
        *,
        embed_fn: EmbedFn | None = None,
    ) -> None:
        self.config = config or HippocampusConfig()
        self.db = db
        # store_embeddings=True keeps vectors at hand for cluster()/rebuild
        # even after process restarts.
        self.episodic = db.collection(self.config.episodic_collection, store_embeddings=True)
        self.semantic = db.collection(self.config.semantic_collection, store_embeddings=True)
        self._embed = _normalise_embed_fn(embed_fn) if embed_fn is not None else None
        # Lazily constructed once we observe the embedding dimension.
        self._dg: _DGProjector | None = None
        self._sr: _SR | None = (
            _SR(
                gamma=self.config.sr_gamma,
                alpha=self.config.sr_alpha,
                window_seconds=self.config.sr_window_seconds,
                omega_max=self.config.sr_omega_max,
                cold_start_ratio=self.config.sr_cold_start_ratio,
            )
            if self.config.sr_enabled
            else None
        )

    # ------------------------------------------------------------------ embed
    async def vectorise(self, texts: Sequence[str]) -> list[list[float]] | None:
        """Return embeddings for ``texts`` or ``None`` to defer to simplevecdb."""
        if self._embed is None:
            return None
        return [list(await self._embed(t)) for t in texts]

    async def query_arg(self, query: str) -> str | list[float]:
        """Shape a query for simplevecdb: text (auto-embed) or precomputed vec."""
        if self._embed is None:
            return query
        return list(await self._embed(query))

    def _ensure_projector(self, dim: int) -> _DGProjector:
        """Build (or rebuild on dim change) the DG projector lazily."""
        if self._dg is None or self._dg.dim != dim:
            self._dg = _DGProjector(
                dim=dim,
                expansion=self.config.dg_expansion,
                sparsity=self.config.dg_sparsity,
                seed=self.config.dg_seed,
            )
        return self._dg

    # ----------------------------------------------------------------- access
    async def bump_retrieval(self, doc_id: int, current: dict[str, Any]) -> None:
        rc = int(current.get("retrieval_count", 0)) + 1
        now = now_seconds()
        history = list(current.get("access_history") or [])
        history.append(now)
        # FIFO-evict the oldest access timestamps once we exceed the cap.
        cap = self.config.access_history_cap
        if cap > 0 and len(history) > cap:
            history = history[-cap:]
        await self.episodic.update_metadata(
            [
                (
                    doc_id,
                    {
                        "retrieval_count": rc,
                        "last_accessed": now,
                        "access_history": history,
                    },
                )
            ]
        )

    async def fetch_episode_group(
        self,
        group_id: int,
        *,
        session_id: str | None = None,
    ) -> list[dict[str, Any]]:
        """Return the ordered sub-records of a theta-segmented episode.

        Sub-records are sorted by ``position_in_episode``. The first
        segment carries ``episode_group_id == its own id`` so passing
        any sub-record's id as ``group_id`` recovers the full ordered
        walk. Pass ``session_id`` for defense-in-depth against any
        future change to id allocation that could let group ids appear
        in more than one namespace.
        """
        flt: dict[str, Any] = {"episode_group_id": group_id}
        if session_id:
            flt["session_id"] = session_id
        rows = await self.episodic.get_documents(flt)
        items: list[dict[str, Any]] = []
        for doc_id, text, md in rows:
            items.append(
                {
                    "id": int(doc_id),
                    "content": text,
                    "position": int(md.get("position_in_episode", 0)),
                    "metadata": md,
                }
            )
        items.sort(key=lambda r: r["position"])
        return items

    # ------------------------------------------------------------------ stats
    async def stats(self) -> dict[str, int]:
        return {
            "episodic_count": await self.episodic.count(),
            "semantic_count": await self.semantic.count(),
        }

    # --------------------------------------------------- operation delegates
    # Sibling-module operations are bound as thin delegates so callers
    # see one cohesive service surface. The sub-modules import only the
    # already-bound symbols above (UNCONSOLIDATED / now_seconds), avoiding
    # an import cycle.
    async def encode_episode(
        self,
        content: str,
        session_id: str,
        *,
        tags: Sequence[str] | None = None,
        salience: float = 0.5,
    ) -> dict[str, Any]:
        return await _encode_mod.run(
            self,
            content=content,
            session_id=session_id,
            tags=tags,
            salience=salience,
        )

    async def recall(
        self,
        query: str,
        *,
        session_id: str | None = None,
        k: int = 8,
        mode: str = "auto",
    ) -> list[dict[str, Any]]:
        return await _recall_mod.run(
            self,
            query=query,
            session_id=session_id,
            k=k,
            mode=mode,
        )

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


# Late imports avoid a top-of-file cycle: each sub-module needs
# UNCONSOLIDATED and ``now_seconds`` (bound above), and references
# HippocampusService only through TYPE_CHECKING.
from .ops import consolidate as _consolidate_mod  # noqa: E402
from .ops import encode as _encode_mod  # noqa: E402
from .ops import forget as _forget_mod  # noqa: E402
from .ops import recall as _recall_mod  # noqa: E402
from .ops import reflect as _reflect_mod  # noqa: E402
