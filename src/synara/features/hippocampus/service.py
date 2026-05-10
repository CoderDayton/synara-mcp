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
from .ops.events import EventBus as _EventBus
from .ops.events import InteractionEvent as _Event
from .ops.events import TriggerPolicy as _Policy
from .ops.events import now_seconds as _now_real
from .primitives.plasticity import PlasticityGraph as _Plasticity
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
    """Wrap sync embedder onto a worker thread so all calls can await uniformly.

    Sync embedders go to a thread; async ones pass through.
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
    """Episodic and semantic memory over two vector collections."""

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
        # Plasticity graph + interaction event bus. Both are persistent
        # (edges live in coll.edges, event log lives in coll.events) so
        # the brain survives process restarts. The reactor only
        # auto-fires when ``self_learning_enabled`` is True.
        self._plasticity: _Plasticity = _Plasticity(
            collection=self.episodic,
            sr=self._sr,
            e_ltp_decay_seconds=self.config.e_ltp_decay_seconds,
            l_ltp_threshold_hits=self.config.l_ltp_threshold_hits,
            habit_threshold_hits=self.config.habit_threshold_hits,
            habit_ltd_multiplier=self.config.habit_ltd_multiplier,
            habit_savings_factor=self.config.habit_savings_factor,
            ltd_decay_per_idle_day=self.config.ltd_decay_per_idle_day,
            time_compression=self.config.time_compression,
        )
        self._bus: _EventBus = _EventBus(
            collection=self.episodic,
            policy=_Policy(
                consolidate_after_novel_encodes=self.config.reactor_consolidate_after_novel,
                consolidate_cooldown_seconds=self.config.reactor_consolidate_cooldown_seconds,
                dream_after_events=self.config.reactor_dream_after_events,
                dream_after_idle_seconds=self.config.reactor_dream_after_idle_seconds,
            ),
            log_capacity=self.config.reactor_event_log_capacity,
        )
        if self.config.self_learning_enabled:
            self._bus.on_consolidate = self._reactor_consolidate
            self._bus.on_dream = self._reactor_dream

    # -------------------------------------------- event emission / reactor
    async def _emit(
        self,
        kind: str,
        *,
        session_id: str | None,
        payload: dict[str, Any] | None = None,
    ) -> None:
        """Record an interaction event and run any due reactor follow-ups."""
        event = _Event(
            kind=kind,  # type: ignore[arg-type]
            timestamp=_now_real(),
            session_id=session_id,
            payload=dict(payload) if payload else {},
        )
        await self._bus.record(event)
        if self.config.self_learning_enabled:
            await self._bus.react(event)

    async def _reactor_consolidate(self, _event: Any) -> None:
        """Reactor callback: cluster pending episodes globally."""
        await _consolidate_mod.run(self, session_id=None, n_clusters=None, min_cluster_size=None)
        await self._emit("consolidate", session_id=None, payload={"trigger": "reactor"})

    async def _reactor_dream(self, _event: Any) -> None:
        """Reactor callback: LTD pass over the plasticity graph."""
        pruned = await self._plasticity.ltd_pass(now=_now_real())
        await self._emit("dream", session_id=None, payload={"pruned_edges": pruned})

    async def event_log(self) -> list[Any]:
        """Return a snapshot of the recent interaction events (for inspection)."""
        return await self._bus.log()

    # ------------------------------------------------------------------ embed
    async def vectorise(self, texts: Sequence[str]) -> list[list[float]] | None:
        """Return embeddings for texts, or None to let simplevecdb embed them."""
        if self._embed is None:
            return None
        return [list(await self._embed(t)) for t in texts]

    async def query_arg(self, query: str) -> str | list[float]:
        """Shape query as text (auto-embed) or precomputed vector."""
        if self._embed is None:
            return query
        return list(await self._embed(query))

    def _ensure_projector(self, dim: int) -> _DGProjector:
        """Build/rebuild DG projector if dimension changed."""
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
        now = now_seconds()
        history = list(current.get("access_history") or [])
        history.append(now)
        cap = self.config.access_history_cap
        if cap > 0 and len(history) > cap:
            history = history[-cap:]
        # Atomic delta on the counter so concurrent recalls cannot lose
        # increments; the timestamp + history list are last-write-wins
        # which is fine — losing a stale timestamp is bounded loss.
        await self.episodic.increment_metadata(doc_id, {"retrieval_count": 1})
        await self.episodic.update_metadata(
            [
                (
                    doc_id,
                    {
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
        """Return theta-segmented episode sub-records, ordered by position.

        First segment has episode_group_id == its own id, so any
        sub-record id recovers the full ordered walk. Pass session_id
        for namespace isolation.
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
        result = await _encode_mod.run(
            self,
            content=content,
            session_id=session_id,
            tags=tags,
            salience=salience,
        )
        await self._emit(
            "encode",
            session_id=session_id,
            payload={"id": result.get("id"), "deduped": bool(result.get("deduped"))},
        )
        return result

    async def recall(
        self,
        query: str,
        *,
        session_id: str | None = None,
        k: int = 8,
        mode: str = "auto",
    ) -> list[dict[str, Any]]:
        result = await _recall_mod.run(
            self,
            query=query,
            session_id=session_id,
            k=k,
            mode=mode,
        )
        await self._emit(
            "recall",
            session_id=session_id,
            payload={"hits": len(result), "mode": mode},
        )
        return result

    async def consolidate(
        self,
        *,
        session_id: str | None = None,
        n_clusters: int | None = None,
        min_cluster_size: int | None = None,
    ) -> list[dict[str, Any]]:
        result = await _consolidate_mod.run(
            self,
            session_id=session_id,
            n_clusters=n_clusters,
            min_cluster_size=min_cluster_size,
        )
        await self._emit(
            "consolidate",
            session_id=session_id,
            payload={"schemas_formed": len(result), "trigger": "user"},
        )
        return result

    async def forget(
        self,
        *,
        strength_floor: float = 0.05,
        decay_tau_seconds: float | None = None,
        dry_run: bool = True,
        max_scan: int = 1000,
    ) -> dict[str, Any]:
        result = await _forget_mod.run(
            self,
            strength_floor=strength_floor,
            decay_tau_seconds=decay_tau_seconds,
            dry_run=dry_run,
            max_scan=max_scan,
        )
        await self._emit(
            "forget",
            session_id=None,
            payload={"removed": int(result.get("removed", 0)), "dry_run": dry_run},
        )
        return result

    async def reflect(
        self,
        *,
        session_id: str,
        query: str | None = None,
        k: int = 5,
    ) -> dict[str, Any]:
        result = await _reflect_mod.run(self, session_id=session_id, query=query, k=k)
        await self._emit(
            "reflect",
            session_id=session_id,
            payload={
                "schemas": len(result.get("schemas") or []),
                "episodes": len(result.get("recent_episodes") or []),
            },
        )
        return result


# Late imports avoid a top-of-file cycle: each sub-module needs
# UNCONSOLIDATED and ``now_seconds`` (bound above), and references
# HippocampusService only through TYPE_CHECKING.
from .ops import consolidate as _consolidate_mod  # noqa: E402
from .ops import encode as _encode_mod  # noqa: E402
from .ops import forget as _forget_mod  # noqa: E402
from .ops import recall as _recall_mod  # noqa: E402
from .ops import reflect as _reflect_mod  # noqa: E402
