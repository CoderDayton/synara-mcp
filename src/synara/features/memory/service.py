"""Memory service.

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
import logging
import time
from collections.abc import Awaitable, Callable, Sequence
from typing import Any

from simplevecdb import AsyncVectorDB

from synara.core.errors import ValidationError

from .basal_ganglia.events import EventBus as _EventBus
from .basal_ganglia.events import InteractionEvent as _Event
from .basal_ganglia.events import TriggerPolicy as _Policy
from .basal_ganglia.events import now_seconds as _now_real
from .config import MemoryConfig
from .hippocampus.plasticity import PlasticityGraph as _Plasticity
from .hippocampus.separate import DGProjector as _DGProjector
from .hippocampus.successor import SuccessorRepresentation as _SR
from .memory_types import (
    MemoryType,
    MemoryTypeRegistry,
    default_registry,
)
from .tracing import start_request as _start_request

_LOG = logging.getLogger(__name__)

# Either a sync embedder ``f(text) -> vec`` or an async one
# ``async f(text) -> vec``. The service normalises both to async at
# construction so internal call sites only ever ``await``.
EmbedFn = Callable[[str], Sequence[float] | Awaitable[Sequence[float]]]

# Sentinel for "this episode has not yet been consolidated into a semantic
# schema". simplevecdb's filter format only supports exact equality and
# IN — not "IS NULL" — so we use 0, with positive ints referencing a real
# semantic doc id.
UNCONSOLIDATED: int = 0

# ``MemoryConfig`` is re-exported so existing code (and tests) that
# imports it from ``service`` keeps working after the split-out.
__all__ = [
    "UNCONSOLIDATED",
    "EmbedFn",
    "MemoryConfig",
    "MemoryService",
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


class MemoryService:
    """Episodic and semantic memory over two vector collections."""

    def __init__(
        self,
        db: AsyncVectorDB,
        config: MemoryConfig | None = None,
        *,
        embed_fn: EmbedFn | None = None,
    ) -> None:
        self.config = config or MemoryConfig()
        self.db = db
        # Serializes consolidation: the reactor trigger and an explicit
        # consolidate() call can otherwise interleave their
        # consolidated_into metadata writes on the same episodes.
        self._consolidate_lock = asyncio.Lock()
        # Memory-type registry: explicit override beats the two
        # ``*_collection`` config fields. Collections are materialised
        # in registration order, so additional kinds just need a spec.
        self.memory_types: MemoryTypeRegistry = (
            self.config.memory_types
            if self.config.memory_types is not None
            else default_registry(
                episodic_collection=self.config.episodic_collection,
                semantic_collection=self.config.semantic_collection,
            )
        )
        # store_embeddings=True keeps vectors at hand for cluster()/rebuild
        # even after process restarts.
        self._collections: dict[MemoryType, Any] = {
            spec.type: db.collection(spec.collection, store_embeddings=True)
            for spec in self.memory_types
        }
        # Convenience handles for the two legacy kinds — preserves the
        # ``service.episodic`` / ``service.semantic`` attribute API.
        self.episodic = self._collections[MemoryType.EPISODIC]
        self.semantic = self._collections[MemoryType.SEMANTIC]
        self._embed = _normalise_embed_fn(embed_fn) if embed_fn is not None else None
        # Lazily constructed once we observe the embedding dimension.
        self._dg: _DGProjector | None = None
        # First-class embedding dim: explicit override beats the lazy
        # probe; ``None`` means "discover and cache on first vector op".
        self._embedding_dimension: int | None = self.config.embedding_dimension
        # Latest captured trace dump (only populated when
        # ``config.tracing_enabled`` and after at least one traced op).
        self.last_trace: dict[str, Any] | None = None
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
        # SR persists its transition tally under ``kind="sr"`` on the
        # episodic collection alongside plasticity's ``kind="plasticity"``
        # rows. Load happens lazily on first DB-touching op (see
        # ``_ensure_sr_loaded``).
        if self._sr is not None:
            self._sr.attach(self.episodic)
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
            # The reactor (consolidate/dream replay) reads the SR, so it
            # must be rehydrated first. Guarding here covers every entry
            # point — including ones that don't touch the SR directly
            # (consolidate/forget/reflect/semantic) but can still trip
            # the reactor via the event counters. Idempotent after the
            # first load.
            await self._ensure_sr_loaded()
            # Reactor callbacks are background side-effects. A failure
            # in consolidation/dream replay must not surface as an error
            # on the unrelated user operation (encode/recall) that
            # happened to trip the trigger.
            try:
                await self._bus.react(event)
            except Exception:
                _LOG.exception("reactor failed for event kind=%s", kind)

    async def _reactor_consolidate(self, _event: Any) -> None:
        """Reactor callback: cluster pending episodes globally."""
        # Advance the trigger gate *before* the work. If consolidation
        # raises (e.g. embedding backend down), the counter must not
        # stay saturated, or every subsequent encode re-fires a failing
        # pass. Advancing the cooldown clock too bounds a persistently
        # failing backend to one retry per cooldown window. The trailing
        # ``_emit`` re-applies the same reset idempotently.
        st = self._bus.state
        st.last_consolidate_at = _now_real()
        st.novel_encodes_since_consolidate = 0
        async with self._consolidate_lock:
            await _consolidate_mod.run(
                self, session_id=None, n_clusters=None, min_cluster_size=None
            )
        await self._emit("consolidate", session_id=None, payload={"trigger": "reactor"})

    async def _reactor_dream(self, _event: Any) -> None:
        """Reactor callback: LTD decay, then off-policy SWR replay.

        Decay runs *first* so the rehearsal that follows is not culled
        by the same pass (a fresh replay edge is E-LTP/bonus-only with
        zero durable weight, which ``ltd_pass`` would otherwise prune).
        One replay is transient potentiation; repeated dreams of the
        same high-priority trace fold it durable via L-LTP.
        """
        t = _now_real()
        pruned = await self._plasticity.ltd_pass(now=t)
        replayed = await _replay_mod.run(self, now=t)
        await self._emit(
            "dream",
            session_id=None,
            payload={"pruned_edges": pruned, "replayed": replayed},
        )

    async def event_log(self) -> list[Any]:
        """Return a snapshot of the recent interaction events (for inspection)."""
        return await self._bus.log()

    # ------------------------------------------------------------------ embed
    async def vectorise(self, texts: Sequence[str]) -> list[list[float]] | None:
        """Return embeddings for texts, or None to let simplevecdb embed them."""
        if self._embed is None:
            return None
        vecs = [list(await self._embed(t)) for t in texts]
        if vecs and self._embedding_dimension is None:
            self._embedding_dimension = len(vecs[0])
        elif (
            vecs
            and self._embedding_dimension is not None
            and len(vecs[0]) != self._embedding_dimension
        ):
            raise ValidationError(
                f"embedding dimension drift: configured {self._embedding_dimension}, "
                f"observed {len(vecs[0])}",
            )
        return vecs

    async def query_arg(self, query: str) -> str | list[float]:
        """Shape query as text (auto-embed) or precomputed vector."""
        if self._embed is None:
            return query
        vec = list(await self._embed(query))
        if self._embedding_dimension is None:
            self._embedding_dimension = len(vec)
        return vec

    async def embedding_dimension(self) -> int | None:
        """Return the active embedding dimension, probing once if needed.

        Returns ``None`` when no embedder is configured (simplevecdb's
        bundled embedder handles vectorisation server-side and the
        service has no view of the dim). Otherwise probes a tiny string
        through the configured embed_fn on the first call and caches.
        """
        if self._embedding_dimension is not None:
            return self._embedding_dimension
        if self._embed is None:
            return None
        probe = list(await self._embed("\u200b"))
        self._embedding_dimension = len(probe) if probe else None
        return self._embedding_dimension

    def collection_for(self, kind: MemoryType) -> Any:
        """Return the simplevecdb collection backing ``kind``.

        Ops that want to address a specific memory kind go through this
        accessor so adding a new kind does not require an attribute on
        the service.
        """
        try:
            return self._collections[kind]
        except KeyError as exc:
            raise ValidationError(
                f"memory type {kind.value!r} is not registered on this service",
            ) from exc

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

    async def delete_episode(
        self,
        episode_id: int,
        *,
        session_id: str | None = None,
    ) -> dict[str, Any]:
        """Delete an episode (and its theta-segment group) from the store.

        Removes the episodic documents via ``delete_by_ids``. Durable
        SR/plasticity edges (``coll.edges``) have an ``ON DELETE
        CASCADE`` FK to documents, so they vanish automatically with the
        doc. Plasticity holds no in-memory state. SR, however, keeps an
        in-memory tally/pending/window: those ids are evicted via
        :meth:`SuccessorRepresentation.evict_nodes` *before* the delete
        so a subsequent :meth:`SuccessorRepresentation.flush` cannot
        upsert a FK-violating edge for a now-deleted id. (``forget``
        relies on the same invariant and evicts identically before its
        own ``delete_by_ids``.) No reactor event is emitted: a targeted
        admin delete should not feed the consolidation/dream trigger.
        """
        target = await self.episodic.get_documents({"id": episode_id})
        if not target:
            raise ValidationError(f"episode {episode_id} not found")
        _doc_id, _text, md = target[0]
        group_id = int(md.get("episode_group_id", episode_id))
        members = await self.fetch_episode_group(group_id, session_id=session_id)
        member_ids = {int(r["id"]) for r in members}
        member_ids.add(int(episode_id))
        ordered = sorted(member_ids)
        if self._sr is not None:
            self._sr.evict_nodes(set(ordered))
        await self.episodic.delete_by_ids(ordered)
        return {"deleted_ids": ordered, "count": len(ordered)}

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
    async def _ensure_sr_loaded(self) -> None:
        """Rehydrate the SR's transition tally from coll.edges (idempotent).

        Called at the top of each public async entry point so the durable
        SR state is available before any observe/boost/omega read.
        """
        if self._sr is not None:
            await self._sr.load()

    async def encode_episode(
        self,
        content: str,
        session_id: str,
        *,
        tags: Sequence[str] | None = None,
        salience: float | None = None,
    ) -> dict[str, Any]:
        await self._ensure_sr_loaded()
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
        await self._ensure_sr_loaded()
        with _start_request("recall", enabled=self.config.tracing_enabled) as _trace_ctx:
            with _trace_ctx.span(
                "recall.run",
                payload={"session_id": session_id, "k": k, "mode": mode},
            ):
                result = await _recall_mod.run(
                    self,
                    query=query,
                    session_id=session_id,
                    k=k,
                    mode=mode,
                )
            if self.config.tracing_enabled:
                self.last_trace = _trace_ctx.as_dict()
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
        async with self._consolidate_lock:
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

    # --------------------------------------------- direct semantic memory
    # The semantic store is also written to by ``consolidate`` (clusters
    # of episodes -> distilled schemas). These two methods give callers a
    # direct lane that bypasses the episodic pipeline — for authored
    # facts, procedures, preferences, and conventions that should persist
    # without raw-trace baggage.
    async def store_semantic_memory(
        self,
        content: str,
        *,
        kind: str = "fact",
        tags: Sequence[str] | None = None,
        confidence: float = 1.0,
    ) -> dict[str, Any]:
        if not content.strip():
            raise ValidationError("content must be non-empty")
        if self.config.max_content_chars and len(content) > self.config.max_content_chars:
            raise ValidationError(
                f"content exceeds max_content_chars ({self.config.max_content_chars})"
            )
        if not 0.0 <= confidence <= 1.0:
            raise ValidationError("confidence must be in [0, 1]")
        if not kind or not kind.strip():
            raise ValidationError("kind must be non-empty")

        now = now_seconds()
        tag_list = sorted({t for t in (tags or []) if isinstance(t, str) and t})
        metadata: dict[str, Any] = {
            "kind": kind,
            "source_episode_ids": [],
            "tags": tag_list,
            "confidence": float(confidence),
            "created_at": now,
            "updated_at": now,
            "authored": True,
        }
        sem_ids = await self.semantic.add_texts(
            [content],
            metadatas=[metadata],
            embeddings=await self.vectorise([content]),
        )
        sem_id = int(sem_ids[0])
        # Patch id back into metadata so downstream recall can read it
        # uniformly (mirrors how consolidate does it).
        await self.semantic.update_metadata([(sem_id, {"id": sem_id})])
        return {
            "id": sem_id,
            "kind": kind,
            "tags": tag_list,
            "confidence": float(confidence),
            "created_at": now,
        }

    async def recall_semantic_memory(
        self,
        query: str,
        *,
        k: int = 8,
        kind: str | None = None,
    ) -> list[dict[str, Any]]:
        if not query.strip():
            raise ValidationError("query must be non-empty")
        if self.config.max_content_chars and len(query) > self.config.max_content_chars:
            raise ValidationError(
                f"query exceeds max_content_chars ({self.config.max_content_chars})"
            )
        if k <= 0:
            return []
        if self.config.max_recall_k and k > self.config.max_recall_k:
            raise ValidationError(f"k exceeds max_recall_k ({self.config.max_recall_k})")
        if await self.semantic.count() == 0:
            return []
        q = await self.query_arg(query)
        # When a kind filter is requested, over-fetch and filter in Python
        # so cosine ranking still drives the final order; semantic store
        # is small (gist-level) so the over-fetch is cheap.
        fetch_k = k if kind is None else max(k * 4, 32)
        hits = await self.semantic.similarity_search(q, k=fetch_k)
        out: list[dict[str, Any]] = []
        for doc, dist in hits:
            md = dict(doc.metadata)
            if kind is not None and str(md.get("kind", "")) != kind:
                continue
            out.append(
                {
                    "id": int(md.get("id", -1)),
                    "content": doc.page_content,
                    "distance": float(dist),
                    "metadata": md,
                }
            )
            if len(out) >= k:
                break
        return out


# Late imports avoid a top-of-file cycle: each sub-module needs
# UNCONSOLIDATED and ``now_seconds`` (bound above), and references
# MemoryService only through TYPE_CHECKING.
from .hippocampus import encode as _encode_mod  # noqa: E402
from .hippocampus import recall as _recall_mod  # noqa: E402
from .hippocampus import replay as _replay_mod  # noqa: E402
from .neocortex import consolidate as _consolidate_mod  # noqa: E402
from .neocortex import forget as _forget_mod  # noqa: E402
from .neocortex import reflect as _reflect_mod  # noqa: E402
