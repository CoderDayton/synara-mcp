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
import contextlib
import inspect
import itertools
import logging
import time
import weakref
from collections.abc import Awaitable, Callable, Coroutine, Sequence
from typing import Any, cast

from simplevecdb import AsyncVectorDB

from synara.core.errors import ValidationError
from synara.storage import STORE_EMBEDDINGS

from .basal_ganglia.events import EventBus as _EventBus
from .basal_ganglia.events import EventKind as _EventKind
from .basal_ganglia.events import InteractionEvent as _Event
from .basal_ganglia.events import TriggerPolicy as _Policy
from .basal_ganglia.events import now_seconds as _now_real
from .config import MemoryConfig, validate_tags
from .hippocampus.background import BackgroundReference as _BackgroundReference
from .hippocampus.plasticity import PlasticityGraph as _Plasticity
from .hippocampus.separate import DGProjector as _DGProjector
from .hippocampus.successor import SuccessorRepresentation as _SR
from .memory_types import (
    SCOPE_SESSION,
    MemoryType,
    MemoryTypeRegistry,
    default_registry,
    resolve_scope,
)
from .port import HygieneCounters
from .recall_report import RecallDiagnostics
from .tracing import start_request as _start_request

_LOG = logging.getLogger(__name__)

# Either a sync embedder ``f(text) -> vec`` or an async one
# ``async f(text) -> vec``. The service normalises both to async at
# construction so internal call sites only ever ``await``.
EmbedFn = Callable[[str], Sequence[float] | Awaitable[Sequence[float]]]

# Optional companion to ``EmbedFn`` for callers that can compute many
# vectors in a single call (e.g. ``Embedder.embed_batch``). When set on
# ``MemoryService``, ``vectorise`` issues one round-trip per call instead
# of N sequential ``await self._embed(...)`` invocations — this is the
# difference between one GPU forward pass (or one HTTP request) and N.
# Must be async; the per-text :data:`EmbedFn` wrapper handles sync→async
# normalisation, but batch backends are async by construction.
EmbedBatchFn = Callable[[Sequence[str]], Awaitable[list[list[float]]]]

# Sentinel for "this episode has not yet been consolidated into a semantic
# schema". simplevecdb's filter format only supports exact equality and
# IN — not "IS NULL" — so we use 0, with positive ints referencing a real
# semantic doc id.
UNCONSOLIDATED: int = 0

# ``MemoryConfig`` is re-exported so existing code (and tests) that
# imports it from ``service`` keeps working after the split-out.
__all__ = [
    "UNCONSOLIDATED",
    "EmbedBatchFn",
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
        embed_batch_fn: EmbedBatchFn | None = None,
        embed_asymmetric: bool = False,
    ) -> None:
        self.config = config or MemoryConfig()
        self.db = db
        # Serializes consolidation: the reactor trigger and an explicit
        # consolidate() call can otherwise interleave their
        # consolidated_into metadata writes on the same episodes.
        self._consolidate_lock = asyncio.Lock()
        # Serializes dream replay: back-to-back event triggers spanning
        # the idle threshold can otherwise fire two LTD passes that
        # double-apply decay or race on the same plasticity edge
        # snapshot. The dream reactor is best-effort background work,
        # so the second contender simply waits its turn.
        self._dream_lock = asyncio.Lock()
        # Memory-type registry: explicit override beats the two
        # ``*_collection`` config fields. Collections are materialised
        # in registration order, so additional kinds just need a spec.
        self.memory_types: MemoryTypeRegistry = (
            self.config.memory_types
            if self.config.memory_types is not None
            else default_registry(
                episodic_collection=self.config.episodic_collection,
                semantic_collection=self.config.semantic_collection,
                schema_candidate_collection=self.config.schema_candidate_collection,
            )
        )
        # STORE_EMBEDDINGS keeps vectors at hand for cluster()/rebuild even
        # after process restarts. It lives in synara.storage because the
        # offline scripts must agree with us on it.
        self._collections: dict[MemoryType, Any] = {
            spec.type: db.collection(spec.collection, store_embeddings=STORE_EMBEDDINGS)
            for spec in self.memory_types
        }
        # Convenience handles for the two legacy kinds — preserves the
        # ``service.episodic`` / ``service.semantic`` attribute API.
        self.episodic = self._collections[MemoryType.EPISODIC]
        self.semantic = self._collections[MemoryType.SEMANTIC]
        # Schema-candidate buffer; the collection is always wired up so
        # tuning consolidate_min_recurrence at runtime doesn't require
        # a schema migration. Present unconditionally on the service.
        self.schema_candidates = self._collections[MemoryType.SCHEMA_CANDIDATE]
        # Background "unrelated" distance per collection, backing the
        # recall relevance ceiling. Samples lazily on first use, so a
        # service that never recalls never pays for it.
        self.background = _BackgroundReference(self.config)
        self._embed = _normalise_embed_fn(embed_fn) if embed_fn is not None else None
        # Optional batch hook. Only consulted by ``vectorise`` (which
        # always has multiple texts available); ``query_arg`` and the
        # dim probe keep the single-text path so a caller can configure
        # only ``embed_fn`` and still work. A batch fn without a single
        # ``embed_fn`` is rejected: ``query_arg`` would have nothing to
        # call for query vectorisation.
        if embed_batch_fn is not None and embed_fn is None:
            raise ValidationError(
                "embed_batch_fn requires embed_fn: query_arg and the dim probe "
                "call the single-text embedder",
            )
        self._embed_batch: EmbedBatchFn | None = embed_batch_fn
        # Does the embedder encode a query differently from a document
        # (task prefixes)? Only consumers that mix a query vector with
        # stored vectors care -- see ``hippocampus/recall._document_space``
        # -- and they use this to skip a redundant second encode on the
        # symmetric models where both sides are the same vector.
        self.embed_asymmetric = embed_asymmetric
        # Lazily constructed once we observe the embedding dimension.
        self._dg: _DGProjector | None = None
        # First-class embedding dim: explicit override beats the lazy
        # probe; ``None`` means "discover and cache on first vector op".
        self._embedding_dimension: int | None = self.config.embedding_dimension
        # Latest captured trace dump (only populated when
        # ``config.tracing_enabled`` and after at least one traced op).
        self.last_trace: dict[str, Any] | None = None
        # Rotating offset cursor for dream-replay's bounded sampling. The
        # underlying ``get_documents`` orders by ID; a fixed ``limit``
        # without ``offset`` would starve newer episodes. Each pass
        # consumes ``dream_replay_max_scan`` rows and advances the
        # cursor; on partial pages we wrap back to 0. Persisted in
        # memory only (rebuilt on restart), which is fine because the
        # power-law strength signal converges across many cycles.
        self._replay_cursor: int = 0
        # Rotating scan cursors for the forget pass, same rationale:
        # a fixed ``limit=max_scan`` without an offset would only ever
        # evaluate the oldest ``max_scan`` rows and permanently starve
        # everything beyond the window (episodic decay pruning and
        # cold-schema GC alike). Advanced only by a non-dry-run pass so
        # a dry-run preview and the delete that follows it see the same
        # window. Process-local like ``_replay_cursor``: the sweep
        # converges across cycles.
        self._forget_cursor: int = 0
        self._forget_schema_cursor: int = 0
        # In-flight background reactor tasks (auto-consolidate / dream).
        # Strong references so the event loop cannot GC a running task;
        # each task discards itself on completion. Tests and shutdown
        # paths drain via :meth:`drain_reactor_tasks`.
        self._reactor_tasks: set[asyncio.Task[None]] = set()
        # Lazy HNSW reconciliation flag. See ``_ensure_index_ready``: we
        # flush+rebuild on first recall to recover from an
        # encode-without-consolidate desync or a crashed-process index.
        self._index_ready: bool = False
        # Monotonic consolidate-pass counter. Advanced at the top of
        # ``neocortex.consolidate.run`` so the candidate gate can
        # require hits to span distinct epochs (cross-pass diversity).
        # Process-local; rebuilt as 0 on restart, which is safe because
        # epoch identity only matters within a single run-stream.
        self._consolidate_epoch: int = 0
        # Hygiene counters: lifetime per-process tallies of how the
        # promotion gate behaves. Surfaced via ``stats()`` /
        # ``memory_stats`` so simulators can observe which gate fires.
        # Per-doc lock map for read-modify-write metadata sequences
        # (``bump_retrieval`` access_history append, ``_accrue_drift``
        # drift_total update). ``WeakValueDictionary`` so locks for
        # rarely-touched docs don't accumulate for the process lifetime;
        # the ``async with`` caller holds a strong ref for the duration
        # of the critical section, so GC cannot evict a live lock.
        self._doc_locks: weakref.WeakValueDictionary[int, asyncio.Lock] = (
            weakref.WeakValueDictionary()
        )
        self._hygiene_counters: HygieneCounters = {
            "schemas_promoted": 0,
            "candidates_parked": 0,
            "candidates_rejected_size": 0,
            "candidates_rejected_confidence": 0,
            "candidates_rejected_session_diversity": 0,
            "candidates_rejected_epoch_diversity": 0,
            "schemas_evicted_unused": 0,
        }
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
        # Reactor handlers are wired at construction; ``None`` when
        # self-learning is off leaves ``react`` a no-op (it only fires
        # handlers that are set).
        _learning = self.config.self_learning_enabled
        self._bus: _EventBus = _EventBus(
            collection=self.episodic,
            policy=_Policy(
                consolidate_after_novel_encodes=self.config.reactor_consolidate_after_novel,
                consolidate_cooldown_seconds=self.config.reactor_consolidate_cooldown_seconds,
                dream_after_events=self.config.reactor_dream_after_events,
                dream_after_idle_seconds=self.config.reactor_dream_after_idle_seconds,
            ),
            log_capacity=self.config.reactor_event_log_capacity,
            on_consolidate=self._schedule_consolidate if _learning else None,
            on_dream=self._schedule_dream if _learning else None,
        )

    # -------------------------------------------- event emission / reactor
    async def _emit(
        self,
        kind: _EventKind,
        *,
        session_id: str | None,
        payload: dict[str, Any] | None = None,
    ) -> None:
        """Record an interaction event and run any due reactor follow-ups."""
        event = _Event.create(kind, session_id=session_id, payload=payload, timestamp=_now_real())
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
        async with self._dream_lock:
            # Bound the LTD sweep with the same knob that sizes replay's
            # scan, so a large edge table cannot turn one dream into an
            # unbounded serial read-modify-write pass. ltd_pass spends
            # the budget on the stalest edges first (they carry the most
            # pending decay), so the sweep still converges across dreams.
            pruned = await self._plasticity.ltd_pass(
                now=t, max_scan=self.config.dream_replay_max_scan or None
            )
            replayed = await _replay_mod.run(self, now=t)
        await self._emit(
            "dream",
            session_id=None,
            payload={"pruned_edges": pruned, "replayed": replayed},
        )

    async def _schedule_consolidate(self, event: Any) -> None:
        """Reactor hook: run consolidation as a background task.

        The trigger gate is advanced *synchronously*, before the task is
        spawned: ``react`` re-evaluates the policy on every subsequent
        event, and without the immediate reset a burst of encodes
        landing before the task's first await would each re-fire a
        duplicate pass. ``_reactor_consolidate`` re-applies the same
        reset idempotently when it actually runs.
        """
        st = self._bus.state
        st.last_consolidate_at = _now_real()
        st.novel_encodes_since_consolidate = 0
        self._spawn_reactor_task(self._reactor_consolidate(event), name="reactor-consolidate")

    async def _schedule_dream(self, event: Any) -> None:
        """Reactor hook: run the dream pass as a background task.

        Mirrors :meth:`_schedule_consolidate`: the dream counters are
        reset synchronously so back-to-back events cannot double-schedule
        a pass while the first task waits to run (the trailing ``dream``
        event re-applies the same reset when the task completes).
        """
        st = self._bus.state
        st.last_dream_at = _now_real()
        st.events_since_dream = 0
        self._spawn_reactor_task(self._reactor_dream(event), name="reactor-dream")

    def _spawn_reactor_task(self, coro: Coroutine[Any, Any, None], *, name: str) -> None:
        """Run a reactor follow-up off the caller's request path.

        Consolidation (embedding round-trips + K-Means) and dreams (LTD
        over the whole edge table + replay) can take seconds; awaiting
        them inside the user op that happened to trip the trigger turns
        a tens-of-ms tool call into a multi-second one and mis-attributes
        the latency in the tool metrics. The internal locks
        (``_consolidate_lock`` / ``_dream_lock``) already serialise the
        work, so backgrounding is safe within the single-writer process.
        Failures are logged, never raised into any user op.
        """
        task = asyncio.create_task(self._run_reactor_guarded(coro, name=name), name=name)
        self._reactor_tasks.add(task)
        task.add_done_callback(self._reactor_tasks.discard)

    async def _run_reactor_guarded(self, coro: Coroutine[Any, Any, None], *, name: str) -> None:
        try:
            await coro
        except asyncio.CancelledError:
            _LOG.warning("background reactor task %s cancelled (drain timeout)", name)
            raise
        except Exception:
            _LOG.exception("background reactor task %s failed", name)

    async def drain_reactor_tasks(self, *, timeout: float | None = None) -> None:
        """Await every in-flight background reactor task.

        For tests that assert on reactor side-effects and for graceful
        shutdown. A reactor task's trailing ``_emit`` can never schedule
        another reactor task (``react`` skips reactor-produced events),
        but loop until the set is empty anyway so a task spawned by a
        user op racing the drain is awaited too. Task failures are
        already logged by the guard; they never propagate here.

        ``timeout`` bounds the total wait (seconds): tasks still running
        at the deadline are cancelled so shutdown cannot hang behind a
        long consolidation. The guard logs the cancellation; a cancelled
        consolidate simply leaves its episodes UNCONSOLIDATED for the
        next pass.
        """
        if timeout is None:
            while self._reactor_tasks:
                await asyncio.gather(*tuple(self._reactor_tasks), return_exceptions=True)
            return
        loop = asyncio.get_running_loop()
        deadline = loop.time() + timeout
        while self._reactor_tasks:
            remaining = deadline - loop.time()
            if remaining <= 0:
                break
            done, _pending = await asyncio.wait(set(self._reactor_tasks), timeout=remaining)
            if not done:
                break
        if self._reactor_tasks:
            for task in tuple(self._reactor_tasks):
                task.cancel()
            await asyncio.gather(*tuple(self._reactor_tasks), return_exceptions=True)

    async def event_log(self) -> list[Any]:
        """Return a snapshot of the recent interaction events (for inspection)."""
        return await self._bus.log()

    # ------------------------------------------------------------------ embed
    async def vectorise(self, texts: Sequence[str]) -> list[list[float]] | None:
        """Return embeddings for texts, or None to let simplevecdb embed them.

        Uses :attr:`_embed_batch` when configured so a multi-text encode
        (theta-segmented episodes, semantic store with many candidates,
        consolidate summaries) costs one backend round-trip instead of
        ``len(texts)``. Falls back to the per-text :attr:`_embed` loop
        otherwise. Both paths produce identical output: the batch fn
        contract is ``len(result) == len(texts)`` with vectors in the
        same order as inputs.

        Empty input returns ``[]`` without invoking either backend.
        """
        if self._embed is None:
            return None
        if not texts:
            return []
        if self._embed_batch is not None:
            raw = await self._embed_batch(texts)
            # Length mismatch silently misaligns vectors to texts in the
            # store, the same hazard ``_index_of`` guards against in the
            # remote backend. Refuse rather than corrupt the index.
            if len(raw) != len(texts):
                raise ValidationError(
                    f"embed_batch_fn returned {len(raw)} vectors for {len(texts)} texts",
                )
            vecs = [list(v) for v in raw]
        else:
            vecs = [list(await self._embed(t)) for t in texts]
        # Dim check on vecs[0] only — matches the original behaviour and
        # is sufficient since a sane backend produces uniform-width
        # vectors; per-vector validation would be redundant work on the
        # hot path.
        if self._embedding_dimension is None:
            self._embedding_dimension = len(vecs[0])
        elif len(vecs[0]) != self._embedding_dimension:
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
    def _doc_lock(self, doc_id: int) -> asyncio.Lock:
        """Per-doc ``asyncio.Lock`` for read-modify-write metadata.

        Package-private — callers within ``features/memory`` use it for
        the ``read access_history / append / write`` and
        ``read drift_total / check cap / write`` sequences. Callers
        MUST hold a strong reference for the entire ``async with``
        block; the ``WeakValueDictionary`` would otherwise let GC drop
        the entry between siblings and break mutual exclusion.
        """
        return self._doc_locks.setdefault(doc_id, asyncio.Lock())

    async def bump_retrieval(self, doc_id: int, current: dict[str, Any]) -> None:
        # Re-read ``access_history`` inside the lock so concurrent
        # ``bump_retrieval`` calls on the same doc cannot lose an
        # appended timestamp. ``current`` (the snapshot returned by
        # similarity_search) is only used as a fast-path fallback when
        # the live row has gone missing (caller deleted it between the
        # search and the bump — unlikely but defensive).
        lock = self._doc_lock(doc_id)
        async with lock:
            now = now_seconds()
            rows = await self.episodic.get_documents({"id": doc_id})
            if rows:
                _, _, fresh = rows[0]
                history = list(fresh.get("access_history") or [])
            else:
                history = list(current.get("access_history") or [])
            history.append(now)
            cap = self.config.access_history_cap
            if cap > 0 and len(history) > cap:
                history = history[-cap:]
            # Two writes, two semantics: ``increment_metadata`` is an
            # atomic delta on a disjoint key (``retrieval_count``);
            # ``update_metadata`` for the history list + timestamp is
            # last-write-wins, but the per-doc lock above keeps the
            # read-append-write atomic for this single doc. The writes
            # target disjoint keys, so ``gather`` is safe and halves
            # the ``to_thread`` round-trips per recall hit.
            await asyncio.gather(
                self.episodic.update_metadata(
                    [
                        (
                            doc_id,
                            {
                                "last_accessed": now,
                                "access_history": history,
                            },
                        )
                    ]
                ),
                self.episodic.increment_metadata(doc_id, {"retrieval_count": 1}),
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

    async def get_episode(
        self,
        episode_id: int,
        *,
        session_id: str | None = None,
    ) -> dict[str, Any]:
        """Return one episode's full, untruncated content by id.

        The companion to bounded :meth:`recall` snippets. A theta-segmented
        episode is reassembled from its ordered sub-records, so the caller
        gets the whole original text rather than a single 1024-char
        segment. ``session_id`` (optional) restricts the group walk to a
        namespace; omit it for a cross-session fetch. A segmented episode
        whose group is not visible in the given ``session_id`` raises
        rather than returning a single unreassembled segment.
        """
        target = await self.episodic.get_documents({"id": episode_id})
        if not target:
            raise ValidationError(f"episode {episode_id} not found")
        _doc_id, text, md = target[0]
        group_id = md.get("episode_group_id")
        if group_id is not None:
            members = await self.fetch_episode_group(int(group_id), session_id=session_id)
            if not members:
                # The episode is segmented (it carries ``episode_group_id``)
                # but the scoped walk returned nothing — ``session_id``
                # excluded the whole group. Falling through to return the
                # single matched segment would pass a fragment off as the
                # whole episode; fail honestly instead.
                raise ValidationError(
                    f"episode {episode_id} belongs to group {group_id} "
                    f"not visible in session {session_id!r}"
                )
            content = "".join(m["content"] for m in members)
            # Re-bond on access: engaging the whole episode strengthens
            # its internal sibling chain (L-LTP for frequently-fetched
            # episodes). Best-effort — a bond failure must not fail the
            # read it decorates.
            with contextlib.suppress(Exception):
                await self._reinforce_segment_chain([int(m["id"]) for m in members], now_seconds())
            return {
                "id": episode_id,
                "content": content,
                "content_chars": len(content),
                "metadata": md,
                "group_id": int(group_id),
                "segment_ids": [int(m["id"]) for m in members],
            }
        return {
            "id": episode_id,
            "content": text,
            "content_chars": len(text),
            "metadata": md,
            "group_id": None,
            "segment_ids": None,
        }

    async def _reinforce_segment_chain(self, seg_ids: list[int], now: float) -> None:
        """Hebbian-bond consecutive sibling segments of one episode.

        Reinforces ``seg_ids[i-1] -> seg_ids[i]`` so spreading activation
        can resurface the whole episode from any one fragment. A no-op
        when SR is disabled (``sr_enabled=False``) or the bond is disabled
        (``segment_assoc_score <= 0``). Called at encode and again on
        ``get_episode`` so a frequently-fetched episode crosses the L-LTP
        threshold and the bond persists.
        """
        score = self.config.segment_assoc_score
        if self._sr is None or score <= 0.0:
            return
        pairs = list(itertools.pairwise(seg_ids))
        if pairs:
            # Consecutive sibling pairs are distinct edges (segment ids are
            # unique within a group), so the per-edge locks never contend;
            # gather pipelines the read+upsert round-trips.
            await asyncio.gather(
                *(self._plasticity.reinforce(a, b, score=score, now=now) for a, b in pairs)
            )

    async def delete_episode(
        self,
        episode_id: int,
        *,
        session_id: str | None = None,
        dry_run: bool = False,
    ) -> dict[str, Any]:
        """Delete an episode (and its theta-segment group) from the store.

        ``dry_run=True`` previews instead: returns the ids that *would*
        be deleted (``candidate_ids`` — the whole segment group) without
        touching the store, mirroring ``forget``'s preview contract.

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
        if dry_run:
            return {"candidate_ids": ordered, "count": len(ordered), "dry_run": True}
        await self._evict_and_delete(ordered)
        return {"deleted_ids": ordered, "count": len(ordered), "dry_run": False}

    async def delete_semantic(self, semantic_id: int) -> dict[str, Any]:
        """Delete a semantic memory (consolidated schema or user-asserted entry).

        Mirrors the cold-schema eviction path (``forget.run``) and the
        ``supersedes`` retire: a plain ``delete_by_ids`` on the semantic
        collection. Semantic docs carry no SR/plasticity edges (those are
        episodic-only), so no eviction or state freeze is needed. Source
        episodes' ``consolidated_into`` is intentionally left untouched —
        identical to the cold-schema GC; a later consolidate re-points the
        survivors. No reactor event is emitted: a targeted admin delete must
        not feed the consolidation/dream trigger.
        """
        if not await self.semantic.get_documents({"id": semantic_id}):
            raise ValidationError(f"semantic memory {semantic_id} not found")
        await self.semantic.delete_by_ids([semantic_id])
        return {"deleted_ids": [semantic_id], "count": 1}

    async def _evict_and_delete(self, ids: list[int]) -> None:
        """Delete episodic docs, evicting their SR in-memory state first.

        ``coll.edges`` has ``ON DELETE CASCADE`` to documents, so durable
        SR/plasticity edges vanish with the docs. But a lingering in-memory
        ``_T``/``_pending``/window entry would make the next SR flush upsert
        a FK-violating edge for a now-deleted id. When SR is active, hold its
        state lock across the evict→delete pair so no concurrent observe
        folds an in-flight recall's co-occurrence back in mid-delete.
        Plasticity holds no in-memory state.
        """
        cm = self._sr.state_freeze() if self._sr is not None else contextlib.nullcontext()
        async with cm:
            if self._sr is not None:
                self._sr.evict_nodes_locked(set(ids))
            await self.episodic.delete_by_ids(ids)

    # ------------------------------------------------------------------ stats
    async def stats(self) -> dict[str, int]:
        # The three counts are independent reads; gather them so a stats
        # poll (dashboard, health check) pays one round-trip of latency
        # instead of three.
        episodic_count, semantic_count, candidate_count = await asyncio.gather(
            self.episodic.count(),
            self.semantic.count(),
            self.schema_candidates.count(),
        )
        out: dict[str, int] = {
            "episodic_count": episodic_count,
            "semantic_count": semantic_count,
            "schema_candidate_count": candidate_count,
            "consolidate_epoch": int(self._consolidate_epoch),
        }
        out.update(cast("dict[str, int]", self._hygiene_counters))
        # Cheap in-memory observability: SR graph mass, reactor trigger
        # state, and in-flight background work — so "why did recall
        # return nothing" or "when did it last consolidate" is one
        # memory_stats call instead of guesswork. Counters rehydrate
        # from the durable event log on first touch.
        await self._bus.ensure_state_loaded()
        st = self._bus.state
        out["sr_edges"] = int(self._sr.total_edges) if self._sr is not None else 0
        out["novel_encodes_since_consolidate"] = int(st.novel_encodes_since_consolidate)
        out["events_since_dream"] = int(st.events_since_dream)
        out["last_consolidate_at"] = int(st.last_consolidate_at)
        out["last_dream_at"] = int(st.last_dream_at)
        out["reactor_tasks_inflight"] = len(self._reactor_tasks)
        return out

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

    async def _ensure_index_ready(self) -> None:
        """Reconcile the HNSW index against the SQLite catalog once.

        simplevecdb buffers ``add_texts`` writes through a pending queue
        and only flushes / rebuilds the usearch index during consolidate
        (or when its own thresholds fire — 5000 pending rows by default).
        Between encode and the first consolidate, ``similarity_search``
        sees an empty index and returns no hits — even though
        ``count() > 0`` and embeddings are stored. The same desync can
        also survive a process crash that loses the in-memory index
        without rolling back the SQLite catalog.

        This guard, invoked at the top of recall, is the canonical fix:
        flush any buffered adds (cheap when none) and call
        ``rebuild_index`` iff the index holds fewer vectors than the
        catalog (``index.size < count()`` — covering both a fully empty
        index and a partially-populated one whose later adds were lost).
        The work runs at most once per service lifetime; thereafter
        writers' own pending flushes keep the index current.
        """
        if self._index_ready:
            return
        # Set early so re-entrant recalls under the same lock don't both
        # try to rebuild. A failure path below resets it so a transient
        # error doesn't permanently disable the recovery.
        self._index_ready = True
        try:
            for coll in (self.episodic, self.semantic):
                cnt = await coll.count()
                if cnt == 0:
                    continue
                # flush_pending is a no-op when there's nothing buffered;
                # cheap to call unconditionally.
                await coll.flush_pending()
                # ``coll.dim is None`` only catches a *fully* empty index.
                # A partial index — some vectors persisted, later adds lost
                # to an unclean exit while the SQLite catalog kept them —
                # leaves dim set yet silently under-serves recall (search
                # hits the index; count() reads the catalog). Rebuild from
                # stored embeddings whenever the index trails the catalog;
                # this subsumes the empty-index case (size 0 < cnt).
                index = getattr(getattr(coll, "_collection", None), "_index", None)
                index_size = getattr(index, "size", None)
                if index_size is None:
                    # simplevecdb's internals moved (the index handle is
                    # double-private); fall back to the library's own
                    # reconciliation heuristic instead of crashing the
                    # recovery path on an upgrade.
                    await coll.rebuild_if_needed()
                elif int(index_size) < cnt:
                    await coll.rebuild_index()
        except Exception:
            self._index_ready = False
            _LOG.exception("index-ready guard failed; will retry on next recall")
            raise

    async def encode_episode(
        self,
        content: str,
        session_id: str,
        *,
        tags: Sequence[str] | None = None,
        salience: float | None = None,
        supersedes: int | None = None,
    ) -> dict[str, Any]:
        await self._ensure_sr_loaded()
        # Validate the retire target before the encode so a bad id fails
        # the whole call up-front (mirrors store_semantic_memory).
        await self._validate_episode_supersedes(supersedes)
        result = await _encode_mod.run(
            self,
            content=content,
            session_id=session_id,
            tags=tags,
            salience=salience,
        )
        if supersedes is not None:
            result["superseded"] = await self._retire_superseded_episode(supersedes, result)
        await self._emit(
            "encode",
            session_id=session_id,
            payload={"id": result.get("id"), "deduped": bool(result.get("deduped"))},
        )
        return result

    async def _validate_episode_supersedes(self, supersedes: int | None) -> None:
        """Reject a ``supersedes`` episode target that is malformed or absent."""
        if supersedes is None:
            return
        if supersedes < 0:
            raise ValidationError("supersedes must be a non-negative id")
        if not await self.episodic.get_documents({"id": supersedes}):
            raise ValidationError(f"supersedes id {supersedes} not found in episodic store")

    async def _retire_superseded_episode(
        self, supersedes: int, result: dict[str, Any]
    ) -> int | None:
        """Delete the episode (and its theta-segment group) the encode replaces.

        Returns the retired id, or ``None`` when the retire was skipped
        because the new content deduped onto the superseded episode itself
        (or one of its group siblings) — deleting then would destroy the
        only remaining copy of the trace. A target that vanished between
        validation and retire counts as retired (the goal state holds).
        If the delete fails after a fresh insert, the insert is rolled
        back and the error re-raised so the pre-call state is restored
        (mirrors ``store_semantic_memory``'s supersedes rollback); a
        dedup-merge result is left in place since its retrieval bump is
        harmless.
        """
        new_id = int(result.get("id", -1))
        if result.get("deduped"):
            target = await self.episodic.get_documents({"id": supersedes})
            if not target:
                return supersedes
            group_id = int(target[0][2].get("episode_group_id", supersedes))
            members = await self.fetch_episode_group(group_id)
            member_ids = {int(m["id"]) for m in members} | {supersedes}
            if new_id in member_ids:
                return None
        try:
            await self.delete_episode(supersedes)
        except ValidationError:
            return supersedes
        except Exception:
            if not result.get("deduped"):
                ids = sorted({new_id, *(int(s) for s in (result.get("segment_ids") or []))} - {-1})
                if ids:
                    with contextlib.suppress(Exception):
                        await self._evict_and_delete(ids)
            raise
        return supersedes

    async def recall(
        self,
        query: str,
        *,
        session_id: str | None = None,
        k: int = 8,
        mode: str = "auto",
        scope_session: bool | None = None,
        tags: list[str] | None = None,
        diagnostics: RecallDiagnostics | None = None,
    ) -> list[dict[str, Any]]:
        return await self._recall(
            query=query,
            session_id=session_id,
            k=k,
            mode=mode,
            scope_session=scope_session,
            tags=tags,
            reinforce=True,
            diagnostics=diagnostics,
        )

    async def recall_readonly(
        self,
        query: str,
        *,
        session_id: str | None = None,
        k: int = 8,
        mode: str = "auto",
        scope_session: bool | None = None,
    ) -> list[dict[str, Any]]:
        """Non-reinforcing recall for GET-like entry points (MCP resource reads).

        Same ranked pipeline as :meth:`recall`, but it does not bump
        ``retrieval_count``, update the SR/plasticity graph, or emit an
        interaction event (so it never appends to the durable event log
        or advances dream pressure). A host may prefetch, cache, or poll
        a resource, so a read must not mutate durable memory state.

        ``scope_session`` carries the same tri-state semantics as
        :meth:`recall` (unset = scope when a ``session_id`` is given).
        """
        return await self._recall(
            query=query,
            session_id=session_id,
            k=k,
            mode=mode,
            reinforce=False,
            scope_session=scope_session,
        )

    async def _recall(
        self,
        *,
        query: str,
        session_id: str | None,
        k: int,
        mode: str,
        reinforce: bool,
        scope_session: bool | None = None,
        tags: list[str] | None = None,
        diagnostics: RecallDiagnostics | None = None,
    ) -> list[dict[str, Any]]:
        await self._ensure_sr_loaded()
        await self._ensure_index_ready()
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
                    scope_session=scope_session,
                    tags=tags,
                    reinforce=reinforce,
                    diagnostics=diagnostics,
                )
            if self.config.tracing_enabled:
                self.last_trace = _trace_ctx.as_dict()
        if reinforce:
            # A read-only recall is GET-like: it must not append to the
            # durable interaction log or advance dream pressure
            # (events_since_dream), or a host that prefetches/polls the
            # resource could trip background replay unbidden — making
            # memory dynamics depend on host I/O, not agent cognition.
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
        # Serialise under the consolidation lock: forget deletes episodes
        # and cold schemas while consolidate is forming/patching schemas
        # over the same rows, and two concurrent forget() passes would
        # compute the same ``weak`` id list and double-delete it. Consolidate
        # never calls forget, so this is not re-entrant.
        async with self._consolidate_lock:
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
    async def _validate_supersedes(self, supersedes: int | None) -> None:
        """Reject a ``supersedes`` target that is malformed or absent."""
        if supersedes is None:
            return
        if supersedes < 0:
            raise ValidationError("supersedes must be a non-negative id")
        if not await self.semantic.get_documents({"id": supersedes}):
            raise ValidationError(f"supersedes id {supersedes} not found in semantic store")

    async def store_semantic_memory(
        self,
        content: str,
        *,
        kind: str = "fact",
        tags: Sequence[str] | None = None,
        confidence: float = 1.0,
        supersedes: int | None = None,
        scope: str | None = None,
        session_id: str | None = None,
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
        validate_tags(self.config, tags)
        await self._validate_supersedes(supersedes)
        # Scope axis: resolve_scope owns the whole policy — explicit
        # ``scope`` wins (validated), a bare session_id means
        # session-scoped, and a bare call (no scope, no session_id) is
        # rejected so a write can never silently default to global.
        eff_scope = resolve_scope(scope, session_id)

        now = now_seconds()
        tag_list = sorted({t for t in (tags or []) if t})
        metadata: dict[str, Any] = {
            "kind": kind,
            "source_episode_ids": [],
            "tags": tag_list,
            "confidence": float(confidence),
            "created_at": now,
            "updated_at": now,
            "authored": True,
            "scope": eff_scope,
        }
        if eff_scope == SCOPE_SESSION:
            metadata["session_id"] = session_id
        sem_ids = await self.semantic.add_texts(
            [content],
            metadatas=[metadata],
            embeddings=await self.vectorise([content]),
        )
        sem_id = int(sem_ids[0])
        # Patch id back into metadata so downstream recall can read it
        # uniformly (mirrors how consolidate does it). Roll back on
        # failure so an orphan (no ``id`` field) cannot persist — see
        # ``encode._insert_single`` for the matching pattern.
        try:
            await self.semantic.update_metadata([(sem_id, {"id": sem_id})])
        except Exception:
            with contextlib.suppress(Exception):
                await self.semantic.delete_by_ids([sem_id])
            raise
        if supersedes is not None:
            # Retire the stale entry so the correction replaces it instead
            # of layering: both would otherwise keep surfacing in recall.
            # Existence was validated above, before the new insert. If the
            # retire fails we'd be left with BOTH entries layered — the very
            # state supersedes exists to prevent — so roll the new insert
            # back and re-raise, restoring the pre-call state for a clean
            # retry (mirrors the metadata-patch rollback above).
            try:
                await self.semantic.delete_by_ids([supersedes])
            except Exception:
                with contextlib.suppress(Exception):
                    await self.semantic.delete_by_ids([sem_id])
                raise
        return {
            "id": sem_id,
            "kind": kind,
            "tags": tag_list,
            "confidence": float(confidence),
            "created_at": now,
            "superseded": supersedes,
            "scope": eff_scope,
        }

    async def recall_semantic_memory(
        self,
        query: str,
        *,
        k: int = 8,
        kind: str | None = None,
        session_id: str | None = None,
        scope_session: bool | None = None,
        diagnostics: RecallDiagnostics | None = None,
    ) -> list[dict[str, Any]]:
        """Semantic-only recall; thin delegate like every other operation.

        Validation, the over-fetch policy, the scope filter, the
        relevance gate, and the cold-schema access bump all live in
        ``hippocampus.recall.run_semantic`` so the semantic and hybrid
        recall paths share one home and cannot drift apart.
        """
        return await _recall_mod.run_semantic(
            self,
            query=query,
            k=k,
            kind=kind,
            session_id=session_id,
            scope_session=scope_session,
            diagnostics=diagnostics,
        )


# Late imports avoid a top-of-file cycle: each sub-module needs
# UNCONSOLIDATED and ``now_seconds`` (bound above), and references
# MemoryService only through TYPE_CHECKING.
from .hippocampus import encode as _encode_mod  # noqa: E402
from .hippocampus import recall as _recall_mod  # noqa: E402
from .hippocampus import replay as _replay_mod  # noqa: E402
from .neocortex import consolidate as _consolidate_mod  # noqa: E402
from .neocortex import forget as _forget_mod  # noqa: E402
from .neocortex import reflect as _reflect_mod  # noqa: E402
