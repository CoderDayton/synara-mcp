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

import asyncio
import inspect
import time
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from typing import Any

from simplevecdb import AsyncVectorDB

from synara.core.errors import ValidationError

from .separate import DGProjector as _DGProjector
from .separate import jaccard as _dg_jaccard

# Either a sync embedder ``f(text) -> vec`` or an async one
# ``async f(text) -> vec``. The service normalises both to async at
# construction so internal call sites only ever ``await``.
EmbedFn = Callable[[str], Sequence[float] | Awaitable[Sequence[float]]]

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
    # Power-law (Wickelgren/Wixted) decay exponent used by ``forget``:
    #   S(t) = salience * sum_k (1 + (t - t_k))^(-d)
    # d ~ 0.5 fits behavioural retention curves better than the exponential
    # Ebbinghaus form. Larger d = faster decay.
    forget_d: float = 0.5
    # Cap on retained access-history timestamps per episode; older entries
    # are FIFO-evicted. Keeps storage bounded while preserving Anderson's
    # base-level activation shape for typical usage.
    access_history_cap: int = 32
    # Minimum cluster size that yields a semantic schema during consolidation.
    consolidate_min_cluster: int = 2
    # Cosine distance below which an unconsolidated episode is *absorbed*
    # into the nearest existing schema (instead of contributing to a new
    # cluster). Implements schema-fitting fast-track consolidation
    # (Tse et al. 2007). Smaller = stricter fit required.
    consolidate_absorb_distance: float = 0.4
    # CA3 iterative pattern completion (modern Hopfield, Ramsauer 2020).
    # ``recall_completion_iters > 0`` enables the iteration; each step
    # softmax-recombines stored vectors at inverse-temperature
    # ``recall_completion_beta`` and anchors the new query a fraction
    # ``recall_completion_anchor`` toward the original. 0 = single-pass
    # k-NN (the legacy behaviour).
    recall_completion_iters: int = 0
    recall_completion_beta: float = 8.0
    recall_completion_anchor: float = 0.6
    # DG-style pattern separation (random expansive projection + k-WTA).
    # When enabled, ``encode_episode`` checks Jaccard overlap of sparse
    # codes against the top cosine candidates instead of the brittle
    # cosine-distance threshold. Stored episodes carry their support set
    # in ``dg_support`` metadata.
    dg_pattern_separation: bool = True
    dg_expansion: int = 4
    dg_sparsity: float = 0.05
    dg_jaccard_threshold: float = 0.5
    dg_dedup_candidates: int = 4
    dg_seed: int = 0


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

    async def _dedup_jaccard(
        self,
        new_emb: list[float],
        new_support: tuple[int, ...],
        session_id: str,
    ) -> dict[str, Any] | None:
        """DG-Jaccard dedup. Pulls the top cosine candidates and returns
        a dedup result if any candidate's stored support has Jaccard
        overlap >= ``dg_jaccard_threshold`` with ``new_support``."""
        cands = await self.episodic.similarity_search(
            new_emb,
            k=self.config.dg_dedup_candidates,
            filter={"session_id": session_id},
        )
        best_j = 0.0
        best_doc: Any = None
        best_dist = 0.0
        for doc, dist in cands:
            cand_support = doc.metadata.get("dg_support") or []
            if not cand_support:
                continue
            j = _dg_jaccard(new_support, cand_support)
            if j > best_j:
                best_j = j
                best_doc = doc
                best_dist = float(dist)
        if best_doc is None or best_j < self.config.dg_jaccard_threshold:
            return None
        doc_id = int(best_doc.metadata.get("id", -1))
        if doc_id >= 0:
            await self.bump_retrieval(doc_id, best_doc.metadata)
        return {
            "id": doc_id,
            "deduped": True,
            "distance": best_dist,
            "jaccard": float(best_j),
            "session_id": session_id,
        }

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

        # Pattern separation: cosine threshold OR DG-Jaccard, depending
        # on config. The DG path embeds once, computes the sparse code,
        # and compares against the stored ``dg_support`` of the top
        # cosine candidates within the same session.
        new_embs = await self.vectorise([content])
        new_emb = new_embs[0] if new_embs else None
        new_support: tuple[int, ...] = ()
        use_dg = self.config.dg_pattern_separation and new_emb is not None
        dedup_hit: dict[str, Any] | None = None
        if use_dg and new_emb is not None:
            new_support = self._ensure_projector(len(new_emb)).support(new_emb)
            dedup_hit = await self._dedup_jaccard(new_emb, new_support, session_id)
        else:
            q_arg: str | list[float] = (
                new_emb if new_emb is not None else await self.query_arg(content)
            )
            existing = await self.episodic.similarity_search(
                q_arg, k=1, filter={"session_id": session_id}
            )
            if existing and existing[0][1] <= self.config.dedup_distance:
                doc, dist = existing[0]
                doc_id = int(doc.metadata.get("id", -1))
                if doc_id >= 0:
                    await self.bump_retrieval(doc_id, doc.metadata)
                dedup_hit = {
                    "id": doc_id,
                    "deduped": True,
                    "distance": float(dist),
                    "session_id": session_id,
                }
        if dedup_hit is not None:
            return dedup_hit

        encoded_at = now_seconds()
        meta: dict[str, Any] = {
            "session_id": session_id,
            "tags": list(tags) if tags else [],
            "salience": float(salience),
            "encoded_at": encoded_at,
            "last_accessed": encoded_at,
            "retrieval_count": 0,
            # Power-law strength integrates over every retrieval event, so
            # we keep a (capped) timestamp list — not just the count.
            "access_history": [encoded_at],
            "consolidated_into": UNCONSOLIDATED,
        }
        if use_dg and new_support:
            meta["dg_support"] = list(new_support)
        ids = await self.episodic.add_texts([content], metadatas=[meta], embeddings=new_embs)
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
        q = await self.query_arg(query)
        # CA3 iterative pattern completion: refine the query by
        # softmax-recombining its near-neighbours before the final
        # search. Skipped when no embed_fn is configured (q is a string)
        # or when iters == 0.
        iters = self.config.recall_completion_iters
        if iters > 0 and isinstance(q, list):
            result = await _complete_mod.run(
                self,
                q,
                ep_filter=ep_filter,
                k_inner=max(k, 8),
                iters=iters,
                beta=self.config.recall_completion_beta,
                eta0=self.config.recall_completion_anchor,
            )
            q = result.query
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
from . import complete as _complete_mod  # noqa: E402
from . import consolidate as _consolidate_mod  # noqa: E402
from . import forget as _forget_mod  # noqa: E402
from . import reflect as _reflect_mod  # noqa: E402
