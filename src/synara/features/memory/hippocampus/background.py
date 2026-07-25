"""Query-conditioned estimate of an embedding model's "unrelated" distance.

The relevance gate needs to answer "is this hit actually close, on this
model's scale?" without a hardcoded threshold, because the scale is a
property of the model and nobody retunes constants after a model swap.
That requires a *reference* distance meaning "unrelated", and the whole
design rests on picking the right one.

``dynamic_ceiling`` originally took that reference from the far end of
the recall's own candidate list (its p90). The assumption was that
over-fetched candidates run from relevant to irrelevant, so their tail
approximates "unrelated". It does not. The candidates are the corpus's
*nearest* neighbours by construction — the top 32-64 of thousands — so
even the p90 is still a fairly good match. The reference came out far
too small, the ceiling with it, and on a model whose distances are
compressed the ceiling landed below the *best* hit: every query returned
nothing, including exact matches.

The reference has to come from documents the query did not select. A
uniform random sample of the collection is exactly that. Measured on a
661-episode store with ``nomic-embed-text-v1``: the mean distance from a
query to a random sample is ~0.59, true targets land at 0.25-0.42
(0.44-0.74 of the reference) and off-topic queries bottom out at
0.45-0.57 (0.76-0.85 of it). One multiplicative ``alpha`` cleanly
separates the two, which is what the gate wanted all along.

The sample is per collection, cached, and refreshed only when the
collection has changed materially — one matrix-vector product per recall
against a few hundred cached vectors, which is nothing next to the ANN
search it gates.

The sample is drawn from the *index's* copy of the vectors, not from the
canonical float32 column in SQLite. Both hold the same vectors, but the
index holds them quantized, and the distances this reference is compared
against come out of that index — measuring the reference on the other
copy would put the two sides of ``distance <= alpha * reference`` on
two different scales. See ``_index_vectors``.
"""

from __future__ import annotations

import asyncio
import logging
import math
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:  # pragma: no cover - typing only
    from ..config import MemoryConfig

_LOG = logging.getLogger(__name__)


class _Sample:
    """One collection's cached background sample.

    ``matrix`` rows are unit-length, so the cosine distance to a
    unit-length query is ``1 - matrix @ query``.
    """

    __slots__ = ("matrix", "population")

    def __init__(self, matrix: np.ndarray, population: int) -> None:
        self.matrix = matrix
        self.population = population

    def drifted(self, population: int, *, ratio: float) -> bool:
        """Has the collection changed enough to invalidate this sample?

        Relative to the population at sampling time, so a young store
        resamples often (where each write shifts the distribution a lot)
        and a large one rarely (where it barely moves).
        """
        if self.population <= 0:
            return True
        return abs(population - self.population) / self.population > ratio


class BackgroundReference:
    """Per-collection background samples with lazy refresh.

    One instance is owned by the memory service and shared by every
    recall. Sampling is guarded per collection so a burst of concurrent
    recalls on a cold cache pays for one sample, not one each.
    """

    def __init__(self, config: MemoryConfig) -> None:
        self._config = config
        self._samples: dict[str, _Sample] = {}
        self._locks: dict[str, asyncio.Lock] = {}
        # Dimension mismatches are a persistent misconfiguration, not a
        # transient fault; log the first one per collection and stay
        # quiet after that rather than one line per recall.
        self._warned: set[str] = set()

    @staticmethod
    def _unit_query(query: object) -> np.ndarray | None:
        """``query`` as a unit vector, or ``None`` if it is not usable.

        A recall configured without an embedder passes the raw query
        string straight through to simplevecdb; there is no vector to
        measure a background against, so the ceiling simply does not
        apply.
        """
        if not isinstance(query, list) or not query:
            return None
        vector = np.asarray(query, dtype=np.float32)
        if vector.ndim != 1:
            return None
        norm = float(np.linalg.norm(vector))
        if norm == 0.0 or not math.isfinite(norm):
            return None
        return vector / norm

    async def reference(self, collection: Any, query: object) -> float | None:
        """Mean distance from ``query`` to a random sample of the collection.

        ``None`` means "no defensible reference" and the caller must skip
        the absolute ceiling entirely — never substitute a guess. That
        happens when the query is not a vector (no embedder wired), when
        the collection is too small to sample, or when the stored vectors
        do not match the query's width.
        """
        cfg = self._config
        q = self._unit_query(query)
        if q is None or cfg.recall_background_sample <= 0:
            return None
        name = getattr(collection, "name", None) or repr(collection)
        try:
            sample = await self._sample_for(collection, name)
        except Exception:
            # The gate is an optimisation; a failure to characterise the
            # background must degrade to "no ceiling", never to a failed
            # recall.
            _LOG.warning("background sampling failed for %s", name, exc_info=True)
            return None
        if sample is None:
            return None

        if q.shape[0] != sample.matrix.shape[1]:
            if name not in self._warned:
                self._warned.add(name)
                _LOG.warning(
                    "background sample for %s is %d-d but the query is %d-d; "
                    "skipping the distance ceiling. Re-embed the store "
                    "(scripts/reembed.py) after an embedding-model change.",
                    name,
                    sample.matrix.shape[1],
                    q.shape[0],
                )
            return None
        distances = 1.0 - (sample.matrix @ q)
        value = float(np.mean(distances))
        return value if math.isfinite(value) else None

    def invalidate(self, collection_name: str | None = None) -> None:
        """Drop cached samples; the next recall re-samples."""
        if collection_name is None:
            self._samples.clear()
        else:
            self._samples.pop(collection_name, None)

    async def _sample_for(self, collection: Any, name: str) -> _Sample | None:
        cfg = self._config
        population = int(await collection.count())
        # Checked before the cache, not after: a collection that has been
        # emptied out (a bulk forget) is too small to characterise *now*,
        # and a stale sample from when it was larger must not keep gating
        # recalls against a population that no longer exists.
        if population < cfg.recall_background_min_sample:
            return None
        cached = self._samples.get(name)
        if cached is not None and not cached.drifted(
            population, ratio=cfg.recall_background_refresh_ratio
        ):
            return cached
        lock = self._locks.setdefault(name, asyncio.Lock())
        async with lock:
            # Re-check: another coroutine may have sampled while we waited.
            cached = self._samples.get(name)
            if cached is not None and not cached.drifted(
                population, ratio=cfg.recall_background_refresh_ratio
            ):
                return cached
            sample = await self._build(collection, population)
            if sample is not None:
                self._samples[name] = sample
            return sample

    async def _build(self, collection: Any, population: int) -> _Sample | None:
        cfg = self._config
        ids = await self._all_ids(collection)
        if len(ids) < cfg.recall_background_min_sample:
            return None
        # Seeded so a given collection samples reproducibly within a
        # process — a gate whose threshold jitters between identical
        # recalls is untestable and unreportable.
        rng = np.random.default_rng(cfg.recall_background_seed)
        size = min(cfg.recall_background_sample, len(ids))
        picked = [int(ids[i]) for i in rng.choice(len(ids), size=size, replace=False)]

        stored = await collection.get_embeddings_by_ids(picked)
        kept: list[int] = []
        rows: list[np.ndarray] = []
        for doc_id in picked:
            vector = stored.get(doc_id)
            if vector is None:
                continue
            row = np.asarray(vector, dtype=np.float32)
            if row.ndim != 1:
                continue
            # Ids stay aligned with rows so the index lookup below can
            # ask for exactly the vectors that survived this filter.
            kept.append(doc_id)
            rows.append(row)
        # A ragged sample means the store is mid-migration (rows at two
        # different widths). Keep the majority width rather than letting
        # numpy build an object array.
        if not rows:
            return None
        widths = {int(r.shape[0]) for r in rows}
        width = max(widths, key=lambda w: sum(1 for r in rows if r.shape[0] == w))
        at_width = [i for i, row in enumerate(rows) if row.shape[0] == width]
        matrix = np.asarray([rows[i] for i in at_width])
        from_index = await self._index_vectors(collection, [kept[i] for i in at_width], width)
        if from_index is not None:
            matrix = from_index
        norms = np.linalg.norm(matrix, axis=1)
        # Zero and non-finite vectors would produce nan distances that
        # poison the mean; drop them rather than propagate.
        usable = np.isfinite(norms) & (norms > 0)
        if int(usable.sum()) < cfg.recall_background_min_sample:
            return None
        matrix = matrix[usable] / norms[usable][:, None]
        return _Sample(matrix, population)

    @staticmethod
    async def _index_vectors(collection: Any, ids: list[int], width: int) -> np.ndarray | None:
        """The sampled vectors as the ANN index actually holds them.

        The distances this reference gates are computed by usearch over
        its own quantized copy of the vectors, not over the canonical
        float32 column in SQLite, so measuring the reference on the
        column puts the two sides of ``distance <= alpha * reference`` on
        slightly different scales. Measured at 768-d under
        ``Quantization.INT8``: a sample reconstructed from the index
        tracks the engine's own distances to 0.005, the canonical column
        only to 0.012.

        ``None`` means "keep the canonical vectors" — a valid reference,
        just the other scale. That is the answer whenever the index
        cannot speak for this sample: no index to ask, a row it does not
        hold (it substitutes a zero vector and warns), or a width that
        disagrees with the column. The last case is what
        ``Quantization.BIT`` looks like — it packs eight dimensions into
        a byte, so its rows are not vectors in the query's space at all.
        """
        index = getattr(getattr(collection, "_collection", None), "_index", None)
        if index is None or not ids:
            return None
        try:
            raw = await collection._run(index.get, np.asarray(ids, dtype=np.uint64))
            matrix = np.asarray(raw, dtype=np.float32)
        except Exception:
            _LOG.debug("index-space background sample unavailable", exc_info=True)
            return None
        if matrix.shape != (len(ids), width):
            return None
        # A zero row is how ``UsearchIndex.get`` reports a key it does
        # not hold, and it means the index has drifted from the catalog.
        # Fall back wholesale rather than sample a mixture of the two
        # representations.
        if not np.all(np.isfinite(matrix)) or not np.all(np.linalg.norm(matrix, axis=1) > 0.0):
            return None
        return matrix

    @staticmethod
    async def _all_ids(collection: Any) -> list[int]:
        """Row ids only.

        Deliberately not ``get_documents()``: that pulls every document's
        text and metadata into memory, which is fine at a few hundred
        episodes and ruinous at a hundred thousand. simplevecdb exposes
        no public id-only listing, so this reaches through to the catalog
        the same way the edge bookkeeping does.
        """
        sync = collection._collection
        return list(await collection._run(sync._catalog.list_all_ids))
