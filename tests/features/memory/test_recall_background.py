"""Background "unrelated" distance reference behind the relevance ceiling."""

from __future__ import annotations

import asyncio
import math
from typing import Any

import numpy as np
import pytest

from synara.features.memory.config import MemoryConfig
from synara.features.memory.hippocampus.background import BackgroundReference
from synara.storage import MEMORY_DB_PATH, STORE_EMBEDDINGS, open_database

_DIM = 8


def _unit(values: list[float]) -> list[float]:
    arr = np.asarray(values, dtype=np.float32)
    return list(arr / np.linalg.norm(arr))


class _Catalog:
    def __init__(self, ids: list[int]) -> None:
        self._ids = ids

    def list_all_ids(self) -> list[int]:
        return list(self._ids)


class _FakeIndex:
    """Enough of ``UsearchIndex`` for the sampler.

    ``rows`` is the index's own copy of the vectors — in production the
    quantized one, here just "different from the column" so a test can
    tell which copy the sample was drawn from. A key the index does not
    hold comes back as a zero row, exactly as usearch reports it.
    """

    def __init__(self, rows: dict[int, list[float]], *, width: int = _DIM) -> None:
        self._rows = rows
        self._width = width

    def get(self, keys: Any) -> np.ndarray:
        return np.asarray(
            [self._rows.get(int(k), [0.0] * self._width) for k in keys], dtype=np.float32
        )


class _Sync:
    def __init__(self, ids: list[int], index: Any = None) -> None:
        self._catalog = _Catalog(ids)
        if index is not None:
            self._index = index


class _FakeCollection:
    """Enough of the simplevecdb async collection for the sampler."""

    def __init__(
        self,
        vectors: dict[int, list[float] | None],
        *,
        name: str = "memory_episodic",
        index: Any = None,
    ) -> None:
        self.name = name
        self._vectors = vectors
        self._collection = _Sync(list(vectors), index)
        self.id_calls = 0

    async def count(self) -> int:
        return len(self._vectors)

    async def _run(self, fn: Any, *args: Any, **kwargs: Any) -> Any:
        self.id_calls += 1
        return fn(*args, **kwargs)

    async def get_embeddings_by_ids(self, ids: list[int]) -> dict[int, Any]:
        return {
            i: np.asarray(self._vectors[i], dtype=np.float32)
            for i in ids
            if self._vectors.get(i) is not None
        }


def _orthogonal_corpus(n: int = 64) -> dict[int, list[float] | None]:
    """A corpus spread evenly over a circle in the first two dimensions.

    The mean distance from any unit query in that plane to the corpus is
    1.0, which makes the expected reference exact rather than approximate.
    """
    out: dict[int, list[float] | None] = {}
    for i in range(n):
        angle = 2.0 * math.pi * i / n
        vec = [math.cos(angle), math.sin(angle)] + [0.0] * (_DIM - 2)
        out[i + 1] = _unit(vec)
    return out


def _config(**overrides: Any) -> MemoryConfig:
    base: dict[str, Any] = {"recall_background_min_sample": 8, "recall_background_sample": 64}
    base.update(overrides)
    return MemoryConfig(**base)


async def test_reference_is_mean_distance_to_the_corpus() -> None:
    coll = _FakeCollection(_orthogonal_corpus())
    bg = BackgroundReference(_config())
    query = _unit([1.0, 0.0] + [0.0] * (_DIM - 2))
    value = await bg.reference(coll, query)
    assert value == pytest.approx(1.0, abs=1e-5)


async def test_reference_ignores_query_magnitude() -> None:
    coll = _FakeCollection(_orthogonal_corpus())
    bg = BackgroundReference(_config())
    base = [1.0, 1.0] + [0.0] * (_DIM - 2)
    small = await bg.reference(coll, list(np.asarray(base) * 0.001))
    large = await bg.reference(coll, list(np.asarray(base) * 1000.0))
    assert small is not None
    assert large is not None
    assert small == pytest.approx(large, abs=1e-5)


async def test_sample_is_cached_until_the_collection_drifts() -> None:
    coll = _FakeCollection(_orthogonal_corpus())
    bg = BackgroundReference(_config(recall_background_refresh_ratio=0.25))
    query = _unit([1.0, 0.0] + [0.0] * (_DIM - 2))
    await bg.reference(coll, query)
    await bg.reference(coll, query)
    assert coll.id_calls == 1

    # Under the refresh ratio: still cached.
    for i in range(65, 75):
        coll._vectors[i] = _unit([0.0, 1.0] + [0.0] * (_DIM - 2))
    await bg.reference(coll, query)
    assert coll.id_calls == 1

    # Over it: resampled.
    for i in range(75, 130):
        coll._vectors[i] = _unit([0.0, 1.0] + [0.0] * (_DIM - 2))
    coll._collection = _Sync(list(coll._vectors))
    await bg.reference(coll, query)
    assert coll.id_calls == 2


async def test_concurrent_first_use_samples_once() -> None:
    coll = _FakeCollection(_orthogonal_corpus())
    bg = BackgroundReference(_config())
    query = _unit([1.0, 0.0] + [0.0] * (_DIM - 2))
    await asyncio.gather(*(bg.reference(coll, query) for _ in range(8)))
    assert coll.id_calls == 1


async def test_invalidate_forces_a_resample() -> None:
    coll = _FakeCollection(_orthogonal_corpus())
    bg = BackgroundReference(_config())
    query = _unit([1.0, 0.0] + [0.0] * (_DIM - 2))
    await bg.reference(coll, query)
    bg.invalidate(coll.name)
    await bg.reference(coll, query)
    assert coll.id_calls == 2


@pytest.mark.parametrize("query", [None, "a text query", [], [0.0] * _DIM])
async def test_unusable_query_yields_no_reference(query: Any) -> None:
    coll = _FakeCollection(_orthogonal_corpus())
    bg = BackgroundReference(_config())
    assert await bg.reference(coll, query) is None


async def test_small_collection_yields_no_reference() -> None:
    # Too few rows to characterise: the ceiling must be skipped, not
    # estimated from noise. This is what spares tiny synthetic corpora.
    corpus = dict(list(_orthogonal_corpus().items())[:4])
    coll = _FakeCollection(corpus)
    bg = BackgroundReference(_config(recall_background_min_sample=8))
    assert await bg.reference(coll, _unit([1.0] + [0.0] * (_DIM - 1))) is None


async def test_disabled_sample_size_yields_no_reference() -> None:
    coll = _FakeCollection(_orthogonal_corpus())
    bg = BackgroundReference(_config(recall_background_sample=0))
    assert await bg.reference(coll, _unit([1.0] + [0.0] * (_DIM - 1))) is None


async def test_dimension_mismatch_yields_no_reference() -> None:
    # A store re-embedded at a new width, or one not yet migrated: gating
    # against a reference from the wrong space is worse than not gating.
    coll = _FakeCollection(_orthogonal_corpus())
    bg = BackgroundReference(_config())
    assert await bg.reference(coll, _unit([1.0, 0.0, 0.0])) is None


async def test_degenerate_stored_vectors_are_dropped() -> None:
    corpus = _orthogonal_corpus()
    corpus[1] = [0.0] * _DIM
    corpus[2] = [float("nan")] * _DIM
    corpus[3] = None
    coll = _FakeCollection(corpus)
    bg = BackgroundReference(_config())
    value = await bg.reference(coll, _unit([1.0, 0.0] + [0.0] * (_DIM - 2)))
    assert value is not None
    assert math.isfinite(value)


async def test_sampling_failure_degrades_to_no_reference() -> None:
    class _Broken(_FakeCollection):
        async def count(self) -> int:
            raise RuntimeError("catalog unavailable")

    bg = BackgroundReference(_config())
    value = await bg.reference(_Broken(_orthogonal_corpus()), _unit([1.0] + [0.0] * (_DIM - 1)))
    assert value is None


def _query() -> list[float]:
    return _unit([1.0, 0.0] + [0.0] * (_DIM - 2))


def _index_matching(query: list[float], ids: list[int]) -> _FakeIndex:
    """An index whose every row sits exactly on the query direction.

    Deliberately nothing like the canonical corpus (which averages a
    distance of 1.0 from that query), so the resulting reference says
    unambiguously which copy of the vectors the sample came from.
    """
    return _FakeIndex({i: list(query) for i in ids})


async def test_sample_is_drawn_from_the_index_representation() -> None:
    # The distances this reference gates come out of the index, so the
    # reference has to be measured on the index's copy of the vectors.
    corpus = _orthogonal_corpus()
    coll = _FakeCollection(corpus, index=_index_matching(_query(), list(corpus)))
    bg = BackgroundReference(_config())
    assert await bg.reference(coll, _query()) == pytest.approx(0.0, abs=1e-5)


async def test_missing_index_row_falls_back_to_the_column() -> None:
    # A zero row means the index has drifted from the catalog. Sampling a
    # mixture of the two copies would be worse than using either.
    corpus = _orthogonal_corpus()
    rows = {i: list(_query()) for i in list(corpus)[:-1]}
    coll = _FakeCollection(corpus, index=_FakeIndex(rows))
    bg = BackgroundReference(_config())
    assert await bg.reference(coll, _query()) == pytest.approx(1.0, abs=1e-5)


async def test_index_width_disagreement_falls_back_to_the_column() -> None:
    # What ``Quantization.BIT`` looks like from here: eight dimensions
    # packed per byte, so the rows are not vectors in the query's space.
    corpus = _orthogonal_corpus()
    packed = {i: [1.0, 0.0] for i in corpus}
    coll = _FakeCollection(corpus, index=_FakeIndex(packed, width=2))
    bg = BackgroundReference(_config())
    assert await bg.reference(coll, _query()) == pytest.approx(1.0, abs=1e-5)


async def test_index_failure_falls_back_to_the_column() -> None:
    class _Broken(_FakeIndex):
        def get(self, keys: Any) -> np.ndarray:
            raise RuntimeError("index closed")

    corpus = _orthogonal_corpus()
    coll = _FakeCollection(corpus, index=_Broken({}))
    bg = BackgroundReference(_config())
    assert await bg.reference(coll, _query()) == pytest.approx(1.0, abs=1e-5)


async def test_index_vectors_bind_to_a_real_collection() -> None:
    """The attributes ``_index_vectors`` reaches through are simplevecdb's,
    not ours. If one is renamed upstream the sampler would quietly fall
    back to the column forever, so pin the binding against a real store.
    """
    db = open_database(MEMORY_DB_PATH)
    try:
        coll = db.collection("background_probe", store_embeddings=STORE_EMBEDDINGS)
        corpus = _orthogonal_corpus(32)
        texts = [f"doc {i}" for i in corpus]
        ids = [
            int(i)
            for i in await coll.add_texts(
                texts,
                [{} for _ in texts],
                [list(v) for v in corpus.values() if v is not None],
            )
        ]
        matrix = await BackgroundReference._index_vectors(coll, ids, _DIM)
        assert matrix is not None
        assert matrix.shape == (len(ids), _DIM)
        # Same vectors as the canonical column, just quantized: each row
        # must still point the same way.
        stored = await coll.get_embeddings_by_ids(ids)
        for row, doc_id in zip(matrix, ids, strict=True):
            column = np.asarray(stored[doc_id], dtype=np.float32)
            cosine = float(np.dot(row, column) / (np.linalg.norm(row) * np.linalg.norm(column)))
            assert cosine > 0.99
    finally:
        await db.close()


async def test_ragged_store_uses_the_majority_width() -> None:
    # Mid-migration: most rows at the new width, a few stragglers at the
    # old one. numpy must not be handed a ragged array.
    corpus = _orthogonal_corpus()
    for i in (1, 2, 3):
        corpus[i] = _unit([1.0, 0.0, 0.0])
    coll = _FakeCollection(corpus)
    bg = BackgroundReference(_config())
    value = await bg.reference(coll, _unit([1.0, 0.0] + [0.0] * (_DIM - 2)))
    assert value == pytest.approx(1.0, abs=0.05)
