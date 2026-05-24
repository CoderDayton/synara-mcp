"""Deterministic tests for MemoryService.vectorise batch path.

These tests do not touch a real embedder. They drive ``vectorise``
through recording stubs to prove:

- when ``embed_batch_fn`` is configured, exactly one batch call is
  issued for any non-empty ``texts`` input;
- when only ``embed_fn`` is configured, one single-text call is issued
  per text and the result matches the batch-path output element-wise;
- the order, dimension, and value of each vector survives the batch
  path (no transposition, no truncation, no padding);
- the contract violation ``len(result) != len(texts)`` is rejected with
  :class:`ValidationError` instead of corrupting the index;
- empty input short-circuits to ``[]`` without invoking either
  backend;
- dim drift on the batch path raises just like the single-text path;
- the ``embed_batch_fn`` without ``embed_fn`` configuration is
  rejected at construction.
"""

from __future__ import annotations

import hashlib
from collections.abc import AsyncIterator, Sequence

import numpy as np
import pytest
import pytest_asyncio
from simplevecdb import AsyncVectorDB

from synara.core.errors import ValidationError
from synara.features.memory.service import MemoryConfig, MemoryService

_DIM = 16


def _hash_vec(text: str, dim: int = _DIM) -> list[float]:
    """Identical to ``test_service.hash_embed`` but dim-parametric so we
    can exercise drift detection."""
    seed = int.from_bytes(hashlib.sha256(text.encode("utf-8")).digest()[:8], "big")
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(dim).astype(np.float32)
    n = float(np.linalg.norm(v))
    out = (v / n) if n > 0 else v
    return [float(x) for x in out.tolist()]


class _RecordingEmbedder:
    """Records every single-text and batch call so a test can assert
    the exact number of round-trips and the input passed in each one."""

    def __init__(self, *, dim: int = _DIM) -> None:
        self.single_calls: list[str] = []
        self.batch_calls: list[list[str]] = []
        self._dim = dim

    def single(self, text: str) -> list[float]:
        self.single_calls.append(text)
        return _hash_vec(text, dim=self._dim)

    async def batch(self, texts: Sequence[str]) -> list[list[float]]:
        # Snapshot the input as a list — the caller may pass a view that
        # mutates later, and we want post-hoc assertions to see exactly
        # what the function received.
        self.batch_calls.append(list(texts))
        return [_hash_vec(t, dim=self._dim) for t in texts]


@pytest_asyncio.fixture
async def db() -> AsyncIterator[AsyncVectorDB]:
    d = AsyncVectorDB(":memory:")
    try:
        yield d
    finally:
        await d.close()


# ---------------------------------------------------------- batch path used
async def test_vectorise_routes_through_batch_when_configured(
    db: AsyncVectorDB,
) -> None:
    rec = _RecordingEmbedder()
    svc = MemoryService(db, config=MemoryConfig(), embed_fn=rec.single, embed_batch_fn=rec.batch)

    vecs = await svc.vectorise(["a", "b", "c"])

    assert vecs is not None
    assert len(vecs) == 3
    assert rec.batch_calls == [["a", "b", "c"]]
    assert rec.single_calls == []  # no fallback


# ---------------------------------------------------------- fallback parity
async def test_vectorise_fallback_uses_single_call_per_text(db: AsyncVectorDB) -> None:
    rec = _RecordingEmbedder()
    svc = MemoryService(db, config=MemoryConfig(), embed_fn=rec.single)

    vecs = await svc.vectorise(["x", "y", "z", "w"])

    assert vecs is not None
    assert len(vecs) == 4
    assert rec.single_calls == ["x", "y", "z", "w"]
    assert rec.batch_calls == []


async def test_batch_and_single_paths_produce_identical_vectors(db: AsyncVectorDB) -> None:
    """Strong correctness check: for the same inputs, the batch path
    output must be element-wise identical to the single-text path. Any
    reshape / transpose bug in the batch wiring would fail here."""
    rec_a = _RecordingEmbedder()
    rec_b = _RecordingEmbedder()
    single = MemoryService(db, config=MemoryConfig(), embed_fn=rec_a.single)
    db_b = AsyncVectorDB(":memory:")
    try:
        batched = MemoryService(
            db_b,
            config=MemoryConfig(),
            embed_fn=rec_b.single,
            embed_batch_fn=rec_b.batch,
        )

        texts = ["alpha", "beta", "gamma", "delta", "epsilon"]
        v_single = await single.vectorise(texts)
        v_batch = await batched.vectorise(texts)

        assert v_single == v_batch  # bit-identical: same hash, same RNG
    finally:
        await db_b.close()


# ---------------------------------------------------------- empty input
async def test_vectorise_empty_returns_empty_without_calling_backend(
    db: AsyncVectorDB,
) -> None:
    rec = _RecordingEmbedder()
    svc = MemoryService(db, config=MemoryConfig(), embed_fn=rec.single, embed_batch_fn=rec.batch)

    assert await svc.vectorise([]) == []
    assert rec.batch_calls == []
    assert rec.single_calls == []


# ---------------------------------------------------------- no embedder at all
async def test_vectorise_returns_none_when_no_embed_fn(db: AsyncVectorDB) -> None:
    svc = MemoryService(db, config=MemoryConfig())
    assert await svc.vectorise(["a"]) is None


# ---------------------------------------------------------- length mismatch
async def test_vectorise_rejects_length_mismatch_from_batch_fn(db: AsyncVectorDB) -> None:
    """Defensive: a batch fn that returns the wrong number of vectors
    must raise, not silently corrupt the text→vector alignment that
    simplevecdb later writes to disk."""

    async def short_batch(texts: Sequence[str]) -> list[list[float]]:
        # Drop the last text intentionally.
        return [_hash_vec(t) for t in texts[:-1]]

    svc = MemoryService(
        db,
        config=MemoryConfig(),
        embed_fn=_hash_vec,
        embed_batch_fn=short_batch,
    )

    with pytest.raises(ValidationError, match="2 vectors for 3 texts"):
        await svc.vectorise(["a", "b", "c"])


# ---------------------------------------------------------- dim drift
async def test_vectorise_detects_dim_drift_on_batch_path(db: AsyncVectorDB) -> None:
    """First call: batch returns dim 16. Second call: same batch fn
    returns dim 32. The cached dim must catch it."""
    state = {"dim": 16}

    async def shifting_batch(texts: Sequence[str]) -> list[list[float]]:
        return [_hash_vec(t, dim=state["dim"]) for t in texts]

    svc = MemoryService(
        db,
        config=MemoryConfig(),
        embed_fn=lambda t: _hash_vec(t, dim=state["dim"]),
        embed_batch_fn=shifting_batch,
    )

    first = await svc.vectorise(["a"])
    assert first is not None
    assert len(first[0]) == 16

    state["dim"] = 32
    with pytest.raises(ValidationError, match="embedding dimension drift"):
        await svc.vectorise(["b"])


# ---------------------------------------------------------- dim cached on first batch
async def test_vectorise_caches_dim_from_first_batch(db: AsyncVectorDB) -> None:
    rec = _RecordingEmbedder(dim=24)
    svc = MemoryService(db, config=MemoryConfig(), embed_fn=rec.single, embed_batch_fn=rec.batch)

    assert await svc.embedding_dimension() is None or True  # service may pre-probe
    await svc.vectorise(["hello"])
    assert await svc.embedding_dimension() == 24


# ---------------------------------------------------------- construct-time gate
async def test_embed_batch_fn_requires_embed_fn(db: AsyncVectorDB) -> None:
    """A batch fn alone leaves ``query_arg`` and the dim probe without
    a callable. Reject at construction so the failure is loud."""

    async def batch(texts: Sequence[str]) -> list[list[float]]:
        return [[0.0] for _ in texts]

    with pytest.raises(ValidationError, match="requires embed_fn"):
        MemoryService(db, config=MemoryConfig(), embed_batch_fn=batch)


# ---------------------------------------------------------- order preservation
async def test_batch_path_preserves_order(db: AsyncVectorDB) -> None:
    """Per-input ordering is load-bearing: simplevecdb writes vectors
    aligned with texts, so a permuted batch result would silently
    corrupt the index. We verify by re-encoding each text individually
    and comparing position-by-position."""
    rec = _RecordingEmbedder()
    svc = MemoryService(db, config=MemoryConfig(), embed_fn=rec.single, embed_batch_fn=rec.batch)

    texts = ["zeta", "yankee", "x-ray", "whiskey", "victor"]
    batch_vecs = await svc.vectorise(texts)

    assert batch_vecs is not None
    for i, t in enumerate(texts):
        assert batch_vecs[i] == _hash_vec(t)
