"""Validation guards + _gather_candidates failure branches for hippocampus.complete."""

from __future__ import annotations

import hashlib
from collections.abc import AsyncIterator
from typing import Any

import numpy as np
import pytest
import pytest_asyncio
from simplevecdb import AsyncVectorDB

from synara.features.memory.config import MemoryConfig
from synara.features.memory.hippocampus import complete as complete_mod
from synara.features.memory.service import MemoryService


def hash_embed(text: str, dim: int = 32) -> list[float]:
    seed = int.from_bytes(hashlib.sha256(text.encode("utf-8")).digest()[:8], "big", signed=False)
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(dim).astype(np.float32)
    n = float(np.linalg.norm(v))
    out = (v / n) if n > 0 else v
    return [float(x) for x in out.tolist()]


@pytest_asyncio.fixture
async def service() -> AsyncIterator[MemoryService]:
    db = AsyncVectorDB(":memory:")
    try:
        yield MemoryService(db, config=MemoryConfig(), embed_fn=hash_embed)
    finally:
        await db.close()


# ============================================================ _normalize floor
def test_normalize_returns_zero_vector_when_norm_below_floor() -> None:
    """Line 64: norm < _NORM_FLOOR → returns zeros_like(v)."""
    tiny = np.array([1e-20, 1e-20, 1e-20], dtype=np.float64)
    out = complete_mod._normalize(tiny)
    np.testing.assert_array_equal(out, np.zeros_like(tiny))


def test_normalize_preserves_unit_direction_otherwise() -> None:
    v = np.array([3.0, 4.0], dtype=np.float64)
    out = complete_mod._normalize(v)
    assert np.allclose(out, np.array([0.6, 0.8]))


# ============================================================ run() input guards
async def test_run_rejects_non_positive_beta(service: MemoryService) -> None:
    with pytest.raises(ValueError, match="beta"):
        await complete_mod.run(
            service, [1.0, 0.0], ep_filter=None, k_inner=4, iters=3, beta=0.0, eta0=0.6
        )


async def test_run_rejects_non_finite_beta(service: MemoryService) -> None:
    with pytest.raises(ValueError, match="beta"):
        await complete_mod.run(
            service,
            [1.0, 0.0],
            ep_filter=None,
            k_inner=4,
            iters=3,
            beta=float("inf"),
            eta0=0.6,
        )


@pytest.mark.parametrize("bad_eta0", [0.0, -0.1, 1.5])
async def test_run_rejects_out_of_range_eta0(service: MemoryService, bad_eta0: float) -> None:
    with pytest.raises(ValueError, match="eta0"):
        await complete_mod.run(
            service,
            [1.0, 0.0],
            ep_filter=None,
            k_inner=4,
            iters=3,
            beta=8.0,
            eta0=bad_eta0,
        )


async def test_run_rejects_negative_eps(service: MemoryService) -> None:
    with pytest.raises(ValueError, match="eps"):
        await complete_mod.run(
            service,
            [1.0, 0.0],
            ep_filter=None,
            k_inner=4,
            iters=3,
            beta=8.0,
            eta0=0.6,
            eps=-1e-6,
        )


async def test_run_rejects_non_finite_eps(service: MemoryService) -> None:
    with pytest.raises(ValueError, match="eps"):
        await complete_mod.run(
            service,
            [1.0, 0.0],
            ep_filter=None,
            k_inner=4,
            iters=3,
            beta=8.0,
            eta0=0.6,
            eps=float("nan"),
        )


async def test_run_rejects_non_positive_k_inner(service: MemoryService) -> None:
    with pytest.raises(ValueError, match="k_inner"):
        await complete_mod.run(
            service, [1.0, 0.0], ep_filter=None, k_inner=0, iters=3, beta=8.0, eta0=0.6
        )


async def test_run_rejects_empty_q0(service: MemoryService) -> None:
    with pytest.raises(ValueError, match="q0"):
        await complete_mod.run(service, [], ep_filter=None, k_inner=4, iters=3, beta=8.0, eta0=0.6)


async def test_run_rejects_non_finite_q0(service: MemoryService) -> None:
    with pytest.raises(ValueError, match="finite"):
        await complete_mod.run(
            service,
            [1.0, float("nan")],
            ep_filter=None,
            k_inner=4,
            iters=3,
            beta=8.0,
            eta0=0.6,
        )


# ============================================================ _gather_candidates errors
async def test_gather_candidates_swallows_similarity_search_failure(
    service: MemoryService, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Lines 128-133: similarity_search raises → log + continue to next leg."""
    # Seed both stores so coll.count() > 0 → the try-block actually runs.
    await service.encode_episode("episode one", "s1")
    await service.store_semantic_memory("semantic fact", kind="fact")

    async def boom_similarity(*args: Any, **kwargs: Any) -> Any:
        raise RuntimeError("synthetic backend hiccup")

    # Break only the episodic leg; the semantic leg still anchors the
    # candidate matrix.
    monkeypatch.setattr(service.episodic, "similarity_search", boom_similarity)
    X = await complete_mod._gather_candidates(service, [0.1] * 32, ep_filter=None, k=4)
    # Semantic leg still contributes at least one row.
    assert X.shape[0] >= 1


async def test_gather_candidates_skips_when_embedding_lookup_returns_none(
    service: MemoryService, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Line 137: emap.get(did) is None → continue past the row."""
    await service.encode_episode("trace alpha", "s1")

    # Replace get_embeddings_by_ids with a stub that returns an empty
    # mapping so every did's lookup returns None.
    async def empty_emap(ids: list[int]) -> dict[int, list[float]]:
        return {}

    monkeypatch.setattr(service.episodic, "get_embeddings_by_ids", empty_emap)
    monkeypatch.setattr(service.semantic, "get_embeddings_by_ids", empty_emap)
    X = await complete_mod._gather_candidates(service, [0.1] * 32, ep_filter=None, k=4)
    # No rows survive — both legs produced ids but no vectors.
    assert X.shape == (0,)


async def test_gather_candidates_skips_non_finite_embeddings(
    service: MemoryService, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Line 141: stored vector contains NaN/Inf → skip that row."""
    await service.encode_episode("trace beta", "s1")

    async def poisoned_emap(ids: list[int]) -> dict[int, list[float]]:
        # NaN-poisoned vector.
        return {ids[0]: [float("nan")] * 32} if ids else {}

    async def empty_emap(ids: list[int]) -> dict[int, list[float]]:
        return {}

    monkeypatch.setattr(service.episodic, "get_embeddings_by_ids", poisoned_emap)
    monkeypatch.setattr(service.semantic, "get_embeddings_by_ids", empty_emap)
    X = await complete_mod._gather_candidates(service, [0.1] * 32, ep_filter=None, k=4)
    assert X.shape == (0,)
