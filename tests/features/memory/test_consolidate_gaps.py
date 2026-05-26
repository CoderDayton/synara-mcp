"""Validation, edge-case, and helper-level coverage for neocortex.consolidate."""

from __future__ import annotations

import hashlib
from collections.abc import AsyncIterator
from typing import Any

import numpy as np
import pytest
import pytest_asyncio
from simplevecdb import AsyncVectorDB

from synara.core.errors import ValidationError
from synara.features.memory.config import MemoryConfig
from synara.features.memory.neocortex import consolidate as cm
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


# ============================================================ _validate_run_inputs
def test_validate_run_rejects_non_positive_n_clusters(service: MemoryService) -> None:
    with pytest.raises(ValidationError, match="n_clusters"):
        cm._validate_run_inputs(service, session_id=None, n_clusters=0, min_cluster_size=None)


def test_validate_run_rejects_non_positive_min_cluster_size(service: MemoryService) -> None:
    with pytest.raises(ValidationError, match="min_cluster_size"):
        cm._validate_run_inputs(service, session_id=None, n_clusters=None, min_cluster_size=0)


def test_validate_run_rejects_oversized_session_id(service: MemoryService) -> None:
    cap = service.config.max_session_id_chars
    assert cap
    assert cap > 0
    over = "x" * (cap + 1)
    with pytest.raises(ValidationError, match="session_id"):
        cm._validate_run_inputs(service, session_id=over, n_clusters=None, min_cluster_size=None)


# ============================================================ _cap_candidates
def test_cap_candidates_returns_top_by_retrieval_count() -> None:
    """Line 218: cap exceeded → sort by retrieval_count desc, slice."""
    cfg = MemoryConfig(consolidate_max_candidates=2)
    candidates: list[tuple[int, str, dict[str, Any]]] = [
        (1, "a", {"retrieval_count": 0}),
        (2, "b", {"retrieval_count": 5}),
        (3, "c", {"retrieval_count": 3}),
        (4, "d", {"retrieval_count": 9}),
    ]
    out = cm._cap_candidates(cfg, candidates)
    ids = [c[0] for c in out]
    assert ids == [4, 2]


def test_cap_candidates_passthrough_when_under_cap() -> None:
    cfg = MemoryConfig(consolidate_max_candidates=10)
    candidates: list[tuple[int, str, dict[str, Any]]] = [
        (1, "a", {"retrieval_count": 0}),
        (2, "b", {"retrieval_count": 1}),
    ]
    assert cm._cap_candidates(cfg, candidates) is candidates


def test_cap_candidates_disabled_when_cap_zero() -> None:
    cfg = MemoryConfig(consolidate_max_candidates=0)
    candidates: list[tuple[int, str, dict[str, Any]]] = [
        (1, "a", {"retrieval_count": 0}),
        (2, "b", {"retrieval_count": 1}),
    ]
    assert cm._cap_candidates(cfg, candidates) is candidates


# ============================================================ _nearest_schema
class _StubSemantic:
    def __init__(self, hits: list[Any] | Exception, count_val: int = 1) -> None:
        self._hits = hits
        self._count = count_val

    async def count(self) -> int:
        return self._count

    async def similarity_search(self, q: Any, k: int) -> list[Any]:
        if isinstance(self._hits, Exception):
            raise self._hits
        return self._hits

    async def get_documents(self, query: Any, limit: int | None = None) -> list[Any]:
        return []

    async def update_metadata(self, items: list[Any]) -> None:
        return None

    async def add_texts(self, *args: Any, **kwargs: Any) -> list[int]:
        return []


class _StubService:
    def __init__(self, semantic: _StubSemantic, episodic: Any = None) -> None:
        self.semantic = semantic
        self.episodic = episodic
        self.config = MemoryConfig()

    async def query_arg(self, text: str) -> list[float]:
        return [0.0]

    async def vectorise(self, texts: list[str]) -> list[list[float]]:
        return [[0.0]]


class _Doc:
    def __init__(self, _id: int) -> None:
        self.metadata = {"id": _id}


async def test_nearest_schema_returns_none_on_search_failure() -> None:
    """Lines 112-113: similarity_search raises → return None."""
    svc = _StubService(_StubSemantic(hits=ValueError("synthetic")))
    assert await cm._nearest_schema(svc, "text") is None  # type: ignore[arg-type]


async def test_nearest_schema_returns_none_when_no_hits() -> None:
    """Line 115: empty hits → return None."""
    svc = _StubService(_StubSemantic(hits=[]))
    assert await cm._nearest_schema(svc, "text") is None  # type: ignore[arg-type]


async def test_nearest_schema_returns_none_when_id_metadata_missing() -> None:
    """Line 119: schema doc has no id metadata → return None."""

    class _BadDoc:
        def __init__(self) -> None:
            self.metadata: dict[str, Any] = {}  # no 'id'

    svc = _StubService(_StubSemantic(hits=[(_BadDoc(), 0.1)]))
    assert await cm._nearest_schema(svc, "text") is None  # type: ignore[arg-type]


async def test_nearest_schema_returns_tuple_when_hit_resolves() -> None:
    svc = _StubService(_StubSemantic(hits=[(_Doc(7), 0.1), (_Doc(8), 0.3)]))
    out = await cm._nearest_schema(svc, "text")  # type: ignore[arg-type]
    assert out is not None
    sch_id, d1, d2 = out
    assert sch_id == 7
    assert d1 == pytest.approx(0.1)
    assert d2 == pytest.approx(0.3)


# ============================================================ _absorb edge cases
class _EpisodicStub:
    """Captures update_metadata calls so the absorb path is observable."""

    def __init__(self) -> None:
        self.meta_updates: list[Any] = []

    async def update_metadata(self, items: list[Any]) -> None:
        self.meta_updates.extend(items)


class _AbsorbSemantic(_StubSemantic):
    def __init__(self, hits: list[Any], schema_docs: dict[int, tuple[str, dict[str, Any]]]) -> None:
        super().__init__(hits=hits, count_val=len(schema_docs))
        self._schema_docs = schema_docs
        self.confidence_updates: list[Any] = []

    async def get_documents(self, query: Any, limit: int | None = None) -> list[Any]:
        sch_id = query.get("id") if isinstance(query, dict) else None
        if sch_id in self._schema_docs:
            text, md = self._schema_docs[sch_id]
            return [(sch_id, text, md)]
        return []

    async def update_metadata(self, items: list[Any]) -> None:
        self.confidence_updates.extend(items)


async def test_absorb_skips_candidates_with_no_nearest_schema() -> None:
    """Line 146: nearest is None → continue."""
    semantic = _AbsorbSemantic(hits=[], schema_docs={})
    # semantic.count() must be > 0 so _absorb doesn't take the early return
    # at line 132. Lie about the count so the loop reaches line 146.
    semantic._count = 1
    svc = _StubService(semantic=semantic, episodic=_EpisodicStub())
    absorbed, formed = await cm._absorb(
        svc,  # type: ignore[arg-type]
        candidates=[(1, "text", {"salience": 0.1})],
        now=10.0,
    )
    assert absorbed == set()
    assert formed == []


async def test_absorb_skips_when_distance_exceeds_threshold() -> None:
    """Line 160: dist > absorb_dist → continue (no schema picked)."""
    # Inflate d1 so even d_near > absorb_dist.
    far = 2.0  # absorb_dist defaults below this
    semantic = _AbsorbSemantic(
        hits=[(_Doc(7), far), (_Doc(8), far)],
        schema_docs={7: ("schema body", {"source_episode_ids": []})},
    )
    svc = _StubService(semantic=semantic, episodic=_EpisodicStub())
    absorbed, formed = await cm._absorb(
        svc,  # type: ignore[arg-type]
        candidates=[(1, "text", {"salience": 0.5})],
        now=10.0,
    )
    assert absorbed == set()
    assert formed == []


async def test_absorb_skips_when_schema_doc_missing() -> None:
    """Line 168: schema doc deleted between scoring and absorption → skip."""
    semantic = _AbsorbSemantic(
        hits=[(_Doc(7), 0.0), (_Doc(8), 0.5)],
        schema_docs={},  # schema 7 not retrievable by get_documents
    )
    semantic._count = 1  # bypass the count==0 early return
    svc = _StubService(semantic=semantic, episodic=_EpisodicStub())
    absorbed, formed = await cm._absorb(
        svc,  # type: ignore[arg-type]
        candidates=[(1, "text", {"salience": 0.5})],
        now=10.0,
    )
    assert absorbed == set()
    assert formed == []


async def test_absorb_skips_when_no_new_sources_added() -> None:
    """Line 173: candidate already in source_episode_ids → no growth, skip."""
    semantic = _AbsorbSemantic(
        hits=[(_Doc(7), 0.0), (_Doc(8), 0.5)],
        # Schema already contains the candidate (1) as a source.
        schema_docs={7: ("schema body", {"source_episode_ids": [1]})},
    )
    semantic._count = 1  # bypass the count==0 early return
    svc = _StubService(semantic=semantic, episodic=_EpisodicStub())
    absorbed, formed = await cm._absorb(
        svc,  # type: ignore[arg-type]
        candidates=[(1, "text", {"salience": 0.5})],
        now=10.0,
    )
    assert absorbed == set()
    assert formed == []
    # And no metadata writes happened.
    assert semantic.confidence_updates == []
    assert svc.episodic.meta_updates == []


# ============================================================ run() ImportError fallback
async def test_run_returns_absorbed_formed_when_kmeans_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Lines 308-309: cluster() raises ValueError/ImportError → return
    whatever was already absorbed without crashing."""
    # Custom config to bypass min_age / min_retrievals gates so freshly
    # encoded episodes are eligible candidates.
    db = AsyncVectorDB(":memory:")
    try:
        cfg = MemoryConfig(consolidate_min_age_seconds=0.0, consolidate_min_retrievals=0)
        service = MemoryService(db, config=cfg, embed_fn=hash_embed)
        for i in range(3):
            await service.encode_episode(f"trace {i}", "s1", salience=0.5)

        calls = {"n": 0}

        async def boom_cluster(*args: Any, **kwargs: Any) -> Any:
            calls["n"] += 1
            raise ImportError("synthetic: scikit-learn not installed")

        # Direct attribute swap; AsyncVectorCollection allows reassignment
        # of bound methods at instance scope.
        service.episodic.cluster = boom_cluster
        out = await service.consolidate(session_id="s1")
        assert calls["n"] == 1, "cluster must be invoked before the except fires"
        assert out == []
    finally:
        await db.close()


async def test_run_returns_absorbed_formed_when_kmeans_raises_value_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Lines 308-309 (ValueError arm): malformed input to MiniBatchKMeans
    surfaces as ValueError; consolidate must swallow and return formed."""
    db = AsyncVectorDB(":memory:")
    try:
        cfg = MemoryConfig(consolidate_min_age_seconds=0.0, consolidate_min_retrievals=0)
        service = MemoryService(db, config=cfg, embed_fn=hash_embed)
        for i in range(3):
            await service.encode_episode(f"trace {i}", "s1", salience=0.5)

        async def bad_cluster(*args: Any, **kwargs: Any) -> Any:
            raise ValueError("synthetic kmeans rejected input")

        service.episodic.cluster = bad_cluster
        out = await service.consolidate(session_id="s1")
        assert out == []
    finally:
        await db.close()


# ============================================================ salience floor
def test_eligibility_gate_drops_low_salience_episodes() -> None:
    """Salience floor keeps low-salience noise from accumulating into
    durable clusters across re-runs. Episodes at or above the floor pass;
    below it are dropped. Missing-salience defaults to the neutral base
    (0.3) so legacy episodes are not prune-on-sight."""
    cfg = MemoryConfig(
        consolidate_min_age_seconds=0.0,
        consolidate_min_retrievals=0,
        consolidate_min_salience=0.3,
    )
    candidates: list[tuple[int, str, dict[str, Any]]] = [
        (1, "noise", {"salience": 0.1, "encoded_at": 0.0, "retrieval_count": 0}),
        (2, "neutral", {"salience": 0.3, "encoded_at": 0.0, "retrieval_count": 0}),
        (3, "salient", {"salience": 0.8, "encoded_at": 0.0, "retrieval_count": 0}),
        (4, "legacy_missing", {"encoded_at": 0.0, "retrieval_count": 0}),
    ]
    out = cm._apply_eligibility_gates(cfg, candidates)
    assert [ep_id for ep_id, _, _ in out] == [2, 3, 4]


def test_eligibility_gate_disabled_returns_all() -> None:
    """All three sub-gates at <=0 should short-circuit and return the
    full candidate list unchanged."""
    cfg = MemoryConfig(
        consolidate_min_age_seconds=0.0,
        consolidate_min_retrievals=0,
        consolidate_min_salience=0.0,
    )
    candidates: list[tuple[int, str, dict[str, Any]]] = [
        (1, "noise", {"salience": 0.0, "encoded_at": 0.0, "retrieval_count": 0}),
        (2, "salient", {"salience": 0.9, "encoded_at": 0.0, "retrieval_count": 0}),
    ]
    out = cm._apply_eligibility_gates(cfg, candidates)
    assert [ep_id for ep_id, _, _ in out] == [1, 2]
