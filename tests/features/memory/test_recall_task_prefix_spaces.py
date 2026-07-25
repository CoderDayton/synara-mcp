"""Query-space vs document-space arithmetic under asymmetric task prefixes.

A model trained with task prefixes puts the same sentence in two
different places depending on whether it was encoded as a query or as a
document. Search is unaffected — that split is what it is *for* — but
two code paths add a query vector to stored vectors instead of merely
comparing them, and those have to bring both onto one scale first:
CA3's recombination step and the reconsolidation cue.
"""

from __future__ import annotations

import hashlib
from collections.abc import AsyncIterator, Sequence
from typing import Any

import numpy as np
import pytest
import pytest_asyncio
from simplevecdb import AsyncVectorDB

from synara.features.memory.config import MemoryConfig
from synara.features.memory.hippocampus import complete as complete_mod
from synara.features.memory.hippocampus import recall as recall_mod
from synara.features.memory.service import MemoryService

_DIM = 16
# The synthetic prefix offset: document encodings are the query encoding
# pushed along this axis. Big enough to dominate any cosine assertion, so
# a test that passes cannot be passing by accident.
_OFFSET_AXIS = _DIM - 1


def _unit(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    return vector / norm if norm > 0 else vector


def _query_vector(text: str) -> np.ndarray:
    seed = int.from_bytes(hashlib.sha256(text.encode()).digest()[:8], "big")
    rng = np.random.default_rng(seed)
    vector = rng.standard_normal(_DIM)
    # Keep the offset axis clear so "did this vector move into document
    # space?" has an unambiguous answer.
    vector[_OFFSET_AXIS] = 0.0
    return _unit(vector)


def _document_vector(text: str) -> np.ndarray:
    shifted = _query_vector(text).copy()
    shifted[_OFFSET_AXIS] = 1.0
    return _unit(shifted)


async def _embed_query(text: str) -> list[float]:
    return [float(x) for x in _query_vector(text)]


async def _embed_documents(texts: Sequence[str]) -> list[list[float]]:
    return [[float(x) for x in _document_vector(t)] for t in texts]


async def _seed(service: MemoryService, texts: Sequence[str], **metadata: Any) -> list[int]:
    """Store ``texts`` as documents, with the ``id`` metadata recall keys on.

    simplevecdb does not put the row id in the metadata blob; synara's
    encode path writes it, and every read path expects it there.
    """
    ids = await service.episodic.add_texts(
        list(texts),
        [dict(metadata) for _ in texts],
        [[float(x) for x in _document_vector(t)] for t in texts],
    )
    ids = [int(i) for i in ids]
    await service.episodic.update_metadata([(i, {"id": i}) for i in ids])
    return ids


@pytest_asyncio.fixture
async def asymmetric() -> AsyncIterator[MemoryService]:
    """A service whose two encoders disagree, exactly as nomic's do."""
    db = AsyncVectorDB(":memory:")
    try:
        yield MemoryService(
            db,
            config=MemoryConfig(),
            embed_fn=_embed_query,
            embed_batch_fn=_embed_documents,
            embed_asymmetric=True,
        )
    finally:
        await db.close()


@pytest_asyncio.fixture
async def symmetric() -> AsyncIterator[MemoryService]:
    db = AsyncVectorDB(":memory:")
    try:
        yield MemoryService(db, config=MemoryConfig(), embed_fn=_embed_query)
    finally:
        await db.close()


# =========================================================== _document_space
async def test_document_space_returns_the_document_encoding(
    asymmetric: MemoryService,
) -> None:
    q = await asymmetric.query_arg("a cue")
    assert isinstance(q, list)
    resolved = await recall_mod._document_space(asymmetric, "a cue", q)
    assert resolved is not None
    assert np.allclose(resolved, _document_vector("a cue"))
    # ... and that really is somewhere else in the space.
    assert 1.0 - float(np.dot(resolved, q)) > 0.1


async def test_document_space_is_a_no_op_for_a_symmetric_embedder(
    symmetric: MemoryService,
) -> None:
    """No second encode is paid where the distinction does not exist."""
    q = await symmetric.query_arg("a cue")
    calls = 0

    async def _counting(texts: Sequence[str]) -> list[list[float]]:
        nonlocal calls
        calls += 1
        return await _embed_documents(texts)

    symmetric._embed_batch = _counting
    assert await recall_mod._document_space(symmetric, "a cue", q) is q
    assert calls == 0


async def test_document_space_without_a_query_vector_is_none(
    asymmetric: MemoryService,
) -> None:
    """No embedder wired: simplevecdb embeds server-side and there is no
    vector on our side to correct."""
    assert await recall_mod._document_space(asymmetric, "a cue", "a cue") is None


async def test_document_space_keeps_the_query_on_a_width_disagreement(
    asymmetric: MemoryService,
) -> None:
    async def _wrong_width(texts: Sequence[str]) -> list[list[float]]:
        return [[1.0, 0.0, 0.0] for _ in texts]

    asymmetric._embed_batch = _wrong_width
    q = await asymmetric.query_arg("a cue")
    # The width guard in ``vectorise`` raises; recall must survive a
    # broken document encoder rather than propagate it to the caller.
    assert await recall_mod._document_space(asymmetric, "a cue", q) is q


async def test_document_space_survives_a_silent_width_disagreement(
    asymmetric: MemoryService,
) -> None:
    async def _wrong_width(texts: Sequence[str]) -> list[list[float]]:
        return [[1.0, 0.0, 0.0] for _ in texts]

    asymmetric._embed_batch = _wrong_width
    q = [0.0] * (_DIM - 1) + [1.0]
    # Nothing cached for the dimension guard to compare against, so the
    # mismatch reaches ``_document_space`` itself.
    asymmetric._embedding_dimension = None
    assert await recall_mod._document_space(asymmetric, "a cue", q) is q


# ============================================================== space_offset
def test_space_offset_recovers_the_prefix_displacement() -> None:
    query = _query_vector("a cue")
    offset = complete_mod.space_offset(query, [float(x) for x in _document_vector("a cue")])
    assert offset is not None
    assert np.allclose(query + offset, _document_vector("a cue"))


@pytest.mark.parametrize(
    "q_doc",
    [None, [1.0, 0.0, 0.0], [float("nan")] * _DIM],
    ids=["absent", "wrong-width", "non-finite"],
)
def test_space_offset_abstains_on_unusable_input(q_doc: Any) -> None:
    assert complete_mod.space_offset(_query_vector("a cue"), q_doc) is None


def test_space_offset_is_zero_for_a_symmetric_model() -> None:
    query = _query_vector("a cue")
    offset = complete_mod.space_offset(query, [float(x) for x in query])
    assert offset is not None
    assert np.allclose(offset, 0.0)


# ============================================================ attractor_step
def _stored_patterns() -> np.ndarray:
    return np.stack([_document_vector(f"doc {i}") for i in range(6)])


def test_attractor_step_without_an_offset_is_unchanged() -> None:
    """The correction must be opt-in: a symmetric embedder passes no
    offset and has to get byte-identical behaviour."""
    q = _query_vector("a cue")
    X = _stored_patterns()
    plain, plain_score = complete_mod.attractor_step(q, X, beta=8.0, q0=q, eta0=0.6)
    zeroed, zeroed_score = complete_mod.attractor_step(
        q, X, beta=8.0, q0=q, eta0=0.6, offset=np.zeros(_DIM)
    )
    assert np.allclose(plain, zeroed)
    assert plain_score == pytest.approx(zeroed_score)


def test_attractor_step_keeps_the_iterate_in_query_space() -> None:
    """Uncorrected, the recombination drags the iterate onto the document
    manifold; corrected, it stays on the query one."""
    q = _query_vector("a cue")
    X = _stored_patterns()
    offset = complete_mod.space_offset(q, [float(x) for x in _document_vector("a cue")])
    assert offset is not None

    drifted, _ = complete_mod.attractor_step(q, X, beta=8.0, q0=q, eta0=0.6)
    corrected, _ = complete_mod.attractor_step(q, X, beta=8.0, q0=q, eta0=0.6, offset=offset)

    # The offset axis is zero for every query encoding and positive for
    # every document one, so it reads out which space a vector is in.
    assert drifted[_OFFSET_AXIS] > 0.1
    assert corrected[_OFFSET_AXIS] == pytest.approx(0.0, abs=1e-6)


async def test_completion_run_threads_the_document_encoding(
    asymmetric: MemoryService,
) -> None:
    await _seed(asymmetric, [f"doc {i}" for i in range(6)])
    q = [float(x) for x in _query_vector("a cue")]
    doc = [float(x) for x in _document_vector("a cue")]

    drifted = await complete_mod.run(asymmetric, q, k_inner=6, iters=3, beta=8.0, eta0=0.6)
    corrected = await complete_mod.run(
        asymmetric, q, k_inner=6, iters=3, beta=8.0, eta0=0.6, q0_document=doc
    )
    assert drifted.query[_OFFSET_AXIS] > 0.1
    assert corrected.query[_OFFSET_AXIS] == pytest.approx(0.0, abs=1e-6)


# ======================================================= reconsolidation cue
async def test_reconsolidation_cue_is_a_document_vector(
    asymmetric: MemoryService,
) -> None:
    """The cue is blended *into* a stored vector, so a query-space cue
    would drag every recalled episode off the document manifold, once per
    recall, up to the drift cap."""
    asymmetric.config = MemoryConfig(
        reconsolidation_alpha=0.5,
        reconsolidation_min_score=0.0,
        reconsolidation_max_total_drift=10.0,
    )
    [doc_id] = await _seed(asymmetric, ["a cue"], session_id="s1", created_at=0.0)
    captured: list[list[float]] = []

    async def _capture(service: Any, row: Any, *, t: float, cue: Any = None) -> None:
        captured.append(list(cue))

    original = recall_mod._accrue_drift
    recall_mod._accrue_drift = _capture
    try:
        await recall_mod.run(asymmetric, query="a cue", session_id="s1", k=4, reinforce=True)
    finally:
        recall_mod._accrue_drift = original

    assert captured, "the drift path did not run"
    assert np.allclose(captured[0], _document_vector("a cue"))
    assert int(doc_id) >= 0
