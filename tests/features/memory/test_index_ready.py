"""Lazy HNSW reconciliation: recall fixes a desynced index on first call.

The simplevecdb store keeps the HNSW graph in a usearch file next to
the SQLite catalog. The two are not atomically synchronized — if the
``.usearch`` file is lost (crash, manual cleanup, encode that buffered
through ``pending`` and was never flushed), the catalog still reports
rows but ``similarity_search`` returns nothing. This was the failure
mode the live dashboard surfaced: 15 stored episodes, but
``coll.dim is None`` and recall came back empty.

``MemoryService._ensure_index_ready`` is the recovery hook: it flushes
any pending adds and rebuilds the HNSW exactly once per process when
the catalog disagrees with the index. These tests cover the happy
path, idempotency, the empty-store no-op, and failure-then-retry.
"""

from __future__ import annotations

import hashlib
import os
import tempfile

import numpy as np
import pytest
from simplevecdb import AsyncVectorDB

from synara.features.memory import MemoryService
from synara.features.memory.service import MemoryConfig


def _hash_embed(text: str, dim: int = 32) -> list[float]:
    seed = int.from_bytes(hashlib.sha256(text.encode("utf-8")).digest()[:8], "big", signed=False)
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(dim).astype(np.float32)
    n = float(np.linalg.norm(v))
    out = (v / n) if n > 0 else v
    return [float(x) for x in out.tolist()]


async def _service_with_desynced_index() -> tuple[AsyncVectorDB, MemoryService, str]:
    """Build a file-backed service whose HNSW index file has been wiped.

    Mirrors the live-dashboard state: SQLite catalog has documents and
    stored embeddings, but the usearch index file is missing — so
    ``similarity_search`` returns nothing until ``rebuild_index`` runs.
    """
    td = tempfile.mkdtemp()
    db_path = os.path.join(td, "store.db")
    db = AsyncVectorDB(db_path)
    service = MemoryService(db, config=MemoryConfig(), embed_fn=_hash_embed)
    await service.encode_episode("cache layer needs work", "s1")
    await service.encode_episode("orbital mechanics overview", "s1")
    await db.close()

    # Wipe the per-collection usearch files. SQLite catalog survives.
    for name in os.listdir(td):
        if name.endswith(".usearch"):
            os.remove(os.path.join(td, name))

    db2 = AsyncVectorDB(db_path)
    svc2 = MemoryService(db2, config=MemoryConfig(), embed_fn=_hash_embed)
    return db2, svc2, td


async def test_recall_fires_index_ready_and_returns_hits() -> None:
    db, service, _ = await _service_with_desynced_index()
    try:
        # Catalog reports 2 episodes but the index is empty —
        # similarity_search would silently return [].
        assert await service.episodic.count() == 2
        assert service.episodic.dim is None
        assert service._index_ready is False

        hits = await service.recall("cache", k=5)
        assert len(hits) == 2, hits
        assert service._index_ready is True
        assert service.episodic.dim is not None
    finally:
        await db.close()


async def test_index_ready_is_idempotent_after_first_call() -> None:
    """Once flipped, the guard short-circuits — rebuild_index must not
    fire again on subsequent recalls."""
    db, service, _ = await _service_with_desynced_index()
    try:
        await service.recall("cache", k=1)
        called = {"n": 0}
        original = service.episodic.rebuild_index

        async def tripwire(*a: object, **kw: object) -> None:
            called["n"] += 1
            await original(*a, **kw)

        service.episodic.rebuild_index = tripwire
        await service.recall("cache", k=1)
        assert called["n"] == 0
    finally:
        await db.close()


async def test_index_ready_skips_rebuild_for_empty_collections() -> None:
    """No catalog rows → no flush, no rebuild — guard still flips so
    later recalls don't pay the cost."""
    db = AsyncVectorDB(":memory:")
    service = MemoryService(db, config=MemoryConfig(), embed_fn=_hash_embed)
    try:
        rebuilt = {"n": 0}
        original = service.episodic.rebuild_index

        async def tripwire(*a: object, **kw: object) -> None:
            rebuilt["n"] += 1
            await original(*a, **kw)

        service.episodic.rebuild_index = tripwire
        service.semantic.rebuild_index = tripwire
        await service._ensure_index_ready()
        assert rebuilt["n"] == 0
        assert service._index_ready is True
    finally:
        await db.close()


async def test_index_ready_resets_on_failure_for_retry() -> None:
    """A transient rebuild failure must not permanently lock the guard;
    the flag flips back so the next call retries."""
    db, service, _ = await _service_with_desynced_index()
    try:
        original = service.episodic.rebuild_index

        async def boom(*a: object, **kw: object) -> None:
            raise RuntimeError("simulated rebuild failure")

        service.episodic.rebuild_index = boom
        with pytest.raises(RuntimeError):
            await service._ensure_index_ready()
        assert service._index_ready is False

        # Restore and retry — guard should run cleanly the second time.
        service.episodic.rebuild_index = original
        await service._ensure_index_ready()
        assert service._index_ready is True
        assert service.episodic.dim is not None
    finally:
        await db.close()


async def test_index_ready_no_op_when_collections_are_already_indexed() -> None:
    """Fresh in-memory store with eager indexing: guard runs, sees a
    healthy index, and never invokes ``rebuild_index``."""
    db = AsyncVectorDB(":memory:")
    service = MemoryService(db, config=MemoryConfig(), embed_fn=_hash_embed)
    try:
        await service.encode_episode("alpha", "s1")
        # add_texts indexed immediately, so dim is already populated.
        assert service.episodic.dim is not None

        rebuilt = {"n": 0}
        original = service.episodic.rebuild_index

        async def tripwire(*a: object, **kw: object) -> None:
            rebuilt["n"] += 1
            await original(*a, **kw)

        service.episodic.rebuild_index = tripwire
        await service.recall("alpha", k=1)
        assert rebuilt["n"] == 0
    finally:
        await db.close()


async def _service_with_partial_index() -> tuple[AsyncVectorDB, MemoryService, str]:
    """Build a file-backed service whose index lost a *later* add.

    Mirrors an unclean exit: the SQLite catalog holds two episodes (both
    with stored embeddings) but the usearch index was only ever saved
    with the first. ``dim`` stays set — so the old ``dim is None`` guard
    skipped the rebuild and recall silently under-served, the index
    (size 1) trailing the catalog (count 2).
    """
    td = tempfile.mkdtemp()
    db_path = os.path.join(td, "store.db")
    db = AsyncVectorDB(db_path)
    service = MemoryService(db, config=MemoryConfig(), embed_fn=_hash_embed)
    # First episode is indexed and saved into the .usearch file on close.
    await service.encode_episode("cache layer needs work", "s1")
    await db.close()

    db2 = AsyncVectorDB(db_path)
    svc2 = MemoryService(db2, config=MemoryConfig(), embed_fn=_hash_embed)
    # Inject a second episode into the SQLite catalog *only* (with a stored
    # embedding), leaving the index at size 1 while dim stays set — the
    # partial desync the rebuild guard must heal.
    sync = svc2.episodic._collection
    sync._catalog.add_documents(
        ["orbital mechanics overview"],
        [{}],
        None,
        embeddings=[_hash_embed("orbital mechanics overview")],
    )
    return db2, svc2, td


async def test_index_ready_rebuilds_partial_index() -> None:
    """Regression: a partially-populated index (dim set, size < count) must
    be rebuilt. The old ``dim is None`` guard only caught a fully empty
    index and left the catalog-only rows permanently unsearchable."""
    db, service, _ = await _service_with_partial_index()
    try:
        assert await service.episodic.count() == 2
        assert service.episodic.dim is not None
        assert service.episodic._collection._index.size == 1
        assert service._index_ready is False

        await service._ensure_index_ready()

        assert service._index_ready is True
        assert service.episodic._collection._index.size == 2
    finally:
        await db.close()
