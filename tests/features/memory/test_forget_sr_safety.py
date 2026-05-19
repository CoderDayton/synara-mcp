"""forget(dry_run=False) must not orphan in-memory SR state.

``coll.edges`` has an ON DELETE CASCADE FK to documents. A live forget
that deletes episode docs without evicting them from the Successor
Representation leaves their ids in ``_sessions``/``_T_counts``; the next
recall (which flushes the SR) then upserts a FK-violating edge for a
now-deleted id and raises ``sqlite3.IntegrityError``.

This is the same invariant ``MemoryService.delete_episode`` relies on
(see ``test_delete_episode.test_recall_still_works_after_delete``);
this module exercises it through the ``forget`` path.

Uses the same deterministic hash embedder + :memory: fixture pattern as
test_service.py / test_delete_episode.py (conftest is intentionally
empty; modules self-contain).
"""

from __future__ import annotations

import hashlib
from collections.abc import AsyncIterator

import numpy as np
import pytest_asyncio
from simplevecdb import AsyncVectorDB

from synara.features.memory.service import MemoryConfig, MemoryService


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


async def test_recall_after_forget_does_not_violate_fk(service: MemoryService) -> None:
    """Live forget must evict pruned ids from in-memory SR before delete.

    Pre-fix this raised ``sqlite3.IntegrityError: FOREIGN KEY constraint
    failed`` at the post-forget recall's SR flush.
    """
    # Explicit salience is stored verbatim (auto-salience only fills an
    # omitted value): keep stays well above the forget floor, drop is
    # zeroed so power-law strength is 0 and it is always pruned.
    keep = await service.encode_episode("keep this memory about otters", "s1", salience=1.0)
    drop = await service.encode_episode("drop this memory about turbines", "s1", salience=0.0)

    # Build durable SR edges and seat both ids in the s1 session window.
    await service.recall("memory", session_id="s1", k=5)

    out = await service.forget(strength_floor=1.0, dry_run=False)
    assert drop["id"] in out["candidate_ids"]
    assert out["removed"] >= 1
    assert await service.episodic.get_documents({"id": drop["id"]}) == []

    # Pre-fix: drop is still in the s1 window, so observe_recall_set
    # re-records an edge for it and the flush upserts a FK-violating row.
    hits = await service.recall("otters", session_id="s1", k=5)
    ids = {h.get("id") for h in hits}
    assert keep["id"] in ids
    assert drop["id"] not in ids
