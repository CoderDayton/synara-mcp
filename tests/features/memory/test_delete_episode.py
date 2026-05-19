"""MemoryService.delete_episode — forget-consistent single/group delete.

Uses the same deterministic hash embedder + :memory: fixture pattern as
test_service.py (conftest is intentionally empty; modules self-contain).
"""

from __future__ import annotations

import hashlib
from collections.abc import AsyncIterator

import numpy as np
import pytest
import pytest_asyncio
from simplevecdb import AsyncVectorDB

from synara.core.errors import ValidationError
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


async def test_delete_single_episode(service: MemoryService) -> None:
    r = await service.encode_episode("a singular short trace", "s1")
    ep_id = r["id"]
    before = (await service.stats())["episodic_count"]

    out = await service.delete_episode(ep_id)

    assert out["count"] >= 1
    assert ep_id in out["deleted_ids"]
    assert (await service.stats())["episodic_count"] == before - out["count"]
    assert await service.episodic.get_documents({"id": ep_id}) == []


async def test_delete_removes_whole_theta_group(service: MemoryService) -> None:
    long_text = (
        "First we set up the environment carefully. "
        "Then we ran the migration against staging. "
        "After that the rollout proceeded to production. "
        "Finally we verified the dashboards were green."
    )
    r = await service.encode_episode(long_text, "s1")
    group_id = r.get("group_id", r["id"])
    members = await service.fetch_episode_group(group_id)
    member_ids = sorted(int(m["id"]) for m in members) or [int(r["id"])]

    # Deleting via any member id removes every member of its group.
    out = await service.delete_episode(member_ids[-1])

    assert set(member_ids).issubset(set(out["deleted_ids"]))
    for mid in member_ids:
        assert await service.episodic.get_documents({"id": mid}) == []


async def test_delete_missing_raises(service: MemoryService) -> None:
    with pytest.raises(ValidationError, match="not found"):
        await service.delete_episode(999_999)


async def test_recall_still_works_after_delete(service: MemoryService) -> None:
    """Forget-consistent: dangling SR edges stay inert; recall unaffected."""
    keep = await service.encode_episode("keep this memory about otters", "s1")
    drop = await service.encode_episode("drop this memory about turbines", "s1")
    # Build SR edges between the two within the session window.
    await service.recall("memory", session_id="s1", k=5)

    await service.delete_episode(drop["id"])

    hits = await service.recall("otters", session_id="s1", k=5)
    ids = {h.get("id") for h in hits}
    assert keep["id"] in ids
    assert drop["id"] not in ids
