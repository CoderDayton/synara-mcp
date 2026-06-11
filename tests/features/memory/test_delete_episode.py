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


async def test_delete_semantic_removes_entry(service: MemoryService) -> None:
    r = await service.store_semantic_memory("prefer ruff over flake8", kind="preference")
    sem_id = r["id"]
    before = (await service.stats())["semantic_count"]

    out = await service.delete_semantic(sem_id)

    assert out == {"deleted_ids": [sem_id], "count": 1}
    assert (await service.stats())["semantic_count"] == before - 1
    assert await service.semantic.get_documents({"id": sem_id}) == []


async def test_delete_semantic_missing_raises(service: MemoryService) -> None:
    with pytest.raises(ValidationError, match="not found"):
        await service.delete_semantic(999_999)


async def test_delete_semantic_keeps_source_episodes(service: MemoryService) -> None:
    """Semantic delete must not cascade into the episodic store."""
    ep = await service.encode_episode("an otter trace to consolidate", "s1")
    sch = await service.store_semantic_memory("otters are aquatic", kind="fact")

    await service.delete_semantic(sch["id"])

    assert await service.semantic.get_documents({"id": sch["id"]}) == []
    assert await service.episodic.get_documents({"id": ep["id"]}) != []


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


# ----------------------------------------------- encode supersedes retire
async def test_encode_supersedes_retires_stale_episode(service: MemoryService) -> None:
    """A corrective store deletes the episode it replaces."""
    old = await service.encode_episode("we deploy with fabric scripts", "s1")
    new = await service.encode_episode(
        "we deploy with terraform now, fabric is retired",
        "s1",
        supersedes=old["id"],
    )

    assert new["superseded"] == old["id"]
    assert await service.episodic.get_documents({"id": old["id"]}) == []
    assert await service.episodic.get_documents({"id": new["id"]}) != []


async def test_encode_supersedes_unknown_id_raises(service: MemoryService) -> None:
    with pytest.raises(ValidationError, match="not found"):
        await service.encode_episode("orphan correction", "s1", supersedes=999_999)


async def test_encode_supersedes_negative_id_raises(service: MemoryService) -> None:
    with pytest.raises(ValidationError, match="non-negative"):
        await service.encode_episode("bad target", "s1", supersedes=-3)


async def test_encode_supersedes_skips_retire_on_self_dedup(service: MemoryService) -> None:
    """Content that dedups onto the superseded episode keeps it: deleting
    the dedup target would destroy the only remaining copy of the trace."""
    old = await service.encode_episode("the staging database password rotates monthly", "s1")
    again = await service.encode_episode(
        "the staging database password rotates monthly",
        "s1",
        supersedes=old["id"],
    )

    assert again["deduped"] is True
    assert again["id"] == old["id"]
    assert again["superseded"] is None
    assert await service.episodic.get_documents({"id": old["id"]}) != []


async def test_encode_supersedes_retires_whole_segment_group(service: MemoryService) -> None:
    """Superseding any member of a theta-segmented episode retires the group."""
    long_text = (
        "First we set up the environment carefully. "
        "Then we ran the migration against staging. "
        "After that the rollout proceeded to production. "
        "Finally we verified the dashboards were green."
    )
    old = await service.encode_episode(long_text, "s1")
    group_id = old.get("group_id", old["id"])
    member_ids = sorted(int(m["id"]) for m in await service.fetch_episode_group(group_id)) or [
        int(old["id"])
    ]

    new = await service.encode_episode(
        "rollouts are fully automated now", "s1", supersedes=member_ids[-1]
    )

    assert new["superseded"] == member_ids[-1]
    for mid in member_ids:
        assert await service.episodic.get_documents({"id": mid}) == []
    assert await service.episodic.get_documents({"id": new["id"]}) != []


# ----------------------------------------------------- dry-run preview
async def test_delete_episode_dry_run_previews_without_deleting(
    service: MemoryService,
) -> None:
    r = await service.encode_episode("a trace we might remove", "s1")
    out = await service.delete_episode(r["id"], dry_run=True)

    assert out["dry_run"] is True
    assert r["id"] in out["candidate_ids"]
    assert "deleted_ids" not in out
    assert await service.episodic.get_documents({"id": r["id"]}) != []


async def test_delete_episode_dry_run_lists_whole_group(service: MemoryService) -> None:
    long_text = (
        "First we set up the environment carefully. "
        "Then we ran the migration against staging. "
        "After that the rollout proceeded to production. "
        "Finally we verified the dashboards were green."
    )
    r = await service.encode_episode(long_text, "s1")
    group_id = r.get("group_id", r["id"])
    member_ids = sorted(int(m["id"]) for m in await service.fetch_episode_group(group_id)) or [
        int(r["id"])
    ]

    out = await service.delete_episode(member_ids[-1], dry_run=True)

    assert out["dry_run"] is True
    assert set(member_ids).issubset(set(out["candidate_ids"]))
    for mid in member_ids:
        assert await service.episodic.get_documents({"id": mid}) != []
