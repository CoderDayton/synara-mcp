"""Regression tests for the scope / GC-liveness / segmentation fix pack.

One test per fixed behaviour:

* reflect's schema leg honours the scope axis (no cross-session leak),
* hybrid recall keeps returned schemas alive for the cold-schema GC,
* theta-segmented episodes reassemble byte-for-byte via ``get_episode``,
* re-storing identical long content dedups onto the existing group,
* the dream idle gate measures the gap between events, not time since
  the last dream,
* ``recall_semantic_memory`` enforces the session_id length cap,
* forget's scan window rotates across non-dry-run passes.
"""

from __future__ import annotations

import hashlib
from collections.abc import AsyncIterator

import numpy as np
import pytest
import pytest_asyncio
from simplevecdb import AsyncVectorDB

from synara.core.errors import ValidationError
from synara.features.memory.basal_ganglia.events import ReactorState, TriggerPolicy
from synara.features.memory.config import MemoryConfig
from synara.features.memory.hippocampus.segment import split_into_segments
from synara.features.memory.service import MemoryService


def hash_embed(text: str, dim: int = 32) -> list[float]:
    seed = int.from_bytes(hashlib.sha256(text.encode("utf-8")).digest()[:8], "big", signed=False)
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(dim).astype(np.float32)
    n = float(np.linalg.norm(v))
    out = (v / n) if n > 0 else v
    return [float(x) for x in out.tolist()]


@pytest_asyncio.fixture
async def db() -> AsyncIterator[AsyncVectorDB]:
    db = AsyncVectorDB(":memory:")
    try:
        yield db
    finally:
        await db.close()


# --------------------------------------------------- reflect scope axis
async def test_reflect_schema_leg_honours_scope_axis(db: AsyncVectorDB) -> None:
    """Reflect in s1 returns s1's and global schemas, never s2's."""
    svc = MemoryService(db, config=MemoryConfig(), embed_fn=hash_embed)
    await svc.store_semantic_memory("alpha rule for session one", kind="fact", session_id="s1")
    await svc.store_semantic_memory("alpha rule for session two", kind="fact", session_id="s2")
    await svc.store_semantic_memory("alpha rule for everyone", kind="fact", scope="global")
    await svc.encode_episode("alpha work happening in session one", "s1", tags=["alpha"])

    out = await svc.reflect(session_id="s1", query="alpha rule")
    summaries = [s["summary"] for s in out["schemas"]]
    assert "alpha rule for session two" not in summaries
    assert "alpha rule for session one" in summaries
    assert "alpha rule for everyone" in summaries


# ------------------------------------- cold-schema GC liveness (hybrid)
async def test_hybrid_recall_keeps_returned_schema_alive(db: AsyncVectorDB) -> None:
    """A schema surfaced by recall_episodes must refresh ``last_accessed``
    when the cold-schema GC is on; a read-only recall must not."""
    cfg = MemoryConfig(forget_schema_unused_seconds=3600.0, recall_relevance_gate=False)
    svc = MemoryService(db, config=cfg, embed_fn=hash_embed)
    sem = await svc.store_semantic_memory("the sky is blue", kind="fact", scope="global")
    sem_id = int(sem["id"])

    async def _age_stamp() -> None:
        await svc.semantic.update_metadata([(sem_id, {"last_accessed": 1.0})])

    async def _stamp() -> float:
        rows = await svc.semantic.get_documents({"id": sem_id})
        return float(rows[0][2]["last_accessed"])

    await _age_stamp()
    out = await svc.recall(query="the sky is blue", k=8, mode="hybrid")
    assert any(r["source"] == "semantic" for r in out)
    assert await _stamp() > 1.0  # bumped: hybrid hit counts as use

    await _age_stamp()
    await svc.recall_readonly(query="the sky is blue", k=8, mode="hybrid")
    assert await _stamp() == 1.0  # read-only recall stays write-free


async def test_hybrid_recall_stays_write_free_when_gc_off(db: AsyncVectorDB) -> None:
    cfg = MemoryConfig(recall_relevance_gate=False)  # forget_schema_unused_seconds=0
    svc = MemoryService(db, config=cfg, embed_fn=hash_embed)
    sem = await svc.store_semantic_memory("the sky is blue", kind="fact", scope="global")
    sem_id = int(sem["id"])
    await svc.semantic.update_metadata([(sem_id, {"last_accessed": 1.0})])
    await svc.recall(query="the sky is blue", k=8, mode="hybrid")
    rows = await svc.semantic.get_documents({"id": sem_id})
    assert float(rows[0][2]["last_accessed"]) == 1.0


# ------------------------------------------- lossless theta segmentation
def test_split_into_segments_is_lossless() -> None:
    content = "\n\n".join(
        f"Paragraph {i} states one plain fact. It adds detail." for i in range(30)
    )
    segs = split_into_segments(content, max_chars=120, max_items=7)
    assert len(segs) > 1
    assert "".join(segs) == content


async def test_get_episode_reassembles_segmented_content_exactly(db: AsyncVectorDB) -> None:
    svc = MemoryService(db, config=MemoryConfig(), embed_fn=hash_embed)
    content = " ".join(f"Sentence number {i} carries unique payload {i}." for i in range(40))
    assert len(content) > 1024  # forces theta segmentation at defaults
    r = await svc.encode_episode(content, "s1")
    assert r.get("group_id") is not None
    full = await svc.get_episode(int(r["id"]))
    assert full["content"] == content


# ----------------------------------------- dedup identity for segments
async def test_reencoding_identical_long_content_dedups_onto_group(db: AsyncVectorDB) -> None:
    svc = MemoryService(db, config=MemoryConfig(), embed_fn=hash_embed)
    content = " ".join(f"Alpha topic sentence {i} about one shared subject." for i in range(40))
    first = await svc.encode_episode(content, "s1")
    assert first["deduped"] is False
    group_id = int(first["group_id"])
    stored = await svc.episodic.count()

    second = await svc.encode_episode(content, "s1")
    assert second["deduped"] is True
    assert int(second["id"]) == group_id  # matched the group head
    assert await svc.episodic.count() == stored  # no duplicate group


# --------------------------------------------------- dream idle gate
def test_dream_idle_gate_measures_gap_between_events() -> None:
    policy = TriggerPolicy(dream_after_events=0, dream_after_idle_seconds=600.0)
    # No prior event on record: a fresh store never dreams on its very
    # first interaction, however large ``now`` is.
    st = ReactorState(events_since_dream=1, prev_event_at=0.0)
    assert policy.dream_due(st, 10_000.0) is False
    # Continuous activity (10 s between events) must not dream, even
    # long after the last dream — the old ``now - last_dream_at`` form
    # fired here.
    st = ReactorState(events_since_dream=50, prev_event_at=9_990.0, last_dream_at=0.0)
    assert policy.dream_due(st, 10_000.0) is False
    # The first event after a >= idle-window pause dreams.
    st = ReactorState(events_since_dream=50, prev_event_at=9_000.0, last_dream_at=8_999.0)
    assert policy.dream_due(st, 10_000.0) is True


# ------------------------------- semantic recall input-cap parity
async def test_recall_semantic_memory_validates_session_id_length(db: AsyncVectorDB) -> None:
    svc = MemoryService(db, config=MemoryConfig(max_session_id_chars=8), embed_fn=hash_embed)
    with pytest.raises(ValidationError, match="max_session_id_chars"):
        await svc.recall_semantic_memory("some query", session_id="x" * 9)


# --------------------------------------------- forget scan rotation
async def test_forget_scan_window_rotates_across_passes(db: AsyncVectorDB) -> None:
    svc = MemoryService(db, config=MemoryConfig(), embed_fn=hash_embed)
    for i in range(5):
        await svc.encode_episode(f"distinct trace number {i} with its own payload", "s1")

    # Dry-run passes preview without advancing, so a preview and the
    # delete that follows it see the same window.
    r1 = await svc.forget(dry_run=True, max_scan=2)
    assert r1["scan_offset"] == 0
    assert r1["scanned"] == 2
    r2 = await svc.forget(dry_run=True, max_scan=2)
    assert r2["scan_offset"] == 0

    # Non-dry-run passes sweep forward and wrap on a partial page.
    r3 = await svc.forget(dry_run=False, max_scan=2)
    assert r3["scan_offset"] == 0
    r4 = await svc.forget(dry_run=False, max_scan=2)
    assert r4["scan_offset"] == 2
    r5 = await svc.forget(dry_run=False, max_scan=2)
    assert r5["scan_offset"] == 4
    assert r5["scanned"] == 1  # partial page -> wrap
    r6 = await svc.forget(dry_run=False, max_scan=2)
    assert r6["scan_offset"] == 0
