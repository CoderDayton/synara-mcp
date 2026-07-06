"""Regression tests for the scope / GC-liveness / segmentation fix pack
and the production-hardening pass that followed it (tag-validation
parity, scope_session strictness, recency on semantic hits, stats
observability, bounded LTD sweep, bounded shutdown drain).

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

import asyncio
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


# ===================== production-hardening pass (second batch) =====


# ------------------------------------------ tag-validation parity
async def test_tag_validation_is_uniform_across_entry_points(db: AsyncVectorDB) -> None:
    """encode, semantic store, and recall share one tag validator:
    non-string tags are a clean ValidationError (never a TypeError),
    and the size caps apply to recall's filter input too."""
    cfg = MemoryConfig(max_tags=2, max_tag_chars=5)
    svc = MemoryService(db, config=cfg, embed_fn=hash_embed)
    with pytest.raises(ValidationError, match="tags must be strings"):
        await svc.encode_episode("a note", "s1", tags=["ok", 123])  # type: ignore[list-item]
    with pytest.raises(ValidationError, match="tags must be strings"):
        await svc.store_semantic_memory("a fact", session_id="s1", tags=[None])  # type: ignore[list-item]
    with pytest.raises(ValidationError, match="too many tags"):
        await svc.recall(query="a note", session_id="s1", tags=["a", "b", "c"])
    with pytest.raises(ValidationError, match="max_tag_chars"):
        await svc.recall(query="a note", session_id="s1", tags=["toolong"])


# ------------------------------------- scope_session strictness
async def test_scope_session_true_without_session_id_rejected(db: AsyncVectorDB) -> None:
    """An explicit scope_session=true with nothing to scope to used to be
    a warn-and-proceed no-op; it must now be rejected on both recall
    paths so the caller cannot silently get cross-session results."""
    svc = MemoryService(db, config=MemoryConfig(), embed_fn=hash_embed)
    await svc.encode_episode("some trace", "s1")
    with pytest.raises(ValidationError, match="scope_session"):
        await svc.recall(query="some trace", scope_session=True)
    with pytest.raises(ValidationError, match="scope_session"):
        await svc.recall_semantic_memory("some trace", scope_session=True)


# --------------------------------------------- stats observability
async def test_memory_stats_exposes_reactor_and_sr_state(db: AsyncVectorDB) -> None:
    svc = MemoryService(db, config=MemoryConfig(), embed_fn=hash_embed)
    await svc.encode_episode("one novel trace", "s1")
    stats = await svc.stats()
    for key in (
        "sr_edges",
        "novel_encodes_since_consolidate",
        "events_since_dream",
        "last_consolidate_at",
        "last_dream_at",
        "reactor_tasks_inflight",
    ):
        assert key in stats
    assert stats["novel_encodes_since_consolidate"] >= 1
    assert stats["events_since_dream"] >= 1
    assert stats["reactor_tasks_inflight"] == 0


# --------------------------------------- recency on semantic hits
async def test_semantic_hits_carry_recency_fields(db: AsyncVectorDB) -> None:
    cfg = MemoryConfig(recall_relevance_gate=False)
    svc = MemoryService(db, config=cfg, embed_fn=hash_embed)
    await svc.store_semantic_memory("the sky is blue", kind="fact", scope="global")
    hybrid = await svc.recall(query="the sky is blue", k=8, mode="hybrid")
    sem = next(r for r in hybrid if r["source"] == "semantic")
    assert sem["created_at"] is not None
    assert sem["age_days"] is not None
    assert sem["age_days"] >= 0.0
    direct = await svc.recall_semantic_memory("the sky is blue", k=8)
    assert direct
    assert direct[0]["created_at"] is not None
    assert direct[0]["updated_age_days"] is not None


# --------------------------------------------- bounded LTD sweep
async def test_ltd_pass_bounds_per_pass_work(db: AsyncVectorDB) -> None:
    """``max_scan`` caps the per-pass read-modify-write work; the rest of
    the table is reached by later passes instead of one unbounded sweep."""
    svc = MemoryService(db, config=MemoryConfig(), embed_fn=hash_embed)
    for i in range(4):
        await svc.encode_episode(f"edge endpoint trace {i}", "s1")
    rows = await svc.episodic.get_documents({"session_id": "s1"})
    ids = [int(r[0]) for r in rows[:4]]
    # Three bonus-only edges whose transient potentiation is far below
    # the prune floor once decayed — every one is prunable.
    for j in ids[1:4]:
        await svc._plasticity.reinforce(ids[0], j, score=0.001, now=0.0)
    assert await svc._plasticity.ltd_pass(now=1e7, max_scan=1) == 1
    assert await svc._plasticity.ltd_pass(now=1e7) == 2


# --------------------------------------------- bounded shutdown drain
async def test_drain_reactor_tasks_timeout_cancels_stragglers(db: AsyncVectorDB) -> None:
    svc = MemoryService(db, config=MemoryConfig(), embed_fn=hash_embed)
    svc._spawn_reactor_task(asyncio.sleep(30), name="test-hang")
    assert len(svc._reactor_tasks) == 1
    loop = asyncio.get_running_loop()
    t0 = loop.time()
    await svc.drain_reactor_tasks(timeout=0.1)
    assert loop.time() - t0 < 5.0  # did not wait out the 30 s sleep
    assert not svc._reactor_tasks  # cancelled task removed itself


# ------------------------------------- segmentation fuzz invariant
@pytest.mark.parametrize("seed", range(8))
def test_split_into_segments_lossless_fuzz(seed: int) -> None:
    """Losslessness must hold for arbitrary shapes: prose, code, unicode,
    odd whitespace, and boundary-free blobs — not just tidy sentences."""
    rng = np.random.default_rng(seed)
    vocab = (
        "Alpha beta gamma.",
        " x = compute(1); y = 2\n",
        "Ünïcode büffer, ér? Gó! ",
        "```\ndef f():\n    return 1\n```\n",
        "   \t ",
        "noboundaryblob" * 25,
        "Tail! Next (one). 'Quoted.' ",
    )
    content = "".join(
        vocab[int(rng.integers(0, len(vocab)))] for _ in range(int(rng.integers(5, 60)))
    )
    for max_chars, max_items in ((64, 5), (128, 7), (1024, 7)):
        segs = split_into_segments(content, max_chars=max_chars, max_items=max_items)
        assert 1 <= len(segs) <= max_items
        assert "".join(segs) == content


# --------------------------------------- group dedup: negative case
async def test_different_long_content_is_not_deduped_onto_group(db: AsyncVectorDB) -> None:
    svc = MemoryService(db, config=MemoryConfig(), embed_fn=hash_embed)
    a = " ".join(f"Alpha topic sentence {i} about one shared subject." for i in range(40))
    b = " ".join(f"Different theme {i} regarding weather and tides." for i in range(40))
    r1 = await svc.encode_episode(a, "s1")
    r2 = await svc.encode_episode(b, "s1")
    assert r1["deduped"] is False
    assert r2["deduped"] is False
    assert r2["group_id"] != r1["group_id"]
