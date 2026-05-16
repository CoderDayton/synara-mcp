"""Last reachable residual branches: signals/events/encode/successor."""

from __future__ import annotations

import asyncio
import hashlib
from collections.abc import AsyncIterator

import numpy as np
import pytest_asyncio
from simplevecdb import AsyncVectorDB

from synara.features.memory.amygdala.signals import SignalRegistry
from synara.features.memory.basal_ganglia.events import EventBus
from synara.features.memory.config import MemoryConfig
from synara.features.memory.hippocampus.successor import SuccessorRepresentation
from synara.features.memory.service import MemoryService


def hash_embed(text: str, dim: int = 32) -> list[float]:
    seed_bytes = hashlib.sha256(text.encode("utf-8")).digest()[:8]
    seed = int.from_bytes(seed_bytes, "big", signed=False)
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(dim).astype(np.float32)
    n = float(np.linalg.norm(v))
    out = (v / n) if n > 0 else v
    return [float(x) for x in out.tolist()]


@pytest_asyncio.fixture
async def db() -> AsyncIterator[AsyncVectorDB]:
    d = AsyncVectorDB(":memory:")
    try:
        yield d
    finally:
        await d.close()


def test_signal_registry_salience_adds_long_content_weight() -> None:
    reg = SignalRegistry()  # legacy structural pass on
    long_text = "word " * 400  # comfortably in the "long" length class
    sig = reg.derive(long_text)
    assert sig["length_class"] == "long"
    s = reg.salience(sig)
    # base + long_content_weight contribution present.
    assert s >= reg.base_salience + reg.long_content_weight


async def test_event_bus_maybe_prune_without_collection_is_noop() -> None:
    bus = EventBus(log_capacity=8)
    await bus._maybe_prune()  # no collection -> early return, no error
    assert await bus.log() == []


async def test_encode_dedup_skips_candidate_without_dg_support(
    db: AsyncVectorDB,
) -> None:
    svc = MemoryService(db, config=MemoryConfig(), embed_fn=hash_embed)
    text = "a sufficiently long candidate without any dg support set"
    # Pre-insert a doc that has NO dg_support metadata; the DG dedup
    # candidate loop must skip it instead of crashing.
    await svc.episodic.add_texts(
        texts=[text],
        embeddings=[hash_embed(text)],
        ids=[1],
        metadatas=[{"id": 1, "session_id": "s1", "consolidated_into": 0}],
    )
    r = await svc.encode_episode(text, "s1")
    # No usable support to match against -> not a dedup hit, stored fresh.
    assert r["deduped"] is False


async def test_sr_load_skips_zero_hit_edges(db: AsyncVectorDB) -> None:
    coll = db.collection("ep")
    await coll.add_texts(
        texts=["d1", "d2"],
        embeddings=[[1.0, 0.0], [0.0, 1.0]],
        ids=[1, 2],
    )
    # Persist an SR edge with hits=0; load() must skip it (count <= 0).
    await asyncio.to_thread(
        coll._collection.edges.upsert,
        1,
        2,
        kind="sr",
        weight=0.0,
        bonus=0.0,
        hits=0,
        metadata={},
    )
    sr = SuccessorRepresentation(kind="sr")
    sr.attach(coll)
    await sr.load()
    assert sr.total_edges == 0.0
    assert sr.boost(1, [2]) == {2: 0.0}
