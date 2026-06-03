"""Segment-group recall enrichment + recency surfacing + sibling plasticity.

Covers the three locked enrichments:
  * recall surfaces ``group_id``/``segment_count`` plus created/updated
    recency so callers can tell old memories from new,
  * recall collapses same-group fragments into a single best-ranked hit,
  * theta-segmented encode (and a ``get_episode`` re-bond) chain the
    sibling sub-records together in the plasticity graph.
"""

from __future__ import annotations

import hashlib
from collections.abc import AsyncIterator
from itertools import pairwise

import numpy as np
import pytest_asyncio
from simplevecdb import AsyncVectorDB

from synara.features.memory.config import MemoryConfig
from synara.features.memory.service import MemoryService

# Three short sentences; with a small theta window each becomes its own
# segment, so an ordinary store produces a multi-segment episode group.
_SEG_CONTENT = "Alpha beta gamma here. Delta epsilon zeta now. Eta theta iota end."


def hash_embed(text: str, dim: int = 32) -> list[float]:
    seed_bytes = hashlib.sha256(text.encode("utf-8")).digest()[:8]
    seed = int.from_bytes(seed_bytes, "big", signed=False)
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(dim).astype(np.float32)
    n = float(np.linalg.norm(v))
    out = (v / n) if n > 0 else v
    return [float(x) for x in out.tolist()]


@pytest_asyncio.fixture
async def service() -> AsyncIterator[MemoryService]:
    db = AsyncVectorDB(":memory:")
    try:
        yield MemoryService(
            db,
            config=MemoryConfig(theta_segment_max_chars=24),
            embed_fn=hash_embed,
        )
    finally:
        await db.close()


# ---- Part 3: config defaults -----------------------------------------


def test_segment_config_defaults() -> None:
    cfg = MemoryConfig()
    assert cfg.recall_collapse_groups is True
    assert cfg.segment_assoc_score == 0.8


# ---- Part 1a: recall surfaces recency + group fields -----------------


async def test_recall_surfaces_recency_and_group_fields(
    service: MemoryService,
) -> None:
    # Short enough to stay a single segment under the fixture's small
    # theta window, so it is genuinely unsegmented.
    await service.encode_episode("a lone memory", "s1")
    hits = await service.recall(query="a lone memory", k=4)
    assert hits
    hit = hits[0]
    assert hit["group_id"] is None
    assert hit["segment_count"] == 1
    assert isinstance(hit["created_at"], float)
    assert isinstance(hit["updated_at"], float)
    assert hit["updated_at"] >= hit["created_at"]
    # Just-encoded: ages are tiny but defined and non-negative.
    assert hit["age_days"] >= 0.0
    assert hit["updated_age_days"] >= 0.0


# ---- Part 1b: collapse same-group fragments --------------------------


async def test_recall_collapses_same_group_fragments(
    service: MemoryService,
) -> None:
    res = await service.encode_episode(_SEG_CONTENT, "s1")
    seg_ids = res["segment_ids"]
    assert len(seg_ids) >= 2  # content actually segmented
    # Only this episode's segments exist, so all land in the candidate
    # set; collapse must fold them into one hit for the group.
    hits = await service.recall(query=_SEG_CONTENT, k=8)
    group_hits = [h for h in hits if h.get("group_id") == seg_ids[0]]
    assert len(group_hits) == 1
    assert group_hits[0]["segment_count"] == len(seg_ids)


async def test_recall_collapse_disabled_keeps_fragments() -> None:
    db = AsyncVectorDB(":memory:")
    try:
        svc = MemoryService(
            db,
            config=MemoryConfig(theta_segment_max_chars=24, recall_collapse_groups=False),
            embed_fn=hash_embed,
        )
        res = await svc.encode_episode(_SEG_CONTENT, "s1")
        seg_ids = res["segment_ids"]
        hits = await svc.recall(query=_SEG_CONTENT, k=8)
        group_hits = [h for h in hits if h.get("group_id") == seg_ids[0]]
        assert len(group_hits) == len(seg_ids)
    finally:
        await db.close()


# ---- Part 2a: encode chains siblings in the plasticity graph ---------


async def test_encode_creates_sibling_plasticity_chain(
    service: MemoryService,
) -> None:
    res = await service.encode_episode(_SEG_CONTENT, "s1")
    seg_ids = res["segment_ids"]
    assert len(seg_ids) >= 2
    for a, b in pairwise(seg_ids):
        edges = await service.episodic.get_edges(src=a, dst=b, kind="plasticity", limit=1)
        assert edges, f"missing sibling plasticity edge {a}->{b}"
        assert int(edges[0].hits) >= 1


# ---- Part 2b: get_episode re-bonds the chain on access ---------------


async def test_get_episode_rebonds_sibling_chain(
    service: MemoryService,
) -> None:
    res = await service.encode_episode(_SEG_CONTENT, "s1")
    seg_ids = res["segment_ids"]
    a, b = seg_ids[0], seg_ids[1]

    async def chain_hits() -> int:
        edges = await service.episodic.get_edges(src=a, dst=b, kind="plasticity", limit=1)
        return int(edges[0].hits) if edges else 0

    before = await chain_hits()
    await service.get_episode(seg_ids[0])
    after = await chain_hits()
    assert after == before + 1
