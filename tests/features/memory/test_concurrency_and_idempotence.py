"""Wave 3 — concurrency + idempotence regressions.

* The reactor's dream pass must serialise: two concurrent triggers
  cannot overlap, or ``ltd_pass`` doubles its decay on the same edge
  snapshot.
* Consolidation must be idempotent on a stable episode set: re-running
  it after every candidate has been absorbed must form no further
  schemas (the candidate filter excludes ``consolidated_into`` rows).
"""

from __future__ import annotations

import asyncio
import hashlib
from collections.abc import AsyncIterator
from typing import Any

import numpy as np
import pytest_asyncio
from simplevecdb import AsyncVectorDB

from synara.features.memory.config import MemoryConfig
from synara.features.memory.service import MemoryService


def _hash_embed(text: str, dim: int = 32) -> list[float]:
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


async def test_reactor_dream_concurrent_invocations_serialise(
    db: AsyncVectorDB,
) -> None:
    """Two concurrent ``_reactor_dream`` calls must not overlap.

    Wraps ``ltd_pass`` so its body sleeps once it has entered; the
    second invocation must wait on the dream lock before its own
    ``ltd_pass`` even starts. Without the lock, a second concurrent
    pass would observe ``in_flight > 0`` and we'd see ``max_in_flight``
    rise to 2.
    """
    cfg = MemoryConfig(
        reactor_consolidate_after_novel=999,
        dg_pattern_separation=False,
        dream_replay_top_k=0,  # keep the body lean; LTD-only suffices
    )
    svc = MemoryService(db, config=cfg, embed_fn=_hash_embed)
    for word in ("alpha", "bravo", "charlie"):
        await svc.encode_episode(word, "s1", salience=0.9)

    original_ltd = svc._plasticity.ltd_pass

    in_flight = 0
    max_in_flight = 0
    enter_event = asyncio.Event()

    async def slow_ltd(**kwargs: Any) -> int:
        nonlocal in_flight, max_in_flight
        in_flight += 1
        max_in_flight = max(max_in_flight, in_flight)
        # Yield long enough that a concurrent caller would have
        # entered too, were the lock missing.
        enter_event.set()
        await asyncio.sleep(0.05)
        result = await original_ltd(**kwargs)
        in_flight -= 1
        return result

    svc._plasticity.ltd_pass = slow_ltd  # type: ignore[method-assign]

    # Fire two concurrent dream reactor calls.
    await asyncio.gather(svc._reactor_dream(None), svc._reactor_dream(None))

    assert max_in_flight == 1, (
        f"dream lock failed: observed {max_in_flight} concurrent ltd_pass entries"
    )
    assert in_flight == 0


async def test_consolidate_is_idempotent_on_stable_episode_set(
    db: AsyncVectorDB,
) -> None:
    """A second consolidate over the same absorbed set forms no new schemas.

    Running ``consolidate`` once absorbs eligible episodes into schemas
    and marks them ``consolidated_into != UNCONSOLIDATED``. The
    candidate filter must exclude them on the next pass, so no
    additional schemas are formed — the schema collection stays the
    same size.
    """
    cfg = MemoryConfig(
        # Aggressive enough to actually form schemas at unit-test volume.
        consolidate_min_age_seconds=0.0,
        consolidate_min_retrievals=0,
        consolidate_min_cluster=2,
        dg_pattern_separation=False,
        reactor_consolidate_after_novel=999,  # disable reactor trigger
    )
    svc = MemoryService(db, config=cfg, embed_fn=_hash_embed)

    # Two distinct clusters by topical similarity.
    for w in ("apple", "banana", "cherry", "date"):
        await svc.encode_episode(w, "s1", salience=0.8)
    for w in ("python", "javascript", "rust", "haskell"):
        await svc.encode_episode(w, "s2", salience=0.8)

    first = await svc.consolidate(session_id=None, n_clusters=2, min_cluster_size=2)
    schemas_after_first = await svc.semantic.count()
    assert len(first) >= 1, first
    assert schemas_after_first >= 1

    second = await svc.consolidate(session_id=None, n_clusters=2, min_cluster_size=2)
    schemas_after_second = await svc.semantic.count()

    # The second pass had no fresh candidates: zero new schemas, and
    # the semantic collection size did not grow.
    assert second == [], second
    assert schemas_after_second == schemas_after_first
