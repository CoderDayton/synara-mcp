"""Off-policy replay skip-branches + neocortex (forget/reflect) units."""

from __future__ import annotations

import hashlib
from collections.abc import AsyncIterator

import numpy as np
import pytest
import pytest_asyncio
from simplevecdb import AsyncVectorDB

from synara.core.errors import ValidationError
from synara.features.memory.config import MemoryConfig
from synara.features.memory.hippocampus import replay as replay_mod
from synara.features.memory.neocortex.forget import (
    access_times_from_meta,
    memory_strength,
)
from synara.features.memory.service import UNCONSOLIDATED, MemoryService


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
        yield MemoryService(db, config=MemoryConfig(), embed_fn=hash_embed)
    finally:
        await db.close()


# ---- replay.run skip branches ----------------------------------------


async def test_replay_disabled_when_gain_zero() -> None:
    db = AsyncVectorDB(":memory:")
    try:
        svc = MemoryService(db, config=MemoryConfig(dream_replay_gain=0.0), embed_fn=hash_embed)
        assert await replay_mod.run(svc, now=1.0) == 0
    finally:
        await db.close()


async def test_replay_no_candidates_returns_zero(service: MemoryService) -> None:
    assert await replay_mod.run(service, now=1.0) == 0


async def test_replay_nothing_scored_when_all_below_min_salience() -> None:
    db = AsyncVectorDB(":memory:")
    try:
        cfg = MemoryConfig(dream_replay_min_salience=0.99)
        svc = MemoryService(db, config=cfg, embed_fn=hash_embed)
        await svc.encode_episode("low salience trace one", "s1", salience=0.1)
        await svc.encode_episode("low salience trace two", "s1", salience=0.1)
        assert await replay_mod.run(svc, now=10.0) == 0
    finally:
        await db.close()


async def test_replay_zero_gain_when_mean_salience_zero() -> None:
    """Two zero-salience co-session episodes: group forms but
    gain = base_gain * 0 == 0 -> the gain<=0 continue branch."""
    db = AsyncVectorDB(":memory:")
    try:
        svc = MemoryService(
            db,
            config=MemoryConfig(surprise_salience_boost=0.0),
            embed_fn=hash_embed,
        )
        await svc.encode_episode("zero salience alpha", "s1", salience=0.0)
        await svc.encode_episode("zero salience beta", "s1", salience=0.0)
        assert await replay_mod.run(svc, now=10.0) == 0
    finally:
        await db.close()


async def test_replay_treats_missing_salience_as_neutral_base(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An episode whose metadata lacks ``salience`` must be treated as the
    neutral base (``_DEFAULT_SALIENCE`` = 0.3), so it clears the default
    replay floor (``dream_replay_min_salience`` = 0.3) — mirroring the
    consolidation eligibility gate (``_apply_eligibility_gates``).

    Regression: the replay scorer defaulted missing salience to 0.0, so a
    legacy/externally-inserted episode with no salience field was silently
    dropped from every replay pass at the default config.
    """
    db = AsyncVectorDB(":memory:")
    try:
        svc = MemoryService(db, config=MemoryConfig(), embed_fn=hash_embed)
        # Two real co-session episodes (so reinforcement's edge writes
        # satisfy the FK to documents), but presented to the scorer with
        # metadata that OMITS ``salience`` — the legacy/external-row case.
        r1 = await svc.encode_episode("alpha trace", "s1")
        r2 = await svc.encode_episode("beta trace", "s1")
        candidates = [
            (
                int(r1["id"]),
                "alpha trace",
                {"session_id": "s1", "consolidated_into": UNCONSOLIDATED},
            ),
            (
                int(r2["id"]),
                "beta trace",
                {"session_id": "s1", "consolidated_into": UNCONSOLIDATED},
            ),
        ]

        async def _fake_fetch(
            service: MemoryService, *, scan_cap: int
        ) -> list[tuple[int, str, dict[str, object]]]:
            return candidates

        monkeypatch.setattr(replay_mod, "_fetch_candidates", _fake_fetch)
        reinforced = await replay_mod.run(svc, now=10.0)
        assert reinforced > 0, "missing-salience episodes were dropped from replay"
    finally:
        await db.close()


# ---- forget units ----------------------------------------------------


def test_memory_strength_empty_access_times_returns_salience() -> None:
    assert memory_strength(0.7, [], now=100.0) == pytest.approx(0.7)


def test_memory_strength_rejects_non_positive_d() -> None:
    with pytest.raises(ValidationError, match="d must be positive"):
        memory_strength(0.5, [1.0], now=2.0, d=0.0)


def test_access_times_legacy_meta_expansion() -> None:
    md = {"encoded_at": 10.0, "last_accessed": 50.0, "retrieval_count": 3}
    assert access_times_from_meta(md, fallback_now=0.0) == [10.0, 50.0, 50.0, 50.0]


def test_access_times_prefers_explicit_history() -> None:
    md = {"access_history": [1.0, 2.0], "encoded_at": 99.0}
    assert access_times_from_meta(md, fallback_now=0.0) == [1.0, 2.0]


async def test_forget_rejects_bad_args(service: MemoryService) -> None:
    with pytest.raises(ValidationError, match="strength_floor"):
        await service.forget(strength_floor=1.5)
    with pytest.raises(ValidationError, match="decay_tau_seconds"):
        await service.forget(decay_tau_seconds=-1.0)


async def test_forget_drops_weak_consolidated_episode(service: MemoryService) -> None:
    await service.encode_episode("a consolidated weak trace", "s1", salience=0.01)
    rows = await service.episodic.get_documents({"session_id": "s1"})
    doc_id = int(rows[0][0])
    # Mark consolidated + ancient so strength < floor and the
    # consolidated-drop branch fires (not the floor/2 branch).
    await service.episodic.update_metadata(
        [(doc_id, {"consolidated_into": 12345, "encoded_at": 0.0, "last_accessed": 0.0})]
    )
    res = await service.forget(strength_floor=0.5, dry_run=False)
    assert doc_id in res["candidate_ids"]
    assert res["removed"] >= 1
    assert UNCONSOLIDATED == -1 or isinstance(UNCONSOLIDATED, int)


# ---- reflect units ---------------------------------------------------


async def test_reflect_rejects_bad_args(service: MemoryService) -> None:
    with pytest.raises(ValidationError, match="session_id must be non-empty"):
        await service.reflect(session_id="")
    with pytest.raises(ValidationError, match="k must be positive"):
        await service.reflect(session_id="s1", k=0)


async def test_reflect_seeds_semantic_search_from_tag(service: MemoryService) -> None:
    await service.store_semantic_memory("ruff is the linter", kind="fact", tags=["tooling"])
    await service.encode_episode("used the linter today", "s1", tags=["tooling"])
    out = await service.reflect(session_id="s1", k=5)
    assert out["session_id"] == "s1"
    # seed taken from the recent episode's first tag -> semantic hit.
    assert any("ruff" in s["summary"] for s in out["schemas"])
    assert out["recent_episodes"]
