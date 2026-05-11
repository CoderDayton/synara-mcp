"""Plasticity-constant simulator for ``MemoryConfig``.

Standalone behavioural sanity check for the neuroplasticity timescales
(E-LTP, L-LTP, reconsolidation, LTD, habit thresholds, savings). Does
NOT touch the runtime service. It mirrors the *intended* dynamics in
plain Python so we can validate the constants in isolation before
wiring anything up.

Run:
    python scripts/sim/plasticity_sim.py
"""

from __future__ import annotations

import math
import random
import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from synara.features.memory.config import MemoryConfig  # noqa: I001


_PRUNE_FLOOR = 1e-3
_MULTI_TARGET_PROB = 0.5


def _slow(cfg: MemoryConfig, real_seconds: float) -> float:
    """Compress a slow-process duration into app-time."""
    return real_seconds / cfg.time_compression


@dataclass
class Episode:
    salience: float
    encoded_at: float
    last_recall: float
    retrieval_count: int = 0
    drift: float = 0.0
    drift_locked: bool = False


@dataclass
class Edge:
    weight: float = 0.0
    bonus: float = 0.0
    bonus_set_at: float = -math.inf
    hits_in_window: int = 0
    total_hits: int = 0
    ever_habit: bool = False
    last_touch: float = -math.inf

    def is_habit(self) -> bool:
        return self.ever_habit


class PlasticitySim:
    """Pure-Python model of the plasticity dynamics implied by the config."""

    def __init__(self, cfg: MemoryConfig, *, seed: int = 0) -> None:
        self.cfg = cfg
        self.rand = random.Random(seed)
        self.t_app = 0.0
        self.episodes: dict[int, Episode] = {}
        self.edges: dict[tuple[int, int], Edge] = {}

    def advance_app(self, app_seconds: float) -> None:
        self.t_app += app_seconds

    def _real_elapsed(self, app_then: float) -> float:
        return max(0.0, (self.t_app - app_then) * self.cfg.time_compression)

    def encode(self, ep_id: int, salience: float, *, surprising: bool = False) -> None:
        s = salience
        if surprising:
            s = min(1.0, s + self.cfg.surprise_salience_boost)
        self.episodes[ep_id] = Episode(
            salience=s,
            encoded_at=self.t_app,
            last_recall=self.t_app,
        )

    def _bump_edge(self, a: int, b: int, score: float) -> None:
        if a == b:
            return
        key = (a, b) if a < b else (b, a)
        e = self.edges.get(key) or Edge()

        # Decay the transient bonus by REAL wall-clock since last set.
        elapsed_real = self._real_elapsed(e.bonus_set_at)
        e.bonus *= math.exp(-elapsed_real / self.cfg.e_ltp_decay_seconds)

        in_window = elapsed_real <= self.cfg.e_ltp_decay_seconds
        e.hits_in_window = e.hits_in_window + 1 if in_window else 1

        # Hit increment, with savings if this edge has ever been a habit.
        increment = score * (self.cfg.habit_savings_factor if e.ever_habit else 1.0)
        e.bonus += 0.5 * increment
        e.bonus_set_at = self.t_app

        # L-LTP: in-window hits clearing threshold fold the bonus into
        # durable weight.
        if e.hits_in_window >= self.cfg.l_ltp_threshold_hits:
            e.weight += e.bonus
            e.bonus = 0.0
            e.hits_in_window = 0

        e.total_hits += 1
        if e.total_hits >= self.cfg.habit_threshold_hits:
            e.ever_habit = True
        e.last_touch = self.t_app
        self.edges[key] = e

    def recall(self, query_id: int, target_ids: list[int], *, score: float = 0.8) -> None:
        for tid in target_ids:
            self._bump_edge(query_id, tid, score=score)
        cfg = self.cfg
        for tid in target_ids:
            ep = self.episodes.get(tid)
            if ep is None or ep.drift_locked or score < cfg.reconsolidation_min_score:
                if ep is not None:
                    ep.last_recall = self.t_app
                    ep.retrieval_count += 1
                continue
            elapsed_real = (self.t_app - ep.last_recall) * cfg.time_compression
            in_window = elapsed_real <= cfg.reconsolidation_window_seconds
            if in_window:
                step = cfg.reconsolidation_alpha * score
                new_drift = ep.drift + step
                if new_drift >= cfg.reconsolidation_max_total_drift:
                    ep.drift = cfg.reconsolidation_max_total_drift
                    ep.drift_locked = True
                else:
                    ep.drift = new_drift
            ep.last_recall = self.t_app
            ep.retrieval_count += 1

    def offline_pass(self) -> int:
        """Apply LTD over idle real-days. Return number of pruned edges."""
        cfg = self.cfg
        pruned: list[tuple[int, int]] = []
        for key, e in self.edges.items():
            idle_days = self._real_elapsed(e.last_touch) / 86400.0
            decay = cfg.ltd_decay_per_idle_day
            if e.is_habit():
                decay *= cfg.habit_ltd_multiplier
            e.weight *= max(0.0, 1.0 - decay) ** idle_days
            if e.weight < _PRUNE_FLOOR and not e.is_habit():
                pruned.append(key)
        for key in pruned:
            del self.edges[key]
        return len(pruned)

    def memory_strength(self, ep_id: int) -> float:
        ep = self.episodes.get(ep_id)
        if ep is None:
            return 0.0
        d = self.cfg.forget_d
        access = [ep.encoded_at]
        if ep.retrieval_count > 0:
            access.append(ep.last_recall)
        s = 0.0
        for a in access:
            age = max(1e-3, self.t_app - a)
            s += (1.0 + age) ** (-d)
        return ep.salience * s


# ---- usage profiles -------------------------------------------------

CORE_IDS = [1, 2, 3, 4, 5]
TOPIC_IDS = [101, 102, 103]  # recurring "user concerns" - act as queries


def _seed_core(sim: PlasticitySim) -> None:
    for cid in CORE_IDS:
        sim.encode(cid, salience=0.7)
    for tid in TOPIC_IDS:
        sim.encode(tid, salience=0.6)


def _drive(
    sim: PlasticitySim,
    *,
    days: int,
    turns_per_day: int,
    core_focus: float,
    idle_days_between_blocks: float = 0.0,
    block_days: int = 1,
) -> int:
    """Turn = one user query against the recurring topic bank.

    With probability ``core_focus`` the query is a TOPIC_ID and the
    recall pulls 1-2 CORE_IDs (so the (topic, core) edge accumulates
    hits over time). Otherwise a fresh filler query/encode happens.
    """
    cfg = sim.cfg
    sec_per_turn_app = _slow(cfg, 86400.0) / max(1, turns_per_day)
    next_id = 1000
    for day_idx in range(days):
        for _ in range(turns_per_day):
            sim.advance_app(sec_per_turn_app)
            if sim.rand.random() < core_focus:
                topic = sim.rand.choice(TOPIC_IDS)
                # Pull 1-2 core memories per recall (multi-target case
                # also exercises target<->target SR edges).
                k = 1 + (1 if sim.rand.random() < _MULTI_TARGET_PROB else 0)
                targets = sim.rand.sample(CORE_IDS, k=k)
                sim.recall(query_id=topic, target_ids=targets, score=0.85)
            else:
                sim.encode(next_id, salience=0.4)
                next_id += 1
        if idle_days_between_blocks > 0 and (day_idx + 1) % block_days == 0:
            sim.advance_app(_slow(cfg, idle_days_between_blocks * 86400.0))
            sim.offline_pass()
    return next_id


def profile_steady(sim: PlasticitySim) -> None:
    _seed_core(sim)
    _drive(sim, days=30, turns_per_day=30, core_focus=0.85)


def profile_bursty(sim: PlasticitySim) -> None:
    _seed_core(sim)
    _drive(
        sim,
        days=10,
        turns_per_day=50,
        core_focus=0.85,
        idle_days_between_blocks=3.0,
        block_days=1,
    )


def profile_sparse(sim: PlasticitySim) -> None:
    _seed_core(sim)
    _drive(sim, days=60, turns_per_day=2, core_focus=0.85)


def profile_disuse_then_resume(sim: PlasticitySim) -> None:
    """Build habits, idle a long time, then resume - tests savings."""
    _seed_core(sim)
    _drive(sim, days=20, turns_per_day=30, core_focus=0.9)
    # 60 real days of disuse.
    sim.advance_app(_slow(sim.cfg, 60 * 86400.0))
    sim.offline_pass()
    # Resume.
    _drive(sim, days=5, turns_per_day=20, core_focus=0.9)


# ---- reporting ------------------------------------------------------


def _summary(sim: PlasticitySim) -> dict[str, float]:
    edges = list(sim.edges.values())
    habits = [e for e in edges if e.is_habit()]
    drifts = [ep.drift for ep in sim.episodes.values()]
    locked = sum(1 for ep in sim.episodes.values() if ep.drift_locked)
    return {
        "edges": float(len(edges)),
        "habits": float(len(habits)),
        "max_hits": float(max((e.total_hits for e in edges), default=0)),
        "habit_max_hits": float(max((e.total_hits for e in habits), default=0)),
        "max_weight": max((e.weight for e in edges), default=0.0),
        "habit_max_weight": max((e.weight for e in habits), default=0.0),
        "nonhabit_max_weight": max((e.weight for e in edges if not e.is_habit()), default=0.0),
        "max_drift": max(drifts, default=0.0),
        "drift_locked": float(locked),
    }


def main() -> int:
    cfg = MemoryConfig()
    print(
        f"# config: time_compression={cfg.time_compression} "
        f"habit_thr={cfg.habit_threshold_hits} "
        f"ltd/day={cfg.ltd_decay_per_idle_day} "
        f"habit_ltd_mult={cfg.habit_ltd_multiplier} "
        f"savings={cfg.habit_savings_factor} "
        f"recon_alpha={cfg.reconsolidation_alpha} "
        f"recon_cap={cfg.reconsolidation_max_total_drift}"
    )
    runs = [
        ("steady-30d-30tpd", profile_steady),
        ("bursty-10x50t-3dgap", profile_bursty),
        ("sparse-60d-2tpd", profile_sparse),
        ("disuse-then-resume", profile_disuse_then_resume),
    ]
    headers = ("profile", "edges", "habit", "maxH", "habH", "maxW", "habW", "drift", "lock")
    print("{:<22} {:>5} {:>5} {:>5} {:>5} {:>7} {:>7} {:>6} {:>4}".format(*headers))
    for name, fn in runs:
        sim = PlasticitySim(cfg, seed=0)
        fn(sim)
        s = _summary(sim)
        print(
            "{:<22} {:>5d} {:>5d} {:>5d} {:>5d} {:>7.2f} {:>7.2f} {:>6.3f} {:>4d}".format(
                name,
                int(s["edges"]),
                int(s["habits"]),
                int(s["max_hits"]),
                int(s["habit_max_hits"]),
                s["max_weight"],
                s["habit_max_weight"],
                s["max_drift"],
                int(s["drift_locked"]),
            )
        )
    print()
    print("# checks (rough sanity, not asserts):")
    print("#  steady should produce >=1 habit edge")
    print("#  sparse should produce 0 habits (insufficient repetition)")
    print("#  bursty habits should still form despite gaps")
    print("#  disuse-then-resume: habits survive the gap (habW > 0)")
    print("#  max_drift should not exceed reconsolidation_max_total_drift")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
