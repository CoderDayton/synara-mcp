"""Long-horizon self-learning simulation over the real MemoryService.

Unlike ``scripts/sim/plasticity_sim.py`` (a standalone constant
validator), this drives the *actual* runtime: episodes are stored and
recalled through ``MemoryService`` across many simulated sessions and
days. Nothing here calls ``consolidate`` or ``forget`` to make schemas
appear -- the basal-ganglia reactor fires them on its own once the
event thresholds are crossed (``self_learning_enabled=True``,
``reactor_consolidate_after_novel=32``, ``reactor_dream_after_events
=128``). The script just *observes* the structure the system grows by
itself and prints a longitudinal timeline.

What "self-evolving" means here, concretely:

  * SR transition graph (kind="sr") accretes co-occurrence edges and
    its discounted closure densifies -- a learned relational prior.
  * The reactor auto-consolidates recurring topic clusters into
    neocortical semantic schemas (semantic_count climbs with zero
    explicit consolidate calls).
  * Dream replay fires off-policy and reinforces associations.
  * Low-salience noise is pruned by periodic forgetting, so the
    episodic store stays bounded while the distilled gist persists.

Only deviation from production defaults: ``consolidate_min_age_seconds``
/ ``consolidate_min_retrievals`` are lowered (as in
``memory_lifecycle_demo.py``) so the reactor can consolidate within a
fast synchronous run instead of waiting a real 60 s. Offline: a
deterministic topical embedder is injected, so no model download and
the run is reproducible.

Run:  uv run --no-sync python scripts/sim/self_learning_sim.py
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import importlib
import itertools
import logging
import math
import random
import sys
import time
from pathlib import Path

# Default: mute INFO/DEBUG/WARNING runtime chatter during the ~1500-session
# run while still surfacing real errors. --verbose dials this back up.
logging.disable(logging.WARNING)

# Two sys.path inserts: the src tree (synara/...) and this script's own
# directory (so `from _report import ...` works without making scripts/sim
# a package).
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

# ruff: noqa: E402  (sys.path tweaks above must precede imports)
from _report import Snapshot, render_html
from simplevecdb import AsyncVectorDB

from synara.features.memory import MemoryConfig
from synara.features.memory.service import MemoryService

# ---- topical offline embedder ---------------------------------------

DIM = 64
N_TOPICS = 8
# Populated by _build_centroids() from the CLI seed before any embedding
# is requested. Kept module-level so topical_embed() (passed as embed_fn
# to MemoryService) can stay a plain function.
_CENTROIDS: list[list[float]] = []


def _build_centroids(seed: int) -> None:
    rng = random.Random(seed)
    _CENTROIDS.clear()
    for _ in range(N_TOPICS):
        v = [rng.gauss(0.0, 1.0) for _ in range(DIM)]
        n = math.sqrt(sum(x * x for x in v)) or 1.0
        _CENTROIDS.append([x / n for x in v])


def _topic_of(text: str) -> int:
    # Episodes are emitted as "t<k>: ..."; the embedder reads the topic
    # tag so same-topic episodes land in the same cosine neighbourhood
    # (which is what lets the reactor cluster + consolidate them).
    head = text.split(":", 1)[0]
    return int(head[1:]) if head.startswith("t") and head[1:].isdigit() else 0


def topical_embed(text: str) -> list[float]:
    k = _topic_of(text) % N_TOPICS
    c = _CENTROIDS[k]
    seed = int.from_bytes(hashlib.sha256(text.encode()).digest()[:8], "big")
    jr = random.Random(seed)
    v = [c[i] + 0.15 * jr.gauss(0.0, 1.0) for i in range(DIM)]
    n = math.sqrt(sum(x * x for x in v)) or 1.0
    return [x / n for x in v]


# ---- simulation defaults --------------------------------------------
# These are the CLI defaults; the live values used at runtime live on
# the parsed argparse.Namespace inside main().

DEFAULT_DAYS = 60
DEFAULT_SESSIONS_PER_DAY = 6
DEFAULT_EP_PER_SESSION = 4
DEFAULT_SNAPSHOT_EVERY = 10
DEFAULT_FORGET_EVERY = 15
# Per topic, a small set of durable high-salience "knowledge" anchors
# encoded once. Every same-topic session re-recalls them: the fixed
# anchor<->anchor plasticity edges accumulate hits across the whole
# run (-> habits latch) while the recall refreshes their access in
# virtual time (-> they survive forgetting). Unique per-session work
# and the salience-0.02 noise are the forgettable churn.
DEFAULT_ANCHORS_PER_TOPIC = 3
ANCHOR_SALIENCE = 0.97
# Frequency-dependent consolidation: only the highest-frequency topic(s)
# earn durable anchors. Tail topics see filler only -> their traces are
# pruned by forget, so the post-forget substrate (and the recall prior)
# concentrates on the hub the agent actually practised. This is the
# emergent property, not a probe hack: a learner masters its most-used
# domain and lets rarely-revisited ones fade.
DEFAULT_CORE_TOPICS = 1
DEFAULT_CENTROID_SEED = 20260518
DEFAULT_SIM_SEED = 7

# Zipfian topic interest: a few topics recur far more than the tail,
# so recurring structure (habits, schemas) has something to latch onto.
_TOPIC_WEIGHTS = [1.0 / (i + 1) for i in range(N_TOPICS)]
_PHRASES = [
    "root cause traced to {n}",
    "patched the {n} path and verified",
    "regression test added for {n}",
    "follow-up: {n} edge case under load",
    "design note on {n} ownership",
]


def _instrument_reactor(svc: MemoryService) -> tuple[list[int], list[int]]:
    """Count true cumulative reactor firings.

    ``event_log`` is a capped ring buffer, so summing it under-counts
    over a long run. Wrap the bus handlers instead -- this is exactly
    the self-triggered activity we want to observe.
    """
    cons = [0]
    dreams = [0]
    for attr in ("_reactor_consolidate", "_reactor_dream", "_bus"):
        if not hasattr(svc, attr):
            raise AttributeError(
                f"MemoryService.{attr} not found; reactor instrumentation "
                "would silently no-op. Rename in production code?"
            )
    for attr in ("on_consolidate", "on_dream"):
        if not hasattr(svc._bus, attr):
            raise AttributeError(
                f"MemoryService._bus.{attr} not found; reactor instrumentation "
                "would silently no-op. Rename in production code?"
            )
    orig_c = svc._reactor_consolidate
    orig_d = svc._reactor_dream

    async def wrapped_c(ev: object) -> None:
        cons[0] += 1
        await orig_c(ev)

    async def wrapped_d(ev: object) -> None:
        dreams[0] += 1
        await orig_d(ev)

    svc._bus.on_consolidate = wrapped_c
    svc._bus.on_dream = wrapped_d
    return cons, dreams


async def _snapshot(svc: MemoryService, day: int, cons: list[int], dreams: list[int]) -> Snapshot:
    await svc._ensure_sr_loaded()
    st = await svc.stats()
    pl = await svc._plasticity.stats()
    sr = svc._sr
    sr_rows = len(sr._T_counts) if sr is not None else 0
    sr_edges = sum(len(r) for r in sr._T_counts.values()) if sr is not None else 0
    epis = int(st["episodic_count"])
    sem = int(st["semantic_count"])
    p_edges = int(pl["edges"])
    max_hits = int(pl["max_total_hits"])
    # Derived ratios expose what raw counts hide:
    #  ep_per_sch - episodes distilled per schema (compression; lower=tighter)
    #  sr_dens    - SR fan-out per node           (graph densification)
    #  edge_conc  - hits on the busiest edge / edge (concentration)
    ep_per_sch = round(epis / sem, 2) if sem else 0.0
    sr_dens = round(sr_edges / sr_rows, 2) if sr_rows else 0.0
    edge_conc = round(max_hits / p_edges, 2) if p_edges else 0.0
    return Snapshot(
        day=day,
        epis=epis,
        sem=sem,
        p_edges=p_edges,
        habits=int(pl["habits"]),
        max_hits=max_hits,
        sr_rows=sr_rows,
        sr_edges=sr_edges,
        cons=cons[0],
        dreams=dreams[0],
        ep_per_sch=ep_per_sch,
        sr_dens=sr_dens,
        edge_conc=edge_conc,
    )


# ---- virtual clock --------------------------------------------------
# The runtime's strength/decay reads a real wall clock (now_seconds),
# so in a fast run every trace is "just now" and forgetting can't bite.
# Patch the bound now_seconds in every consumer module to a virtual
# clock the sim advances by one simulated day per loop iteration, so
# encode timestamps, recall access times, the consolidate age gate, the
# reactor idle trigger and forget's decay all share one coherent
# simulated timeline. Sim-only instrumentation (no runtime change) --
# the same technique as the reactor-handler wrapping above.

_SIM_DAY_SECONDS = 86400.0
_VT = [time.time()]

# Virtualise the whole episode age/access path: encode (encoded_at),
# recall (refreshes access_history on retrieval) and forget (reads
# both to compute strength). recall MUST be virtual: durable anchor
# episodes are re-recalled every same-topic session, and only a
# virtual access stamp keeps their strength above the floor as
# virtual time advances -- that is what stops the aggressive decay
# from cascade-deleting (FK ON DELETE CASCADE) the plasticity edges
# the habit accumulates on. The habit gate is cumulative ``hits``
# (gap-independent), so virtual cross-day reinforcement spacing does
# not block latching; the real-clock ltd_pass run by the dream
# reactor sees a virtual ``last_touch`` ahead of its real ``now``,
# clamps idle to 0 and so does not prune these edges.
# Deliberately NOT service/events/consolidate: the reactor stamps
# trigger state via a separate unpatched alias (events.now_seconds,
# imported into service as ``_now_real``); a virtual ``now`` vs a
# real ``last_*_at`` makes every dream_due/consolidate_due
# perpetually true -> dream-replay runaway.
_CLOCK_MODULES = (
    "synara.features.memory.neocortex.forget",
    "synara.features.memory.hippocampus.recall",
    "synara.features.memory.hippocampus.encode",
)


def _vclock() -> float:
    return _VT[0]


def _advance_day() -> None:
    _VT[0] += _SIM_DAY_SECONDS


def _install_virtual_clock() -> None:
    for name in _CLOCK_MODULES:
        mod = importlib.import_module(name)
        if not hasattr(mod, "now_seconds"):
            raise AttributeError(
                f"{name}.now_seconds not found; virtual clock would silently "
                "no-op and forget would run on real time. Rename upstream?"
            )
        mod.now_seconds = _vclock


def _fmt_row(vals: tuple[object, ...]) -> str:
    cells = [f"{v:>8.2f}" if isinstance(v, float) else f"{v:>8}" for v in vals]
    return " ".join(cells)


def _heartbeat(day: int, total: int, *, enabled: bool) -> None:
    if not enabled:
        return
    print(f"\r# day {day:>4}/{total} ", end="", flush=True)


def _heartbeat_clear(*, enabled: bool) -> None:
    if not enabled:
        return
    print("\r" + " " * 32 + "\r", end="", flush=True)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Long-horizon self-learning simulation over the real "
            "MemoryService. Writes an HTML report and prints a longitudinal "
            "table to stdout."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--days", type=int, default=DEFAULT_DAYS, help="simulated days")
    p.add_argument(
        "--sessions",
        type=int,
        default=DEFAULT_SESSIONS_PER_DAY,
        help="sessions per simulated day",
    )
    p.add_argument(
        "--episodes-per-session",
        type=int,
        default=DEFAULT_EP_PER_SESSION,
        help="forgettable episodes encoded per session",
    )
    p.add_argument(
        "--snapshot-every",
        type=int,
        default=DEFAULT_SNAPSHOT_EVERY,
        help="days between snapshot rows",
    )
    p.add_argument(
        "--forget-every",
        type=int,
        default=DEFAULT_FORGET_EVERY,
        help="days between forget passes",
    )
    p.add_argument(
        "--anchors-per-topic",
        type=int,
        default=DEFAULT_ANCHORS_PER_TOPIC,
        help="durable knowledge anchors per core topic",
    )
    p.add_argument(
        "--core-topics",
        type=int,
        default=DEFAULT_CORE_TOPICS,
        help="how many topics get anchored (frequency-dependent consolidation)",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SIM_SEED,
        help="RNG seed for per-session topic sampling",
    )
    p.add_argument(
        "--centroid-seed",
        type=int,
        default=DEFAULT_CENTROID_SEED,
        help="RNG seed for the offline topical embedder's centroids",
    )
    p.add_argument(
        "--out",
        type=Path,
        default=Path(__file__).resolve().parent / "self_learning_report.html",
        help="HTML report destination",
    )
    p.add_argument(
        "--quiet",
        action="store_true",
        help="suppress per-snapshot rows and progress heartbeat",
    )
    p.add_argument(
        "--verbose",
        action="store_true",
        help="re-enable runtime INFO/DEBUG logs (default mutes WARNING and below)",
    )
    args = p.parse_args(argv)
    for name in (
        "days",
        "sessions",
        "episodes_per_session",
        "snapshot_every",
        "forget_every",
        "anchors_per_topic",
        "core_topics",
    ):
        if getattr(args, name) <= 0:
            p.error(f"--{name.replace('_', '-')} must be positive")
    if args.core_topics > N_TOPICS:
        p.error(f"--core-topics must be <= N_TOPICS ({N_TOPICS})")
    return args


async def _recall_prior(svc: MemoryService) -> tuple[int, int]:
    """Recall-prior probe, measured on the MAINTAINED (pre-forget) state.

    Investigation finding: post-forget ``ep_count`` is tiny, so the SR
    rank weight ``omega = _sr.omega(ep_count)`` is large and the learned
    t0-anchor<->non-t0-filler bridges out-rank ~2 of the topical anchors
    (recall.py ``_sr_rank_keys``). That is real emergent structure, but
    sampling it at the post-forget trough is the same trough bias the
    plasticity table already avoids. Probing the maintained state (large
    ep_count -> small omega -> cosine-led) measures the prior the system
    actually operates with between maintenance sweeps.
    """
    requested_k = 5
    hot = await svc.recall("t0: core knowledge anchor", session_id="probe", k=requested_k)
    if len(hot) < requested_k:
        # Synthesis reports "same/total top hits"; if cosine returned fewer
        # than asked, surface that rather than letting the denominator quietly
        # shrink and inflate the apparent hit ratio.
        print(
            f"# WARN: recall prior probe got {len(hot)}/{requested_k} hits "
            "(store may be smaller than expected)"
        )
    same = sum(1 for h in hot if str(h.get("content", "")).startswith("t0:"))
    return same, len(hot)


HEADERS: tuple[str, ...] = (
    "day",
    "epis",
    "sem",
    "pEdg",
    "hab",
    "maxH",
    "srRow",
    "srEdg",
    "cons",
    "drm",
    "ep/sch",
    "srE/srR",
    "mH/pE",
)


async def _seed_anchors(svc: MemoryService, core_topics: int, anchors_per_topic: int) -> None:
    """Encode the durable per-topic knowledge anchors once (high salience)."""
    for t in range(core_topics):
        for a in range(anchors_per_topic):
            await svc.encode_episode(
                f"t{t}: core knowledge anchor {a}",
                session_id=f"anchors-t{t}",
                tags=[f"topic-{t}", "anchor"],
                salience=ANCHOR_SALIENCE,
            )


async def _run_session(
    svc: MemoryService,
    *,
    sid: str,
    topic: int,
    day: int,
    s: int,
    anchors_per_topic: int,
    ep_per_session: int,
) -> None:
    """One simulated session: hub-anchor recall, anchor rehearsal, churn."""
    # 1. Re-activate the durable HUB anchors (topic 0) every session
    #    regardless of which topic the filler is about, so the most-
    #    practised substrate consolidates. Co-recall lays down the stable
    #    anchor<->anchor SR + plasticity edges (their hit count climbs
    #    every same-topic session until it crosses habit_threshold_hits)
    #    and the retrieval refreshes anchor access in virtual time so
    #    they stay above the forget floor.
    await svc.recall(
        "t0: core knowledge anchor",
        session_id=sid,
        k=anchors_per_topic,
    )
    # 1b. Deterministic anchor rehearsal: the ranked recall above can be
    #     crowded out by same-topic filler, so the actual anchor episodes
    #     never get their access-time refreshed and forget prunes them
    #     (FK CASCADE then wipes their accumulated plasticity edges).
    #     Retrieve each anchor by its exact stored text so every anchor
    #     episode is touched every session: Dt~=0 at the forget pass ->
    #     strength ~= salience (0.97) >> floor -> the recurring substrate
    #     survives every cascade and its anchor<->anchor edges keep
    #     climbing toward habit_threshold_hits.
    for a in range(anchors_per_topic):
        await svc.recall(f"t0: core knowledge anchor {a}", session_id=sid, k=1)
    # 2. Forgettable churn: unique per-session work plus a near-zero-
    #    salience throwaway, so the store self-bounds.
    for e in range(ep_per_session):
        phrase = _PHRASES[e % len(_PHRASES)].format(n=f"topic-{topic}")
        await svc.encode_episode(
            f"t{topic}: {phrase}",
            session_id=sid,
            tags=[f"topic-{topic}"],
            salience=0.6,
        )
    await svc.encode_episode(
        f"t{topic}: incidental scratch note {day}-{s}",
        session_id=sid,
        tags=["noise"],
        salience=0.02,
    )


def _build_synthesis(
    rows: list[Snapshot],
    post: Snapshot,
    cons_count: int,
    dreams_count: int,
    forget_passes: int,
    forget_removed: int,
    probe: tuple[int, int],
) -> list[str]:
    """Format the emergent-structure summary lines from the collected rows."""
    first, last = rows[0], rows[-1]
    epis_series = [r.epis for r in rows]
    rising = all(b > a for a, b in itertools.pairwise(epis_series))
    peak_habits = max(r.habits for r in rows)
    peak_maxh = max(r.max_hits for r in rows)
    same, total = probe
    return [
        f"reactor self-fired: consolidations={cons_count} dreams={dreams_count} "
        "(explicit svc.consolidate() calls in this script: 0)",
        f"semantic schemas: {first.sem} -> {last.sem} (one cooldown-gated "
        "reactor pass distils the recurring topics)",
        f"SR transition rows: {first.sr_rows} -> {last.sr_rows}, edges "
        f"{first.sr_edges} -> {last.sr_edges} (learned relational prior)",
        f"plasticity edges: {first.p_edges} -> {last.p_edges}, habits "
        f"{first.habits} -> {last.habits} (peak {peak_habits}), "
        f"maxHits {last.max_hits} (peak {peak_maxh}) "
        + (
            "-- anchor<->anchor edges self-cross habit_threshold_hits"
            if peak_habits
            else "-- no habit latched"
        ),
        f"episodic store: {epis_series[0]} -> {post.epis} after the "
        f"terminal forget (mid-cycle peak {max(epis_series)}); strictly-rising="
        f"{rising} -> virtual-time decay "
        + ("did NOT bound it" if rising else "BOUNDED it")
        + f" (forget passes={forget_passes} removed={forget_removed})",
        f"ratios (post-forget): {post.ep_per_sch} episodes/schema (compression), "
        f"{post.sr_dens} SR-edges/row (densification), "
        f"{post.edge_conc} maxhits/edge (concentration)",
        f"recall prior check (maintained pre-forget state): {same}/{total} "
        "top hits are the recurring hot topic -> structure shapes retrieval",
    ]


def _print_preface(
    args: argparse.Namespace,
    cfg: MemoryConfig,
    days: int,
    sessions_per_day: int,
) -> None:
    if args.quiet:
        return
    print(
        f"# real-runtime self-learning sim: {days}d x {sessions_per_day} "
        f"sessions, reactor self_learning={cfg.self_learning_enabled}"
    )
    print(
        f"# consolidate_after_novel={cfg.reactor_consolidate_after_novel} "
        f"dream_after_events={cfg.reactor_dream_after_events} "
        f"dream_replay_top_k={cfg.dream_replay_top_k}"
    )
    print(f"# seed={args.seed} centroid_seed={args.centroid_seed}")
    print(_fmt_row(HEADERS))


async def _run_simulation(
    svc: MemoryService,
    *,
    days: int,
    sessions_per_day: int,
    ep_per_session: int,
    anchors_per_topic: int,
    snapshot_every: int,
    forget_every: int,
    rng: random.Random,
    cons: list[int],
    dreams: list[int],
    show_progress: bool,
    quiet: bool,
) -> tuple[list[Snapshot], Snapshot, tuple[int, int], int, int]:
    """Day-by-day simulation loop.

    Returns ``(rows, post_snapshot, probe, forget_passes, forget_removed)``.
    """
    rows: list[Snapshot] = []
    snap = await _snapshot(svc, 0, cons, dreams)
    rows.append(snap)
    if not quiet:
        print(_fmt_row(snap.as_row()))

    forget_passes = 0
    forget_removed = 0
    probe: tuple[int, int] = (0, 0)
    for day in range(1, days + 1):
        _advance_day()  # one simulated day of virtual aging
        _heartbeat(day, days, enabled=show_progress)
        for s in range(sessions_per_day):
            sid = f"d{day}-s{s}"
            topic = rng.choices(range(N_TOPICS), weights=_TOPIC_WEIGHTS)[0]
            await _run_session(
                svc,
                sid=sid,
                topic=topic,
                day=day,
                s=s,
                anchors_per_topic=anchors_per_topic,
                ep_per_session=ep_per_session,
            )

        # Snapshot BEFORE forget. snapshot_every and forget_every can collide
        # at integer multiples; sampling after the forget cascade would
        # measure the sawtooth at its trough (FK CASCADE has just wiped the
        # high-hit edges) and misreport the steady state the system actually
        # maintains between maintenance sweeps. The sawtooth itself stays
        # visible via peak fields + forget_removed.
        if day % snapshot_every == 0:
            snap = await _snapshot(svc, day, cons, dreams)
            rows.append(snap)
            _heartbeat_clear(enabled=show_progress)
            if not quiet:
                print(_fmt_row(snap.as_row()))

        if day == days:
            # Recall prior measured on the maintained state, i.e. BEFORE the
            # terminal forget (see _recall_prior for why this is the
            # faithful sampling point).
            probe = await _recall_prior(svc)

        if day % forget_every == 0:
            # Aggressive floor: anchors are refreshed by exact-text
            # rehearsal every session (Dt~=0 -> S ~= 0.97 here), so they
            # clear this floor with margin while never-rehearsed filler and
            # salience-0.02 noise fall through. A low floor (0.005) was
            # measured to keep filler alive and dilute anchor recall
            # (maxHits collapsed 223 -> 0); 0.05 restores the concentrated-
            # substrate regime so habits actually latch.
            fr = await svc.forget(strength_floor=0.05, dry_run=False, max_scan=5000)
            forget_passes += 1
            forget_removed += int(fr.get("removed", 0))

    # The table/charts sample BEFORE each forget so plasticity steady state
    # is visible; the store-bounded and compression claims must be judged
    # AFTER the terminal forget. These metrics are anti-phase on the same
    # sawtooth, so report both phases rather than pick one.
    _heartbeat_clear(enabled=show_progress)
    post = await _snapshot(svc, days, cons, dreams)
    return rows, post, probe, forget_passes, forget_removed


def _write_report(
    out: Path,
    rows: list[Snapshot],
    syn: list[str],
    *,
    cfg: MemoryConfig,
    days: int,
    sessions_per_day: int,
) -> None:
    meta = (
        f"{days} days x {sessions_per_day} sessions; reactor "
        f"self_learning={cfg.self_learning_enabled}, "
        f"consolidate_after_novel={cfg.reactor_consolidate_after_novel}, "
        f"dream_after_events={cfg.reactor_dream_after_events}; "
        f"virtual clock: 1 sim-day = {int(_SIM_DAY_SECONDS)}s aging"
    )
    out.write_text(render_html(rows, HEADERS, meta, syn), encoding="utf-8")


async def main(args: argparse.Namespace) -> int:
    if args.verbose:
        logging.disable(logging.NOTSET)
    _build_centroids(args.centroid_seed)
    show_progress = sys.stdout.isatty() and not args.quiet
    days = args.days
    sessions_per_day = args.sessions
    ep_per_session = args.episodes_per_session
    snapshot_every = args.snapshot_every
    forget_every = args.forget_every
    anchors_per_topic = args.anchors_per_topic
    core_topics = args.core_topics

    db = AsyncVectorDB(":memory:")
    # Single documented deviation: let the reactor consolidate without a
    # real 60 s maturation wait. Everything else is a production default
    # (self_learning_enabled, reactor thresholds, dream replay, etc.).
    cfg = MemoryConfig(
        consolidate_min_age_seconds=0.0,
        consolidate_min_retrievals=0,
    )
    _install_virtual_clock()
    svc = MemoryService(db, config=cfg, embed_fn=topical_embed)
    cons, dreams = _instrument_reactor(svc)
    rng = random.Random(args.seed)

    _print_preface(args, cfg, days, sessions_per_day)
    await _seed_anchors(svc, core_topics, anchors_per_topic)
    rows, post, probe, forget_passes, forget_removed = await _run_simulation(
        svc,
        days=days,
        sessions_per_day=sessions_per_day,
        ep_per_session=ep_per_session,
        anchors_per_topic=anchors_per_topic,
        snapshot_every=snapshot_every,
        forget_every=forget_every,
        rng=rng,
        cons=cons,
        dreams=dreams,
        show_progress=show_progress,
        quiet=args.quiet,
    )
    syn = _build_synthesis(
        rows,
        post,
        cons_count=cons[0],
        dreams_count=dreams[0],
        forget_passes=forget_passes,
        forget_removed=forget_removed,
        probe=probe,
    )
    if not args.quiet:
        print("\n# self-evolution observed (snapshot[0] -> snapshot[-1]):")
        for line in syn:
            print(f"#  {line}")

    _write_report(
        args.out,
        rows,
        syn,
        cfg=cfg,
        days=days,
        sessions_per_day=sessions_per_day,
    )
    if not args.quiet:
        print(f"\n# HTML report written: {args.out}")

    await db.close()
    if not args.quiet:
        print("\n# DONE - in-memory DB discarded, nothing persisted")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main(_parse_args())))
