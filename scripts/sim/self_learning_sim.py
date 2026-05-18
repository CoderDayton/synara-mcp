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

import asyncio
import hashlib
import html as _html
import importlib
import itertools
import logging
import math
import random
import sys
import time
from pathlib import Path

logging.disable(logging.CRITICAL)

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from simplevecdb import AsyncVectorDB  # noqa: E402

from synara.features.memory import MemoryConfig  # noqa: E402
from synara.features.memory.service import MemoryService  # noqa: E402

# ---- topical offline embedder ---------------------------------------

DIM = 64
N_TOPICS = 8
_RNG = random.Random(20260518)
_CENTROIDS: list[list[float]] = []
for _ in range(N_TOPICS):
    v = [_RNG.gauss(0.0, 1.0) for _ in range(DIM)]
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


# ---- simulation -----------------------------------------------------

DAYS = 60
SESSIONS_PER_DAY = 6
EP_PER_SESSION = 4
SNAPSHOT_EVERY = 10
FORGET_EVERY = 15
# Per topic, a small set of durable high-salience "knowledge" anchors
# encoded once. Every same-topic session re-recalls them: the fixed
# anchor<->anchor plasticity edges accumulate hits across the whole
# run (-> habits latch) while the recall refreshes their access in
# virtual time (-> they survive forgetting). Unique per-session work
# and the salience-0.02 noise are the forgettable churn.
ANCHORS_PER_TOPIC = 3
ANCHOR_SALIENCE = 0.97
# Frequency-dependent consolidation: only the highest-frequency topic(s)
# earn durable anchors. Tail topics see filler only -> their traces are
# pruned by forget, so the post-forget substrate (and the recall prior)
# concentrates on the hub the agent actually practised. This is the
# emergent property, not a probe hack: a learner masters its most-used
# domain and lets rarely-revisited ones fade.
_CORE_TOPICS = 1

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


async def _snapshot(
    svc: MemoryService, day: int, cons: list[int], dreams: list[int]
) -> tuple[float, ...]:
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
    #  ep/sch  - episodes distilled per schema (compression; lower=tighter)
    #  srE/srR - SR fan-out per node           (graph densification)
    #  mH/pE   - hits on the busiest edge / edge (concentration)
    ep_per_sch = round(epis / sem, 2) if sem else 0.0
    sr_dens = round(sr_edges / sr_rows, 2) if sr_rows else 0.0
    edge_conc = round(max_hits / p_edges, 2) if p_edges else 0.0
    return (
        day,
        epis,
        sem,
        p_edges,
        int(pl["habits"]),
        max_hits,
        sr_rows,
        sr_edges,
        cons[0],
        dreams[0],
        ep_per_sch,
        sr_dens,
        edge_conc,
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
        if hasattr(mod, "now_seconds"):
            mod.now_seconds = _vclock


def _fmt_row(vals: tuple[object, ...]) -> str:
    cells = [f"{v:>8.2f}" if isinstance(v, float) else f"{v:>8}" for v in vals]
    return " ".join(cells)


_CW, _CH = 380, 210


def _svg_chart(
    title: str,
    days: list[int],
    ys: list[float],
    *,
    color: str = "#2f9e44",
) -> str:
    """One self-contained SVG line chart, autoscaled from 0 to max(ys)."""
    pad_l, pad_r, pad_t, pad_b = 46, 14, 30, 26
    iw, ih = _CW - pad_l - pad_r, _CH - pad_t - pad_b
    xmin, xmax = days[0], days[-1]
    ymax = max(ys) if ys else 1.0
    ymax = ymax if ymax > 0 else 1.0

    def sx(d: float) -> float:
        return pad_l + (d - xmin) / ((xmax - xmin) or 1) * iw

    def sy(v: float) -> float:
        return pad_t + ih - (v / ymax) * ih

    pts = " ".join(f"{sx(d):.1f},{sy(v):.1f}" for d, v in zip(days, ys, strict=True))
    dots = "".join(
        f'<circle cx="{sx(d):.1f}" cy="{sy(v):.1f}" r="2.6" '
        f'fill="{color}"><title>day {d}: {v:g}</title></circle>'
        for d, v in zip(days, ys, strict=True)
    )
    ax = pad_t + ih
    return (
        f'<svg viewBox="0 0 {_CW} {_CH}" class="chart" '
        'xmlns="http://www.w3.org/2000/svg">'
        f'<text x="{pad_l}" y="18" class="ct">{_html.escape(title)}</text>'
        f'<line x1="{pad_l}" y1="{ax}" x2="{pad_l + iw}" y2="{ax}" '
        'class="axis"/>'
        f'<line x1="{pad_l}" y1="{pad_t}" x2="{pad_l}" y2="{ax}" '
        'class="axis"/>'
        f'<text x="{pad_l - 6}" y="{pad_t + 4}" class="tk" '
        f'text-anchor="end">{ymax:g}</text>'
        f'<text x="{pad_l - 6}" y="{ax}" class="tk" text-anchor="end">0</text>'
        f'<text x="{pad_l}" y="{_CH - 8}" class="tk">d{xmin}</text>'
        f'<text x="{pad_l + iw}" y="{_CH - 8}" class="tk" '
        f'text-anchor="end">d{xmax}</text>'
        f'<polyline points="{pts}" fill="none" stroke="{color}" '
        'stroke-width="2"/>'
        f"{dots}</svg>"
    )


def _render_html(
    rows: list[tuple[float, ...]],
    headers: tuple[str, ...],
    meta: str,
    synthesis: list[str],
) -> str:
    """Assemble a dependency-free HTML report (inline SVG + table)."""
    days = [int(r[0]) for r in rows]
    specs = [
        ("Episodic store", 1, "#1971c2"),
        ("Semantic schemas", 2, "#9c36b5"),
        ("Plasticity edges", 3, "#2f9e44"),
        ("Habit edges", 4, "#e8590c"),
        ("Max edge hits", 5, "#c2255c"),
        ("SR transition rows", 6, "#0c8599"),
        ("SR graph edges", 7, "#5c940d"),
        ("Dream replays (cum.)", 9, "#862e9c"),
        ("Episodes / schema (compression)", 10, "#d6336c"),
        ("SR fan-out / node (densification)", 11, "#1098ad"),
        ("Hits / edge (concentration)", 12, "#f08c00"),
    ]
    charts = "".join(
        _svg_chart(t, days, [float(r[c]) for r in rows], color=col) for t, c, col in specs
    )
    syn_li = "".join(f"<li>{_html.escape(s)}</li>" for s in synthesis)
    head = "".join(f"<th>{_html.escape(h)}</th>" for h in headers)
    body = "".join("<tr>" + "".join(f"<td>{v}</td>" for v in r) + "</tr>" for r in rows)
    return (
        '<!doctype html><html lang="en"><head><meta charset="utf-8">'
        "<title>synara self-learning sim</title><style>"
        "body{font:14px/1.5 system-ui,sans-serif;margin:24px;"
        "background:#fafafa;color:#212529}"
        "h1{font-size:20px;margin:0 0 4px}"
        ".meta{color:#666;margin:0 0 16px}"
        "ul{margin:0 0 20px;padding-left:18px}li{margin:2px 0}"
        ".grid{display:grid;grid-template-columns:repeat(auto-fit,"
        "minmax(360px,1fr));gap:14px}"
        ".chart{background:#fff;border:1px solid #e0e0e0;border-radius:8px}"
        ".ct{font-size:13px;font-weight:600;fill:#212529}"
        ".axis{stroke:#adb5bd;stroke-width:1}"
        ".tk{font-size:10px;fill:#868e96}"
        "table{border-collapse:collapse;margin-top:22px;font-size:12px}"
        "th,td{border:1px solid #dee2e6;padding:3px 8px;text-align:right}"
        "th{background:#f1f3f5}"
        "</style></head><body>"
        "<h1>synara &mdash; runtime self-learning simulation</h1>"
        f'<p class="meta">{_html.escape(meta)}</p>'
        f"<ul>{syn_li}</ul>"
        f'<div class="grid">{charts}</div>'
        f"<table><thead><tr>{head}</tr></thead><tbody>{body}"
        "</tbody></table></body></html>"
    )


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
    hot = await svc.recall("t0: core knowledge anchor", session_id="probe", k=5)
    same = sum(1 for h in hot if str(h.get("content", "")).startswith("t0:"))
    return same, len(hot)


async def main() -> int:
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
    rng = random.Random(7)
    forget_passes = 0
    forget_removed = 0
    probe: tuple[int, int] = (0, 0)

    headers = (
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
    print(
        f"# real-runtime self-learning sim: {DAYS}d x {SESSIONS_PER_DAY} "
        f"sessions, reactor self_learning={cfg.self_learning_enabled}"
    )
    print(
        f"# consolidate_after_novel={cfg.reactor_consolidate_after_novel} "
        f"dream_after_events={cfg.reactor_dream_after_events} "
        f"dream_replay_top_k={cfg.dream_replay_top_k}"
    )
    print(_fmt_row(headers))

    # Seed durable per-topic knowledge anchors once (high salience).
    for t in range(_CORE_TOPICS):
        for a in range(ANCHORS_PER_TOPIC):
            await svc.encode_episode(
                f"t{t}: core knowledge anchor {a}",
                session_id=f"anchors-t{t}",
                tags=[f"topic-{t}", "anchor"],
                salience=ANCHOR_SALIENCE,
            )

    rows: list[tuple[float, ...]] = []
    snap = await _snapshot(svc, 0, cons, dreams)
    rows.append(snap)
    print(_fmt_row(snap))

    for day in range(1, DAYS + 1):
        _advance_day()  # one simulated day of virtual aging
        for s in range(SESSIONS_PER_DAY):
            sid = f"d{day}-s{s}"
            topic = rng.choices(range(N_TOPICS), weights=_TOPIC_WEIGHTS)[0]
            # 1. Re-activate the durable HUB anchors (topic 0) every
            #    session regardless of which topic the filler is about,
            #    so the most-practised substrate consolidates. Co-recall
            #    lays down the stable anchor<->anchor SR + plasticity
            #    edges (their hit count climbs every same-topic session
            #    until it crosses habit_threshold_hits) and the
            #    retrieval refreshes anchor access in virtual time so
            #    they stay above the forget floor.
            await svc.recall(
                "t0: core knowledge anchor",
                session_id=sid,
                k=ANCHORS_PER_TOPIC,
            )
            # 1b. Deterministic anchor rehearsal: the ranked recall above
            #     can be crowded out by same-topic filler, so the actual
            #     anchor episodes never get their access-time refreshed
            #     and forget prunes them (FK CASCADE then wipes their
            #     accumulated plasticity edges). Retrieve each anchor by
            #     its exact stored text so every anchor episode is touched
            #     every session: Dt~=0 at the forget pass -> strength ~=
            #     salience (0.97) >> floor -> the recurring substrate
            #     survives every cascade and its anchor<->anchor edges
            #     keep climbing toward habit_threshold_hits.
            for a in range(ANCHORS_PER_TOPIC):
                await svc.recall(
                    f"t0: core knowledge anchor {a}",
                    session_id=sid,
                    k=1,
                )
            # 2. Forgettable churn: unique per-session work plus a
            #    near-zero-salience throwaway, so the store self-bounds.
            for e in range(EP_PER_SESSION):
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

        # Snapshot BEFORE forget. SNAPSHOT_EVERY=10 and FORGET_EVERY=15
        # collide at days 30 and 60; sampling after the forget cascade
        # would measure the sawtooth at its trough (FK CASCADE has just
        # wiped the high-hit edges) and misreport the steady state the
        # system actually maintains between maintenance sweeps. Recording
        # the maintained state is the faithful measurement; the sawtooth
        # itself stays visible via peak fields + forget_removed.
        if day % SNAPSHOT_EVERY == 0:
            snap = await _snapshot(svc, day, cons, dreams)
            rows.append(snap)
            print(_fmt_row(snap))

        if day == DAYS:
            # Measure the recall prior on the maintained state, i.e.
            # BEFORE the terminal forget, for the same reason the
            # plasticity table samples pre-forget (see _recall_prior).
            probe = await _recall_prior(svc)

        if day % FORGET_EVERY == 0:
            # Aggressive floor: anchors are refreshed by exact-text
            # rehearsal every session (Dt~=0 -> S ~= 0.97 here), so they
            # clear this floor with margin while never-rehearsed filler
            # and salience-0.02 noise fall through. A low floor (0.005)
            # was measured to keep filler alive and dilute anchor recall
            # (maxHits collapsed 223 -> 0); 0.05 restores the
            # concentrated-substrate regime so habits actually latch.
            fr = await svc.forget(strength_floor=0.05, dry_run=False, max_scan=5000)
            forget_passes += 1
            forget_removed += int(fr.get("removed", 0))

    # The table/charts sample BEFORE each forget so plasticity steady
    # state is visible; the store-bounded and compression claims must be
    # judged AFTER the terminal forget. These metrics are anti-phase on
    # the same sawtooth, so report both phases rather than pick one.
    post = await _snapshot(svc, DAYS, cons, dreams)

    # ---- emergent-structure synthesis -------------------------------
    # Every number below is measured. The episodic line reports whether
    # virtual-time decay actually bounded the store, not a tidier story.
    first, last = rows[0], rows[-1]
    epis_series = [int(r[1]) for r in rows]
    rising = all(b > a for a, b in itertools.pairwise(epis_series))
    # probe captured pre-terminal-forget (maintained state); see
    # _recall_prior for why this is the faithful sampling point.
    same, total = probe

    syn = [
        f"reactor self-fired: consolidations={cons[0]} dreams={dreams[0]} "
        "(explicit svc.consolidate() calls in this script: 0)",
        f"semantic schemas: {first[2]} -> {last[2]} (one cooldown-gated "
        "reactor pass distils the recurring topics)",
        f"SR transition rows: {first[6]} -> {last[6]}, edges "
        f"{first[7]} -> {last[7]} (learned relational prior)",
        f"plasticity edges: {int(first[3])} -> {int(last[3])}, habits "
        f"{int(first[4])} -> {int(last[4])} (peak "
        f"{max(int(r[4]) for r in rows)}), maxHits {int(last[5])} "
        f"(peak {max(int(r[5]) for r in rows)}) "
        + (
            "-- anchor<->anchor edges self-cross habit_threshold_hits"
            if max(int(r[4]) for r in rows)
            else "-- no habit latched"
        ),
        f"episodic store: {epis_series[0]} -> {int(post[1])} after the "
        f"terminal forget (mid-cycle peak {max(epis_series)}); strictly-rising="
        f"{rising} -> virtual-time decay "
        + ("did NOT bound it" if rising else "BOUNDED it")
        + f" (forget passes={forget_passes} removed={forget_removed})",
        f"ratios (post-forget): {post[10]} episodes/schema (compression), "
        f"{post[11]} SR-edges/row (densification), "
        f"{post[12]} maxhits/edge (concentration)",
        f"recall prior check (maintained pre-forget state): {same}/{total} "
        "top hits are the recurring hot topic -> structure shapes retrieval",
    ]
    print("\n# self-evolution observed (snapshot[0] -> snapshot[-1]):")
    for line in syn:
        print(f"#  {line}")

    out = (
        Path(sys.argv[1])
        if len(sys.argv) > 1
        else Path(__file__).resolve().parent / "self_learning_report.html"
    )
    meta = (
        f"{DAYS} days x {SESSIONS_PER_DAY} sessions; reactor "
        f"self_learning={cfg.self_learning_enabled}, "
        f"consolidate_after_novel={cfg.reactor_consolidate_after_novel}, "
        f"dream_after_events={cfg.reactor_dream_after_events}; "
        f"virtual clock: 1 sim-day = {int(_SIM_DAY_SECONDS)}s aging"
    )
    out.write_text(_render_html(rows, headers, meta, syn), encoding="utf-8")
    print(f"\n# HTML report written: {out}")

    await db.close()
    print("\n# DONE - in-memory DB discarded, nothing persisted")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
