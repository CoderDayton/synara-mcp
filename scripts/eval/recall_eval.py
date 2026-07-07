"""Ranked-IR recall eval for Synara MCP MemoryService.

Standalone, runnable like ``scripts/sim/self_learning_sim.py``. Seeds a
corpus of episodes tagged with known topics, runs held-out queries
through the real ``svc.recall`` pipeline (real embedder by default),
and reports MRR@k, Recall@k, Precision@k. Optionally runs a terminal
``svc.forget`` and re-evaluates so the pre/post delta is visible -- the
specific gap flagged in the ChatGPT critique: the sim only tracked a
binary "top hit on hot topic" probe, not a ranked-IR metric.

Ground truth: each stored episode carries ``tags=["topic-<n>"]``; a
recall hit is relevant iff its metadata.tags contains the query's
topic. This matches the sim's own ground-truth scheme.

Run:
    uv run --no-sync python scripts/eval/recall_eval.py
    uv run --no-sync python scripts/eval/recall_eval.py --no-forget
    uv run --no-sync python scripts/eval/recall_eval.py --k 5 --queries-per-topic 30
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import logging
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, median
from typing import Any

# Mute runtime chatter; the eval prints its own structured output.
logging.disable(logging.WARNING)

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from simplevecdb import AsyncVectorDB  # noqa: E402

from synara.features.embedding import EmbeddingConfig, build_embedder  # noqa: E402
from synara.features.memory import MemoryConfig  # noqa: E402
from synara.features.memory.service import MemoryService  # noqa: E402

# Distinct keyword vocabularies so the real embedder cluster boundaries
# fall along our ground-truth topics. Eight topics matches the sim's
# N_TOPICS so eval and sim share a workload shape.
_TOPIC_KEYWORDS: dict[int, list[str]] = {
    0: ["python", "decorator", "asyncio", "pytest", "venv", "typing", "lambda", "iterator"],
    1: ["sourdough", "yeast", "knead", "proof", "crumb", "baguette", "rye", "ferment"],
    2: ["galaxy", "nebula", "redshift", "quasar", "telescope", "parsec", "cosmology", "spectra"],
    3: ["sonata", "octave", "arpeggio", "cadenza", "tempo", "minor", "fugue", "polyphony"],
    4: ["mortgage", "amortization", "escrow", "appraisal", "deed", "lien", "equity", "closing"],
    5: ["kayak", "paddle", "rapids", "portage", "eddy", "wetsuit", "drysuit", "current"],
    6: [
        "mitochondria",
        "ribosome",
        "enzyme",
        "chromosome",
        "cytoplasm",
        "ATP",
        "membrane",
        "organelle",
    ],
    7: [
        "transformer",
        "attention",
        "embedding",
        "tokenizer",
        "gradient",
        "softmax",
        "fine-tune",
        "logits",
    ],
}

# Near-duplicate subtopics in two tight clusters (Python concurrency, NN
# training). Each topic = 5 shared cluster terms + 4 distinct terms, so a
# real embedder genuinely confuses siblings within a cluster (and the fake
# embedder inherits the same overlap through the shared terms). Used by
# --hard; the default _TOPIC_KEYWORDS above keeps topics far apart.
_HARD_CLUSTER_A = ["concurrency", "parallelism", "scheduling", "throughput", "workload"]
_HARD_CLUSTER_B = ["training", "gradient", "lossfunction", "optimization", "convergence"]
_HARD_TOPIC_KEYWORDS: dict[int, list[str]] = {
    0: [*_HARD_CLUSTER_A, "asyncio", "coroutine", "awaitable", "eventloop"],
    1: [*_HARD_CLUSTER_A, "threadpool", "mutex", "semaphore", "racecondition"],
    2: [*_HARD_CLUSTER_A, "multiprocess", "forkserver", "processpool", "ipcqueue"],
    3: [*_HARD_CLUSTER_A, "executor", "futureobj", "submitcall", "ascompleted"],
    4: [*_HARD_CLUSTER_B, "adamopt", "sgdopt", "momentumterm", "rmsprop"],
    5: [*_HARD_CLUSTER_B, "dropoutlayer", "weightdecay", "earlystopping", "l2penalty"],
    6: [*_HARD_CLUSTER_B, "warmupsteps", "cosineschedule", "stepdecay", "annealing"],
    7: [*_HARD_CLUSTER_B, "batchnorm", "layernorm", "standardize", "rescaling"],
}


def _active_keywords(hard: bool) -> dict[int, list[str]]:
    return _HARD_TOPIC_KEYWORDS if hard else _TOPIC_KEYWORDS


_FILLER = [
    "Notes from a working session.",
    "Quick recap for the team.",
    "Reminder for later review.",
    "Found this useful today.",
    "Following up on yesterday.",
]

# Generic terms with no single home topic. In --hard mode they dilute the
# topic signal (and, on the real embedder, pull unrelated episodes nearer).
_BRIDGE_KEYWORDS = [
    "overview",
    "summary",
    "review",
    "update",
    "analysis",
    "system",
    "process",
    "structure",
    "model",
    "pattern",
]


@dataclass(frozen=True)
class Sample:
    text: str
    topic: int


def _generate_episode(rng: random.Random, topic: int, kw_table: dict[int, list[str]]) -> str:
    kws = rng.sample(kw_table[topic], k=min(4, len(kw_table[topic])))
    filler = rng.choice(_FILLER)
    return f"{filler} Topic terms: {', '.join(kws)}."


def _generate_magnet(
    rng: random.Random, home: int, n_topics: int, kw_table: dict[int, list[str]], *, span: int = 4
) -> str:
    """A cross-topic 'magnet' episode: one keyword from each of ``span``
    consecutive topics. It matches many topics' queries, so during warm-up
    it is co-recalled with episodes from all of them and becomes a
    high-out-degree hub in the plasticity graph -- exactly what the
    spreading-activation fan-out cap is meant to contain. Tagged with its
    ``home`` topic only, so for the other spanned topics it is a
    relevant-looking distractor."""
    span = min(span, n_topics)
    spanned = [(home + d) % n_topics for d in range(span)]
    kws = [rng.choice(kw_table[t]) for t in spanned]
    kws.extend(rng.sample(_BRIDGE_KEYWORDS, k=min(2, len(_BRIDGE_KEYWORDS))))
    filler = rng.choice(_FILLER)
    return f"{filler} Cross-topic note: {', '.join(kws)}."


_QUERY_TERMS = 3


def _generate_query(
    rng: random.Random, topic: int, kw_table: dict[int, list[str]], *, exclude: set[str]
) -> str:
    pool = [k for k in kw_table[topic] if k not in exclude]
    if len(pool) < _QUERY_TERMS:
        pool = kw_table[topic]
    kws = rng.sample(pool, k=_QUERY_TERMS)
    return f"What do we know about {kws[0]}, {kws[1]}, and {kws[2]}?"


def _build_corpus(
    rng: random.Random,
    *,
    topics: list[int],
    episodes_per_topic: int,
    noise_fraction: float,
    kw_table: dict[int, list[str]],
    hard: bool = False,
    magnet_fraction: float = 0.1,
) -> list[tuple[Sample, float]]:
    """Return (sample, salience) pairs. A fraction get sub-floor salience
    so the terminal forget actually prunes something and the pre/post
    delta is meaningful. ``hard`` mode draws from the adjacent-subtopic
    table (heavy shared vocabulary, so cosine genuinely confuses topics
    within a cluster) and turns ``magnet_fraction`` of the high-salience
    episodes per topic into cross-topic hubs."""
    out: list[tuple[Sample, float]] = []
    n_topics = len(topics)
    n_noise = int(episodes_per_topic * noise_fraction)
    n_magnet = int(episodes_per_topic * magnet_fraction) if hard else 0
    for topic in topics:
        for i in range(episodes_per_topic):
            # Sub-floor salience so the terminal forget actually prunes.
            # memory_strength = salience * sum_k (1 + age_k)^-d, where
            # the sum has 1 + retrieval_count terms (forget.py:75-76).
            # The pre-eval recall warms each retrieved episode's
            # retrieval_count by 1; at ages~0 each term is ~1.0, so
            # strength ~= salience * (1 + retrieval_count). Pick a
            # noise salience that stays under strength_floor/2 = 0.025
            # even after a handful of accidental top-k hits.
            salience = 0.005 if i < n_noise else 0.6
            # Magnets come from the high-salience band so they survive
            # forget and persist as hubs.
            if hard and n_noise <= i < n_noise + n_magnet:
                text = _generate_magnet(rng, topic, n_topics, kw_table)
            else:
                text = _generate_episode(rng, topic, kw_table)
            out.append((Sample(text=text, topic=topic), salience))
    rng.shuffle(out)
    return out


def _build_queries(
    rng: random.Random,
    *,
    topics: list[int],
    queries_per_topic: int,
    kw_table: dict[int, list[str]],
) -> list[Sample]:
    # Held-out queries share topic vocabulary with the corpus (not strict
    # train/test text isolation). In hard mode they draw from the same
    # adjacent-subtopic table, so a query's terms overlap sibling topics.
    qs: list[Sample] = []
    for topic in topics:
        for _ in range(queries_per_topic):
            qs.append(
                Sample(text=_generate_query(rng, topic, kw_table, exclude=set()), topic=topic)
            )
    rng.shuffle(qs)
    return qs


def _hit_topics(hit: dict[str, Any]) -> set[int]:
    md = hit.get("metadata") or {}
    tags = md.get("tags") or []
    out: set[int] = set()
    for t in tags:
        if isinstance(t, str) and t.startswith("topic-"):
            try:
                out.add(int(t.split("-", 1)[1]))
            except ValueError:
                continue
    return out


@dataclass(frozen=True)
class EvalResult:
    mrr: float
    recall: float
    precision: float
    irrelevant_rate: float
    pack_chars: float
    n_queries: int
    n_zero_hits: int


@dataclass(frozen=True)
class SchemaHealth:
    total: int
    singletons: int
    median_ep: float
    pure: int  # schemas tagged with exactly one topic
    mixed: int  # schemas tagged with >=2 topics (over-absorption signal)
    untagged: int
    max_per_topic: int  # worst-fragmenting topic's schema count
    per_topic: dict[int, int]


async def _schema_health(svc: MemoryService) -> SchemaHealth:
    """Walk all schemas once and report bloat/over-absorption indicators.

    Unlike the sim's noisy-vs-tight split (which depends on the embedder
    being deliberately wide on selected topics), here every topic is
    real and equal. The acceptance signal is per-topic schema count:
    healthy clustering is ~1 schema per topic; under-clustering shows as
    high ``max_per_topic`` and singletons; over-absorption shows as
    ``mixed`` schemas fusing multiple topics into one gist.
    """
    rows = await svc.semantic.get_documents(filter_dict=None)
    total = len(rows)
    singletons = 0
    pure = 0
    mixed = 0
    untagged = 0
    per_topic: dict[int, int] = {}
    ep_counts: list[int] = []
    for _id, _text, md in rows:
        eps = list(md.get("source_episode_ids") or [])
        n_eps = len(eps)
        ep_counts.append(n_eps)
        if n_eps <= 1:
            singletons += 1
        topic_idxs: set[int] = set()
        for tag in md.get("tags") or []:
            if isinstance(tag, str) and tag.startswith("topic-") and tag[6:].isdigit():
                topic_idxs.add(int(tag[6:]))
        if not topic_idxs:
            untagged += 1
        elif len(topic_idxs) == 1:
            pure += 1
            for t in topic_idxs:
                per_topic[t] = per_topic.get(t, 0) + 1
        else:
            mixed += 1
            for t in topic_idxs:
                per_topic[t] = per_topic.get(t, 0) + 1
    return SchemaHealth(
        total=total,
        singletons=singletons,
        median_ep=float(median(ep_counts)) if ep_counts else 0.0,
        pure=pure,
        mixed=mixed,
        untagged=untagged,
        max_per_topic=max(per_topic.values()) if per_topic else 0,
        per_topic=per_topic,
    )


async def _eval(
    svc: MemoryService,
    queries: list[Sample],
    *,
    k: int,
    relevant_per_topic: int,
) -> EvalResult:
    mrrs: list[float] = []
    recalls: list[float] = []
    precisions: list[float] = []
    irrelevants: list[float] = []
    packs: list[float] = []
    zero = 0
    for q in queries:
        hits = await svc.recall(q.text, k=k)
        if not hits:
            zero += 1
            mrrs.append(0.0)
            recalls.append(0.0)
            precisions.append(0.0)
            irrelevants.append(0.0)
            packs.append(0.0)
            continue
        relevance = [1 if q.topic in _hit_topics(h) else 0 for h in hits]
        mrr = 0.0
        for i, r in enumerate(relevance):
            if r:
                mrr = 1.0 / (i + 1)
                break
        mrrs.append(mrr)
        # Recall@k is capped at 1.0 in case the corpus shrank below
        # relevant_per_topic after forgetting.
        recalls.append(min(1.0, sum(relevance) / max(1, relevant_per_topic)))
        precisions.append(sum(relevance) / k)
        n_hits = len(hits)
        # Irrelevant rate = fraction of *returned* hits that are off-topic
        # (distinct from Prec@k, which divides by k). Pack chars = total
        # content delivered back -- the recall token cost (~chars/4 tokens).
        irrelevants.append((n_hits - sum(relevance)) / n_hits)
        packs.append(float(sum(len(str(h.get("content") or "")) for h in hits)))
    return EvalResult(
        mrr=mean(mrrs),
        recall=mean(recalls),
        precision=mean(precisions),
        irrelevant_rate=mean(irrelevants),
        pack_chars=mean(packs),
        n_queries=len(queries),
        n_zero_hits=zero,
    )


def _fmt_health(label: str, pre: SchemaHealth, post: SchemaHealth | None) -> list[str]:
    def line(h: SchemaHealth) -> str:
        return (
            f"total={h.total} singletons={h.singletons} "
            f"median_ep={h.median_ep:.1f} "
            f"pure={h.pure} mixed={h.mixed} untagged={h.untagged} "
            f"max_per_topic={h.max_per_topic} "
            f"per_topic={{{', '.join(f'{t}:{c}' for t, c in sorted(h.per_topic.items()))}}}"
        )

    rows = [f"  pre  {line(pre)}"]
    if post is not None:
        rows.append(f"  post {line(post)}")
    return [label, *rows]


def _fmt(label: str, pre: EvalResult, post: EvalResult | None, k: int) -> list[str]:
    rows = [
        f"  MRR@{k:<3} pre={pre.mrr:.3f}"
        + (f"  post={post.mrr:.3f}  Δ={post.mrr - pre.mrr:+.3f}" if post else ""),
        f"  Rec@{k:<3} pre={pre.recall:.3f}"
        + (f"  post={post.recall:.3f}  Δ={post.recall - pre.recall:+.3f}" if post else ""),
        f"  Prec@{k:<3} pre={pre.precision:.3f}"
        + (f"  post={post.precision:.3f}  Δ={post.precision - pre.precision:+.3f}" if post else ""),
        f"  Irrel@{k:<2} pre={pre.irrelevant_rate:.3f}"
        + (
            f" post={post.irrelevant_rate:.3f} Δ={post.irrelevant_rate - pre.irrelevant_rate:+.3f}"
            if post
            else ""
        ),
        f"  Pack(ch)  pre={pre.pack_chars:.0f}"
        + (
            f"  post={post.pack_chars:.0f}  Δ={post.pack_chars - pre.pack_chars:+.0f}"
            if post
            else ""
        ),
        f"  queries={pre.n_queries} zero-hit-pre={pre.n_zero_hits}"
        + (f" zero-hit-post={post.n_zero_hits}" if post else ""),
    ]
    return [label, *rows]


def _make_fake_embed(kw_table: dict[int, list[str]]) -> Any:
    """Deterministic offline embedder: each topic keyword bumps its topic's
    dimension. Because adjacent subtopics in the hard table share most of
    their vocabulary, a shared term bumps every topic that owns it -- so the
    same cluster overlap that confuses the real embedder shows up here too,
    with no hand-tuned smear."""
    dim = 32

    async def fake_embed(text: str) -> list[float]:
        vec = [0.0] * dim
        for t, kws in kw_table.items():
            for kw in kws:
                if kw in text:
                    vec[t % dim] += 1.0
        h = hashlib.blake2b(text.encode(), digest_size=dim).digest()
        for i in range(dim):
            vec[i] += (h[i] - 128) / 1024.0
        return vec

    return fake_embed


async def _make_embed(args: argparse.Namespace) -> tuple[Any, Any]:
    """Build the embed fns once so a sweep can reuse them across configs.

    Returns ``(embed_fn, embed_batch_fn_or_None)``.
    """
    if args.fake_embedder:
        return _make_fake_embed(_active_keywords(args.hard)), None
    embedder = build_embedder(EmbeddingConfig())
    await embedder.warmup_async()
    return embedder.embed, embedder.embed_batch


def _make_cfg(args: argparse.Namespace, **overrides: Any) -> MemoryConfig:
    """Base eval config plus any CLI ablation overrides. Explicit
    ``overrides`` (e.g. a sweep grid row) win over the CLI flags."""
    base: dict[str, Any] = {
        "consolidate_min_age_seconds": 0.0,
        "consolidate_min_retrievals": 0,
        "consolidate_min_recurrence": args.min_recurrence,
    }
    if args.sr_omega_max is not None:
        base["sr_omega_max"] = args.sr_omega_max
    if args.spread_hops is not None:
        base["spreading_activation_hops"] = args.spread_hops
    if args.spread_decay is not None:
        base["spreading_activation_decay"] = args.spread_decay
    if args.spread_weight is not None:
        base["spreading_activation_weight"] = args.spread_weight
    if args.max_fanout is not None:
        base["spreading_activation_max_fanout"] = args.max_fanout
    base.update(overrides)
    return MemoryConfig(**base)


async def _build_and_encode(
    args: argparse.Namespace,
    cfg: MemoryConfig,
    embed_fn: Any,
    embed_batch_fn: Any,
    seed: int | None = None,
) -> tuple[Any, MemoryService, list[Sample]]:
    """Fresh in-memory service + encoded corpus + held-out queries. The
    seeded RNG keeps the corpus identical across sweep configs."""
    rng = random.Random(args.seed if seed is None else seed)
    topics = list(range(args.topics))
    kw_table = _active_keywords(args.hard)
    db = AsyncVectorDB(":memory:")
    if embed_batch_fn is None:
        svc = MemoryService(db, config=cfg, embed_fn=embed_fn)
    else:
        svc = MemoryService(db, config=cfg, embed_fn=embed_fn, embed_batch_fn=embed_batch_fn)
    corpus = _build_corpus(
        rng,
        topics=topics,
        episodes_per_topic=args.episodes_per_topic,
        noise_fraction=args.noise_fraction,
        kw_table=kw_table,
        hard=args.hard,
        magnet_fraction=args.magnet_fraction,
    )
    queries = _build_queries(
        rng, topics=topics, queries_per_topic=args.queries_per_topic, kw_table=kw_table
    )
    for i, (sample, salience) in enumerate(corpus):
        await svc.encode_episode(
            sample.text,
            session_id=f"corpus-{i % 16}",
            tags=[f"topic-{sample.topic}"],
            salience=salience,
        )
    # Drain the unconsolidated tail so schema-health and the sweep see the
    # full corpus (the reactor won't have fired on the last partial batch).
    if not args.no_consolidate:
        await svc.consolidate()
    return db, svc, queries


async def _warm(svc: MemoryService, queries: list[Sample], *, passes: int, k: int) -> None:
    """Re-run the query set so co-recalled episodes reinforce durable
    plasticity edges. Spreading activation reads durable ``weight`` only,
    so on a cold graph (``passes=0``) the fan-out cap is dormant -- warm
    first to exercise it."""
    for _ in range(passes):
        for q in queries:
            await svc.recall(q.text, k=k)


# hops x fan-out grid. ``fo0`` = unbounded (the cap off); higher hops make
# the fan-out cap actually bite. Edit freely; --warm-passes feeds it edges.
_SWEEP_GRID: list[dict[str, Any]] = [
    {
        "label": "h1 fo8 (default)",
        "spreading_activation_hops": 1,
        "spreading_activation_max_fanout": 8,
    },
    {
        "label": "h1 fo0 (cap off)",
        "spreading_activation_hops": 1,
        "spreading_activation_max_fanout": 0,
    },
    {"label": "h2 fo0", "spreading_activation_hops": 2, "spreading_activation_max_fanout": 0},
    {"label": "h2 fo8", "spreading_activation_hops": 2, "spreading_activation_max_fanout": 8},
    {"label": "h2 fo4", "spreading_activation_hops": 2, "spreading_activation_max_fanout": 4},
    {"label": "h3 fo0", "spreading_activation_hops": 3, "spreading_activation_max_fanout": 0},
    {"label": "h3 fo4", "spreading_activation_hops": 3, "spreading_activation_max_fanout": 4},
]


def _seed_list(args: argparse.Namespace) -> list[int]:
    if args.seeds:
        return [int(s) for s in args.seeds.split(",") if s.strip()]
    return [args.seed]


async def _run_sweep(args: argparse.Namespace) -> int:
    embed_fn, embed_batch_fn = await _make_embed(args)
    seeds = _seed_list(args)
    print(
        f"# recall_eval sweep: topics={args.topics} "
        f"episodes/topic={args.episodes_per_topic} queries/topic={args.queries_per_topic} "
        f"k={args.k} warm_passes={args.warm_passes} hard={args.hard} "
        f"seeds={seeds} embedder={'fake' if args.fake_embedder else 'real'}"
    )
    note = "  (mean over seeds; ±=half-range on Prec)" if len(seeds) > 1 else ""
    print(
        f"# {'config':<18}{'MRR':>8}{'Rec':>8}{'Prec':>8}{'Irrel':>8}"
        f"{'Pack':>9}{'edges':>8}{'maxW':>8}{note}"
    )
    for spec in _SWEEP_GRID:
        overrides = {kk: vv for kk, vv in spec.items() if kk != "label"}
        cfg = _make_cfg(args, **overrides)
        results: list[EvalResult] = []
        edges: list[float] = []
        maxws: list[float] = []
        for sd in seeds:
            db, svc, queries = await _build_and_encode(args, cfg, embed_fn, embed_batch_fn, seed=sd)
            await _warm(svc, queries, passes=args.warm_passes, k=args.k)
            results.append(
                await _eval(svc, queries, k=args.k, relevant_per_topic=args.episodes_per_topic)
            )
            pstats = await svc._plasticity.stats()
            edges.append(pstats["edges"])
            maxws.append(pstats["max_weight"])
            await db.close()
        precs = [r.precision for r in results]
        spread = f" ±{(max(precs) - min(precs)) / 2:.3f}" if len(seeds) > 1 else ""
        print(
            f"  {spec['label']!s:<18}{mean(r.mrr for r in results):>8.3f}"
            f"{mean(r.recall for r in results):>8.3f}{mean(precs):>8.3f}"
            f"{mean(r.irrelevant_rate for r in results):>8.3f}"
            f"{mean(r.pack_chars for r in results):>9.0f}"
            f"{mean(edges):>8.0f}{mean(maxws):>8.3f}{spread}"
        )
    return 0


async def _run(args: argparse.Namespace) -> int:
    if args.sweep:
        return await _run_sweep(args)

    embed_fn, embed_batch_fn = await _make_embed(args)
    cfg = _make_cfg(args)
    db, svc, queries = await _build_and_encode(args, cfg, embed_fn, embed_batch_fn)

    if not args.quiet:
        print(
            f"# recall_eval: topics={args.topics} "
            f"episodes={args.topics * args.episodes_per_topic} "
            f"(noise_fraction={args.noise_fraction}) "
            f"queries={len(queries)} k={args.k} warm_passes={args.warm_passes} "
            f"hard={args.hard} embedder={'fake' if args.fake_embedder else 'real'}"
        )

    await _warm(svc, queries, passes=args.warm_passes, k=args.k)
    health_pre = await _schema_health(svc)
    high_sal_per_topic = int(args.episodes_per_topic * (1 - args.noise_fraction))
    pre = await _eval(svc, queries, k=args.k, relevant_per_topic=args.episodes_per_topic)

    post: EvalResult | None = None
    health_post: SchemaHealth | None = None
    forget_removed = 0
    if not args.no_forget:
        fr = await svc.forget(strength_floor=0.05, dry_run=False, max_scan=10_000)
        forget_removed = int(fr.get("removed", 0))
        post = await _eval(svc, queries, k=args.k, relevant_per_topic=high_sal_per_topic)
        health_post = await _schema_health(svc)

    if not args.quiet:
        if post is not None:
            print(f"# forget: removed={forget_removed} (strength_floor=0.05)")
        for line in _fmt("# recall metrics:", pre, post, args.k):
            print(line)
        for line in _fmt_health("# schema health:", health_pre, health_post):
            print(line)

    await db.close()
    return 0


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--topics", type=int, default=8)
    p.add_argument("--episodes-per-topic", type=int, default=40)
    p.add_argument("--queries-per-topic", type=int, default=15)
    p.add_argument("--k", type=int, default=10)
    p.add_argument(
        "--noise-fraction",
        type=float,
        default=0.4,
        help="fraction of episodes encoded at sub-floor salience (pruned by forget)",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--seeds",
        type=str,
        default=None,
        help="comma-separated seeds to average the sweep over, e.g. 42,1,7 "
        "(sweep only; overrides --seed)",
    )
    p.add_argument("--no-forget", action="store_true", help="skip terminal forget + post-eval")
    p.add_argument(
        "--fake-embedder",
        action="store_true",
        help="use a deterministic in-process embedder (fast, no model download)",
    )
    p.add_argument(
        "--no-consolidate",
        action="store_true",
        help="skip the final explicit consolidate pass (reactor-only schemas)",
    )
    p.add_argument(
        "--min-recurrence",
        type=int,
        default=2,
        help=(
            "v2 candidate-promotion gate: a new stage-2 cluster's gist must "
            "recur within consolidate_schema_merge_distance across N "
            "consolidate passes before promotion to a durable schema. 1 = "
            "legacy one-pass behaviour."
        ),
    )
    p.add_argument("--sr-omega-max", type=float, default=None, help="override sr_omega_max")
    p.add_argument(
        "--spread-hops", type=int, default=None, help="override spreading_activation_hops"
    )
    p.add_argument(
        "--spread-decay", type=float, default=None, help="override spreading_activation_decay"
    )
    p.add_argument(
        "--spread-weight", type=float, default=None, help="override spreading_activation_weight"
    )
    p.add_argument(
        "--max-fanout",
        type=int,
        default=None,
        help="override spreading_activation_max_fanout (0 = unbounded, the cap off)",
    )
    p.add_argument(
        "--warm-passes",
        type=int,
        default=0,
        help="extra recall passes before measuring; builds durable plasticity edges "
        "so spreading activation and the fan-out cap are exercised",
    )
    p.add_argument(
        "--sweep",
        action="store_true",
        help="run a hops x fan-out grid (see _SWEEP_GRID) and tabulate metrics",
    )
    p.add_argument(
        "--hard",
        action="store_true",
        help="ambiguous workload: neighbour-topic contamination + cross-topic "
        "'magnet' hub episodes, so cosine is imperfect and spreading/fan-out "
        "have headroom to move the metrics",
    )
    p.add_argument(
        "--magnet-fraction",
        type=float,
        default=0.1,
        help="fraction of high-salience episodes per topic made cross-topic hubs (--hard only)",
    )
    p.add_argument("--quiet", action="store_true")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    return asyncio.run(_run(_parse_args(argv)))


if __name__ == "__main__":
    raise SystemExit(main())
