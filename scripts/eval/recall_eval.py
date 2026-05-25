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

_FILLER = [
    "Notes from a working session.",
    "Quick recap for the team.",
    "Reminder for later review.",
    "Found this useful today.",
    "Following up on yesterday.",
]


@dataclass(frozen=True)
class Sample:
    text: str
    topic: int


def _generate_episode(rng: random.Random, topic: int) -> str:
    kws = rng.sample(_TOPIC_KEYWORDS[topic], k=4)
    filler = rng.choice(_FILLER)
    return f"{filler} Topic terms: {', '.join(kws)}."


_QUERY_TERMS = 3


def _generate_query(rng: random.Random, topic: int, *, exclude: set[str]) -> str:
    pool = [k for k in _TOPIC_KEYWORDS[topic] if k not in exclude]
    if len(pool) < _QUERY_TERMS:
        pool = _TOPIC_KEYWORDS[topic]
    kws = rng.sample(pool, k=_QUERY_TERMS)
    return f"What do we know about {kws[0]}, {kws[1]}, and {kws[2]}?"


def _build_corpus(
    rng: random.Random,
    *,
    topics: list[int],
    episodes_per_topic: int,
    noise_fraction: float,
) -> list[tuple[Sample, float]]:
    """Return (sample, salience) pairs. A fraction get sub-floor salience
    so the terminal forget actually prunes something and the pre/post
    delta is meaningful."""
    out: list[tuple[Sample, float]] = []
    for topic in topics:
        for i in range(episodes_per_topic):
            text = _generate_episode(rng, topic)
            # Sub-floor salience so the terminal forget actually prunes.
            # memory_strength = salience * sum_k (1 + age_k)^-d, where
            # the sum has 1 + retrieval_count terms (forget.py:75-76).
            # The pre-eval recall warms each retrieved episode's
            # retrieval_count by 1; at ages~0 each term is ~1.0, so
            # strength ~= salience * (1 + retrieval_count). Pick a
            # noise salience that stays under strength_floor/2 = 0.025
            # even after a handful of accidental top-k hits.
            salience = 0.005 if i < int(episodes_per_topic * noise_fraction) else 0.6
            out.append((Sample(text=text, topic=topic), salience))
    rng.shuffle(out)
    return out


def _build_queries(
    rng: random.Random,
    *,
    topics: list[int],
    queries_per_topic: int,
) -> list[Sample]:
    # Reserve the first 3 keywords per topic for queries; corpus draws
    # from the full list. Overlap is fine -- this models held-out queries
    # that share topic vocabulary, not strict train/test text isolation.
    qs: list[Sample] = []
    for topic in topics:
        for _ in range(queries_per_topic):
            qs.append(Sample(text=_generate_query(rng, topic, exclude=set()), topic=topic))
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
    zero = 0
    for q in queries:
        hits = await svc.recall(q.text, k=k)
        if not hits:
            zero += 1
            mrrs.append(0.0)
            recalls.append(0.0)
            precisions.append(0.0)
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
    return EvalResult(
        mrr=mean(mrrs),
        recall=mean(recalls),
        precision=mean(precisions),
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
            f"per_topic={{{', '.join(f'{t}:{c}' for t, c in sorted(pre.per_topic.items()))}}}"
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
        f"  queries={pre.n_queries} zero-hit-pre={pre.n_zero_hits}"
        + (f" zero-hit-post={post.n_zero_hits}" if post else ""),
    ]
    return [label, *rows]


async def _run(args: argparse.Namespace) -> int:
    rng = random.Random(args.seed)
    topics = list(range(args.topics))

    db = AsyncVectorDB(":memory:")
    cfg = MemoryConfig(
        consolidate_min_age_seconds=0.0,
        consolidate_min_retrievals=0,
        consolidate_min_recurrence=args.min_recurrence,
    )
    if args.fake_embedder:
        # Deterministic offline path. Mirrors the sim's topical_embed
        # idea: each topic gets one fixed direction in a small space,
        # plus a tiny per-text jitter. Useful for fast iteration where
        # you don't want to pay the model load.
        dim = 32

        async def fake_embed(text: str) -> list[float]:
            # Pull topic from the keyword overlap; otherwise hash.
            vec = [0.0] * dim
            for t, kws in _TOPIC_KEYWORDS.items():
                for kw in kws:
                    if kw in text:
                        vec[t % dim] += 1.0
            h = hashlib.blake2b(text.encode(), digest_size=dim).digest()
            for i in range(dim):
                vec[i] += (h[i] - 128) / 1024.0
            return vec

        svc = MemoryService(db, config=cfg, embed_fn=fake_embed)
    else:
        embedder = build_embedder(EmbeddingConfig())
        await embedder.warmup_async()
        svc = MemoryService(
            db,
            config=cfg,
            embed_fn=embedder.embed,
            embed_batch_fn=embedder.embed_batch,
        )

    corpus = _build_corpus(
        rng,
        topics=topics,
        episodes_per_topic=args.episodes_per_topic,
        noise_fraction=args.noise_fraction,
    )
    queries = _build_queries(rng, topics=topics, queries_per_topic=args.queries_per_topic)

    if not args.quiet:
        print(
            f"# recall_eval: topics={len(topics)} "
            f"episodes={len(corpus)} (noise_fraction={args.noise_fraction}) "
            f"queries={len(queries)} k={args.k} "
            f"embedder={'fake' if args.fake_embedder else 'real'}"
        )

    for i, (sample, salience) in enumerate(corpus):
        await svc.encode_episode(
            sample.text,
            session_id=f"corpus-{i % 16}",
            tags=[f"topic-{sample.topic}"],
            salience=salience,
        )

    # Drain any pending unconsolidated tail. The reactor fires during
    # encode (self_learning_enabled=True by default), but the last batch
    # of <consolidate_after_novel novel episodes won't have triggered;
    # this explicit pass guarantees the schema-health metrics see the
    # full corpus.
    if not args.no_consolidate:
        await svc.consolidate()
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
    p.add_argument("--quiet", action="store_true")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    return asyncio.run(_run(_parse_args(argv)))


if __name__ == "__main__":
    raise SystemExit(main())
