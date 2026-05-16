"""One-shot, no-persistence walkthrough of a memory's life.

Wires the real synara memory feature in-process against a ``:memory:``
DB (nothing touches disk) and narrates each stage:

  1. ENCODING        episodes enter the hippocampus
  2. RELATIONS       the Successor Representation (transition tally T,
                     discounted closure M) + Hebbian plasticity edges
                     that form *between* co-occurring episodes
  3. RECALL          how those relations re-rank retrieval
  4. CONSOLIDATION   a cluster folds into a neocortical semantic schema
  5. REFLECTION      session-scoped synthesis
  6. FORGETTING      power-law decay → strength-ranked pruning

Run:  uv run --no-sync python /tmp/memory_lifecycle_demo.py

Consolidation is normally gated behind a 60 s maturation window and a
≥1-retrieval requirement; this demo lowers both to 0 so the full
lifecycle is observable in one synchronous run. Everything else uses
production defaults.
"""

from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import AsyncIterator
from typing import Any

# Silence the server's ctx.info/debug stream so the narration is clean.
logging.disable(logging.CRITICAL)

from fastmcp import Client, FastMCP  # noqa: E402
from fastmcp.server.lifespan import lifespan  # noqa: E402
from simplevecdb import AsyncVectorDB, Quantization  # noqa: E402

from synara.features import memory  # noqa: E402
from synara.features.embedding import EmbeddingConfig, build_embedder  # noqa: E402
from synara.features.memory import MemoryConfig  # noqa: E402

W = 70


def hr(title: str) -> None:
    print(f"\n{'=' * W}\n {title}\n{'=' * W}")


def show(label: str, data: object) -> None:
    print(f"  {label}: {json.dumps(data, default=str, indent=2)[:850]}")


async def main() -> None:  # noqa: PLR0915 - linear narrated walkthrough
    db = AsyncVectorDB(":memory:", quantization=Quantization.INT8)
    embedder = build_embedder(EmbeddingConfig())

    @lifespan
    async def app_lifespan(_s: FastMCP) -> AsyncIterator[dict[str, Any]]:
        try:
            yield {"db": db, "embedder": embedder}
        finally:
            await embedder.aclose()
            await db.close()

    mcp = FastMCP(name="memory-lifecycle-demo", lifespan=app_lifespan)
    # Only deviation from prod defaults: make consolidation observable
    # without a real 60 s wall-clock wait.
    cfg = MemoryConfig(consolidate_min_age_seconds=0.0, consolidate_min_retrievals=0)
    service = memory.register(mcp, db, config=cfg, embedder=embedder)

    async with Client(mcp) as c:

        async def call(tool: str, **kw: object) -> Any:
            return (await c.call_tool(tool, kw)).data

        # ------------------------------------------------------------------
        hr("1. ENCODING — episodes enter the hippocampus")
        # One coherent debugging thread in a single session. Episodes
        # encoded inside the same session window co-occur, which is what
        # builds the relational structure inspected in stage 2.
        thread = [
            "Investigating a memory leak in the websocket connection pool",
            "Found it: ConnectionPool.release() never clears the read buffer",
            "Patched release() to drop the buffer; leak gone under load test",
        ]
        tid: list[int] = []
        for text in thread:
            d = await call(
                "store_episode",
                content=text,
                session_id="sess-A",
                tags=["bug", "websocket"],
                salience=0.7,
            )
            tid.append(d["id"])
            print(f"  encoded id={d['id']:>2}  «{text}»")

        # Unrelated, low-salience aside in another session — our future
        # forget candidate. Deliberately never recalled, so its strength
        # stays at the salience floor.
        d = await call(
            "store_episode",
            content="Renamed a local variable from tmp to scratch",
            session_id="sess-B",
            tags=["chore"],
            salience=0.1,
        )
        chore_id = d["id"]
        print(f"  encoded id={chore_id:>2}  (low-salience chore, session B)")
        show("stats", await call("memory_stats"))

        # ------------------------------------------------------------------
        hr("2. RELATIONS — Successor Representation + plasticity graph")
        # Recall once so the co-retrieval Hebbian rule fires and lays down
        # plasticity edges between the thread's episodes.
        await call("recall_episodes", query="websocket leak fix", session_id="sess-A", k=4)
        await service._ensure_sr_loaded()
        sr = service._sr

        print("\n  Successor transition tally T[i][j]  (episode i seen,")
        print("  then j within the session window — durable, kind='sr'):")
        for i in tid:
            row = {j: round(v, 2) for j, v in sr._T_counts.get(i, {}).items()}
            if row:
                print(f"    T[{i}] -> {row}")

        print("\n  Discounted closure M = (I - gamma*T)^-1 style TD pass — the")
        print("  multi-step relational prior actually used at recall:")
        for i in tid:
            row = {j: round(v, 3) for j, v in sr._M.get(i, {}).items() if v}
            if row:
                print(f"    M[{i}] -> {row}")

        print("\n  Hebbian plasticity edge weights (co-recall potentiation):")
        for a in tid:
            for b in tid:
                if a != b:
                    w = await service._plasticity.edge_weight(a, b)
                    if w:
                        print(f"    W[{a}->{b}] = {w:.4f}")
        print("\n  ⇒ episodes are no longer independent points; they form a")
        print("    directed graph the recall ranker reads as a prior.")

        # ------------------------------------------------------------------
        hr("3. RECALL — relations re-rank retrieval")
        print(f"  Query strongly matches only episode {tid[0]}. Episodes {tid[1:]} have")
        print("  weak lexical overlap with the query but are pulled up by")
        print("  the SR/plasticity prior. Printed `distance` is RAW cosine;")
        print("  result *order* is the blended rank — so a larger distance")
        print("  appearing above a smaller one is the relational boost:\n")
        res = await call(
            "recall_episodes",
            query="websocket connection pool memory leak",
            session_id="sess-A",
            k=4,
        )
        for rank, r in enumerate(res, 1):
            print(
                f"   #{rank}  id={r['id']:>2}  cos_dist={r['distance']:.3f}  «{r['content'][:48]}…»"
            )

        # ------------------------------------------------------------------
        hr("4. CONSOLIDATION — hippocampus → neocortical schema")
        print("  Add more thematically-close episodes so a cluster forms,")
        print("  then consolidate (SWR replay → schema integration):\n")
        for text in [
            "Another leak: HTTP client sessions not closed on timeout",
            "Connection cleanup rule: always release buffers in a finally block",
            "Audited the gRPC channel pool for the same buffer-retention bug",
        ]:
            d = await call(
                "store_episode", content=text, session_id="sess-A", tags=["bug"], salience=0.6
            )
            print(f"  encoded id={d['id']:>2}  «{text}»")

        schemas = await call("consolidate_episodes", min_cluster_size=2)
        show("schemas formed (episodic → semantic)", schemas)
        show("stats after consolidation", await call("memory_stats"))

        print("\n  The gist is now retrievable as SEMANTIC memory, decoupled")
        print("  from any single episode:")
        sem = await call(
            "recall_semantic_memory", query="how do I prevent connection resource leaks", k=3
        )
        for s in sem:
            print(f"    «{str(s.get('content'))[:80]}…»")

        # ------------------------------------------------------------------
        hr("5. REFLECTION — session-scoped synthesis")
        refl = await call(
            "reflect_session", session_id="sess-A", query="what did we learn about leaks?", k=3
        )
        print("  schemas:", json.dumps(refl.get("schemas"), default=str)[:300])
        print("  recent :", [e["id"] for e in refl.get("recent_episodes", [])])

        # ------------------------------------------------------------------
        hr("6. FORGETTING — power-law decay & strength-ranked pruning")
        print("  S(t) = salience · Σ_k (1 + Δt_k)^(-d). Every recall above")
        print("  re-touched the thread episodes (each retrieval appends an")
        print("  access time, lifting S), so they are now strong. We add")
        print("  one isolated, never-recalled, near-zero-salience trace —")
        print("  the textbook decay candidate:\n")
        d = await call(
            "store_episode",
            content="Bumped the copyright year in the file header",
            session_id="sess-Z",
            tags=["chore"],
            salience=0.01,
        )
        faint_id = d["id"]
        print(f"  encoded id={faint_id:>2}  (salience=0.01, never recalled)")

        plan = await call("forget_episodes", strength_floor=0.06, dry_run=True)
        show("forget plan (dry run, floor=0.06)", plan)

        print("\n  Commit the prune — the faint unconsolidated trace is")
        print("  dropped; consolidated & strong episodes survive:\n")
        culled = await call("forget_episodes", strength_floor=0.06, dry_run=False)
        show("forget result", culled)
        show("final stats", await call("memory_stats"))

        hr("DONE — in-memory DB discarded, nothing persisted")


if __name__ == "__main__":
    asyncio.run(main())
