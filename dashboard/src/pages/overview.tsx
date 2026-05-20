import { lazy, Suspense } from "react";
import { ArrowUpRight } from "lucide-react";
import { useHealth, useMemories, useStats } from "@/lib/queries";
import { ErrorState, Loading } from "@/components/common/states";
import { Panel } from "@/components/common/panel";
import { Skeleton } from "@/components/ui/skeleton";

const StoreChart = lazy(() => import("@/components/overview/store-chart"));

const TOOLS: Array<[string, string]> = [
  ["store_episode", "Encode an episodic trace"],
  ["recall_episodes", "Cross-session episodic recall"],
  ["consolidate_episodes", "Cluster traces into schemas"],
  ["forget_episodes", "Power-law decay prune"],
  ["reflect_session", "Summarise a session"],
  ["store_semantic_memory", "Write a semantic memory"],
  ["recall_semantic_memory", "Semantic memory recall"],
  ["memory_stats", "Store + tunable snapshot"],
];

const PIPELINE: Array<{ label: string; sub: string }> = [
  { label: "Encode", sub: "store_episode → embedding" },
  { label: "Hippocampus", sub: "episodic · successor rep · plasticity" },
  { label: "Consolidate", sub: "cluster → schemas" },
  { label: "Neocortex", sub: "durable semantic memory" },
];

const MCP_SNIPPET = `{
  "mcpServers": {
    "synara": { "command": "uvx", "args": ["synara-mcp"] }
  }
}`;

export default function Overview() {
  const stats = useStats();
  const health = useHealth();
  const recent = useMemories({ kind: "episodic", limit: 6 });

  if (stats.isLoading || health.isLoading) return <Loading />;
  if (stats.error) return <ErrorState error={stats.error} />;

  const s = stats.data;
  const h = health.data;
  const ep = s?.episodic_count ?? 0;
  const sem = s?.semantic_count ?? 0;
  const total = ep + sem;
  const hasData = total > 0;
  const chart = [
    { name: "Episodic", count: ep },
    { name: "Semantic", count: sem },
  ];
  const items =
    recent.data && "items" in recent.data
      ? (recent.data.items as Array<{ id: number; content?: string }>)
      : [];

  const status: Array<[string, string]> = [
    ["state", h ? "online" : "—"],
    ...(h
      ? ([
          ["transport", h.transport],
          ["version", `v${h.version}`],
          ["uptime", `${Math.round(h.uptime_seconds)}s`],
          ["embedding", `${h.embedding_backend}/${h.embedding_model}`],
        ] as Array<[string, string]>)
      : []),
  ];

  const kpis: Array<[string, number, string]> = [
    ["Episodic", ep, "raw traces"],
    ["Semantic", sem, "distilled schemas"],
    ["Total", total, "addressable memories"],
  ];

  return (
    <div className="space-y-px">
      {/* ── HERO ─────────────────────────────────────────────── */}
      <section className="relative overflow-hidden border border-border/70 bg-card">
        <div
          aria-hidden
          className="pointer-events-none absolute inset-0"
          style={{
            backgroundImage:
              "radial-gradient(80% 140% at 92% -30%, color-mix(in oklab, var(--primary) 22%, transparent), transparent 58%)",
          }}
        />
        <div className="relative px-6 py-10 sm:px-10 sm:py-14">
          <div className="eyebrow flex items-center gap-2 text-primary">
            <span className="size-1.5 animate-pulse rounded-full bg-primary" />
            Neural memory server
          </div>
          <h1 className="display mt-5 text-[clamp(2.6rem,1.6rem+5vw,5rem)] leading-[0.95]">
            Synara
          </h1>
          <p className="mt-4 max-w-xl text-sm leading-relaxed text-muted-foreground">
            Episodic + semantic memory over MCP, ranked by a
            successor-representation prior over recall.
          </p>
          <dl className="mt-8 flex flex-wrap gap-x-8 gap-y-3">
            {status.map(([k, v]) => (
              <div key={k} className="flex flex-col gap-1">
                <dt className="eyebrow">{k}</dt>
                <dd className="font-mono text-sm text-foreground/90">{v}</dd>
              </div>
            ))}
          </dl>
        </div>
      </section>

      {/* ── KPI STRIP ───────────────────────────────────────── */}
      <section className="grid grid-cols-1 border border-border/70 bg-card sm:grid-cols-3">
        {kpis.map(([label, value, sub], i) => (
          <div
            key={label}
            className={`px-6 py-7 ${
              i > 0 ? "border-border/70 sm:border-l" : ""
            } ${i > 0 ? "border-t sm:border-t-0" : ""}`}
          >
            <div className="eyebrow">{label}</div>
            <div className="metric mt-3 text-5xl tracking-tight sm:text-6xl">
              {value.toLocaleString()}
            </div>
            <div className="mt-2 text-xs text-muted-foreground">{sub}</div>
          </div>
        ))}
      </section>

      {/* ── COMPOSITION + PIPELINE ──────────────────────────── */}
      <div className="grid grid-cols-1 gap-px lg:grid-cols-3">
        <Panel
          eyebrow={hasData ? "Store composition" : "Get started"}
          className="lg:col-span-2"
          aside={
            hasData ? (
              <span className="eyebrow flex items-center gap-1.5 text-primary">
                <span className="size-1.5 animate-pulse rounded-full bg-primary" />
                live
              </span>
            ) : undefined
          }
        >
          {hasData ? (
            <div className="h-64 sm:h-72">
              <Suspense fallback={<Skeleton className="size-full" />}>
                <StoreChart data={chart} />
              </Suspense>
            </div>
          ) : (
            <div className="space-y-4 text-sm">
              <p className="text-muted-foreground">
                The store is empty — point an MCP client at this server, then
                call{" "}
                <code className="bg-muted px-1.5 py-0.5 font-mono text-xs text-primary">
                  store_episode
                </code>
                .
              </p>
              <pre className="overflow-x-auto border border-border/70 bg-surface-canvas p-4 font-mono text-xs leading-relaxed text-foreground/90">
                {MCP_SNIPPET}
              </pre>
            </div>
          )}
        </Panel>

        <Panel eyebrow="Memory pipeline">
          <ol className="relative space-y-5 before:absolute before:bottom-2 before:left-[5px] before:top-2 before:w-px before:bg-border">
            {PIPELINE.map((stage) => (
              <li key={stage.label} className="relative flex gap-4">
                <span className="z-10 mt-1 size-2.5 shrink-0 rounded-full bg-primary ring-4 ring-primary/15" />
                <div className="min-w-0">
                  <div className="text-sm font-medium">{stage.label}</div>
                  <div className="font-mono text-xs text-muted-foreground">
                    {stage.sub}
                  </div>
                </div>
              </li>
            ))}
          </ol>
        </Panel>
      </div>

      {/* ── MCP TOOL SURFACE ────────────────────────────────── */}
      <Panel
        eyebrow="MCP tools"
        aside={
          <span className="font-mono text-xs text-muted-foreground">
            {TOOLS.length} exposed
          </span>
        }
      >
        <div className="grid grid-cols-1 gap-px bg-border/70 sm:grid-cols-2 lg:grid-cols-4">
          {TOOLS.map(([name, desc]) => (
            <div
              key={name}
              className="group bg-card p-4 transition-colors hover:bg-muted/40"
            >
              <div className="truncate font-mono text-xs font-medium text-primary">
                {name}
              </div>
              <div className="mt-2 text-xs leading-relaxed text-muted-foreground">
                {desc}
              </div>
            </div>
          ))}
        </div>
      </Panel>

      {/* ── RECENT ──────────────────────────────────────────── */}
      {items.length > 0 && (
        <Panel
          eyebrow="Recent episodes"
          aside={
            <ArrowUpRight
              className="size-4 text-muted-foreground"
              aria-hidden
            />
          }
        >
          <ul className="divide-y divide-border/60">
            {items.map((it) => (
              <li
                key={it.id}
                className="flex items-start gap-4 py-3 text-sm first:pt-0 last:pb-0"
              >
                <span className="shrink-0 font-mono text-xs tabular-nums text-muted-foreground">
                  #{it.id}
                </span>
                <span className="line-clamp-2 text-foreground/90">
                  {it.content ?? "—"}
                </span>
              </li>
            ))}
          </ul>
        </Panel>
      )}
    </div>
  );
}
