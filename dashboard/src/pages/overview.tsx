import { lazy, Suspense, type ReactNode } from "react";
import type { LucideIcon } from "lucide-react";
import { ArrowRight, Brain, GitBranch, Layers, Sparkles } from "lucide-react";
import { useHealth, useMemories, useStats } from "@/lib/queries";
import { ErrorState, Loading } from "@/components/common/states";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
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
    "synara": {
      "command": "uvx",
      "args": ["synara-mcp"]
    }
  }
}`;

function Pill({ children }: { children: ReactNode }) {
  return (
    <span className="inline-flex items-center gap-1.5 rounded-full border border-border/60 bg-background/40 px-2.5 py-1 font-mono text-xs text-muted-foreground">
      {children}
    </span>
  );
}

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
  const hasData = ep + sem > 0;
  const chart = [
    { name: "Episodic", count: ep },
    { name: "Semantic", count: sem },
  ];
  const items =
    recent.data && "items" in recent.data
      ? (recent.data.items as Array<{ id: number; content?: string }>)
      : [];

  return (
    <div className="space-y-6">
      {/* Status band — identity + live state + headline counts */}
      <section className="relative overflow-hidden rounded-2xl border border-border/60 bg-card shadow-card">
        <div
          aria-hidden
          className="pointer-events-none absolute inset-0"
          style={{
            backgroundImage:
              "radial-gradient(70% 130% at 88% -20%, color-mix(in oklab, var(--primary) 16%, transparent), transparent 60%)",
          }}
        />
        <div className="relative flex flex-col gap-8 p-6 sm:p-8 lg:flex-row lg:items-center lg:justify-between">
          <div className="min-w-0">
            <div className="flex items-center gap-2 text-xs font-semibold uppercase tracking-[0.18em] text-primary">
              <span className="size-1.5 animate-pulse rounded-full bg-primary" />
              Neural memory server
            </div>
            <h1 className="mt-3 text-3xl font-semibold tracking-tight sm:text-4xl">
              Synara
            </h1>
            <p className="mt-2 max-w-md text-sm text-muted-foreground">
              Episodic + semantic memory over MCP, ranked by a
              successor-representation prior over recall.
            </p>
            <div className="mt-5 flex flex-wrap items-center gap-2">
              <Pill>
                <span className="size-1.5 rounded-full bg-success" />
                {h ? "online" : "—"}
              </Pill>
              {h && (
                <>
                  <Pill>{h.transport}</Pill>
                  <Pill>v{h.version}</Pill>
                  <Pill>{Math.round(h.uptime_seconds)}s up</Pill>
                  <Pill>
                    {h.embedding_backend} · {h.embedding_model}
                  </Pill>
                </>
              )}
            </div>
          </div>
          <div className="grid shrink-0 grid-cols-2 gap-px overflow-hidden rounded-xl border border-border/60 bg-border/60">
            {(
              [
                { label: "Episodic", value: ep, icon: Brain },
                { label: "Semantic", value: sem, icon: Layers },
              ] satisfies Array<{
                label: string;
                value: number;
                icon: LucideIcon;
              }>
            ).map(({ label, value, icon: Icon }) => (
              <div key={label} className="bg-card px-7 py-5">
                <div className="flex items-center gap-1.5 text-[0.7rem] font-semibold uppercase tracking-[0.13em] text-muted-foreground">
                  <Icon className="size-3.5 text-primary" aria-hidden />
                  {label}
                </div>
                <div className="mt-3 font-mono text-4xl font-semibold tracking-tight tabular-nums">
                  {value.toLocaleString()}
                </div>
              </div>
            ))}
          </div>
        </div>
      </section>

      <div className="grid grid-cols-1 gap-4 lg:grid-cols-3">
        {/* Main: composition chart, or a real "connect a client" guide */}
        <Card className="lg:col-span-2">
          <CardHeader className="flex-row items-start justify-between gap-4">
            <div>
              <CardTitle>
                {hasData ? "Store composition" : "Get started"}
              </CardTitle>
              <p className="mt-1 text-xs text-muted-foreground">
                {hasData
                  ? "Episodic traces vs. distilled semantic schemas"
                  : "The store is empty — point an MCP client at this server"}
              </p>
            </div>
            {hasData && (
              <span className="inline-flex items-center gap-1.5 rounded-full bg-primary/10 px-2.5 py-1 text-[0.7rem] font-medium text-primary ring-1 ring-inset ring-primary/20">
                <span className="size-1.5 animate-pulse rounded-full bg-primary" />
                live
              </span>
            )}
          </CardHeader>
          <CardContent>
            {hasData ? (
              <div className="h-64 sm:h-72">
                <Suspense
                  fallback={<Skeleton className="size-full rounded-md" />}
                >
                  <StoreChart data={chart} />
                </Suspense>
              </div>
            ) : (
              <div className="space-y-4">
                <ol className="space-y-3 text-sm">
                  <li className="flex gap-3">
                    <span className="grid size-5 shrink-0 place-items-center rounded-full bg-primary/15 font-mono text-[0.7rem] text-primary ring-1 ring-inset ring-primary/25">
                      1
                    </span>
                    Add Synara to your MCP client config:
                  </li>
                </ol>
                <pre className="overflow-x-auto rounded-lg border border-border/60 bg-background/60 p-4 font-mono text-xs leading-relaxed text-foreground/90">
                  {MCP_SNIPPET}
                </pre>
                <ol
                  className="space-y-3 text-sm"
                  start={2}
                  style={{ listStyle: "none" }}
                >
                  <li className="flex gap-3">
                    <span className="grid size-5 shrink-0 place-items-center rounded-full bg-primary/15 font-mono text-[0.7rem] text-primary ring-1 ring-inset ring-primary/25">
                      2
                    </span>
                    Call{" "}
                    <code className="rounded bg-muted px-1.5 py-0.5 font-mono text-xs text-primary">
                      store_episode
                    </code>{" "}
                    — traces, the successor graph and stats populate here
                    live.
                  </li>
                </ol>
              </div>
            )}
          </CardContent>
        </Card>

        {/* Memory pipeline — domain identity, not a generic widget */}
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <GitBranch className="size-4 text-primary" aria-hidden />
              Memory pipeline
            </CardTitle>
          </CardHeader>
          <CardContent>
            <ol className="relative space-y-4 before:absolute before:left-[7px] before:top-2 before:bottom-2 before:w-px before:bg-border">
              {PIPELINE.map((stage) => (
                <li
                  key={stage.label}
                  className="relative flex gap-3 pl-0"
                >
                  <span className="z-10 mt-1 size-3.5 shrink-0 rounded-full border-2 border-primary bg-card" />
                  <div className="min-w-0">
                    <div className="text-sm font-medium">{stage.label}</div>
                    <div className="font-mono text-xs text-muted-foreground">
                      {stage.sub}
                    </div>
                  </div>
                </li>
              ))}
            </ol>
          </CardContent>
        </Card>
      </div>

      {/* MCP tool surface — always-useful, on-brand content */}
      <Card>
        <CardHeader className="flex-row items-center justify-between gap-4">
          <CardTitle className="flex items-center gap-2">
            <Sparkles className="size-4 text-primary" aria-hidden />
            MCP tools
          </CardTitle>
          <span className="font-mono text-xs text-muted-foreground">
            {TOOLS.length} exposed
          </span>
        </CardHeader>
        <CardContent className="grid grid-cols-1 gap-2 sm:grid-cols-2 lg:grid-cols-4">
          {TOOLS.map(([name, desc]) => (
            <div
              key={name}
              className="group rounded-lg border border-border/60 bg-muted/25 p-3 transition-colors hover:border-primary/30 hover:bg-muted/40"
            >
              <div className="truncate font-mono text-xs font-medium text-primary">
                {name}
              </div>
              <div className="mt-1.5 text-xs text-muted-foreground">
                {desc}
              </div>
            </div>
          ))}
        </CardContent>
      </Card>

      {items.length > 0 && (
        <Card>
          <CardHeader className="flex-row items-center justify-between gap-4">
            <CardTitle>Recent episodes</CardTitle>
            <ArrowRight
              className="size-4 text-muted-foreground"
              aria-hidden
            />
          </CardHeader>
          <CardContent>
            <ul className="space-y-1.5">
              {items.map((it) => (
                <li
                  key={it.id}
                  className="flex items-start gap-3 rounded-lg border border-transparent px-3 py-2.5 text-sm transition-colors hover:border-border/60 hover:bg-muted/40"
                >
                  <span className="mt-0.5 shrink-0 rounded-md bg-muted px-1.5 py-0.5 font-mono text-[0.7rem] text-muted-foreground tabular-nums ring-1 ring-inset ring-border/60">
                    #{it.id}
                  </span>
                  <span className="line-clamp-2 text-foreground/90">
                    {it.content ?? "—"}
                  </span>
                </li>
              ))}
            </ul>
          </CardContent>
        </Card>
      )}
    </div>
  );
}
