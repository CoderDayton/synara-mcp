import { lazy, Suspense } from "react";
import { Activity, Brain, Cpu, Database } from "lucide-react";
import { useHealth, useMemories, useStats } from "@/lib/queries";
import { ErrorState, Loading } from "@/components/common/states";
import { Panel } from "@/components/common/panel";
import { StatCard } from "@/components/common/stat-card";
import { Skeleton } from "@/components/ui/skeleton";

const StoreChart = lazy(() => import("@/components/overview/store-chart"));

const TOOLS: Array<[string, string]> = [
  ["store_episode", "encode an episodic trace"],
  ["recall_episodes", "cross-session episodic recall"],
  ["consolidate_episodes", "cluster traces → schemas"],
  ["forget_episodes", "power-law decay prune"],
  ["reflect_session", "summarise a session"],
  ["store_semantic_memory", "write a semantic memory"],
  ["recall_semantic_memory", "semantic memory recall"],
  ["memory_stats", "store + tunable snapshot"],
];

const PIPELINE: Array<{ label: string; sub: string }> = [
  { label: "encode", sub: "store_episode → embedding" },
  { label: "hippocampus", sub: "episodic · SR · plasticity" },
  { label: "consolidate", sub: "cluster → schemas" },
  { label: "neocortex", sub: "durable semantic memory" },
];

const MCP_SNIPPET = `{
  "mcpServers": {
    "synara": { "command": "uvx", "args": ["synara-mcp"] }
  }
}`;

export default function Overview() {
  const stats = useStats();
  const health = useHealth();
  const recent = useMemories({ kind: "episodic", limit: 8 });

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

  return (
    <div className="grid grid-cols-1 gap-3 lg:grid-cols-6 lg:gap-4">
      {/* HERO ID strip — replaces the old big-name hero with a compact
          terminal banner that names the service and shows live state. */}
      <Panel
        variant="raised"
        className="lg:col-span-6"
        bodyClassName="flex flex-col gap-3 px-5 py-4 sm:flex-row sm:items-center sm:justify-between"
      >
        <div className="min-w-0">
          <div className="eyebrow flex items-center gap-2 text-primary">
            <span className="pulse-dot" aria-hidden />
            neural memory server · online
          </div>
          <div className="mt-2 font-mono text-xl sm:text-2xl">
            <span className="text-primary">synara</span>
            <span className="text-muted-foreground">@</span>
            <span className="text-foreground">{h?.transport ?? "—"}</span>
            <span className="text-muted-foreground">:</span>
            <span className="text-foreground">v{h?.version ?? "—"}</span>
          </div>
          <p className="mt-1.5 max-w-xl text-xs text-muted-foreground">
            Episodic + semantic memory over MCP, ranked by a
            successor-representation prior.
          </p>
        </div>
        <dl className="grid grid-cols-3 gap-x-6 gap-y-1 text-[0.7rem]">
          {[
            ["uptime", h ? `${Math.round(h.uptime_seconds)}s` : "—"],
            ["embed", h ? h.embedding_backend : "—"],
            ["model", h ? h.embedding_model.split("/").slice(-1)[0] : "—"],
          ].map(([k, v]) => (
            <div key={k} className="space-y-0.5">
              <dt className="eyebrow text-[0.6rem]">{k}</dt>
              <dd
                className="font-mono text-xs text-foreground/90"
                title={String(v)}
              >
                {v}
              </dd>
            </div>
          ))}
        </dl>
      </Panel>

      {/* KPI strip — three terminal metric tiles */}
      <div className="lg:col-span-6 grid grid-cols-1 gap-3 sm:grid-cols-3">
        <StatCard
          label="episodic"
          value={ep.toLocaleString()}
          hint="raw traces"
          icon={Activity}
        />
        <StatCard
          label="semantic"
          value={sem.toLocaleString()}
          hint="distilled schemas"
          icon={Brain}
        />
        <StatCard
          label="total"
          value={total.toLocaleString()}
          hint="addressable memories"
          icon={Database}
        />
      </div>

      {/* Composition (chart) — wide */}
      <Panel
        title="store composition"
        eyebrow="distribution"
        className="lg:col-span-4"
        aside={
          hasData ? (
            <span className="flex items-center gap-1.5 text-primary">
              <span className="pulse-dot" aria-hidden />
              live
            </span>
          ) : (
            <span>empty</span>
          )
        }
      >
        {hasData ? (
          <div className="h-60">
            <Suspense fallback={<Skeleton className="size-full" />}>
              <StoreChart data={chart} />
            </Suspense>
          </div>
        ) : (
          <div className="space-y-3 text-xs">
            <p className="text-muted-foreground">
              Store is empty. Point an MCP client at this server, then call{" "}
              <code className="border border-border bg-muted px-1 py-0.5 font-mono text-primary">
                store_episode
              </code>
              .
            </p>
            <pre className="overflow-x-auto border border-border bg-surface-canvas p-3 font-mono text-[0.7rem] leading-relaxed text-foreground/90">
              {MCP_SNIPPET}
            </pre>
          </div>
        )}
      </Panel>

      {/* Pipeline — narrow */}
      <Panel
        title="memory pipeline"
        icon={<Cpu className="size-3.5" aria-hidden />}
        className="lg:col-span-2"
      >
        <ol className="relative space-y-3 pl-1 before:absolute before:bottom-1 before:left-[3px] before:top-1 before:w-px before:bg-primary/30">
          {PIPELINE.map((stage) => (
            <li key={stage.label} className="relative flex gap-3">
              <span className="z-10 mt-1 size-1.5 shrink-0 rounded-full bg-primary shadow-[0_0_6px_var(--primary)]" />
              <div className="min-w-0">
                <div className="font-mono text-[0.72rem] uppercase tracking-wider text-foreground">
                  {stage.label}
                </div>
                <div className="mt-0.5 font-mono text-[0.65rem] text-muted-foreground">
                  {stage.sub}
                </div>
              </div>
            </li>
          ))}
        </ol>
      </Panel>

      {/* MCP tools grid */}
      <Panel
        title="exposed mcp tools"
        eyebrow="surface"
        className="lg:col-span-4"
        aside={<span>{TOOLS.length} listed</span>}
        bodyClassName="p-0"
      >
        <div className="grid grid-cols-1 divide-y divide-border sm:grid-cols-2 sm:divide-y-0 sm:[&>*]:border-b sm:[&>*]:border-border">
          {TOOLS.map(([name, desc], i) => (
            <div
              key={name}
              className={
                "group flex items-center justify-between gap-3 px-4 py-2.5 transition-colors hover:bg-primary/5 " +
                (i % 2 === 0 ? "sm:border-r sm:border-border" : "")
              }
            >
              <div className="min-w-0">
                <div className="truncate font-mono text-[0.72rem] text-primary group-hover:text-foreground">
                  {name}
                </div>
                <div className="mt-0.5 truncate text-[0.65rem] text-muted-foreground">
                  {desc}
                </div>
              </div>
              <span
                className="font-mono text-[0.6rem] text-muted-foreground opacity-0 transition-opacity group-hover:opacity-100"
                aria-hidden
              >
                ↗
              </span>
            </div>
          ))}
        </div>
      </Panel>

      {/* Recent episodes — live feed */}
      <Panel
        title="recent episodes"
        eyebrow="tail -f"
        className="lg:col-span-2"
        aside={
          items.length > 0 ? (
            <span className="flex items-center gap-1.5 text-primary">
              <span className="pulse-dot" aria-hidden />
              live
            </span>
          ) : undefined
        }
        bodyClassName="p-0"
      >
        {items.length === 0 ? (
          <div className="p-4 text-[0.7rem] text-muted-foreground">
            No episodes yet.
          </div>
        ) : (
          <ul className="divide-y divide-border">
            {items.slice(0, 6).map((it) => (
              <li
                key={it.id}
                className="flex items-start gap-3 px-4 py-2 text-xs"
              >
                <span className="shrink-0 font-mono text-[0.65rem] tabular-nums text-primary">
                  #{it.id}
                </span>
                <span className="line-clamp-2 leading-relaxed text-foreground/80">
                  {it.content ?? "—"}
                </span>
              </li>
            ))}
          </ul>
        )}
      </Panel>
    </div>
  );
}
