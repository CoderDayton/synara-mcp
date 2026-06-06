import { lazy, Suspense } from "react";
import { Activity, Brain, Database, History, Sparkles, Wrench } from "lucide-react";
import { toast } from "sonner";
import {
  useConsolidate,
  useHealth,
  useMemories,
  useStats,
  useToolMetrics,
} from "@/lib/queries";
import type { ToolMetricRow } from "@/lib/api";
import { formatDuration, relativeTime } from "@/lib/format";
import { Empty, ErrorState, Loading } from "@/components/common/states";
import { Panel } from "@/components/common/panel";
import { StatCard } from "@/components/common/stat-card";
import { Button } from "@/components/ui/button";
import { Skeleton } from "@/components/ui/skeleton";

const StoreChart = lazy(() => import("@/components/overview/store-chart"));

const MCP_SNIPPET = `{
  "mcpServers": {
    "synara": { "command": "uvx", "args": ["synara-mcp"] }
  }
}`;

export default function Overview() {
  const stats = useStats();
  const health = useHealth();
  const recentCap = 25;
  const recentOffset = Math.max(0, (stats.data?.episodic_count ?? 0) - recentCap);
  const recent = useMemories({
    kind: "episodic",
    limit: recentCap,
    offset: recentOffset,
  });

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
  // The store returns rows id-ascending, so the offset above fetches the
  // tail — the newest `recentCap` episodes — which we then sort
  // id-descending for a newest-first "tail -f" feed.
  const items =
    recent.data && "items" in recent.data
      ? [...(recent.data.items as Array<{ id: number; content?: string }>)].sort(
          (a, b) => b.id - a.id,
        )
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
        <dl className="grid grid-cols-2 gap-x-6 gap-y-1 text-[0.7rem] sm:grid-cols-4">
          {[
            ["uptime", h ? formatDuration(h.uptime_seconds) : "—"],
            ["embed", h ? h.embedding_backend : "—"],
            ["model", h ? h.embedding_model.split("/").slice(-1)[0] : "—"],
            ["client", h ? h.mcp_client : "—"],
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
          <Empty
            dense
            icon={Database}
            label="Store is empty"
            hint={
              <>
                Point an MCP client at this server, then call{" "}
                <code className="border border-border bg-muted px-1 py-0.5 font-mono text-primary">
                  store_episode
                </code>
                .
              </>
            }
          >
            <pre className="w-full overflow-x-auto border border-border bg-surface-canvas p-3 text-left font-mono text-[0.7rem] leading-relaxed text-foreground/90">
              {MCP_SNIPPET}
            </pre>
          </Empty>
        )}
      </Panel>

      <HippocampusPressure ep={ep} sem={sem} />

      <ToolMetricsPanel />

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
        bodyClassName="relative p-0 min-h-[22rem]"
      >
        {items.length === 0 ? (
          <Empty
            dense
            icon={History}
            label="No episodes yet"
            hint="Stored traces will stream in here as the MCP client encodes them."
          />
        ) : (
          <ul className="absolute inset-0 divide-y divide-border overflow-y-auto">
            {items.map((it) => (
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

function formatMs(ms: number | null): string {
  if (ms == null) return "—";
  if (ms < 1) return `${ms.toFixed(2)}ms`;
  if (ms < 10) return `${ms.toFixed(1)}ms`;
  if (ms < 1000) return `${Math.round(ms)}ms`;
  return `${(ms / 1000).toFixed(2)}s`;
}

function ToolMetricsPanel() {
  const m = useToolMetrics();
  const rows: ToolMetricRow[] = m.data?.tools ?? [];
  const totalCalls = rows.reduce((a, r) => a + r.count, 0);
  const totalErrors = rows.reduce((a, r) => a + r.error_count, 0);
  const live = totalCalls > 0;

  return (
    <Panel
      title="exposed mcp tools"
      eyebrow="live telemetry"
      className="lg:col-span-4"
      aside={
        m.isLoading ? (
          <span>loading…</span>
        ) : (
          <span className="flex items-center gap-2">
            <span className="tabular-nums">{totalCalls.toLocaleString()} calls</span>
            {totalErrors > 0 && (
              <span className="text-destructive tabular-nums">
                · {totalErrors} err
              </span>
            )}
            {live && (
              <span className="flex items-center gap-1 text-primary">
                <span className="pulse-dot" aria-hidden />
                live
              </span>
            )}
          </span>
        )
      }
      bodyClassName="p-0"
    >
      {rows.length === 0 ? (
        m.isLoading ? (
          <Loading label="fetching tool surface" />
        ) : (
          <Empty
            dense
            icon={Wrench}
            label="No tools registered"
            hint="The server hasn't declared any MCP tools. Check the server logs."
          />
        )
      ) : (
        <ul className="divide-y divide-border">
          {rows.map((t) => (
            <ToolRow key={t.name} row={t} />
          ))}
        </ul>
      )}
    </Panel>
  );
}

function ToolRow({ row }: { row: ToolMetricRow }) {
  const called = row.count > 0;
  const hasErrors = row.error_count > 0;
  return (
    <li className="grid grid-cols-[1fr_auto] items-center gap-3 px-4 py-2.5 transition-colors hover:bg-primary/5">
      <div className="min-w-0">
        <div className="flex items-center gap-2">
          <span className="truncate font-mono text-[0.72rem] text-primary">
            {row.name}
          </span>
          {hasErrors && (
            <span
              className="font-mono text-[0.6rem] text-destructive tabular-nums"
              title={`${row.error_count} error${row.error_count === 1 ? "" : "s"}`}
            >
              !{row.error_count}
            </span>
          )}
        </div>
        <div className="mt-0.5 truncate text-[0.65rem] text-muted-foreground">
          {row.headline}
        </div>
      </div>
      <div className="flex shrink-0 items-center gap-3 font-mono text-[0.65rem] tabular-nums">
        <span
          className={called ? "text-foreground/90" : "text-muted-foreground/60"}
          title="invocation count"
        >
          {row.count.toLocaleString()}×
        </span>
        <span
          className={called ? "text-foreground/70" : "text-muted-foreground/40"}
          title={
            row.last_called_at
              ? `last called ${new Date(row.last_called_at * 1000).toLocaleString()}`
              : "never called"
          }
        >
          {row.last_called_at ? relativeTime(row.last_called_at) : "—"}
        </span>
        <span
          className={
            "min-w-[3.5rem] text-right " +
            (called ? "text-primary/90" : "text-muted-foreground/40")
          }
          title="p50 / p95 latency over last 200 calls"
        >
          {formatMs(row.p50_ms)} / {formatMs(row.p95_ms)}
        </span>
      </div>
    </li>
  );
}

function HippocampusPressure({ ep, sem }: { ep: number; sem: number }) {
  const m = useConsolidate();
  const ratio = sem > 0 ? ep / sem : null;
  const hasQueue = ep > 0;

  function run() {
    m.mutate(
      { session_id: null, n_clusters: null, min_cluster_size: null },
      {
        onSuccess: (r) =>
          toast.success(
            r.schemas_formed > 0
              ? `Formed ${r.schemas_formed} schema(s).`
              : "Nothing new to consolidate.",
          ),
        onError: (e) =>
          toast.error(e instanceof Error ? e.message : "Consolidate failed"),
      },
    );
  }

  return (
    <Panel
      title="hippocampus pressure"
      eyebrow="queue"
      icon={<Sparkles className="size-3.5" aria-hidden />}
      className="lg:col-span-2"
      bodyClassName="flex flex-col gap-3"
    >
      <div>
        <div className="metric text-3xl text-foreground tabular-nums leading-none">
          {ep.toLocaleString()}
        </div>
        <div className="mt-1 text-[0.7rem] text-muted-foreground">
          raw episodes awaiting consolidation
        </div>
      </div>

      <div className="font-mono text-[0.7rem] text-muted-foreground">
        {sem > 0 ? (
          <>
            <span className="text-foreground tabular-nums">
              {sem.toLocaleString()}
            </span>{" "}
            schema{sem === 1 ? "" : "s"} ·{" "}
            <span className="text-primary tabular-nums">
              {ratio !== null ? `${ratio.toFixed(ratio >= 10 ? 0 : 1)}×` : "—"}
            </span>{" "}
            ratio
          </>
        ) : hasQueue ? (
          <span className="text-primary">no schemas yet</span>
        ) : (
          "store is idle"
        )}
      </div>

      <Button
        size="sm"
        variant="outline"
        onClick={run}
        disabled={!hasQueue || m.isPending}
        className="mt-auto w-full font-mono text-[0.72rem]"
      >
        {m.isPending ? "consolidating…" : "consolidate now"}
      </Button>
    </Panel>
  );
}
