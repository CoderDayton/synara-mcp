import { lazy, Suspense } from "react";
import type { LucideIcon } from "lucide-react";
import { Brain, Clock, Database, Layers } from "lucide-react";
import { useHealth, useMemories, useStats } from "@/lib/queries";
import { PageHeader } from "@/components/common/page-header";
import { ErrorState, Loading } from "@/components/common/states";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Skeleton } from "@/components/ui/skeleton";

const StoreChart = lazy(() => import("@/components/overview/store-chart"));

function Metric({
  label,
  value,
  hint,
  icon: Icon,
}: {
  label: string;
  value: string | number;
  hint?: string;
  icon: LucideIcon;
}) {
  return (
    <div className="bg-card p-5 transition-colors hover:bg-card/60 sm:p-6">
      <div className="flex items-center gap-1.5 text-[0.7rem] font-semibold uppercase tracking-[0.13em] text-muted-foreground">
        <Icon className="size-3.5 text-primary" aria-hidden />
        {label}
      </div>
      <div className="mt-3 truncate font-mono text-3xl font-semibold tracking-tight tabular-nums sm:text-[2.5rem] sm:leading-none">
        {value}
      </div>
      {hint && (
        <div className="mt-2 truncate text-xs text-muted-foreground">
          {hint}
        </div>
      )}
    </div>
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
  const chart = [
    { name: "Episodic", count: s?.episodic_count ?? 0 },
    { name: "Semantic", count: s?.semantic_count ?? 0 },
  ];
  const items =
    recent.data && "items" in recent.data
      ? (recent.data.items as Array<{ id: number; content?: string }>)
      : [];

  return (
    <div className="space-y-6 sm:space-y-8">
      <PageHeader
        title="Overview"
        subtitle="Live state of the Synara memory server."
      />

      {/* Instrument cluster — hairline-separated metric columns */}
      <div className="grid grid-cols-2 gap-px overflow-hidden rounded-xl border border-border/60 bg-border/60 shadow-card lg:grid-cols-4">
        <Metric
          label="Episodic"
          value={(s?.episodic_count ?? "—").toLocaleString?.() ?? "—"}
          icon={Brain}
          hint="raw traces"
        />
        <Metric
          label="Semantic"
          value={(s?.semantic_count ?? "—").toLocaleString?.() ?? "—"}
          icon={Layers}
          hint="distilled schemas"
        />
        <Metric
          label="Embedding"
          value={h?.embedding_backend ?? "—"}
          icon={Database}
          hint={h?.embedding_model}
        />
        <Metric
          label="Uptime"
          value={h ? `${Math.round(h.uptime_seconds)}s` : "—"}
          icon={Clock}
          hint={h ? `${h.transport} · v${h.version}` : undefined}
        />
      </div>

      <div className="grid grid-cols-1 gap-4 sm:gap-6 lg:grid-cols-3">
        <Card className="lg:col-span-2">
          <CardHeader className="flex-row items-start justify-between gap-4 [.border-b]:pb-6">
            <div>
              <CardTitle>Store composition</CardTitle>
              <p className="mt-1 text-xs text-muted-foreground">
                Episodic traces vs. distilled semantic schemas
              </p>
            </div>
            <span className="inline-flex items-center gap-1.5 rounded-full bg-primary/10 px-2.5 py-1 text-[0.7rem] font-medium text-primary ring-1 ring-inset ring-primary/20">
              <span className="size-1.5 animate-pulse rounded-full bg-primary" />
              live
            </span>
          </CardHeader>
          <CardContent className="h-64 sm:h-72">
            <Suspense
              fallback={<Skeleton className="size-full rounded-md" />}
            >
              <StoreChart data={chart} />
            </Suspense>
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle>Recent episodes</CardTitle>
          </CardHeader>
          <CardContent className="space-y-2">
            {recent.error && <ErrorState error={recent.error} />}
            {!recent.error && items.length === 0 && (
              <p className="py-10 text-center text-sm text-muted-foreground">
                No episodes yet.
              </p>
            )}
            <ul className="space-y-1.5">
              {items.map((it) => (
                <li
                  key={it.id}
                  className="group flex items-start gap-3 rounded-lg border border-transparent px-3 py-2.5 text-sm transition-colors hover:border-border/60 hover:bg-muted/40"
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
      </div>
    </div>
  );
}
