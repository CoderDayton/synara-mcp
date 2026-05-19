import { lazy, Suspense } from "react";
import { Brain, Clock, Database, Layers } from "lucide-react";
import { useHealth, useMemories, useStats } from "@/lib/queries";
import { PageHeader } from "@/components/common/page-header";
import { StatCard } from "@/components/common/stat-card";
import { ErrorState, Loading } from "@/components/common/states";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Skeleton } from "@/components/ui/skeleton";

const StoreChart = lazy(() => import("@/components/overview/store-chart"));

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

      <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-4">
        <StatCard
          label="Episodic"
          value={s?.episodic_count ?? "—"}
          icon={Brain}
          hint="raw traces"
        />
        <StatCard
          label="Semantic"
          value={s?.semantic_count ?? "—"}
          icon={Layers}
          hint="distilled schemas"
        />
        <StatCard
          label="Embedding"
          value={h?.embedding_backend ?? "—"}
          icon={Database}
          hint={h?.embedding_model}
        />
        <StatCard
          label="Uptime"
          value={h ? `${Math.round(h.uptime_seconds)}s` : "—"}
          icon={Clock}
          hint={h ? `${h.transport} · v${h.version}` : undefined}
        />
      </div>

      <div className="grid grid-cols-1 gap-4 lg:grid-cols-3">
        <Card className="lg:col-span-2">
          <CardHeader>
            <CardTitle>Store composition</CardTitle>
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
              <p className="py-6 text-center text-sm text-muted-foreground">
                No episodes yet.
              </p>
            )}
            <ul className="space-y-2">
              {items.map((it) => (
                <li
                  key={it.id}
                  className="rounded-md border border-border bg-muted/30 px-3 py-2 text-sm"
                >
                  <span className="mr-2 font-mono text-xs text-muted-foreground">
                    #{it.id}
                  </span>
                  <span className="line-clamp-2 align-middle">
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
