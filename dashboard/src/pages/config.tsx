import { useState } from "react";
import { Info, Search, X } from "lucide-react";
import { useHealth, useParams } from "@/lib/queries";
import { PageHeader } from "@/components/common/page-header";
import { Panel } from "@/components/common/panel";
import { ErrorState, Loading } from "@/components/common/states";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import {
  Table,
  TableBody,
  TableCell,
  TableRow,
} from "@/components/ui/table";

function fmt(v: unknown): string {
  if (v == null) return "—";
  switch (typeof v) {
    case "object":
      return JSON.stringify(v);
    case "string":
      return v;
    case "number":
    case "boolean":
    case "bigint":
      return String(v);
    default:
      // function / symbol — should not appear in a JSON config, but
      // produce a readable sentinel instead of the engine default.
      return `<${typeof v}>`;
  }
}

export default function Config() {
  const params = useParams();
  const health = useHealth();
  const [filter, setFilter] = useState("");

  const allEntries = Object.entries(params.data ?? {}).sort(([a], [b]) =>
    a.localeCompare(b),
  );
  const q = filter.trim().toLowerCase();
  const entries = q
    ? allEntries.filter(([k, v]) => {
        if (k.toLowerCase().includes(q)) return true;
        return fmt(v).toLowerCase().includes(q);
      })
    : allEntries;

  if (params.isLoading) return <Loading />;
  if (params.error) return <ErrorState error={params.error} />;

  const h = health.data;

  return (
    <div>
      <PageHeader
        eyebrow="system"
        title="config"
        subtitle="Effective MemoryConfig and runtime — read-only by design."
      />

      <Alert className="mb-4 border-warning/40 bg-warning/5 font-mono text-xs">
        <Info className="size-4 text-warning" aria-hidden />
        <AlertTitle className="uppercase tracking-wider text-warning">
          read-only
        </AlertTitle>
        <AlertDescription>
          <code className="border border-border bg-muted px-1 py-0.5 text-xs">
            MemoryConfig
          </code>{" "}
          is frozen; live mutation would require rebuilding the SR /
          plasticity layers. Change values via <code>SYNARA_*</code> env vars
          and restart.
        </AlertDescription>
      </Alert>

      <div className="grid grid-cols-1 items-start gap-3 lg:grid-cols-3 lg:gap-4">
        <Panel title="Runtime" bodyClassName="space-y-2 text-sm">
          {h ? (
            <dl className="space-y-2">
              {[
                ["Version", h.version],
                ["Transport", h.transport],
                ["Embedding", `${h.embedding_backend} · ${h.embedding_model}`],
                ["DB path", h.db_path],
              ].map(([k, v]) => (
                <div
                  key={k}
                  className="flex items-baseline justify-between gap-3"
                >
                  <dt className="shrink-0 text-muted-foreground">{k}</dt>
                  <dd
                    className="min-w-0 truncate font-mono text-xs"
                    title={String(v)}
                  >
                    {v}
                  </dd>
                </div>
              ))}
            </dl>
          ) : (
            <p className="text-muted-foreground">Runtime unavailable.</p>
          )}
        </Panel>

        <Panel
          title="MemoryConfig"
          aside={
            <span>
              {filter ? (
                <>
                  <span className="text-foreground tabular-nums">
                    {entries.length}
                  </span>
                  <span className="text-muted-foreground"> / </span>
                  <span className="tabular-nums">{allEntries.length}</span>{" "}
                  keys
                </>
              ) : (
                <>
                  <span className="text-foreground tabular-nums">
                    {allEntries.length}
                  </span>{" "}
                  keys
                </>
              )}
            </span>
          }
          className="lg:col-span-2"
          bodyClassName="p-0"
        >
          {/* Sticky search bar — the panel body owns the scroll, so the
              filter row stays pinned while the table scrolls below it. */}
          <div className="sticky top-0 z-10 flex items-center gap-2 border-b border-border bg-card px-3 py-2">
            <Search
              className="size-3.5 shrink-0 text-muted-foreground"
              aria-hidden
            />
            <Input
              value={filter}
              onChange={(e) => setFilter(e.target.value)}
              placeholder="filter keys or values…"
              aria-label="Filter MemoryConfig"
              className="h-7 border-0 bg-transparent px-0 font-mono text-xs shadow-none focus-visible:ring-0"
            />
            {filter && (
              <Button
                type="button"
                variant="ghost"
                size="icon"
                className="size-6"
                aria-label="Clear filter"
                onClick={() => setFilter("")}
              >
                <X className="size-3.5" aria-hidden />
              </Button>
            )}
          </div>
          <div className="max-h-[28rem] overflow-y-auto">
            {entries.length === 0 ? (
              <div className="px-4 py-8 text-center font-mono text-xs text-muted-foreground">
                no matches for{" "}
                <span className="text-foreground">"{filter}"</span>
              </div>
            ) : (
              <Table>
                <TableBody>
                  {entries.map(([k, v]) => (
                    <TableRow key={k}>
                      <TableCell className="w-1/2 font-mono text-xs">
                        {k}
                      </TableCell>
                      <TableCell className="font-mono text-xs tabular-nums">
                        {fmt(v)}
                      </TableCell>
                    </TableRow>
                  ))}
                </TableBody>
              </Table>
            )}
          </div>
        </Panel>
      </div>
    </div>
  );
}
