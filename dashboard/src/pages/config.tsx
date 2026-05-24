import { Info } from "lucide-react";
import { useHealth, useParams } from "@/lib/queries";
import { PageHeader } from "@/components/common/page-header";
import { Panel } from "@/components/common/panel";
import { ErrorState, Loading } from "@/components/common/states";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
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

  if (params.isLoading) return <Loading />;
  if (params.error) return <ErrorState error={params.error} />;

  const entries = Object.entries(params.data ?? {}).sort(([a], [b]) =>
    a.localeCompare(b),
  );
  const h = health.data;

  return (
    <div className="space-y-6 sm:space-y-8">
      <PageHeader
        eyebrow="System"
        title="Configuration"
        subtitle="Effective MemoryConfig and runtime — read-only by design."
      />

      <Alert>
        <Info className="size-4" aria-hidden />
        <AlertTitle>Read-only</AlertTitle>
        <AlertDescription>
          <code className="rounded bg-muted px-1 py-0.5 text-xs">
            MemoryConfig
          </code>{" "}
          is frozen; live mutation would require rebuilding the SR /
          plasticity layers. Change values via <code>SYNARA_*</code> env vars
          and restart.
        </AlertDescription>
      </Alert>

      <div className="grid grid-cols-1 items-start gap-4 md:gap-6 lg:grid-cols-3">
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
            <span className="font-mono text-xs text-muted-foreground">
              {entries.length} keys
            </span>
          }
          className="lg:col-span-2"
          bodyClassName="overflow-x-auto p-0"
        >
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
        </Panel>
      </div>
    </div>
  );
}
