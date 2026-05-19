import { useState } from "react";
import { Brain, Sparkles, Trash } from "lucide-react";
import { toast } from "sonner";
import { useConsolidate, useForget, useReflect } from "@/lib/queries";
import { PageHeader } from "@/components/common/page-header";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Badge } from "@/components/ui/badge";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";

function ResultBlock({ value }: { value: unknown }) {
  if (value == null) return null;
  return (
    <pre className="mt-4 max-h-56 overflow-auto rounded-md border border-border bg-muted/40 p-3 text-xs">
      {JSON.stringify(value, null, 2)}
    </pre>
  );
}

function ForgetCard() {
  const [floor, setFloor] = useState("0.05");
  const [maxScan, setMaxScan] = useState("1000");
  const [dryRun, setDryRun] = useState(true);
  const m = useForget();

  function run() {
    m.mutate(
      {
        strength_floor: Number(floor),
        dry_run: dryRun,
        max_scan: Number(maxScan),
      },
      {
        onSuccess: (r) =>
          toast.success(
            r.dry_run
              ? `Dry run: ${r.candidate_ids.length} candidate(s) of ${r.scanned} scanned.`
              : `Forgot ${r.removed} episode(s).`,
          ),
        onError: (e) =>
          toast.error(e instanceof Error ? e.message : "Forget failed"),
      },
    );
  }

  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          <Trash className="size-4 text-primary" aria-hidden />
          Forget
        </CardTitle>
        <CardDescription>
          Power-law decay prune. Dry run lists candidates without deleting.
        </CardDescription>
      </CardHeader>
      <CardContent className="space-y-4">
        <div className="grid grid-cols-1 gap-3 sm:grid-cols-2">
          <div className="space-y-1.5">
            <Label htmlFor="floor">Strength floor (0–1)</Label>
            <Input
              id="floor"
              inputMode="decimal"
              value={floor}
              onChange={(e) => setFloor(e.target.value)}
            />
          </div>
          <div className="space-y-1.5">
            <Label htmlFor="scan">Max scan</Label>
            <Input
              id="scan"
              inputMode="numeric"
              value={maxScan}
              onChange={(e) => setMaxScan(e.target.value)}
            />
          </div>
        </div>
        <label className="flex items-center gap-2 text-sm">
          <input
            type="checkbox"
            checked={dryRun}
            onChange={(e) => setDryRun(e.target.checked)}
            className="size-4 accent-[var(--primary)]"
          />
          Dry run (preview only)
        </label>
        <Button
          onClick={run}
          disabled={m.isPending}
          variant={dryRun ? "secondary" : "destructive"}
        >
          {m.isPending ? "Running…" : dryRun ? "Preview" : "Forget now"}
        </Button>
        {!dryRun && (
          <Badge variant="destructive" className="ml-2 align-middle">
            destructive
          </Badge>
        )}
        <ResultBlock value={m.data} />
      </CardContent>
    </Card>
  );
}

function ConsolidateCard() {
  const [session, setSession] = useState("");
  const [nClusters, setNClusters] = useState("");
  const [minSize, setMinSize] = useState("");
  const m = useConsolidate();

  function run() {
    m.mutate(
      {
        session_id: session.trim() || null,
        n_clusters: nClusters ? Number(nClusters) : null,
        min_cluster_size: minSize ? Number(minSize) : null,
      },
      {
        onSuccess: (r) => toast.success(`Formed ${r.schemas_formed} schema(s).`),
        onError: (e) =>
          toast.error(e instanceof Error ? e.message : "Consolidate failed"),
      },
    );
  }

  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          <Sparkles className="size-4 text-primary" aria-hidden />
          Consolidate
        </CardTitle>
        <CardDescription>
          Cluster episodes into distilled semantic schemas.
        </CardDescription>
      </CardHeader>
      <CardContent className="space-y-4">
        <div className="grid grid-cols-1 gap-3 sm:grid-cols-3">
          <div className="space-y-1.5">
            <Label htmlFor="cs">Session (opt.)</Label>
            <Input
              id="cs"
              value={session}
              onChange={(e) => setSession(e.target.value)}
            />
          </div>
          <div className="space-y-1.5">
            <Label htmlFor="nc">N clusters</Label>
            <Input
              id="nc"
              inputMode="numeric"
              value={nClusters}
              onChange={(e) => setNClusters(e.target.value)}
            />
          </div>
          <div className="space-y-1.5">
            <Label htmlFor="mc">Min size</Label>
            <Input
              id="mc"
              inputMode="numeric"
              value={minSize}
              onChange={(e) => setMinSize(e.target.value)}
            />
          </div>
        </div>
        <Button onClick={run} disabled={m.isPending}>
          {m.isPending ? "Working…" : "Consolidate"}
        </Button>
        <ResultBlock value={m.data} />
      </CardContent>
    </Card>
  );
}

function ReflectCard() {
  const [session, setSession] = useState("");
  const [query, setQuery] = useState("");
  const [k, setK] = useState("5");
  const m = useReflect();

  function run() {
    if (!session.trim()) {
      toast.error("Session ID is required for reflect.");
      return;
    }
    m.mutate(
      { session_id: session.trim(), query: query.trim() || null, k: Number(k) },
      {
        onSuccess: () => toast.success("Reflection complete."),
        onError: (e) =>
          toast.error(e instanceof Error ? e.message : "Reflect failed"),
      },
    );
  }

  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          <Brain className="size-4 text-primary" aria-hidden />
          Reflect
        </CardTitle>
        <CardDescription>
          Summarise a session into schemas + recent episodes.
        </CardDescription>
      </CardHeader>
      <CardContent className="space-y-4">
        <div className="grid grid-cols-1 gap-3 sm:grid-cols-3">
          <div className="space-y-1.5">
            <Label htmlFor="rs">Session ID *</Label>
            <Input
              id="rs"
              value={session}
              onChange={(e) => setSession(e.target.value)}
            />
          </div>
          <div className="space-y-1.5">
            <Label htmlFor="rq">Query (opt.)</Label>
            <Input
              id="rq"
              value={query}
              onChange={(e) => setQuery(e.target.value)}
            />
          </div>
          <div className="space-y-1.5">
            <Label htmlFor="rk">k</Label>
            <Input
              id="rk"
              inputMode="numeric"
              value={k}
              onChange={(e) => setK(e.target.value)}
            />
          </div>
        </div>
        <Button onClick={run} disabled={m.isPending}>
          {m.isPending ? "Reflecting…" : "Reflect"}
        </Button>
        <ResultBlock value={m.data} />
      </CardContent>
    </Card>
  );
}

export default function Admin() {
  return (
    <div className="space-y-6">
      <PageHeader
        title="Admin"
        subtitle="Sanctioned maintenance operations. Effects are immediate and shared with the live MCP server."
      />
      <div className="grid grid-cols-1 gap-4 lg:grid-cols-2">
        <ForgetCard />
        <ConsolidateCard />
        <div className="lg:col-span-2">
          <ReflectCard />
        </div>
      </div>
    </div>
  );
}
