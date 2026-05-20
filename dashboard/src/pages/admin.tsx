import { useState } from "react";
import { Brain, Sparkles, Trash } from "lucide-react";
import { toast } from "sonner";
import { useConsolidate, useForget, useReflect } from "@/lib/queries";
import { PageHeader } from "@/components/common/page-header";
import { Panel } from "@/components/common/panel";
import { ResultBlock } from "@/components/common/result-block";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Badge } from "@/components/ui/badge";

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
    <Panel
      icon={<Trash className="size-4 text-primary" aria-hidden />}
      title="Forget"
      subtitle="Power-law decay prune. Dry run lists candidates without deleting."
      bodyClassName="space-y-4"
    >
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
        <div className="flex flex-wrap items-center gap-2">
          <Button
            onClick={run}
            disabled={m.isPending}
            variant={dryRun ? "secondary" : "destructive"}
          >
            {m.isPending ? "Running…" : dryRun ? "Preview" : "Forget now"}
          </Button>
          {!dryRun && <Badge variant="destructive">destructive</Badge>}
        </div>
        <ResultBlock value={m.data} />
    </Panel>
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
    <Panel
      icon={<Sparkles className="size-4 text-primary" aria-hidden />}
      title="Consolidate"
      subtitle="Cluster episodes into distilled semantic schemas."
      bodyClassName="space-y-4"
    >
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
    </Panel>
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
    <Panel
      icon={<Brain className="size-4 text-primary" aria-hidden />}
      title="Reflect"
      subtitle="Summarise a session into schemas + recent episodes."
      bodyClassName="space-y-4"
    >
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
    </Panel>
  );
}

export default function Admin() {
  return (
    <div className="space-y-6 sm:space-y-8">
      <PageHeader
        eyebrow="Operations"
        title="Admin"
        subtitle="Sanctioned maintenance operations. Effects are immediate and shared with the live MCP server."
      />
      <div className="grid grid-cols-1 items-start gap-4 md:gap-6 lg:grid-cols-2">
        <ForgetCard />
        <ConsolidateCard />
        <div className="lg:col-span-2">
          <ReflectCard />
        </div>
      </div>
    </div>
  );
}
