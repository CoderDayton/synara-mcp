import { useMemo, useState } from "react";
import {
  AlertTriangle,
  Brain,
  ChevronDown,
  ChevronRight,
  Eye,
  ListTree,
  Radio,
  Sparkles,
  Trash2,
  Zap,
} from "lucide-react";
import { toast } from "sonner";
import {
  useConsolidate,
  useForget,
  useReflect,
  useStats,
} from "@/lib/queries";
import type {
  ConsolidateResult,
  ForgetResult,
  ReflectResult,
} from "@/lib/api";
import { PageHeader } from "@/components/common/page-header";
import { Panel } from "@/components/common/panel";
import { ResultBlock } from "@/components/common/result-block";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { cn } from "@/lib/utils";

/* ------------------------------------------------------------------ types */

type OpName = "forget" | "consolidate" | "reflect";
type Severity = "info" | "ok" | "warn" | "err";

type LogEntry = {
  ts: number;
  op: OpName;
  summary: string;
  severity: Severity;
};

type AppendLog = (e: Omit<LogEntry, "ts">) => void;

/* -------------------------------------------------------------- transcript
 *
 * Session-scoped in-memory log. The backend has no audit endpoint, so we
 * record what we ourselves issued during this dashboard session. This is
 * the SOTA affordance that turns "fire-and-forget JSON" into a console
 * the operator can actually reason about. */
function useTranscript() {
  const [entries, setEntries] = useState<LogEntry[]>([]);
  const append: AppendLog = (e) =>
    setEntries((prev) => [{ ...e, ts: Date.now() }, ...prev].slice(0, 40));
  const clear = () => setEntries([]);
  return { entries, append, clear };
}

function fmtTime(ts: number): string {
  return new Date(ts).toTimeString().slice(0, 8);
}

/* ------------------------------------------------------------- primitives */

/** Severity gutter — a 2px tinted strip pinned to the panel top. The
 *  three tones encode the operation's intent before the user reads the
 *  title: destructive (primary/salmon), synth (chart-2/green),
 *  read (chart-4/teal). */
function OpStrip({ tone }: { tone: "destructive" | "synth" | "read" }) {
  const color =
    tone === "destructive"
      ? "var(--primary)"
      : tone === "synth"
        ? "var(--chart-2)"
        : "var(--chart-4)";
  return (
    <div
      className="pointer-events-none absolute inset-x-0 top-0 z-10 h-[2px]"
      style={{
        background: `linear-gradient(90deg, transparent, ${color} 25%, ${color} 75%, transparent)`,
      }}
      aria-hidden
    />
  );
}

function Field({
  id,
  label,
  value,
  onChange,
  hint,
  inputMode = "text",
  placeholder,
  required,
}: {
  id: string;
  label: string;
  value: string;
  onChange: (v: string) => void;
  hint?: string;
  inputMode?: "decimal" | "numeric" | "text";
  placeholder?: string;
  required?: boolean;
}) {
  return (
    <div className="space-y-1.5">
      <Label htmlFor={id} className="eyebrow flex items-center gap-1.5">
        {label}
        {required && <span className="text-primary">*</span>}
      </Label>
      <Input
        id={id}
        inputMode={inputMode}
        value={value}
        onChange={(e) => onChange(e.target.value)}
        placeholder={placeholder}
        className="font-mono text-[0.78rem]"
      />
      {hint && (
        <div className="text-[0.65rem] leading-tight text-muted-foreground">
          {hint}
        </div>
      )}
    </div>
  );
}

/** Range slider on the strength floor — instant, visual feedback that an
 *  input box can never provide. Phosphor accent on the thumb. */
function FloorSlider({
  value,
  onChange,
}: {
  value: string;
  onChange: (v: string) => void;
}) {
  const n = Number(value);
  const safe = Number.isFinite(n) ? Math.min(1, Math.max(0, n)) : 0.05;
  return (
    <div className="space-y-1">
      <input
        type="range"
        min={0}
        max={1}
        step={0.01}
        value={safe}
        onChange={(e) => onChange(e.target.value)}
        className="w-full accent-[var(--primary)]"
        aria-label="strength floor"
      />
      <div className="flex justify-between font-mono text-[0.6rem] text-muted-foreground">
        <span>0.00</span>
        <span className="text-foreground">{safe.toFixed(2)}</span>
        <span>1.00</span>
      </div>
    </div>
  );
}

/** Console line — used for impact previews and transcript rows.
 *  Renders a single mono row with a small leading glyph + body. */
function ConsoleLine({
  glyph,
  children,
  className,
}: {
  glyph: string;
  children: React.ReactNode;
  className?: string;
}) {
  return (
    <div className={cn("flex items-start gap-2 font-mono text-[0.7rem]", className)}>
      <span className="select-none text-primary" aria-hidden>
        {glyph}
      </span>
      <span className="min-w-0 flex-1 leading-relaxed">{children}</span>
    </div>
  );
}

function AdvancedToggle({
  open,
  onToggle,
  label,
}: {
  open: boolean;
  onToggle: () => void;
  label: string;
}) {
  return (
    <button
      type="button"
      onClick={onToggle}
      className="flex w-full items-center gap-1.5 border-t border-border pt-2 font-mono text-[0.65rem] uppercase tracking-wider text-muted-foreground transition-colors hover:text-primary"
    >
      {open ? (
        <ChevronDown className="size-3" aria-hidden />
      ) : (
        <ChevronRight className="size-3" aria-hidden />
      )}
      {label}
    </button>
  );
}

/* --------------------------------------------------------------- forget */

function ForgetCard({
  epCount,
  append,
}: {
  epCount: number;
  append: AppendLog;
}) {
  const [floor, setFloor] = useState("0.05");
  const [maxScan, setMaxScan] = useState("1000");
  const [decayTau, setDecayTau] = useState("");
  const [advanced, setAdvanced] = useState(false);
  const [preview, setPreview] = useState<{ res: ForgetResult; sig: string } | null>(null);
  const m = useForget();

  const floorNum = Number(floor);
  const scanNum = Number(maxScan);
  const tauNum = decayTau.trim() ? Number(decayTau) : null;
  // Parameter signature gates the preview: when any input changes, the
  // derived preview goes to null without a setState-in-effect. Keeps the
  // "Forget 23" button honest — never maps to a stale floor.
  const sig = `${floor}|${maxScan}|${decayTau}`;
  const livePreview = preview && preview.sig === sig ? preview.res : null;
  const armedCount = livePreview?.candidate_ids.length ?? 0;

  function buildBody(dry: boolean) {
    return {
      strength_floor: Number.isFinite(floorNum) ? floorNum : 0.05,
      dry_run: dry,
      max_scan: Number.isFinite(scanNum) ? scanNum : 1000,
      decay_tau_seconds: tauNum,
    };
  }

  function runPreview() {
    m.mutate(buildBody(true), {
      onSuccess: (r) => {
        setPreview({ res: r, sig });
        append({
          op: "forget",
          severity: r.candidate_ids.length > 0 ? "info" : "ok",
          summary: `dry-run · ${r.candidate_ids.length}/${r.scanned} ≤ ${floorNum.toFixed(2)}`,
        });
      },
      onError: (e) => {
        const msg = e instanceof Error ? e.message : "preview failed";
        append({ op: "forget", severity: "err", summary: msg });
        toast.error(msg);
      },
    });
  }

  function runLive() {
    if (!livePreview || armedCount === 0) return;
    m.mutate(buildBody(false), {
      onSuccess: (r) => {
        append({
          op: "forget",
          severity: "warn",
          summary: `removed ${r.removed} episode${r.removed === 1 ? "" : "s"} · floor ${floorNum.toFixed(2)}`,
        });
        toast.success(`Forgot ${r.removed} episode(s).`);
        setPreview(null);
      },
      onError: (e) => {
        const msg = e instanceof Error ? e.message : "forget failed";
        append({ op: "forget", severity: "err", summary: msg });
        toast.error(msg);
      },
    });
  }

  return (
    <Panel
      title="forget"
      eyebrow="prune"
      icon={<Trash2 className="size-3.5" aria-hidden />}
      aside={
        <span className="flex items-center gap-1.5 text-destructive">
          <AlertTriangle className="size-3" aria-hidden />
          irreversible
        </span>
      }
      className="relative w-full overflow-hidden"
      bodyClassName="space-y-4"
    >
      <OpStrip tone="destructive" />

      <div className="space-y-2">
        <Label htmlFor="floor" className="eyebrow">
          strength floor
        </Label>
        <Input
          id="floor"
          inputMode="decimal"
          value={floor}
          onChange={(e) => setFloor(e.target.value)}
          className="font-mono text-[0.78rem]"
        />
        <FloorSlider value={floor} onChange={setFloor} />
        <div className="text-[0.65rem] text-muted-foreground">
          episodes with power-law strength below this floor are eligible.
        </div>
      </div>

      <Field
        id="scan"
        label="max scan"
        value={maxScan}
        onChange={setMaxScan}
        inputMode="numeric"
        hint={`scanning up to ${(Number.isFinite(scanNum) ? scanNum : 0).toLocaleString()} of ${epCount.toLocaleString()} episodes`}
      />

      <AdvancedToggle
        open={advanced}
        onToggle={() => setAdvanced((s) => !s)}
        label="advanced"
      />
      {advanced && (
        <Field
          id="tau"
          label="decay τ (seconds)"
          value={decayTau}
          onChange={setDecayTau}
          inputMode="numeric"
          placeholder="default"
          hint="override the server's half-life. empty = inherit."
        />
      )}

      <div className="border border-border bg-surface-canvas px-3 py-2.5">
        {livePreview ? (
          <div className="space-y-1">
            <ConsoleLine glyph="▌">
              <span className="text-foreground">{armedCount}</span>{" "}
              <span className="text-muted-foreground">
                candidate{armedCount === 1 ? "" : "s"} of
              </span>{" "}
              <span className="text-foreground">{livePreview.scanned}</span>{" "}
              <span className="text-muted-foreground">scanned</span>
            </ConsoleLine>
            {armedCount > 0 && (
              <ConsoleLine glyph="↳" className="text-muted-foreground">
                <span className="truncate">
                  ids:{" "}
                  <span className="text-foreground/80">
                    {livePreview.candidate_ids.slice(0, 14).join(" ")}
                    {livePreview.candidate_ids.length > 14 && " …"}
                  </span>
                </span>
              </ConsoleLine>
            )}
          </div>
        ) : (
          <ConsoleLine glyph="$" className="text-muted-foreground">
            run <span className="text-primary">preview</span> to estimate
            impact — no episodes are deleted.
          </ConsoleLine>
        )}
      </div>

      <div className="flex flex-wrap items-center gap-2">
        <Button
          onClick={runPreview}
          disabled={m.isPending}
          variant="outline"
          size="sm"
          className="gap-1.5 font-mono"
        >
          <Eye className="size-3.5" aria-hidden />
          {m.isPending && !livePreview ? "scanning…" : "preview"}
        </Button>
        <Button
          onClick={runLive}
          disabled={!livePreview || armedCount === 0 || m.isPending}
          variant="destructive"
          size="sm"
          className="gap-1.5 font-mono"
        >
          <Zap className="size-3.5" aria-hidden />
          {m.isPending && livePreview
            ? "purging…"
            : `forget ${armedCount || ""}`.trim()}
        </Button>
      </div>
    </Panel>
  );
}

/* ---------------------------------------------------------- consolidate */

function ConsolidateCard({
  epCount,
  append,
}: {
  epCount: number;
  append: AppendLog;
}) {
  const [session, setSession] = useState("");
  const [nClusters, setNClusters] = useState("");
  const [minSize, setMinSize] = useState("");
  const [advanced, setAdvanced] = useState(false);
  const [last, setLast] = useState<ConsolidateResult | null>(null);
  const m = useConsolidate();

  function run() {
    m.mutate(
      {
        session_id: session.trim() || null,
        n_clusters: nClusters ? Number(nClusters) : null,
        min_cluster_size: minSize ? Number(minSize) : null,
      },
      {
        onSuccess: (r) => {
          setLast(r);
          append({
            op: "consolidate",
            severity: r.schemas_formed > 0 ? "ok" : "info",
            summary:
              r.schemas_formed > 0
                ? `formed ${r.schemas_formed} schema${r.schemas_formed === 1 ? "" : "s"}${session ? ` · ${session}` : ""}`
                : "no clusters reached the size floor",
          });
          toast.success(
            r.schemas_formed > 0
              ? `Formed ${r.schemas_formed} schema(s).`
              : "Nothing new to consolidate.",
          );
        },
        onError: (e) => {
          const msg = e instanceof Error ? e.message : "consolidate failed";
          append({ op: "consolidate", severity: "err", summary: msg });
          toast.error(msg);
        },
      },
    );
  }

  const scope = session.trim()
    ? `session ${session.trim()}`
    : `all ${epCount.toLocaleString()} episodes`;
  const k = nClusters ? Number(nClusters) : null;

  return (
    <Panel
      title="consolidate"
      eyebrow="synthesise"
      icon={<Sparkles className="size-3.5" aria-hidden />}
      aside={
        <span className="flex items-center gap-1.5" style={{ color: "var(--chart-2)" }}>
          <Radio className="size-3" aria-hidden />
          non-destructive
        </span>
      }
      className="relative w-full overflow-hidden"
      bodyClassName="flex flex-col gap-4"
    >
      <OpStrip tone="synth" />

      <Field
        id="cs"
        label="session id"
        value={session}
        onChange={setSession}
        placeholder="all sessions"
        hint="empty = consolidate across the whole store."
      />

      <AdvancedToggle
        open={advanced}
        onToggle={() => setAdvanced((s) => !s)}
        label="cluster overrides"
      />
      {advanced && (
        <div className="grid grid-cols-2 gap-3">
          <Field
            id="nc"
            label="n clusters"
            value={nClusters}
            onChange={setNClusters}
            inputMode="numeric"
            placeholder="auto"
          />
          <Field
            id="mc"
            label="min size"
            value={minSize}
            onChange={setMinSize}
            inputMode="numeric"
            placeholder="auto"
          />
        </div>
      )}

      <div className="flex flex-1 flex-col gap-2 border border-border bg-surface-canvas p-3">
        <div className="eyebrow text-[0.6rem]">plan</div>
        <ConsoleLine glyph="$" className="text-muted-foreground">
          cluster <span className="text-foreground">{scope}</span> into{" "}
          <span className="text-foreground">{k ?? "auto"}</span> schema
          {k === 1 ? "" : "s"}
        </ConsoleLine>
        <ConsoleLine glyph=" " className="text-muted-foreground">
          n clusters{" "}
          <span className="text-foreground">{nClusters || "auto"}</span>
          {"  \u00b7  "}min size{" "}
          <span className="text-foreground">{minSize || "auto"}</span>
        </ConsoleLine>
        <div className="mt-auto border-t border-border/60 pt-2">
          {last ? (
            <ConsoleLine glyph="\u21b3" className="text-muted-foreground">
              last run formed{" "}
              <span
                style={{ color: "var(--chart-2)" }}
                className="font-medium"
              >
                {last.schemas_formed}
              </span>{" "}
              schema{last.schemas_formed === 1 ? "" : "s"}
            </ConsoleLine>
          ) : (
            <ConsoleLine glyph="\u00b7" className="text-muted-foreground/70">
              awaiting first run in this session
            </ConsoleLine>
          )}
        </div>
      </div>

      <Button
        onClick={run}
        disabled={m.isPending}
        size="sm"
        className="gap-1.5 self-start font-mono"
      >
        <Sparkles className="size-3.5" aria-hidden />
        {m.isPending ? "consolidating…" : "consolidate"}
      </Button>

      {last && last.schemas.length > 0 && (
        <details className="border border-border bg-surface-canvas">
          <summary className="cursor-pointer px-3 py-1.5 font-mono text-[0.65rem] uppercase tracking-wider text-muted-foreground hover:text-foreground">
            schemas ({last.schemas.length})
          </summary>
          <ResultBlock value={last.schemas} />
        </details>
      )}
    </Panel>
  );
}

/* --------------------------------------------------------------- reflect */

function ReflectCard({ append }: { append: AppendLog }) {
  const [session, setSession] = useState("");
  const [query, setQuery] = useState("");
  const [k, setK] = useState("5");
  const [last, setLast] = useState<ReflectResult | null>(null);
  const m = useReflect();

  function run() {
    if (!session.trim()) {
      toast.error("Session ID is required for reflect.");
      return;
    }
    m.mutate(
      {
        session_id: session.trim(),
        query: query.trim() || null,
        k: Number(k) || 5,
      },
      {
        onSuccess: (r) => {
          setLast(r);
          append({
            op: "reflect",
            severity: "ok",
            summary: `reflect ${session.trim()}${query ? ` · "${query}"` : ""}`,
          });
          toast.success("Reflection complete.");
        },
        onError: (e) => {
          const msg = e instanceof Error ? e.message : "reflect failed";
          append({ op: "reflect", severity: "err", summary: msg });
          toast.error(msg);
        },
      },
    );
  }

  return (
    <Panel
      title="reflect"
      eyebrow="summarise"
      icon={<Brain className="size-3.5" aria-hidden />}
      aside={
        <span
          className="flex items-center gap-1.5"
          style={{ color: "var(--chart-4)" }}
        >
          <ListTree className="size-3" aria-hidden />
          read-only
        </span>
      }
      className="relative overflow-hidden"
      bodyClassName="space-y-4"
    >
      <OpStrip tone="read" />

      <div className="grid grid-cols-1 gap-3 sm:grid-cols-3">
        <Field
          id="rs"
          label="session id"
          required
          value={session}
          onChange={setSession}
          placeholder="session-id"
        />
        <Field
          id="rq"
          label="query"
          value={query}
          onChange={setQuery}
          placeholder="optional anchor"
        />
        <Field
          id="rk"
          label="k"
          value={k}
          onChange={setK}
          inputMode="numeric"
          hint="recent episodes per cluster."
        />
      </div>

      <Button
        onClick={run}
        disabled={m.isPending || !session.trim()}
        size="sm"
        className="gap-1.5 font-mono"
      >
        <Brain className="size-3.5" aria-hidden />
        {m.isPending ? "reflecting…" : "reflect"}
      </Button>

      {last && <ResultBlock value={last} />}
    </Panel>
  );
}

/* ----------------------------------------------------------- transcript */

function TranscriptPanel({
  entries,
  onClear,
}: {
  entries: LogEntry[];
  onClear: () => void;
}) {
  return (
    <Panel
      title="transcript"
      eyebrow="tail -f admin.log"
      aside={
        <button
          type="button"
          onClick={onClear}
          disabled={entries.length === 0}
          className="font-mono uppercase tracking-wider transition-colors hover:text-primary disabled:opacity-50"
        >
          clear
        </button>
      }
      bodyClassName="p-0"
    >
      {entries.length === 0 ? (
        <div className="px-4 py-3 font-mono text-[0.7rem] text-muted-foreground">
          no operations yet · session-scoped log
        </div>
      ) : (
        <ul className="divide-y divide-border">
          {entries.map((e, i) => (
            <li
              key={`${e.ts}-${i}`}
              className="flex items-start gap-3 px-4 py-1.5 font-mono text-[0.7rem]"
            >
              <span className="shrink-0 text-muted-foreground">
                {fmtTime(e.ts)}
              </span>
              <span
                className={cn(
                  "shrink-0 uppercase tracking-wider",
                  e.severity === "err" && "text-destructive",
                  e.severity === "warn" && "text-primary",
                  e.severity === "ok" && "text-foreground",
                  e.severity === "info" && "text-muted-foreground",
                )}
                style={
                  e.severity === "ok"
                    ? { color: "var(--chart-2)" }
                    : undefined
                }
              >
                {e.op}
              </span>
              <span className="min-w-0 flex-1 truncate text-foreground/85">
                {e.summary}
              </span>
            </li>
          ))}
        </ul>
      )}
    </Panel>
  );
}

/* ----------------------------------------------------------------- hero */

function AdminHero({ ep, sem }: { ep: number; sem: number }) {
  return (
    <Panel
      variant="raised"
      bodyClassName="flex flex-col gap-3 px-5 py-4 sm:flex-row sm:items-center sm:justify-between"
    >
      <div className="min-w-0">
        <div className="eyebrow flex items-center gap-2 text-primary">
          <span className="pulse-dot" aria-hidden />
          maintenance console · live
        </div>
        <div className="mt-2 font-mono text-xl sm:text-2xl">
          <span className="text-muted-foreground">root@</span>
          <span className="text-primary">synara</span>
          <span className="text-muted-foreground">:</span>
          <span className="text-foreground">~/ops</span>
          <span className="text-muted-foreground"># </span>
          <span className="text-foreground">admin</span>
        </div>
        <p className="mt-1.5 max-w-xl text-xs text-muted-foreground">
          Effects are immediate and shared with the live MCP server. Destructive
          ops require an explicit preview.
        </p>
      </div>
      <dl className="grid grid-cols-2 gap-x-6 gap-y-1 text-[0.7rem]">
        {[
          ["episodic", ep.toLocaleString()],
          ["semantic", sem.toLocaleString()],
        ].map(([k, v]) => (
          <div key={k} className="space-y-0.5">
            <dt className="eyebrow text-[0.6rem]">{k}</dt>
            <dd className="metric text-sm tabular-nums text-foreground/90">
              {v}
            </dd>
          </div>
        ))}
      </dl>
    </Panel>
  );
}

/* ---------------------------------------------------------------- shell */

export default function Admin() {
  const stats = useStats();
  const { entries, append, clear } = useTranscript();
  const ep = stats.data?.episodic_count ?? 0;
  const sem = stats.data?.semantic_count ?? 0;

  // Memoize so the transcript prop reference is stable across renders; the
  // cards consume it inside mutation callbacks and we don't want a fresh
  // closure to mask a missed callback.
  const appendStable = useMemo(() => append, [append]);

  return (
    <div>
      <PageHeader
        eyebrow="operations"
        title="admin"
        subtitle="Sanctioned maintenance ops. Forget previews before it bites; consolidate and reflect run inline."
      />

      <div className="grid grid-cols-1 gap-3 lg:grid-cols-6 lg:gap-4">
        <div className="lg:col-span-6">
          <AdminHero ep={ep} sem={sem} />
        </div>

        <div className="flex lg:col-span-3">
          <ForgetCard epCount={ep} append={appendStable} />
        </div>
        <div className="flex lg:col-span-3">
          <ConsolidateCard epCount={ep} append={appendStable} />
        </div>

        <div className="lg:col-span-4">
          <ReflectCard append={appendStable} />
        </div>
        <div className="lg:col-span-2">
          <TranscriptPanel entries={entries} onClear={clear} />
        </div>
      </div>
    </div>
  );
}
