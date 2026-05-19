import { Activity, useMemo, useState } from "react";
import cytoscape from "cytoscape";
import fcose from "cytoscape-fcose";
import CytoscapeComponent from "react-cytoscapejs";
import type { ElementDefinition, StylesheetJson } from "cytoscape";
import { Eye, Search } from "lucide-react";
import { useGraph, useMemories } from "@/lib/queries";
import { PageHeader } from "@/components/common/page-header";
import { Empty, ErrorState, Loading } from "@/components/common/states";
import { MemoryDetailDialog } from "@/components/memories/memory-detail-dialog";
import { Button } from "@/components/ui/button";
import { cn } from "@/lib/utils";

cytoscape.use(fcose);

type Kind = "episodic" | "semantic";
const KINDS: Kind[] = ["episodic", "semantic"];
const LIST = 60;

// Canvas needs concrete colors (no oklch); these track the design tokens.
const C = {
  node: "#26272f",
  nodeBorder: "#7c5cff",
  focus: "#a78bfa",
  label: "#d8d8e0",
  sr: "#8b5cf6",
  plastic: "#e0a93b",
};

const STYLESHEET = [
  {
    selector: "node",
    style: {
      "background-color": C.node,
      "border-color": C.nodeBorder,
      "border-width": 1.5,
      label: "data(label)",
      color: C.label,
      "font-size": 9,
      "font-family": "JetBrains Mono Variable, monospace",
      "text-valign": "center",
      "text-halign": "center",
      width: 26,
      height: 26,
    },
  },
  {
    selector: "node.focus",
    style: {
      "background-color": C.focus,
      "border-width": 2.5,
      width: 46,
      height: 46,
      "font-size": 11,
      "z-index": 10,
    },
  },
  {
    selector: "node.dim",
    style: { opacity: 0.25 },
  },
  {
    selector: "edge.sr",
    style: {
      width: "mapData(hits, 1, 20, 1, 6)",
      "line-color": C.sr,
      "target-arrow-color": C.sr,
      "target-arrow-shape": "triangle",
      "curve-style": "bezier",
      opacity: 0.7,
    },
  },
  {
    selector: "edge.plastic",
    style: {
      width: 1.5,
      "line-color": C.plastic,
      "line-style": "dashed",
      "curve-style": "bezier",
      opacity: 0.55,
    },
  },
  // Cytoscape stylesheet css unions are intentionally stringly; a single
  // cast keeps the styling readable without per-property noise.
] as unknown as StylesheetJson;

const LAYOUT = {
  name: "fcose",
  quality: "default",
  animate: false,
  randomize: true,
  nodeSeparation: 75,
} as unknown as cytoscape.LayoutOptions;

export default function Memories() {
  const [kind, setKind] = useState<Kind>("episodic");
  const [query, setQuery] = useState("");
  const [submitted, setSubmitted] = useState("");
  const [focus, setFocus] = useState<number | null>(null);
  const [detailId, setDetailId] = useState<number | null>(null);

  const graph = useGraph(
    focus != null
      ? { focus, depth: 2, max_nodes: 300 }
      : { depth: 1, max_nodes: 300 },
  );
  const list = useMemories({
    kind,
    q: submitted || undefined,
    limit: LIST,
  });

  const data = graph.data;
  const rows =
    list.data && "items" in list.data
      ? (list.data.items as Array<{
          id?: number;
          content?: string;
          score?: number;
        }>)
      : [];
  const isSearch = !!submitted;

  const elements = useMemo<ElementDefinition[]>(() => {
    if (!data) return [];
    const els: ElementDefinition[] = data.nodes.map((id) => ({
      data: { id: String(id), label: `#${id}` },
      classes: focus === id ? "focus" : undefined,
    }));
    data.sr_edges.forEach((e, i) =>
      els.push({
        data: {
          id: `s${i}`,
          source: String(e.src),
          target: String(e.dst),
          hits: e.hits,
        },
        classes: "sr",
      }),
    );
    data.plasticity_edges.forEach((e, i) =>
      els.push({
        data: { id: `p${i}`, source: String(e.src), target: String(e.dst) },
        classes: "plastic",
      }),
    );
    return els;
  }, [data, focus]);

  function select(id: number) {
    setFocus(id);
    setDetailId(id);
  }

  function runSearch(e: React.FormEvent) {
    e.preventDefault();
    setSubmitted(query.trim());
  }

  function switchKind(k: Kind) {
    setKind(k);
    setSubmitted("");
    setQuery("");
  }

  return (
    <div className="space-y-6">
      <PageHeader
        eyebrow="Store"
        title="Memory map"
        subtitle="Every memory as a node. Solid edges are successor transitions, dashed are plasticity. Click a node to inspect its trace."
        actions={
          data && (
            <div className="flex flex-wrap items-center gap-x-5 gap-y-1">
              {[
                ["nodes", data.nodes.length],
                ["sr", data.sr_edges.length],
                ["plastic", data.plasticity_edges.length],
              ].map(([k, v]) => (
                <div key={k} className="flex flex-col">
                  <span className="eyebrow">{k}</span>
                  <span className="metric text-base">{v}</span>
                </div>
              ))}
              {focus != null && (
                <Button
                  variant="outline"
                  size="sm"
                  onClick={() => setFocus(null)}
                >
                  Reset view
                </Button>
              )}
            </div>
          )
        }
      />

      <div className="grid grid-cols-1 gap-px overflow-hidden border border-border/70 bg-border/70 lg:grid-cols-[300px_1fr]">
        {/* ── SIDE RAIL ─────────────────────────────────────── */}
        <div className="flex flex-col bg-card">
          <div className="flex">
            {KINDS.map((k) => (
              <button
                key={k}
                type="button"
                onClick={() => switchKind(k)}
                aria-pressed={kind === k}
                className={cn(
                  "eyebrow flex-1 px-4 py-3.5 transition-colors",
                  kind === k
                    ? "bg-primary/12 text-primary"
                    : "text-muted-foreground hover:bg-muted/50 hover:text-foreground",
                )}
              >
                {k}
              </button>
            ))}
          </div>
          <form
            onSubmit={runSearch}
            className="flex items-center gap-2 border-y border-border/70 px-3 py-2"
          >
            <Search
              className="size-4 shrink-0 text-muted-foreground"
              aria-hidden
            />
            <input
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              placeholder={`search ${kind}…`}
              aria-label="Search memories"
              className="min-w-0 flex-1 bg-transparent font-mono text-sm outline-none placeholder:text-muted-foreground/60"
            />
          </form>
          <div className="max-h-[58vh] flex-1 overflow-auto">
            {list.isLoading && (
              <div className="p-4">
                <Loading label="Loading" />
              </div>
            )}
            {list.error && (
              <div className="p-4">
                <ErrorState error={list.error} />
              </div>
            )}
            {!list.isLoading && rows.length === 0 && (
              <p className="p-4 text-xs text-muted-foreground">
                {isSearch ? "No matches." : `No ${kind} memories yet.`}
              </p>
            )}
            <ul className="divide-y divide-border/50">
              {rows.map((r, i) => (
                <li key={r.id ?? i}>
                  <button
                    type="button"
                    disabled={r.id == null}
                    onClick={() => r.id != null && select(r.id)}
                    className={cn(
                      "flex w-full items-start gap-2 px-4 py-3 text-left text-sm transition-colors hover:bg-muted/40",
                      focus === r.id && "bg-primary/10",
                    )}
                  >
                    <span className="shrink-0 font-mono text-xs tabular-nums text-muted-foreground">
                      #{r.id ?? "—"}
                    </span>
                    <span className="line-clamp-2 text-foreground/85">
                      {r.content ?? "—"}
                    </span>
                  </button>
                </li>
              ))}
            </ul>
          </div>
        </div>

        {/* ── MAP ───────────────────────────────────────────── */}
        <div className="relative min-h-[60vh] bg-card lg:min-h-[72vh]">
          {graph.isLoading && (
            <div className="grid h-full place-items-center">
              <Loading label="Building map" />
            </div>
          )}
          {graph.error && (
            <div className="p-6">
              <ErrorState error={graph.error} />
            </div>
          )}
          {data && elements.length === 0 && (
            <div className="grid h-full place-items-center p-6">
              <Empty
                label="No memory graph yet"
                hint="The map is built from episode co-occurrence within a session window. Encode and recall a few episodes, then return."
              />
            </div>
          )}
          <Activity mode={data && elements.length > 0 ? "visible" : "hidden"}>
            <div className="absolute inset-0">
              <CytoscapeComponent
                elements={elements}
                stylesheet={STYLESHEET}
                layout={LAYOUT}
                style={{ width: "100%", height: "100%" }}
                minZoom={0.2}
                maxZoom={2.5}
                cy={(c) => {
                  c.removeAllListeners();
                  c.on("tap", "node", (e) =>
                    select(Number(e.target.id())),
                  );
                }}
              />
              <div className="pointer-events-none absolute bottom-3 left-3 flex gap-4 border border-border/70 bg-background/80 px-3 py-2 font-mono text-[0.65rem] text-muted-foreground backdrop-blur">
                <span className="flex items-center gap-1.5">
                  <span className="h-px w-4 bg-[#8b5cf6]" /> successor
                </span>
                <span className="flex items-center gap-1.5">
                  <span className="h-px w-4 border-t border-dashed border-[#e0a93b]" />{" "}
                  plasticity
                </span>
              </div>
            </div>
          </Activity>
        </div>
      </div>

      <p className="flex items-center gap-1.5 text-xs text-muted-foreground">
        <Eye className="size-3.5" aria-hidden />
        Selecting a node focuses its 2-hop neighbourhood and opens the trace.
      </p>

      <MemoryDetailDialog
        id={detailId}
        onOpenChange={(o) => !o && setDetailId(null)}
      />
    </div>
  );
}
