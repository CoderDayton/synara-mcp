/**
 * Memory map.
 *
 * The graph *is* the page. Search runs the recall pipeline over the
 * whole store and lights every hit on the map (dimming the rest);
 * selecting a node traces its structure in the inspector. Browsing
 * with no query shows the global successor graph.
 */
import { useEffect, useMemo, useState } from "react";
import { AlertTriangle, RotateCcw, Search, X } from "lucide-react";
import { useGraph, useMemories } from "@/lib/queries";
import type { GraphNode, MemoryList } from "@/lib/api";
import { PageHeader } from "@/components/common/page-header";
import { Empty, ErrorState, Loading } from "@/components/common/states";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { MemoryGraph } from "@/components/memories/memory-graph";
import { MemoryInspector } from "@/components/memories/memory-inspector";

function recallIds(list: MemoryList | undefined): number[] {
  if (!list || !("query" in list)) return [];
  return list.items
    .map((h) => (typeof h.id === "number" ? h.id : -1))
    .filter((id) => id >= 0);
}

export default function Memories() {
  const [draft, setDraft] = useState("");
  const [query, setQuery] = useState("");
  const [focus, setFocus] = useState<number | null>(null);
  const [depth, setDepth] = useState(2);
  const [selected, setSelected] = useState<GraphNode | null>(null);

  const search = useMemories({
    kind: "episodic",
    q: query || undefined,
    limit: 60,
  });
  const graph = useGraph(
    focus != null
      ? { focus, depth, max_nodes: 260 }
      : { depth: 1, max_nodes: 260 },
  );

  const matchIds = useMemo(
    () => (query ? recallIds(search.data) : []),
    [search.data, query],
  );
  const highlight = useMemo(
    () => new Set(matchIds.map((id) => `ep:${id}`)),
    [matchIds],
  );

  // "Search the whole map": if the top hit isn't already on the
  // current canvas, recenter on it so its neighbourhood loads — the
  // highlight then lands on real nodes instead of nothing.
  useEffect(() => {
    if (!query || matchIds.length === 0 || !graph.data) return;
    const present = new Set(
      graph.data.nodes.map((n) => (n.kind === "episodic" ? n.id : -1)),
    );
    // Intentional view-state sync from async recall results — there is
    // no earlier point (results arrive after submit, via react-query).
    // eslint-disable-next-line react-hooks/set-state-in-effect
    if (!matchIds.some((id) => present.has(id))) setFocus(matchIds[0]);
  }, [query, matchIds, graph.data]);

  function reset() {
    setFocus(null);
    setSelected(null);
    setQuery("");
    setDraft("");
  }

  const showEmpty =
    graph.data && graph.data.nodes.length === 0 && !graph.isLoading;

  const showDepth = focus != null;
  const showReset = focus != null || !!query;

  return (
    <section className="flex min-h-0 flex-1 flex-col gap-6 sm:gap-8">
      <PageHeader
        eyebrow="Store"
        title="Memory map"
        subtitle="The successor graph, plasticity associations, and consolidation schemas of the live store — search to light up matches across the whole map."
      />

      {graph.data?.legacy && (
        <Alert variant="destructive" className="border-warning/50">
          <AlertTriangle className="size-4" aria-hidden />
          <AlertTitle>Stale server — restart required</AlertTitle>
          <AlertDescription>
            This dashboard is talking to a memory server that started
            before the enriched graph route was deployed. Nodes have no
            salience, session, or successor/plasticity structure until
            the MCP server process is restarted. The map below is a
            best-effort legacy view.
          </AlertDescription>
        </Alert>
      )}

      <div className="relative flex min-h-[28rem] flex-1 overflow-hidden rounded-xl border border-border/70 bg-surface-canvas shadow-card">
        {/* Search operates on the whole map */}
        <form
          onSubmit={(e) => {
            e.preventDefault();
            setQuery(draft.trim());
          }}
          className="absolute top-3 left-1/2 z-20 flex w-[min(26rem,calc(100%-1.5rem))] -translate-x-1/2 items-center gap-1.5 rounded-lg border border-border/70 bg-surface-overlay p-1.5 shadow-card backdrop-blur sm:top-4"
        >
          <Search
            className="ml-1.5 size-4 shrink-0 text-muted-foreground"
            aria-hidden
          />
          <Input
            value={draft}
            onChange={(e) => setDraft(e.target.value)}
            placeholder="Search the memory map…"
            aria-label="Search the memory map"
            className="h-8 border-0 bg-transparent px-1 shadow-none focus-visible:ring-0"
          />
          {query && (
            <span className="shrink-0 px-1 font-mono text-[0.65rem] text-muted-foreground">
              {matchIds.length} hit{matchIds.length === 1 ? "" : "s"}
            </span>
          )}
          {(query || draft) && (
            <Button
              type="button"
              variant="ghost"
              size="icon-sm"
              aria-label="Clear search"
              onClick={reset}
            >
              <X className="size-4" aria-hidden />
            </Button>
          )}
          <Button type="submit" size="sm" variant="secondary">
            Search
          </Button>
        </form>

        {/* Top-right: focus controls — depth + reset, only shown when relevant */}
        {(showDepth || showReset) && (
          <div className="absolute top-3 right-3 z-20 flex items-center gap-1 rounded-lg border border-border/70 bg-surface-overlay p-1 shadow-card backdrop-blur sm:top-4 sm:right-4">
            {showDepth && (
              <Select
                value={String(depth)}
                onValueChange={(v) => setDepth(Number(v))}
              >
                <SelectTrigger
                  size="sm"
                  aria-label="Neighbourhood depth"
                  className="h-8 gap-2 border-0 bg-transparent px-2 shadow-none focus-visible:ring-0 dark:bg-transparent dark:hover:bg-transparent"
                >
                  <span className="eyebrow">depth</span>
                  <SelectValue className="font-mono text-xs tabular-nums" />
                </SelectTrigger>
                <SelectContent align="end">
                  <SelectItem value="1">1</SelectItem>
                  <SelectItem value="2">2</SelectItem>
                  <SelectItem value="3">3</SelectItem>
                </SelectContent>
              </Select>
            )}
            {showReset && (
              <Button
                type="button"
                variant="ghost"
                size="sm"
                onClick={reset}
                className="h-8 px-2"
              >
                <RotateCcw className="size-3.5" aria-hidden />
                Reset
              </Button>
            )}
          </div>
        )}

        {/* Graph */}
        <div className="relative min-w-0 flex-1">
          {graph.isLoading && <Loading label="Building map" />}
          {graph.error && (
            <div className="p-6">
              <ErrorState error={graph.error} />
            </div>
          )}
          {showEmpty && (
            <div className="grid h-full place-items-center p-6">
              <Empty
                label="No memory graph yet"
                hint="The map is built from successor transitions and plasticity edges. Store and recall a few episodes, then return."
              />
            </div>
          )}
          {graph.data && graph.data.nodes.length > 0 && (
            <MemoryGraph
              data={graph.data}
              selectedKey={selected?.key ?? null}
              highlight={highlight}
              onSelect={setSelected}
              onFocus={(id) => setFocus(id)}
            />
          )}
          {search.error && (
            <div className="absolute right-3 bottom-3 z-20">
              <ErrorState error={search.error} />
            </div>
          )}
        </div>

        {/* Inspector */}
        {selected && (
          <aside className="absolute inset-y-0 right-0 z-30 w-full border-l border-border/60 bg-surface-floating backdrop-blur sm:w-[22rem] lg:static lg:bg-surface-canvas lg:backdrop-blur-none">
            <MemoryInspector
              node={selected}
              onFocus={(id) => {
                setFocus(id);
                setDepth((d) => Math.max(d, 2));
              }}
              onClose={() => setSelected(null)}
            />
          </aside>
        )}
      </div>
    </section>
  );
}
