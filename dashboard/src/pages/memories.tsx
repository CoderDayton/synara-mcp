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
import type { GraphNode, MemoryList, RecallMode } from "@/lib/api";
import { PageHeader } from "@/components/common/page-header";
import { Empty, ErrorState, Loading } from "@/components/common/states";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { MemoryGraph } from "@/components/memories/memory-graph";
import { MemoryInspector } from "@/components/memories/memory-inspector";

function recallIds(list: MemoryList | undefined): number[] {
  if (!list || !("query" in list)) return [];
  return list.items
    .map((h) => (typeof h.id === "number" ? h.id : -1))
    .filter((id) => id >= 0);
}

function recallMode(list: MemoryList | undefined): RecallMode | null {
  if (!list || !("query" in list)) return null;
  return list.recall_mode ?? null;
}

/** Combine the episodic + semantic search modes into a single label
 *  that reflects how the lit-up node set was produced. Substring wins
 *  the override when either leg fell back, so the user knows they're
 *  not seeing pure semantic recall. */
function combineModes(
  a: RecallMode | null,
  b: RecallMode | null,
): RecallMode | null {
  const modes = [a, b].filter((m): m is RecallMode => m != null);
  if (modes.length === 0) return null;
  if (modes.includes("hybrid")) return "hybrid";
  if (modes.includes("substring") && modes.includes("semantic")) return "hybrid";
  if (modes.includes("substring")) return "substring";
  if (modes.includes("semantic")) return "semantic";
  return "empty";
}

const MODE_LABEL: Record<RecallMode, { text: string; hint: string }> = {
  semantic: {
    text: "semantic",
    hint: "Matches ranked by vector similarity (cosine + SR re-ranking).",
  },
  substring: {
    text: "substring",
    hint: "Vector recall returned nothing — falling back to literal substring matches.",
  },
  hybrid: {
    text: "hybrid",
    hint: "Vector recall + literal substring matches both contributed to these hits.",
  },
  empty: {
    text: "no hits",
    hint: "Neither vector recall nor a substring scan found anything.",
  },
};

export default function Memories() {
  const [draft, setDraft] = useState("");
  const [query, setQuery] = useState("");
  // Camera focus only — drives a viewport pan/zoom on the existing
  // UMAP layout. We deliberately do NOT pass this to the server: the
  // global UMAP manifold must stay stable across recenter clicks, so
  // every fetch sees the same node set.
  const [focus, setFocus] = useState<number | null>(null);
  const [selected, setSelected] = useState<GraphNode | null>(null);

  const episodicSearch = useMemories({
    kind: "episodic",
    q: query || undefined,
    limit: 60,
  });
  const semanticSearch = useMemories({
    kind: "semantic",
    q: query || undefined,
    limit: 60,
  });
  const graph = useGraph({ depth: 1, max_nodes: 260 });

  const matchIds = useMemo(
    () => (query ? recallIds(episodicSearch.data) : []),
    [episodicSearch.data, query],
  );
  const semanticMatchIds = useMemo(
    () => (query ? recallIds(semanticSearch.data) : []),
    [semanticSearch.data, query],
  );
  const highlight = useMemo(() => {
    const s = new Set<string>();
    for (const id of matchIds) s.add(`ep:${id}`);
    for (const id of semanticMatchIds) s.add(`sem:${id}`);
    return s;
  }, [matchIds, semanticMatchIds]);
  const totalHits = matchIds.length + semanticMatchIds.length;
  const searchError = episodicSearch.error ?? semanticSearch.error;
  const mode = useMemo(
    () =>
      query
        ? combineModes(recallMode(episodicSearch.data), recallMode(semanticSearch.data))
        : null,
    [query, episodicSearch.data, semanticSearch.data],
  );

  // "Search the whole map": pan the camera to the top hit so its
  // neighbourhood is visible. Camera-only — no refetch, no relayout.
  useEffect(() => {
    if (!query || matchIds.length === 0 || !graph.data) return;
    // eslint-disable-next-line react-hooks/set-state-in-effect
    setFocus(matchIds[0]);
  }, [query, matchIds, graph.data]);

  function reset() {
    setFocus(null);
    setSelected(null);
    setQuery("");
    setDraft("");
  }

  const showEmpty =
    graph.data && graph.data.nodes.length === 0 && !graph.isLoading;

  const showReset = focus != null || !!query;

  return (
    <section className="flex min-h-0 flex-1 flex-col gap-3">
      <PageHeader
        eyebrow="store"
        title="memory map"
        subtitle="The successor graph, plasticity associations, and consolidation schemas of the live store — search to light up matches across the whole map."
      />

      {graph.data?.legacy && (
        <Alert variant="destructive" className="border-warning/50 font-mono text-xs">
          <AlertTriangle className="size-4 text-warning" aria-hidden />
          <AlertTitle className="uppercase tracking-wider text-warning">
            stale server — restart required
          </AlertTitle>
          <AlertDescription>
            This dashboard is talking to a memory server that started
            before the enriched graph route was deployed. Nodes have no
            salience, context, or successor/plasticity structure until
            the MCP server process is restarted. The map below is a
            best-effort legacy view.
          </AlertDescription>
        </Alert>
      )}

      <div className="panel relative flex min-h-[28rem] flex-1 overflow-hidden bg-surface-canvas">
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
            <span
              className="flex shrink-0 items-center gap-1.5 px-1 font-mono text-[0.65rem] text-muted-foreground"
              role="status"
              aria-live="polite"
            >
              <span>
                {totalHits} hit{totalHits === 1 ? "" : "s"}
              </span>
              {mode && (
                <span
                  title={MODE_LABEL[mode].hint}
                  className={
                    "rounded px-1.5 py-0.5 text-[0.6rem] uppercase tracking-wide " +
                    (mode === "substring"
                      ? "bg-warning/15 text-warning"
                      : mode === "hybrid"
                        ? "bg-chart-1/15 text-chart-1"
                        : mode === "empty"
                          ? "bg-muted text-muted-foreground"
                          : "bg-chart-2/15 text-chart-2")
                  }
                >
                  {MODE_LABEL[mode].text}
                </span>
              )}
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

        {/* Top-right: focus controls — depth + reset, only shown when
         *  relevant. The inspector covers the right edge at three
         *  breakpoints:
         *   - <sm: inspector is w-full → reset would be fully buried,
         *     so we hide it (the inspector's own close button lets the
         *     user dismiss the panel first).
         *   - sm–lg: inspector is a 22rem absolute panel → shift the
         *     reset left of it.
         *   - lg+: inspector is static and pushes the graph; the
         *     graph's own right edge is where right-4 lands. */}
        {showReset && (
          <div
            className={
              "absolute top-3 z-40 flex items-center gap-1 rounded-lg border border-border/70 bg-surface-overlay p-1 shadow-card backdrop-blur sm:top-4 " +
              (selected
                ? "hidden sm:flex sm:right-[calc(22rem+1rem)] lg:right-4"
                : "right-3 sm:right-4")
            }
          >
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
              cameraFocus={focus}
              highlight={highlight}
              onSelect={setSelected}
              onFocus={(id) => setFocus(id)}
            />
          )}
          {searchError && (
            <div className="absolute right-3 bottom-3 z-20">
              <ErrorState error={searchError} />
            </div>
          )}
        </div>

        {/* Inspector */}
        {selected && (
          <aside className="absolute inset-y-0 right-0 z-30 w-full border-l border-border/60 bg-surface-floating backdrop-blur sm:w-[22rem] lg:static lg:bg-surface-canvas lg:backdrop-blur-none">
            <MemoryInspector
              node={selected}
              onFocus={(id) => setFocus(id)}
              onClose={() => setSelected(null)}
            />
          </aside>
        )}
      </div>
    </section>
  );
}
