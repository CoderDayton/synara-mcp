/**
 * The memory map.
 *
 * A faithful projection of the live memory system, rendered by the
 * canvas constellation renderer (graph-canvas.tsx):
 *
 *  - Episodic dots   — size ∝ salience + retrieval count; tinted by
 *    session (a *context hint*, not a partition); the focused anchor is
 *    enlarged and ringed.
 *  - Semantic dots   — consolidation schemas, cyan.
 *  - SR edges        — the temporal successor graph; opacity/width track
 *    the discounted closure M (the actual recall-ranking prior).
 *  - Plasticity edges— Hebbian associations, dashed gold; habit edges
 *    (hits ≥ threshold) draw bolder.
 *  - Consolidation   — episode → schema absorption, dotted cyan.
 *
 * Selecting a node spreads activation: its neighbourhood stays lit and
 * everything else dims — the same neighbourhood the recall pipeline
 * would walk. Bridge episodes (routing activation between sessions) get
 * a faint halo ring. Layout is the deterministic session-clustered
 * force embedding (graph-layout.ts), recomputed only when the graph
 * signature changes.
 */
import { useEffect, useRef, useState } from "react";
import type { GraphData, GraphNode } from "@/lib/api";
import { clamp, shortSession } from "@/lib/format";
import { GraphCanvas, type EdgeInfo, type GraphCanvasHandle } from "./graph-canvas";
import { computeLayout, layoutSignature, type Positions } from "./graph-layout";
import {
  bridgeThreshold,
  computeBridgeScores,
  sessionCommunities,
  type CommunityEdge,
} from "./community";

export function MemoryGraph({
  data,
  selectedKey,
  cameraFocus,
  focusNonce = 0,
  onSelect,
  onSelectEdge,
  selectedEdge,
  onFocus,
  onClearFocus,
  highlight,
}: {
  data: GraphData | undefined;
  selectedKey: string | null;
  /** Camera-only focus: pans the viewport to this episode without
   *  refetching or re-laying-out. Keeps the map stable across recenter
   *  clicks. */
  cameraFocus?: number | null;
  /** Bumped on every focus request so repeat recenters re-trigger. */
  focusNonce?: number;
  onSelect: (node: GraphNode | null) => void;
  /** Clicking an edge opens it in the sidebar. */
  onSelectEdge?: (info: EdgeInfo) => void;
  /** The edge currently inspected — stays highlighted and lights its
   *  endpoints. */
  selectedEdge?: EdgeInfo | null;
  onFocus: (episodeId: number) => void;
  /** Clears the camera focus (the × on the focus chip). */
  onClearFocus?: () => void;
  /** Search hits (node keys). When set, the map dims everything else
   *  and rings the matches — search acts on the whole graph. */
  highlight?: Set<string>;
}) {
  // The session-clustered force layout runs synchronously on the main
  // thread — for the graph sizes this view shows it settles in tens of
  // ms. The request id discards a stale result so a slow settle can't
  // overwrite a newer one.
  const [layout, setLayout] = useState<Positions>(new Map());
  const [layoutVersion, setLayoutVersion] = useState(0);
  // Context lens: off by default. When on, same-session episodes are
  // chained in encoded-at order — the timeline of how each context
  // unfolded. Sparse (n-1 links per session, not a clique).
  const [contextLens, setContextLens] = useState(false);
  const [zoom, setZoom] = useState(1);
  const canvasRef = useRef<GraphCanvasHandle | null>(null);
  const reqIdRef = useRef(0);
  const sig = layoutSignature(data);

  useEffect(() => {
    const id = ++reqIdRef.current;
    let canceled = false;
    void computeLayout(data).then((settled) => {
      if (canceled || id !== reqIdRef.current) return;
      setLayout(settled);
      setLayoutVersion((v) => v + 1);
    });
    return () => {
      canceled = true;
    };
  }, [sig]);

  const active = selectedKey;
  const searchOn = !active && !!highlight && highlight.size > 0;

  // Bridge nodes: episodes whose SR/plasticity transitions cross
  // session boundaries. The React Compiler auto-memoizes — this only
  // re-runs when `data` changes.
  const bridgeKeys = new Set<string>();
  if (data && data.nodes.length > 0) {
    const episodicKeys = data.nodes
      .filter((n) => n.kind === "episodic")
      .map((n) => n.key);
    if (episodicKeys.length > 0) {
      const edgeList: CommunityEdge[] = [];
      for (const e of data.sr_edges) {
        edgeList.push({ src: `ep:${e.src}`, dst: `ep:${e.dst}`, w: clamp(e.m, 0, 1) });
      }
      for (const e of data.plasticity_edges) {
        edgeList.push({
          src: `ep:${e.src}`,
          dst: `ep:${e.dst}`,
          w: clamp(e.strength, 0, 1),
        });
      }
      const comms = sessionCommunities(data.nodes);
      const scores = computeBridgeScores(episodicKeys, edgeList, comms);
      const thresh = bridgeThreshold(scores);
      for (const [k, v] of scores) if (v > 0 && v >= thresh) bridgeKeys.add(k);
    }
  }

  // Two ways the map narrows: a selected node lights its
  // spreading-activation neighbourhood; a search lights every hit
  // across the whole graph. Either dims the rest.
  const lit = new Set<string>();
  if (selectedEdge && data) {
    lit.add(`ep:${selectedEdge.src}`);
    lit.add(
      selectedEdge.kind === "consolidation"
        ? selectedEdge.dst
        : `ep:${selectedEdge.dst}`,
    );
  } else if (active && data) {
    lit.add(active);
    for (const e of data.sr_edges) {
      if (`ep:${e.src}` === active) lit.add(`ep:${e.dst}`);
      if (`ep:${e.dst}` === active) lit.add(`ep:${e.src}`);
    }
    for (const e of data.plasticity_edges) {
      if (`ep:${e.src}` === active) lit.add(`ep:${e.dst}`);
      if (`ep:${e.dst}` === active) lit.add(`ep:${e.src}`);
    }
    for (const e of data.consolidation_edges) {
      if (`ep:${e.src}` === active) lit.add(e.dst);
      if (e.dst === active) lit.add(`ep:${e.src}`);
    }
  } else if (searchOn && highlight) {
    for (const k of highlight) lit.add(k);
  }
  const focusing = active != null || searchOn || selectedEdge != null;

  if (!data) return null;

  const activeSession =
    active != null
      ? (() => {
          const n = data.nodes.find((nd) => nd.key === active);
          return n?.kind === "episodic" ? n.session_id : null;
        })()
      : null;

  return (
    <div className="relative h-full w-full">
      <GraphCanvas
        ref={canvasRef}
        data={data}
        positions={layout}
        layoutVersion={layoutVersion}
        selectedKey={selectedKey}
        cameraFocus={cameraFocus ?? null}
        focusNonce={focusNonce}
        lit={lit}
        focusing={focusing}
        highlight={searchOn ? highlight : undefined}
        bridgeKeys={bridgeKeys}
        contextLens={contextLens}
        selectedEdge={selectedEdge}
        onSelect={onSelect}
        onSelectEdge={onSelectEdge}
        onFocus={onFocus}
        onZoom={setZoom}
      />

      {/* Legend + lens toggles, one card */}
      {data.nodes.length > 0 && (
        <div className="absolute bottom-3 left-3 z-20 flex flex-col gap-2">
          <div className="flex flex-col gap-1.5 rounded-md border border-border/70 bg-surface-overlay px-3 py-2.5 font-mono text-[0.6rem] text-muted-foreground backdrop-blur sm:text-[0.62rem]">
            <div className="hidden flex-col gap-1.5 sm:flex">
            <span className="flex items-center gap-1.5">
              <span className="h-0.5 w-5" style={{ background: "var(--color-chart-1)" }} />
              successor (width ∝ M)
            </span>
            <span className="flex items-center gap-1.5">
              <span
                className="h-0 w-5 border-t-2 border-dashed"
                style={{ borderColor: "var(--color-chart-3)" }}
              />
              plasticity (bold = habit)
            </span>
            <span className="flex items-center gap-1.5">
              <span
                className="h-0 w-5 border-t-2 border-dotted"
                style={{ borderColor: "var(--color-chart-2)" }}
              />
              consolidation → schema
            </span>
            <span className="flex items-center gap-1.5">
              <span
                aria-hidden
                className="size-2 rounded-full border border-muted-foreground/40"
                style={{ background: "oklch(0.65 0.13 200 / 0.4)" }}
              />
              dot = memory · halo = bridge
            </span>
            {contextLens && (
              <span className="flex items-center gap-1.5">
                <span className="h-0.5 w-5" style={{ background: "oklch(0.72 0.14 200)" }} />
                context (same session)
              </span>
            )}
            <span className="mt-1 flex items-center gap-2 border-t border-border/60 pt-1.5 text-foreground/70">
              <span>ω {data.omega.toFixed(2)}</span>
              <span>·</span>
              <span>{data.episode_count} ep</span>
              {(data.truncated || data.semantics_truncated) && (
                <span className="text-warning">
                  · capped{data.semantics_truncated && !data.truncated ? " (semantics)" : ""}
                </span>
              )}
            </span>
            </div>
            {/* The lens toggles live with the legend — they change what
             *  the legend describes, so they belong on the same card. */}
            <button
              type="button"
              aria-pressed={contextLens}
              title="Context lens — link same-session episodes in time order"
              onClick={() => setContextLens((v) => !v)}
              className={`flex items-center gap-1.5 rounded px-1.5 py-1 text-left text-[0.62rem] font-medium transition-colors focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-primary sm:mt-0.5 sm:border-t sm:border-border/60 sm:pt-1.5 ${
                contextLens
                  ? "text-primary"
                  : "text-muted-foreground hover:text-foreground"
              }`}
            >
              <span
                aria-hidden
                className={`grid size-3 place-items-center rounded-[3px] border text-[0.5rem] leading-none ${
                  contextLens
                    ? "border-primary bg-primary/20 text-primary"
                    : "border-muted-foreground/50 text-transparent"
                }`}
              >
                ✓
              </span>
              context lens
            </button>
          </div>
        </div>
      )}

      {/* Zoom controls */}
      <div className="absolute right-3 bottom-3 z-20 flex items-center gap-0.5 rounded-md border border-border/70 bg-surface-overlay/85 p-1 font-mono shadow-card backdrop-blur">
        <button
          type="button"
          aria-label="Zoom out"
          title="Zoom out"
          onClick={() => canvasRef.current?.zoomBy(0.8)}
          className="grid size-7 place-items-center rounded text-base leading-none text-muted-foreground transition-colors hover:bg-foreground/10 hover:text-foreground focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-primary"
        >
          −
        </button>
        <button
          type="button"
          aria-label={`Fit view (current zoom ${zoom.toFixed(2)}×)`}
          title="Fit view"
          onClick={() => canvasRef.current?.fit()}
          className="min-w-12 rounded px-2 py-1 text-center text-[0.7rem] font-medium tabular-nums text-foreground/85 transition-colors hover:bg-foreground/10 hover:text-foreground focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-primary"
        >
          {zoom.toFixed(2)}×
        </button>
        <button
          type="button"
          aria-label="Zoom in"
          title="Zoom in"
          onClick={() => canvasRef.current?.zoomBy(1.25)}
          className="grid size-7 place-items-center rounded text-base leading-none text-muted-foreground transition-colors hover:bg-foreground/10 hover:text-foreground focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-primary"
        >
          +
        </button>
      </div>

      {/* Focus chip: shows which episode the camera is parked on; the
       *  crosshair re-centers, the × releases the focus. */}
      {cameraFocus != null && (
        <div className="absolute top-3 left-3 z-20 flex items-center gap-1 rounded-full border border-primary/40 bg-primary/10 py-0.5 pr-1 pl-2.5 font-mono text-[0.6rem] text-primary shadow-card backdrop-blur sm:text-[0.62rem]">
          <button
            type="button"
            title={`Recenter on #${cameraFocus}`}
            onClick={() => onFocus(cameraFocus)}
            className="flex items-center gap-1 rounded-full transition-colors hover:text-primary/80 focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-primary"
          >
            <span aria-hidden>⌖</span>
            focus #{cameraFocus}
          </button>
          {onClearFocus && (
            <button
              type="button"
              aria-label="Clear focus"
              title="Clear focus"
              onClick={onClearFocus}
              className="grid size-4 place-items-center rounded-full text-primary/70 transition-colors hover:bg-primary/20 hover:text-primary focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-primary"
            >
              ×
            </button>
          )}
        </div>
      )}

      {/* Node count / active session */}
      <div className="absolute top-3 right-3 z-10 hidden max-w-[12rem] text-right font-mono text-[0.6rem] leading-relaxed text-muted-foreground/70 sm:block">
        {data.nodes.length > 0
          ? `${data.nodes.length} nodes${
              active ? ` · ${shortSession(activeSession)}` : ""
            }`
          : ""}
      </div>
    </div>
  );
}
