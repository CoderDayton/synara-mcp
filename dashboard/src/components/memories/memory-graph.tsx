/**
 * The memory map.
 *
 * A faithful projection of the live memory system onto @xyflow/react:
 *
 *  - Episodic nodes  — size ∝ salience + retrieval count; tinted by
 *    session (a *context hint*, not a partition); the focused anchor is
 *    enlarged and ringed.
 *  - Semantic nodes  — consolidation schemas; ringed by confidence.
 *  - SR edges        — the temporal successor graph. Opacity/width track
 *    the discounted closure M (the actual recall-ranking prior), arrow
 *    in the transition direction.
 *  - Plasticity edges— Hebbian associations. Dashed gold; width ∝
 *    weight+bonus; habit edges (hits ≥ threshold) are thicker and glow.
 *  - Consolidation   — episode → schema absorption, dashed cyan.
 *
 * Selecting a node spreads activation: its association edges animate and
 * everything else dims — the same neighbourhood the recall pipeline
 * would walk. Layout is ELK's 'layered' (Sugiyama-style hierarchic)
 * pass — the same top-down shape yFiles' HierarchicLayout produces —
 * recomputed whenever the graph signature changes.
 */
import {
  Background,
  BackgroundVariant,
  Handle,
  MarkerType,
  MiniMap,
  Panel,
  Position,
  ReactFlow,
  useReactFlow,
  useStore,
  useStoreApi,
  type Edge,
  type Node,
  type NodeProps,
  type NodeTypes,
} from "@xyflow/react";
import "@xyflow/react/dist/style.css";
import { useEffect, useRef, useState } from "react";
import type { GraphData, GraphNode } from "@/lib/api";
import { clamp, sessionHue, shortSession } from "@/lib/format";
import {
  computeLayout,
  layoutSignature,
  nodeRadius,
  type Positions,
} from "./graph-layout";
import {
  bridgeThreshold,
  buildClusterHulls,
  computeBridgeScores,
  detectCommunities,
  polygonPath,
  type CommunityEdge,
  type HullEntry,
} from "./community";
import {
  HoverCard,
  HoverCardContent,
  HoverCardTrigger,
} from "@/components/ui/hover-card";

type NodeData = {
  node: GraphNode;
  dim: boolean;
  active: boolean;
  /** This node sits above the bridge-score threshold — it routes
   *  activation between communities. The disc renderer draws an
   *  outer halo when true so these routing abstractions are visible
   *  at a glance. */
  isBridge: boolean;
  /** Keyboard handlers threaded through node data so each rendered
   *  node can self-activate without relying on ReactFlow's mouse-only
   *  event hooks. Bound to the same callbacks ReactFlow's click hooks
   *  fire — keyboard then becomes a first-class input. */
  onSelect: (n: GraphNode) => void;
  onFocus?: (id: number) => void;
};
type FlowNode = Node<NodeData>;

/** Shared keyboard handler for episodic/semantic node renderers.
 *
 * - Enter/Space: select the node (mirrors single-click).
 * - 'F'        : recenter the map on this episode (mirrors double-click).
 *                Ignored for schemas — they have no SR focus.
 *
 * `preventDefault` on Space stops page-scroll; on Enter it avoids
 * surfacing the event to ReactFlow's pane handler which would clear
 * the selection. */
function handleNodeKey(
  e: React.KeyboardEvent,
  data: NodeData,
): void {
  if (e.key === "Enter" || e.key === " ") {
    e.preventDefault();
    e.stopPropagation();
    data.onSelect(data.node);
    return;
  }
  if ((e.key === "f" || e.key === "F") && data.node.kind === "episodic") {
    e.preventDefault();
    e.stopPropagation();
    data.onFocus?.(data.node.id);
  }
}

/* ----------------------------------------------------------- node renderers */

function hiddenHandles() {
  return (
    <>
      <Handle
        type="target"
        position={Position.Top}
        isConnectable={false}
        style={{ opacity: 0, width: 1, height: 1, border: 0, minWidth: 0 }}
      />
      <Handle
        type="source"
        position={Position.Bottom}
        isConnectable={false}
        style={{ opacity: 0, width: 1, height: 1, border: 0, minWidth: 0 }}
      />
    </>
  );
}

/** Zoom level above which node captions fade in. Past the `fitView`
 *  ceiling (1.4) so the overview stays clean and captions only appear
 *  once the user has *deliberately* zoomed in to investigate. */
const CAPTION_ZOOM_MIN = 1.5;
/** Pixels of slack around the viewport when culling captions — keeps a
 *  caption visible for a node that's just being panned in. */
const CAPTION_VIEWPORT_PAD = 80;

function EpisodicNode({ data, selected }: NodeProps<FlowNode>) {
  const n = data.node;
  if (n.kind !== "episodic") return null;
  const r = nodeRadius(n);
  const hue = sessionHue(n.session_id);
  const consolidated = n.consolidated_into > 0;
  const label =
    `Episode #${n.id}` +
    (n.session_id ? `, session ${shortSession(n.session_id)}` : "") +
    `, salience ${n.salience.toFixed(2)}` +
    (consolidated ? ", consolidated" : "") +
    ". Enter to inspect, F to recenter.";
  return (
    <HoverCard>
      <HoverCardTrigger asChild>
        <div
          role="button"
          tabIndex={0}
          aria-label={label}
          aria-pressed={selected}
          onKeyDown={(e) => handleNodeKey(e, data)}
          style={{
            width: r * 2,
            height: r * 2,
            opacity: data.dim ? 0.16 : 1,
          }}
          className="relative grid place-items-center font-mono"
        >
          {hiddenHandles()}
          <div
            aria-hidden
            style={{
              background: `oklch(0.34 0.10 ${hue} / 0.65)`,
              borderColor:
                n.is_focus || selected
                  ? "var(--color-primary)"
                  : `oklch(0.80 0.18 ${hue} / 0.85)`,
              // Active states win; the bridge halo only shows on
              // resting nodes so the selection ring stays unambiguous.
              boxShadow:
                n.is_focus || selected || data.active
                  ? "0 0 0 2px var(--color-primary), 0 0 22px -4px var(--color-primary)"
                  : data.isBridge
                    ? `0 0 0 1.5px oklch(0.80 0.15 ${hue} / 0.6), 0 0 0 5px oklch(0.85 0.10 ${hue} / 0.18), 0 0 20px -2px oklch(0.85 0.15 ${hue} / 0.5)`
                    : `0 0 14px -3px oklch(0.72 0.15 ${hue} / 0.35)`,
            }}
            className="absolute inset-0 grid place-items-center rounded-[30%] border-[1.5px] transition-[opacity,box-shadow,transform] duration-300"
          >
            <span
              className="leading-none font-semibold"
              style={{ fontSize: clamp(r * 0.42, 8, 13), color: "var(--color-foreground)" }}
            >
              #{n.id}
            </span>
            {consolidated && (
              <span
                className="absolute -top-1 -right-1 size-2 rounded-full"
                style={{ background: "var(--color-chart-2)" }}
              />
            )}
          </div>

        </div>
      </HoverCardTrigger>
      <HoverCardContent side="top" className="w-72 p-0">
        <div
          className="flex items-center justify-between gap-2 border-b border-border/60 px-3 py-2 text-xs font-mono"
          style={{ background: `oklch(0.34 0.10 ${hue} / 0.35)` }}
        >
          <span className="flex items-center gap-1.5 font-semibold">
            <span
              className="size-2 rounded-sm"
              style={{ background: `oklch(0.80 0.18 ${hue} / 0.9)` }}
              aria-hidden
            />
            Episode #{n.id}
          </span>
          {n.is_focus && (
            <span className="rounded bg-primary/15 px-1.5 py-0.5 text-[0.6rem] font-medium tracking-wide text-primary uppercase">
              focus
            </span>
          )}
        </div>
        {n.preview && (
          <div className="line-clamp-3 px-3 py-2 text-xs leading-snug text-muted-foreground">
            {n.preview}
          </div>
        )}
        <div className="grid grid-cols-2 gap-y-1 border-t border-border/60 px-3 py-2 font-mono text-[0.65rem]">
          <div>
            <div className="text-[0.55rem] tracking-wider text-muted-foreground uppercase">
              Salience
            </div>
            <div className="tabular-nums text-foreground">{n.salience.toFixed(2)}</div>
          </div>
          <div>
            <div className="text-[0.55rem] tracking-wider text-muted-foreground uppercase">
              Retrievals
            </div>
            <div className="tabular-nums text-foreground">{n.retrieval_count}</div>
          </div>
          {n.session_id && (
            <div>
              <div className="text-[0.55rem] tracking-wider text-muted-foreground uppercase">
                Context
              </div>
              <div className="text-foreground">{shortSession(n.session_id)}</div>
            </div>
          )}
          {consolidated && (
            <div>
              <div className="text-[0.55rem] tracking-wider text-muted-foreground uppercase">
                Schema
              </div>
              <div className="text-foreground">⌬{n.consolidated_into}</div>
            </div>
          )}
        </div>
      </HoverCardContent>
    </HoverCard>
  );
}

function SemanticNode({ data, selected }: NodeProps<FlowNode>) {
  const n = data.node;
  if (n.kind !== "semantic") return null;
  const r = nodeRadius(n);
  const conf = clamp(n.confidence, 0, 1);
  const label = `Schema #${n.id}, confidence ${conf.toFixed(2)}, ${n.source_count} sources. Enter to inspect.`;
  return (
    <HoverCard>
      <HoverCardTrigger asChild>
        <div
          role="button"
          tabIndex={0}
          aria-label={label}
          aria-pressed={selected}
          onKeyDown={(e) => handleNodeKey(e, data)}
          style={{
            width: r * 2,
            height: r * 2,
            opacity: data.dim ? 0.16 : 1,
          }}
          className="relative grid place-items-center font-mono"
        >
          {hiddenHandles()}
          <div
            aria-hidden
            style={{
              background: "oklch(0.34 0.10 195 / 0.55)",
              borderColor: selected
                ? "var(--color-primary)"
                : `oklch(0.80 0.16 195 / ${0.4 + conf * 0.55})`,
              boxShadow:
                selected || data.active
                  ? "0 0 0 2px var(--color-primary), 0 0 22px -4px var(--color-primary)"
                  : "0 0 16px -4px oklch(0.78 0.15 195 / 0.5)",
            }}
            className="absolute inset-0 grid place-items-center rounded-full border-2 border-dashed transition-[opacity,box-shadow] duration-300"
          >
            <span
              className="px-1 text-center leading-tight font-semibold"
              style={{ fontSize: clamp(r * 0.3, 8, 12), color: "var(--color-chart-2)" }}
            >
              ⌬{n.id}
            </span>
          </div>

        </div>
      </HoverCardTrigger>
      <HoverCardContent side="top" className="w-72 p-0">
        <div
          className="flex items-center gap-1.5 border-b border-border/60 px-3 py-2 font-mono text-xs font-semibold"
          style={{ background: "oklch(0.34 0.10 195 / 0.35)" }}
        >
          <span
            className="size-2 rounded-sm"
            style={{ background: "var(--color-chart-2)" }}
            aria-hidden
          />
          <span style={{ color: "var(--color-chart-2)" }}>Schema ⌬{n.id}</span>
        </div>
        {n.preview && (
          <div className="line-clamp-3 px-3 py-2 text-xs leading-snug text-muted-foreground">
            {n.preview}
          </div>
        )}
        <div className="grid grid-cols-2 gap-y-1 border-t border-border/60 px-3 py-2 font-mono text-[0.65rem]">
          <div>
            <div className="text-[0.55rem] tracking-wider text-muted-foreground uppercase">
              Confidence
            </div>
            <div className="tabular-nums text-foreground">{conf.toFixed(2)}</div>
          </div>
          <div>
            <div className="text-[0.55rem] tracking-wider text-muted-foreground uppercase">
              Sources
            </div>
            <div className="tabular-nums text-foreground">{n.source_count}</div>
          </div>
        </div>
      </HoverCardContent>
    </HoverCard>
  );
}

const NODE_TYPES: NodeTypes = {
  episodic: EpisodicNode,
  semantic: SemanticNode,
};

/** Community hulls — convex polygons traced around each cluster of
 *  co-activating episodes. Rendered as low-alpha SVG paths behind
 *  every other ReactFlow layer (`zIndex: -1` undercuts the renderer's
 *  stacking context) and follow the viewport via a single `<g>`
 *  transform updated imperatively on pan/zoom. */
function CommunityHullsLayer({ hulls }: { hulls: HullEntry[] }) {
  const store = useStoreApi();
  const gRef = useRef<SVGGElement>(null);

  useEffect(() => {
    let raf = 0;
    const apply = (state?: ReturnType<typeof store.getState>) => {
      raf = 0;
      const s = state ?? store.getState();
      const g = gRef.current;
      if (!g) return;
      const tx = s.transform[0];
      const ty = s.transform[1];
      const zoom = s.transform[2];
      g.setAttribute("transform", `translate(${tx} ${ty}) scale(${zoom})`);
    };
    const schedule = (s: ReturnType<typeof store.getState>) => {
      if (!raf) raf = requestAnimationFrame(() => apply(s));
    };
    const unsub = store.subscribe(schedule);
    apply();
    return () => {
      unsub();
      if (raf) cancelAnimationFrame(raf);
    };
  }, [store]);

  if (hulls.length === 0) return null;
  return (
    <svg
      aria-hidden
      className="pointer-events-none absolute inset-0 h-full w-full"
      style={{ zIndex: -1 }}
    >
      <g ref={gRef}>
        {hulls.map((h) => (
          <path
            key={h.community}
            d={polygonPath(h.polygon)}
            fill={`oklch(0.65 0.12 ${h.hue} / 0.04)`}
            stroke={`oklch(0.72 0.12 ${h.hue} / 0.18)`}
            strokeWidth={1.25}
            strokeLinejoin="round"
            // Keep stroke width visually constant across zooms.
            vectorEffect="non-scaling-stroke"
          />
        ))}
      </g>
    </svg>
  );
}

/** Captions live on a single overlay layer above the canvas — not as
 *  children of each node. The whole layer is `pointer-events: none`,
 *  so labels never intercept clicks/drags: pan-drag and hover-card
 *  still work even when the cursor is over a caption that visually
 *  sits over another node's disc.
 *
 *  Performance shape:
 *  - React renders each caption div **once** (per `nodes` identity).
 *    Selection / focus / data updates regenerate `nodes`, which
 *    rebuilds the static caption list — at most a few times per UI
 *    interaction.
 *  - Pan/zoom updates skip React entirely. We subscribe to the
 *    ReactFlow store directly, coalesce via rAF, and update each
 *    caption's `transform` (GPU compositing) + `display` (off-screen
 *    cull) imperatively. For 500+ nodes the per-frame cost is a tight
 *    O(N) loop of string assignments — no reconciliation, no diffing.
 *  - Below the zoom threshold the whole layer is `display:none`; DOM
 *    stays mounted so crossing the threshold doesn't re-render. */
type CaptionStatic = {
  key: string;
  /** Flow-space centre of the disc. */
  cx: number;
  cy: number;
  /** Disc radius — used to anchor the caption below the disc edge. */
  r: number;
  preview: string;
  dim: boolean;
  className: string;
  style: React.CSSProperties;
};

const EPISODIC_CAPTION_CLS =
  "max-w-[13rem] rounded-md border bg-surface-overlay/85 px-2 py-1 text-center font-mono text-[10px] leading-[1.25] font-medium tracking-tight text-foreground/85 shadow-card backdrop-blur-sm transition-opacity duration-200";
const SEMANTIC_CAPTION_CLS =
  "max-w-[13rem] rounded-md border border-l-2 bg-surface-overlay/85 px-2 py-1 text-center font-mono text-[10px] leading-[1.25] font-medium tracking-tight shadow-card backdrop-blur-sm transition-opacity duration-200";
const SEMANTIC_CAPTION_STYLE: React.CSSProperties = {
  borderColor: "oklch(0.72 0.13 230 / 0.35)",
  borderLeftColor: "var(--color-chart-2)",
  color: "color-mix(in oklab, var(--color-chart-2) 55%, var(--color-foreground))",
};

function CaptionsOverlay({ nodes }: { nodes: FlowNode[] }) {
  const store = useStoreApi();
  const layerRef = useRef<HTMLDivElement>(null);
  const elRefs = useRef<Map<string, HTMLDivElement>>(new Map());

  // Static per-caption data. Identity is preserved across renders
  // when `nodes` is stable — the React Compiler auto-memoizes — so
  // the imperative effect below still re-subscribes only on actual
  // node-list changes (selection, search, data refresh).
  const captions: CaptionStatic[] = [];
  for (const fn of nodes) {
    const n = fn.data.node;
    if (!n.preview) continue;
    const r = nodeRadius(n);
    const cx = fn.position.x + r;
    const cy = fn.position.y + r;
    if (n.kind === "episodic") {
      const hue = sessionHue(n.session_id);
      captions.push({
        key: fn.id,
        cx,
        cy,
        r,
        preview: n.preview,
        dim: fn.data.dim,
        className: EPISODIC_CAPTION_CLS,
        style: { borderColor: `oklch(0.72 0.12 ${hue} / 0.35)` },
      });
    } else {
      captions.push({
        key: fn.id,
        cx,
        cy,
        r,
        preview: n.preview,
        dim: fn.data.dim,
        className: SEMANTIC_CAPTION_CLS,
        style: SEMANTIC_CAPTION_STYLE,
      });
    }
  }

  // Imperative position pump. Subscribes once to the ReactFlow store,
  // coalesces store changes through rAF, and writes positions
  // directly to each caption's DOM node. No React render on pan/zoom.
  useEffect(() => {
    let raf = 0;
    const apply = () => {
      raf = 0;
      const layer = layerRef.current;
      if (!layer) return;
      const s = store.getState();
      const tx = s.transform[0];
      const ty = s.transform[1];
      const zoom = s.transform[2];
      const visible = zoom >= CAPTION_ZOOM_MIN;
      layer.style.display = visible ? "" : "none";
      if (!visible) return;
      const vw = s.width;
      const vh = s.height;
      const minX = -CAPTION_VIEWPORT_PAD;
      const maxX = vw + CAPTION_VIEWPORT_PAD;
      const minY = -CAPTION_VIEWPORT_PAD;
      const maxY = vh + CAPTION_VIEWPORT_PAD;
      for (const c of captions) {
        const el = elRefs.current.get(c.key);
        if (!el) continue;
        const sx = c.cx * zoom + tx;
        const sy = c.cy * zoom + ty + c.r * zoom + 8;
        if (sx < minX || sx > maxX || sy < minY || sy > maxY) {
          if (el.style.display !== "none") el.style.display = "none";
          continue;
        }
        if (el.style.display === "none") el.style.display = "";
        el.style.transform = `translate3d(${sx}px, ${sy}px, 0) translateX(-50%)`;
      }
    };
    const schedule = () => {
      if (!raf) raf = requestAnimationFrame(apply);
    };
    const unsub = store.subscribe(schedule);
    apply();
    return () => {
      unsub();
      if (raf) cancelAnimationFrame(raf);
    };
  }, [store, captions]);

  return (
    <div
      ref={layerRef}
      aria-hidden
      className="pointer-events-none absolute inset-0 z-20 overflow-hidden"
      style={{ display: "none" }}
    >
      {captions.map((c) => (
        <div
          key={c.key}
          ref={(el) => {
            if (el) elRefs.current.set(c.key, el);
            else elRefs.current.delete(c.key);
          }}
          style={{
            position: "absolute",
            top: 0,
            left: 0,
            // Off-screen until the first apply() runs — avoids a
            // one-frame flash at (0,0) before positioning lands.
            transform: "translate3d(-9999px, -9999px, 0)",
            willChange: "transform",
            opacity: c.dim ? 0.16 : 1,
            ...c.style,
          }}
          className={c.className}
        >
          <span className="line-clamp-3 break-words whitespace-normal">{c.preview}</span>
        </div>
      ))}
    </div>
  );
}

/** Compact zoom widget. Replaces ReactFlow's default Controls so the
 *  current zoom is visible (the built-in Controls hide it) and the
 *  styling matches the rest of the dashboard's glass panels. The
 *  centre cell doubles as a "fit view" shortcut — clicking the
 *  readout reframes the whole map. */
function ZoomControls() {
  const { zoomIn, zoomOut, fitView } = useReactFlow();
  const zoom = useStore((s) => s.transform[2]);
  const btn =
    "grid size-7 place-items-center rounded text-base leading-none text-muted-foreground transition-colors hover:bg-foreground/10 hover:text-foreground focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-primary";
  return (
    <Panel
      position="bottom-right"
      className="!m-3 flex items-center gap-0.5 rounded-md border border-border/70 bg-surface-overlay/85 p-1 font-mono shadow-card backdrop-blur"
    >
      <button
        type="button"
        aria-label="Zoom out"
        title="Zoom out"
        onClick={() => void zoomOut({ duration: 200 })}
        className={btn}
      >
        −
      </button>
      <button
        type="button"
        aria-label={`Fit view (current zoom ${zoom.toFixed(2)}×)`}
        title="Fit view"
        onClick={() => void fitView({ padding: 0.25, duration: 300, maxZoom: 1.4 })}
        className="min-w-12 rounded px-2 py-1 text-center text-[0.7rem] font-medium tabular-nums text-foreground/85 transition-colors hover:bg-foreground/10 hover:text-foreground focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-primary"
      >
        {zoom.toFixed(2)}×
      </button>
      <button
        type="button"
        aria-label="Zoom in"
        title="Zoom in"
        onClick={() => void zoomIn({ duration: 200 })}
        className={btn}
      >
        +
      </button>
    </Panel>
  );
}

/** Refits the viewport whenever the layout signature settles — UMAP
 *  is async so ReactFlow's built-in fitView runs before positions
 *  arrive and ends up framing the (0,0) placeholders. When a camera
 *  focus is set ("Recenter map" / 'F' / search hit) we pan to that
 *  node instead of refitting the whole graph, so the user's spatial
 *  understanding of the UMAP manifold is preserved. */
function ViewportController({
  version,
  focusKey,
}: {
  version: number;
  focusKey: string | null;
}) {
  const { fitView, setCenter, getNode, getZoom } = useReactFlow();
  useEffect(() => {
    if (version === 0) return;
    const t = requestAnimationFrame(() => {
      if (focusKey) {
        const node = getNode(focusKey);
        if (node) {
          const w = (node.measured?.width ?? node.width ?? 0) / 2;
          const h = (node.measured?.height ?? node.height ?? 0) / 2;
          void setCenter(node.position.x + w, node.position.y + h, {
            zoom: Math.max(getZoom(), 1.2),
            duration: 450,
          });
          return;
        }
      }
      void fitView({ padding: 0.25, duration: 350, maxZoom: 1.4 });
    });
    return () => cancelAnimationFrame(t);
  }, [version, focusKey, fitView, setCenter, getNode, getZoom]);
  return null;
}

/* -------------------------------------------------------------- the surface */

export function MemoryGraph({
  data,
  selectedKey,
  cameraFocus,
  onSelect,
  onFocus,
  highlight,
}: {
  data: GraphData | undefined;
  selectedKey: string | null;
  /** Camera-only focus: pans the viewport to this episode without
   *  refetching or re-laying-out. Keeps the UMAP manifold stable
   *  across recenter clicks. */
  cameraFocus?: number | null;
  onSelect: (node: GraphNode | null) => void;
  onFocus: (episodeId: number) => void;
  /** Search hits (node keys). When set, the map dims everything else
   *  and rings the matches — search acts on the whole graph. */
  highlight?: Set<string>;
}) {
  // ELK's 'layered' pass runs on the main thread — for the graph sizes
  // this view shows it settles in a few ms. The request id discards
  // stale results so a slow settle can't overwrite a newer one driven
  // by a focus change.
  const posRef = useRef<Positions>(new Map());
  const [layout, setLayout] = useState<Positions>(new Map());
  const [layoutVersion, setLayoutVersion] = useState(0);
  // Context lens: off by default. When on, same-session episodes are
  // chained in encoded-at order — the timeline of how each context
  // unfolded. Sparse (n-1 links per session, not a clique) so it stays
  // cheap; gated so the default view carries no extra edges.
  const [contextLens, setContextLens] = useState(false);
  const reqIdRef = useRef(0);
  const sig = layoutSignature(data);

  useEffect(() => {
    const id = ++reqIdRef.current;
    let canceled = false;
    void computeLayout(data).then((settled) => {
      if (canceled || id !== reqIdRef.current) return;
      posRef.current = settled;
      setLayout(settled);
      setLayoutVersion((v) => v + 1);
    });
    return () => {
      canceled = true;
    };
  }, [sig]);

  const active = selectedKey;
  const searchOn = !active && !!highlight && highlight.size > 0;

  // Community structure + bridge nodes. Computed from the SR +
  // plasticity edge weights — these are the functional/transition
  // graph, distinct from the embedding-similarity manifold UMAP draws
  // positions from. The React Compiler auto-memoizes the result so
  // this only re-runs when `data` or `layout` actually change.
  let hulls: HullEntry[] = [];
  const bridgeKeys = new Set<string>();
  if (data && data.nodes.length > 0) {
    const episodicKeys = data.nodes
      .filter((n) => n.kind === "episodic")
      .map((n) => n.key);
    if (episodicKeys.length > 0) {
      const edgeList: CommunityEdge[] = [];
      for (const e of data.sr_edges) {
        edgeList.push({
          src: `ep:${e.src}`,
          dst: `ep:${e.dst}`,
          w: clamp(e.m, 0, 1),
        });
      }
      for (const e of data.plasticity_edges) {
        edgeList.push({
          src: `ep:${e.src}`,
          dst: `ep:${e.dst}`,
          w: clamp(e.strength, 0, 1),
        });
      }
      const comms = detectCommunities(episodicKeys, edgeList);
      const scores = computeBridgeScores(episodicKeys, edgeList, comms);
      const thresh = bridgeThreshold(scores);
      for (const [k, v] of scores) if (v > 0 && v >= thresh) bridgeKeys.add(k);

      const placed: Array<{ key: string; x: number; y: number; r: number }> = [];
      for (const n of data.nodes) {
        if (n.kind !== "episodic") continue;
        const p = layout.get(n.key);
        if (!p) continue;
        placed.push({ key: n.key, x: p.x, y: p.y, r: nodeRadius(n) });
      }
      hulls = buildClusterHulls(placed, comms);
    }
  }

  // Two ways the map narrows: a selected node lights its
  // spreading-activation neighbourhood; a search lights every hit
  // across the whole graph. Either dims the rest. The compiler
  // memoizes everything below; explicit useMemo isn't needed.
  const lit = new Set<string>();
  if (active && data) {
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
  const focusing = active != null || searchOn;

  const flowNodes: FlowNode[] = !data
    ? []
    : data.nodes.map((n) => {
      const p = layout.get(n.key) ?? { x: 0, y: 0 };
      const r = nodeRadius(n);
      // is_focus is now driven client-side from the camera focus so
      // the global UMAP fetch never has to be re-issued for recenter.
      const node: GraphNode =
        n.kind === "episodic" && cameraFocus != null && n.id === cameraFocus
          ? { ...n, is_focus: true }
          : n.kind === "episodic"
            ? { ...n, is_focus: false }
            : n;
        return {
          id: n.key,
          type: n.kind,
          position: { x: p.x - r, y: p.y - r },
          selected: n.key === selectedKey,
          data: {
            node,
            dim: focusing && !lit.has(n.key),
            active: searchOn
              ? lit.has(n.key)
              : active != null && lit.has(n.key) && n.key !== active,
            isBridge: bridgeKeys.has(n.key),
            onSelect,
            onFocus,
          },
        };
      });

  const dimEdge = (a: string, b: string) =>
    focusing && !(lit.has(a) && lit.has(b));

  const flowEdges: Edge[] = [];
  if (data) {
    if (contextLens) {
      // Same-session connectors: link each session's episodes in
      // encoded-at order. Pushed first so they sit beneath the SR /
      // plasticity / consolidation edges. Straight, no marker / filter /
      // animation, so the extra paint cost stays negligible.
      const bySession = new Map<string, Array<{ id: number; t: number }>>();
      for (const n of data.nodes) {
        if (n.kind !== "episodic" || !n.session_id) continue;
        const entry = { id: n.id, t: n.encoded_at };
        const arr = bySession.get(n.session_id);
        if (arr) arr.push(entry);
        else bySession.set(n.session_id, [entry]);
      }
      let ci = 0;
      for (const [sid, group] of bySession) {
        if (group.length < 2) continue;
        const ordered = [...group].sort((a, b) => a.t - b.t);
        const hue = sessionHue(sid);
        for (let i = 1; i < ordered.length; i++) {
          const s = `ep:${ordered[i - 1].id}`;
          const t = `ep:${ordered[i].id}`;
          const dim = dimEdge(s, t);
          flowEdges.push({
            id: `ctx-${ci++}`,
            source: s,
            target: t,
            type: "straight",
            style: {
              stroke: `oklch(0.72 0.14 ${hue})`,
              strokeWidth: 1,
              opacity: dim ? 0.04 : 0.4,
            },
          });
        }
      }
    }
    data.sr_edges.forEach((e, i) => {
      const s = `ep:${e.src}`;
      const t = `ep:${e.dst}`;
      const dim = dimEdge(s, t);
      flowEdges.push({
        id: `sr-${i}`,
        source: s,
        target: t,
        type: "default",
        markerEnd: { type: MarkerType.ArrowClosed, width: 14, height: 14 },
        style: {
          stroke: "var(--color-chart-1)",
          strokeWidth: 1 + clamp(e.m, 0, 1) * 5,
          opacity: dim ? 0.05 : 0.2 + clamp(e.m, 0, 1) * 0.65,
        },
      });
    });
    data.plasticity_edges.forEach((e, i) => {
      const s = `ep:${e.src}`;
      const t = `ep:${e.dst}`;
      const dim = dimEdge(s, t);
      const touchesActive = active != null && (s === active || t === active);
      flowEdges.push({
        id: `pl-${i}`,
        source: s,
        target: t,
        type: "default",
        animated: touchesActive,
        style: {
          stroke: "var(--color-chart-3)",
          strokeWidth: clamp(0.8 + e.strength * 1.6, 0.8, 7) * (e.is_habit ? 1.5 : 1),
          strokeDasharray: "5 4",
          opacity: dim ? 0.05 : e.is_habit ? 0.85 : 0.5,
        },
      });
    });
    data.consolidation_edges.forEach((e, i) => {
      const s = `ep:${e.src}`;
      const dim = dimEdge(s, e.dst);
      flowEdges.push({
        id: `cs-${i}`,
        source: s,
        target: e.dst,
        type: "default",
        markerEnd: { type: MarkerType.ArrowClosed, width: 12, height: 12 },
        style: {
          stroke: "var(--color-chart-2)",
          strokeWidth: 1.2,
          strokeDasharray: "2 4",
          opacity: dim ? 0.04 : 0.55,
        },
      });
    });
  }
  const nodes = flowNodes;
  const edges = flowEdges;

  const summary = data
    ? `Memory map: ${data.nodes.length} node${data.nodes.length === 1 ? "" : "s"}, ${data.sr_edges.length} successor edge${data.sr_edges.length === 1 ? "" : "s"}.` +
      " Use Tab to step through nodes, Enter to inspect, F to recenter."
    : "Memory map: empty.";
  return (
    <ReactFlow
      nodes={nodes}
      edges={edges}
      nodeTypes={NODE_TYPES}
      onNodeClick={(_, node) => onSelect(node.data.node)}
      onNodeDoubleClick={(_, node) => {
        const n = node.data.node;
        if (n.kind === "episodic") onFocus(n.id);
      }}
      onPaneClick={() => onSelect(null)}
      aria-label={summary}
      role="application"
      fitView
      fitViewOptions={{ padding: 0.25, maxZoom: 1.4 }}
      minZoom={0.15}
      maxZoom={2.4}
      proOptions={{ hideAttribution: true }}
      nodesDraggable={false}
      nodesConnectable={false}
      elementsSelectable
      onlyRenderVisibleElements
      className="bg-transparent"
    >
      <ViewportController
        version={layoutVersion}
        focusKey={cameraFocus != null ? `ep:${cameraFocus}` : null}
      />
      <Background
        variant={BackgroundVariant.Dots}
        gap={22}
        size={1}
        color="color-mix(in oklab, var(--color-foreground) 9%, transparent)"
      />
      {data && data.nodes.length > 0 && (
        <Panel
          position="top-left"
          className="!m-3 rounded-md border border-border/70 bg-surface-overlay/85 p-1 font-mono shadow-card backdrop-blur"
        >
          <button
            type="button"
            aria-pressed={contextLens}
            title="Context lens — link same-session episodes in time order"
            onClick={() => setContextLens((v) => !v)}
            className={`rounded px-2 py-1 text-[0.62rem] font-medium transition-colors focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-primary ${
              contextLens
                ? "bg-primary/15 text-primary"
                : "text-muted-foreground hover:bg-foreground/10 hover:text-foreground"
            }`}
          >
            context lens
          </button>
        </Panel>
      )}
      <ZoomControls />
      <CommunityHullsLayer hulls={hulls} />
      <CaptionsOverlay nodes={nodes} />
      <MiniMap
        pannable
        zoomable
        ariaLabel="Memory map overview"
        bgColor="transparent"
        maskColor="oklch(0.155 0.005 286 / 0.7)"
        className="!rounded-md !border !border-border !bg-surface-floating"
        nodeColor={(nd) => {
          const d = nd.data as NodeData;
          return d.node.kind === "semantic"
            ? "var(--color-chart-2)"
            : `oklch(0.7 0.12 ${sessionHue(
                d.node.kind === "episodic" ? d.node.session_id : null,
              )})`;
        }}
      />
      {data && (
        <Panel
          position="bottom-left"
          className="!m-3 hidden flex-col gap-1.5 rounded-md border border-border/70 bg-surface-overlay px-3 py-2.5 font-mono text-[0.6rem] text-muted-foreground backdrop-blur sm:flex sm:text-[0.62rem]"
        >
          <span className="flex items-center gap-1.5">
            <span
              className="h-0.5 w-5"
              style={{ background: "var(--color-chart-1)" }}
            />
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
              className="size-2 rounded-sm border border-muted-foreground/40"
              style={{ background: "oklch(0.65 0.13 200 / 0.18)" }}
            />
            community · halo = bridge
          </span>
          {contextLens && (
            <span className="flex items-center gap-1.5">
              <span
                className="h-0.5 w-5"
                style={{ background: "oklch(0.72 0.14 200)" }}
              />
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
        </Panel>
      )}
      {cameraFocus != null && (
        <Panel
          position="top-left"
          className="!m-3 rounded-md border border-primary/40 bg-primary/10 px-2.5 py-1 font-mono text-[0.6rem] text-primary sm:text-[0.62rem]"
        >
          focused on #{cameraFocus} — double-click a node to recenter
        </Panel>
      )}
      <Panel
        position="top-right"
        className="!m-3 hidden max-w-[12rem] text-right font-mono text-[0.6rem] leading-relaxed text-muted-foreground/70 sm:block"
      >
        {data && data.nodes.length > 0
          ? `${data.nodes.length} nodes${
              active
                ? ` · ${shortSession(
                    data.nodes.find((n) => n.key === active)?.kind === "episodic"
                      ? (
                          data.nodes.find(
                            (n) => n.key === active,
                          ) as { session_id: string | null }
                        ).session_id
                      : null,
                  )}`
                : ""
            }`
          : ""}
      </Panel>
    </ReactFlow>
  );
}
