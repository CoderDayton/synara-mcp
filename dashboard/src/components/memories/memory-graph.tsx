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
  Controls,
  Handle,
  MarkerType,
  MiniMap,
  Panel,
  Position,
  ReactFlow,
  useReactFlow,
  type Edge,
  type Node,
  type NodeProps,
  type NodeTypes,
} from "@xyflow/react";
import "@xyflow/react/dist/style.css";
import { useEffect, useMemo, useRef, useState } from "react";
import type { GraphData, GraphNode } from "@/lib/api";
import { clamp, sessionHue, shortSession } from "@/lib/format";
import {
  computeLayout,
  layoutSignature,
  nodeRadius,
  type Positions,
} from "./graph-layout";
import {
  HoverCard,
  HoverCardContent,
  HoverCardTrigger,
} from "@/components/ui/hover-card";

type NodeData = {
  node: GraphNode;
  dim: boolean;
  active: boolean;
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
            background: `oklch(0.30 0.05 ${hue} / 0.55)`,
            borderColor:
              n.is_focus || selected
                ? "var(--color-primary)"
                : `oklch(0.72 0.12 ${hue} / 0.75)`,
            boxShadow:
              n.is_focus || selected || data.active
                ? "0 0 0 2px var(--color-primary), 0 0 22px -4px var(--color-primary)"
                : "var(--shadow-card)",
          }}
          className="grid place-items-center rounded-[30%] border-[1.5px] font-mono transition-[opacity,box-shadow,transform] duration-300"
        >
          {hiddenHandles()}
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
      </HoverCardTrigger>
      <HoverCardContent side="top" className="w-72 p-0">
        <div
          className="flex items-center justify-between gap-2 border-b border-border/60 px-3 py-2 text-xs font-mono"
          style={{ background: `oklch(0.30 0.05 ${hue} / 0.30)` }}
        >
          <span className="flex items-center gap-1.5 font-semibold">
            <span
              className="size-2 rounded-sm"
              style={{ background: `oklch(0.72 0.12 ${hue} / 0.85)` }}
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
            background: "oklch(0.30 0.06 230 / 0.4)",
            borderColor: selected
              ? "var(--color-primary)"
              : `oklch(0.72 0.13 230 / ${0.35 + conf * 0.6})`,
            boxShadow:
              selected || data.active
                ? "0 0 0 2px var(--color-primary), 0 0 22px -4px var(--color-primary)"
                : "0 0 18px -8px oklch(0.72 0.13 230 / 0.7)",
          }}
          className="grid place-items-center rounded-full border-2 border-dashed font-mono transition-[opacity,box-shadow] duration-300"
        >
          {hiddenHandles()}
          <span
            className="px-1 text-center leading-tight font-semibold"
            style={{ fontSize: clamp(r * 0.3, 8, 12), color: "var(--color-chart-2)" }}
          >
            ⌬{n.id}
          </span>
        </div>
      </HoverCardTrigger>
      <HoverCardContent side="top" className="w-72 p-0">
        <div
          className="flex items-center gap-1.5 border-b border-border/60 px-3 py-2 font-mono text-xs font-semibold"
          style={{ background: "oklch(0.30 0.06 230 / 0.30)" }}
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
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [sig]);

  const active = selectedKey;
  const searchOn = !active && !!highlight && highlight.size > 0;

  const { nodes, edges } = useMemo<{ nodes: FlowNode[]; edges: Edge[] }>(() => {
    if (!data) return { nodes: [], edges: [] };

    // Two ways the map narrows: a selected node lights its
    // spreading-activation neighbourhood; a search lights every hit
    // across the whole graph. Either dims the rest.
    const lit = new Set<string>();
    if (active) {
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

    const flowNodes: FlowNode[] = data.nodes.map((n) => {
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
          onSelect,
          onFocus,
        },
      };
    });

    const dimEdge = (a: string, b: string) =>
      focusing && !(lit.has(a) && lit.has(b));

    const flowEdges: Edge[] = [];
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
          filter: e.is_habit ? "drop-shadow(0 0 4px var(--color-chart-3))" : undefined,
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
    return { nodes: flowNodes, edges: flowEdges };
  }, [data, layout, selectedKey, active, searchOn, highlight, onSelect, onFocus, cameraFocus]);

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
      <Controls
        showInteractive={false}
        className="!rounded-md !border !border-border !bg-surface-floating !shadow-card !backdrop-blur [&_button]:!border-border [&_button]:!bg-transparent [&_button]:!text-muted-foreground hover:[&_button]:!text-foreground"
      />
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
          <span className="mt-1 flex items-center gap-2 border-t border-border/60 pt-1.5 text-foreground/70">
            <span>ω {data.omega.toFixed(2)}</span>
            <span>·</span>
            <span>{data.episode_count} ep</span>
            {data.truncated && (
              <span className="text-warning">· capped</span>
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
