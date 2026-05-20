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
 * would walk. Layout is a settled d3-force simulation; node positions
 * persist across focus changes so expansion animates instead of
 * reshuffling.
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
  type Edge,
  type Node,
  type NodeProps,
  type NodeTypes,
} from "@xyflow/react";
import "@xyflow/react/dist/style.css";
import {
  forceCenter,
  forceCollide,
  forceLink,
  forceManyBody,
  forceSimulation,
  type SimulationLinkDatum,
  type SimulationNodeDatum,
} from "d3-force";
import { useEffect, useMemo, useRef, useState } from "react";
import type { GraphData, GraphNode } from "@/lib/api";
import { clamp, sessionHue, shortSession } from "@/lib/format";

type NodeData = {
  node: GraphNode;
  dim: boolean;
  active: boolean;
};
type FlowNode = Node<NodeData>;

function nodeRadius(n: GraphNode): number {
  if (n.kind === "semantic") return 30 + clamp(n.source_count, 0, 12) * 1.4;
  const sal = clamp(n.salience, 0, 1);
  const ret = clamp(n.retrieval_count, 0, 10);
  return (n.is_focus ? 26 : 17) + sal * 16 + ret * 1.1;
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
  return (
    <div
      title={`#${n.id}${n.preview ? ` — ${n.preview}` : ""}`}
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
          title="consolidated into a schema"
        />
      )}
    </div>
  );
}

function SemanticNode({ data, selected }: NodeProps<FlowNode>) {
  const n = data.node;
  if (n.kind !== "semantic") return null;
  const r = nodeRadius(n);
  const conf = clamp(n.confidence, 0, 1);
  return (
    <div
      title={`schema #${n.id}${n.preview ? ` — ${n.preview}` : ""}`}
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
  );
}

const NODE_TYPES: NodeTypes = {
  episodic: EpisodicNode,
  semantic: SemanticNode,
};

/* ------------------------------------------------------------------- layout */

interface SimNode extends SimulationNodeDatum {
  id: string;
  r: number;
}

function layoutSignature(data: GraphData | undefined): string {
  return data
    ? `${data.nodes.map((n) => n.key).join(",")}|${data.sr_edges.length}|${
        data.plasticity_edges.length
      }|${data.consolidation_edges.length}|${data.focus ?? ""}`
    : "";
}

/**
 * Settle a d3-force simulation once and return id→position. `prev`
 * (the last settled layout) seeds the next run so a focus/expand
 * re-fetch nudges the graph instead of teleporting every node. Pure —
 * no hooks/refs, so it runs from an effect.
 */
function computeLayout(
  data: GraphData | undefined,
  prev: Map<string, { x: number; y: number }>,
): Map<string, { x: number; y: number }> {
  const out = new Map<string, { x: number; y: number }>();
  if (!data || data.nodes.length === 0) return out;
  {
    const sim: SimNode[] = data.nodes.map((n, i) => {
      const seed = prev.get(n.key);
      const angle = (i / data.nodes.length) * Math.PI * 2;
      return {
        id: n.key,
        r: nodeRadius(n) + 8,
        x: seed?.x ?? Math.cos(angle) * 240,
        y: seed?.y ?? Math.sin(angle) * 240,
        fx: n.kind === "episodic" && n.is_focus ? 0 : undefined,
        fy: n.kind === "episodic" && n.is_focus ? 0 : undefined,
      };
    });
    const present = new Set(sim.map((s) => s.id));
    const links: SimulationLinkDatum<SimNode>[] = [];
    for (const e of data.sr_edges) {
      const s = `ep:${e.src}`;
      const t = `ep:${e.dst}`;
      if (present.has(s) && present.has(t))
        links.push({ source: s, target: t });
    }
    for (const e of data.plasticity_edges) {
      const s = `ep:${e.src}`;
      const t = `ep:${e.dst}`;
      if (present.has(s) && present.has(t))
        links.push({ source: s, target: t });
    }
    for (const e of data.consolidation_edges) {
      const s = `ep:${e.src}`;
      if (present.has(s) && present.has(e.dst))
        links.push({ source: s, target: e.dst });
    }
    forceSimulation(sim)
      .force(
        "link",
        forceLink<SimNode, SimulationLinkDatum<SimNode>>(links)
          .id((d) => d.id)
          .distance(96)
          .strength(0.35),
      )
      .force("charge", forceManyBody().strength(-340))
      .force("center", forceCenter(0, 0))
      .force(
        "collide",
        forceCollide<SimNode>().radius((d) => d.r + 6),
      )
      .stop()
      .tick(320);
    for (const s of sim) {
      out.set(s.id, { x: s.x ?? 0, y: s.y ?? 0 });
    }
  }
  return out;
}

/* -------------------------------------------------------------- the surface */

export function MemoryGraph({
  data,
  selectedKey,
  onSelect,
  onFocus,
  highlight,
}: {
  data: GraphData | undefined;
  selectedKey: string | null;
  onSelect: (node: GraphNode | null) => void;
  onFocus: (episodeId: number) => void;
  /** Search hits (node keys). When set, the map dims everything else
   *  and rings the matches — search acts on the whole graph. */
  highlight?: Set<string>;
}) {
  // d3-force layout settled in an effect (keeps ref access out of
  // render); previous positions seed the next run so focus/expand
  // glides instead of teleporting.
  const posRef = useRef<Map<string, { x: number; y: number }>>(new Map());
  const [layout, setLayout] = useState<Map<string, { x: number; y: number }>>(
    new Map(),
  );
  const sig = layoutSignature(data);
  useEffect(() => {
    const settled = computeLayout(data, posRef.current);
    posRef.current = settled;
    setLayout(settled);
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
      return {
        id: n.key,
        type: n.kind,
        position: { x: p.x - r, y: p.y - r },
        selected: n.key === selectedKey,
        data: {
          node: n,
          dim: focusing && !lit.has(n.key),
          active: searchOn
            ? lit.has(n.key)
            : active != null && lit.has(n.key) && n.key !== active,
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
  }, [data, layout, selectedKey, active, searchOn, highlight]);

  return (
    <ReactFlow
      nodes={nodes}
      edges={edges}
      nodeTypes={NODE_TYPES}
      onNodeClick={(_, node) => onSelect((node.data as NodeData).node)}
      onNodeDoubleClick={(_, node) => {
        const n = (node.data as NodeData).node;
        if (n.kind === "episodic") onFocus(n.id);
      }}
      onPaneClick={() => onSelect(null)}
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
      {data && data.focus != null && (
        <Panel
          position="top-left"
          className="!m-3 rounded-md border border-primary/40 bg-primary/10 px-2.5 py-1 font-mono text-[0.6rem] text-primary sm:text-[0.62rem]"
        >
          focused on #{data.focus} — double-click a node to recenter
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
