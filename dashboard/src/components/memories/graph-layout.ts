/**
 * Pure d3-force layout helpers shared between the main thread and the
 * layout Worker. No React, no DOM — both consumers can import freely.
 */
import {
  forceCenter,
  forceCollide,
  forceLink,
  forceManyBody,
  forceSimulation,
  type SimulationLinkDatum,
  type SimulationNodeDatum,
} from "d3-force";
import type { GraphData, GraphNode } from "@/lib/api";
import { clamp } from "@/lib/format";

export interface SimNode extends SimulationNodeDatum {
  id: string;
  r: number;
}

export type Positions = Map<string, { x: number; y: number }>;

export function nodeRadius(n: GraphNode): number {
  if (n.kind === "semantic") return 30 + clamp(n.source_count, 0, 12) * 1.4;
  const sal = clamp(n.salience, 0, 1);
  const ret = clamp(n.retrieval_count, 0, 10);
  return (n.is_focus ? 26 : 17) + sal * 16 + ret * 1.1;
}

export function layoutSignature(data: GraphData | undefined): string {
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
 * no hooks/refs, so it runs from an effect *or* a Worker.
 */
export function computeLayout(
  data: GraphData | undefined,
  prev: Positions,
): Positions {
  const out: Positions = new Map();
  if (!data || data.nodes.length === 0) return out;
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
  return out;
}
