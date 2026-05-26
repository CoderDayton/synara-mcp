/**
 * Memory-map layout — semantic geometry, not graph topology.
 *
 * The hippocampus encodes memories as positions in an abstract
 * cognitive map (Tolman → O'Keefe place cells → Behrens & Whittington
 * "Tolman-Eichenbaum machine" → "Hippocampal neurons construct a map
 * of an abstract value space", Nature 2020): similar memories sit
 * near each other in a low-dimensional manifold of the embedding
 * space. We faithfully render that manifold by projecting episode &
 * schema embedding vectors into 2D with UMAP — the same technique
 * the Allen Institute uses for single-cell neuron-type atlases.
 *
 * The 2D position then *means* something: closeness on the canvas
 * mirrors closeness in the embedding manifold, which is what the
 * recall pipeline scores against. SR / plasticity / consolidation
 * edges then read as the actual neighbourhoods that spreading
 * activation walks.
 *
 * Fallback: if a node lacks an embedding (legacy server, no embedder
 * configured, or schema with no aggregate vector), we fall back to
 * ELK's force algorithm so the map degrades gracefully instead of
 * collapsing those nodes to the origin.
 */
import ELK, { type ElkNode } from "elkjs/lib/elk.bundled.js";
import { UMAP } from "umap-js";
import type { GraphData, GraphNode } from "@/lib/api";
import { clamp } from "@/lib/format";

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

/* ---------------------------------------------------------------- UMAP path */

/** Pixel envelope of the rendered map. The UMAP output is scaled
 *  uniformly into this box so the layout fills the canvas regardless
 *  of how compact or spread the manifold is. */
const CANVAS = { width: 1100, height: 760 };

/** UMAP needs ≥ this many embedding-bearing nodes to be meaningful —
 *  below that, the projection has too few neighbours to fit and
 *  collapses to a near-line. We fall back to force layout then. */
const MIN_UMAP_NODES = 4;

function rescale(
  raw: Array<[number, number]>,
  box: { width: number; height: number },
): Array<{ x: number; y: number }> {
  if (raw.length === 0) return [];
  let minX = Infinity;
  let maxX = -Infinity;
  let minY = Infinity;
  let maxY = -Infinity;
  for (const [x, y] of raw) {
    if (x < minX) minX = x;
    if (x > maxX) maxX = x;
    if (y < minY) minY = y;
    if (y > maxY) maxY = y;
  }
  const spanX = maxX - minX || 1;
  const spanY = maxY - minY || 1;
  // Uniform scale (preserve aspect) so distances stay comparable in
  // both axes — non-uniform stretch would distort the manifold.
  const scale = Math.min(box.width / spanX, box.height / spanY);
  return raw.map(([x, y]) => ({
    x: (x - minX - spanX / 2) * scale,
    y: (y - minY - spanY / 2) * scale,
  }));
}

function umapLayout(data: GraphData): Positions | null {
  const withEmbed = data.nodes
    .map((n) => ({ key: n.key, embedding: n.embedding }))
    .filter((n): n is { key: string; embedding: number[] } => !!n.embedding);
  if (withEmbed.length < MIN_UMAP_NODES) return null;

  const dim = withEmbed[0].embedding.length;
  if (!withEmbed.every((n) => n.embedding.length === dim)) return null;

  const umap = new UMAP({
    nComponents: 2,
    nNeighbors: Math.min(15, Math.max(2, withEmbed.length - 1)),
    minDist: 0.25,
    spread: 1.4,
    // Deterministic seed so the layout is stable across reloads —
    // moving memories should reflect new data, not RNG.
    random: mulberry32(0x5e0a7a),
  });
  const proj = umap.fit(withEmbed.map((n) => n.embedding));
  const placed = rescale(
    proj.map((p) => [p[0], p[1]]),
    CANVAS,
  );
  const out: Positions = new Map();
  withEmbed.forEach((n, i) => out.set(n.key, placed[i]));
  return out;
}

/**
 * Deterministic PRNG (Mulberry32). UMAP accepts a `random: () => number`
 * so seeding here makes the projection reproducible — the alternative
 * (`Math.random`) jiggles every layout.
 */
function mulberry32(seed: number): () => number {
  let s = seed >>> 0;
  return () => {
    s = (s + 0x6d2b79f5) >>> 0;
    let t = s;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/* --------------------------------------------------------------- force fallback */

const elk = new ELK();

const FORCE_OPTIONS: Record<string, string> = {
  "elk.algorithm": "org.eclipse.elk.force",
  "elk.force.iterations": "400",
  "elk.force.repulsivePower": "1",
  "elk.spacing.nodeNode": "80",
  "elk.force.temperature": "0.001",
};

async function forceLayout(data: GraphData): Promise<Positions> {
  const out: Positions = new Map();
  if (data.nodes.length === 0) return out;
  const present = new Set(data.nodes.map((n) => n.key));
  const children: ElkNode[] = data.nodes.map((n) => {
    const d = nodeRadius(n) * 2;
    return { id: n.key, width: d, height: d };
  });
  const edges: { id: string; sources: string[]; targets: string[] }[] = [];
  let i = 0;
  for (const e of data.sr_edges) {
    const s = `ep:${e.src}`;
    const t = `ep:${e.dst}`;
    if (present.has(s) && present.has(t))
      edges.push({ id: `sr-${i++}`, sources: [s], targets: [t] });
  }
  for (const e of data.plasticity_edges) {
    const s = `ep:${e.src}`;
    const t = `ep:${e.dst}`;
    if (present.has(s) && present.has(t))
      edges.push({ id: `pl-${i++}`, sources: [s], targets: [t] });
  }
  for (const e of data.consolidation_edges) {
    const s = `ep:${e.src}`;
    if (present.has(s) && present.has(e.dst))
      edges.push({ id: `cs-${i++}`, sources: [s], targets: [e.dst] });
  }
  const root: ElkNode = {
    id: "root",
    layoutOptions: FORCE_OPTIONS,
    children,
    edges,
  };
  const result = await elk.layout(root);
  for (const c of result.children ?? []) {
    const w = c.width ?? 0;
    const h = c.height ?? 0;
    out.set(c.id, { x: (c.x ?? 0) + w / 2, y: (c.y ?? 0) + h / 2 });
  }
  return out;
}

/* ------------------------------------------------------------------- public */

/**
 * Lay out the memory map. UMAP on embedding vectors when available
 * (the scientifically faithful path); ELK force as a graceful
 * fallback. Async because UMAP iterates and force is intrinsically
 * Promise-shaped.
 */
export async function computeLayout(
  data: GraphData | undefined,
): Promise<Positions> {
  if (!data || data.nodes.length === 0) return new Map();

  const umap = umapLayout(data);
  if (umap && umap.size === data.nodes.length) return umap;

  // Mixed case: some nodes have embeddings, others don't (e.g. a
  // freshly-consolidated schema before its aggregate vector is built,
  // or a legacy server). Place the embedded ones via UMAP and fall
  // through to force for the rest so nothing collapses to (0,0).
  const force = await forceLayout(data);
  if (umap) {
    // Centre UMAP positions on the force-layout centroid so the two
    // pools don't disagree on origin.
    let cx = 0;
    let cy = 0;
    let n = 0;
    for (const p of force.values()) {
      cx += p.x;
      cy += p.y;
      n++;
    }
    if (n > 0) {
      cx /= n;
      cy /= n;
    }
    for (const [k, p] of umap) {
      force.set(k, { x: p.x + cx, y: p.y + cy });
    }
  }
  return force;
}
