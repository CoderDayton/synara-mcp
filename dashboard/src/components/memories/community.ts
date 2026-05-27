/**
 * Community structure for the memory map.
 *
 * Two interpretability layers on top of the UMAP-positioned graph:
 *
 *  1. **Communities** — label propagation over the union of SR +
 *     plasticity edges. Episodes that co-activate end up in the same
 *     group regardless of where embedding similarity placed them. The
 *     hulls drawn from these groups answer "what activates together"
 *     independent of "what means the same thing."
 *
 *  2. **Bridges** — episodes whose edges cross community boundaries.
 *     Their cross-community weight sum is a cheap-but-honest proxy
 *     for betweenness: at 20-200 nodes it picks out the same routing
 *     abstractions a full Brandes pass would, in a fraction of the
 *     time. Bridges get a halo on the disc.
 *
 * Label propagation is fine at this scale (typical < 200 nodes) — it
 * converges in passes ≈ graph diameter, has no random init, and is
 * fully deterministic given a sorted key order for tie-breaks.
 */

export type CommunityEdge = {
  src: string;
  dst: string;
  w: number;
};

/** Label propagation. Each node starts in its own community; each
 *  pass it adopts the label with the highest summed neighbour weight,
 *  ties broken by lowest label id. Sorted node order makes this
 *  deterministic. Output labels are compacted to dense [0..k). */
export function detectCommunities(
  nodeKeys: readonly string[],
  edges: readonly CommunityEdge[],
): Map<string, number> {
  const labels = new Map<string, number>();
  const order = [...nodeKeys].sort();
  order.forEach((k, i) => labels.set(k, i));

  const adj = new Map<string, Array<{ nbr: string; w: number }>>();
  for (const k of order) adj.set(k, []);
  for (const e of edges) {
    if (!adj.has(e.src) || !adj.has(e.dst)) continue;
    if (e.w <= 0) continue;
    adj.get(e.src)!.push({ nbr: e.dst, w: e.w });
    adj.get(e.dst)!.push({ nbr: e.src, w: e.w });
  }

  const MAX_ITERS = 20;
  for (let iter = 0; iter < MAX_ITERS; iter++) {
    let changed = false;
    for (const k of order) {
      const nbrs = adj.get(k);
      if (!nbrs || nbrs.length === 0) continue;
      const counts = new Map<number, number>();
      for (const { nbr, w } of nbrs) {
        const lbl = labels.get(nbr);
        if (lbl === undefined) continue;
        counts.set(lbl, (counts.get(lbl) ?? 0) + w);
      }
      if (counts.size === 0) continue;
      let bestLbl = Number.MAX_SAFE_INTEGER;
      let bestW = -Infinity;
      for (const [lbl, w] of counts) {
        if (w > bestW || (w === bestW && lbl < bestLbl)) {
          bestLbl = lbl;
          bestW = w;
        }
      }
      if (labels.get(k) !== bestLbl) {
        labels.set(k, bestLbl);
        changed = true;
      }
    }
    if (!changed) break;
  }

  // Compact to dense ids in stable order (sorted-key traversal).
  const remap = new Map<number, number>();
  let next = 0;
  const out = new Map<string, number>();
  for (const k of order) {
    const orig = labels.get(k)!;
    if (!remap.has(orig)) remap.set(orig, next++);
    out.set(k, remap.get(orig)!);
  }
  return out;
}

/** Sum of edge weights leaving each node's community. Cheap betweenness
 *  proxy — at this graph size it surfaces the same routing nodes as a
 *  full Brandes pass but is O(E) instead of O(VE). */
export function computeBridgeScores(
  nodeKeys: readonly string[],
  edges: readonly CommunityEdge[],
  communities: Map<string, number>,
): Map<string, number> {
  const scores = new Map<string, number>();
  for (const k of nodeKeys) scores.set(k, 0);
  for (const e of edges) {
    const cs = communities.get(e.src);
    const cd = communities.get(e.dst);
    if (cs === undefined || cd === undefined) continue;
    if (cs === cd) continue;
    scores.set(e.src, (scores.get(e.src) ?? 0) + e.w);
    scores.set(e.dst, (scores.get(e.dst) ?? 0) + e.w);
  }
  return scores;
}

/** Andrew's monotone-chain convex hull. O(n log n). Returns the hull
 *  as a CCW polygon; empty input → empty output. */
export function convexHull(
  pts: ReadonlyArray<readonly [number, number]>,
): Array<[number, number]> {
  if (pts.length <= 1) return pts.map((p) => [p[0], p[1]]);
  const sorted = pts
    .map((p) => [p[0], p[1]] as [number, number])
    .sort((a, b) => a[0] - b[0] || a[1] - b[1]);
  const cross = (
    o: readonly [number, number],
    a: readonly [number, number],
    b: readonly [number, number],
  ) => (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0]);
  const lower: Array<[number, number]> = [];
  for (const p of sorted) {
    while (
      lower.length >= 2 &&
      cross(lower[lower.length - 2], lower[lower.length - 1], p) <= 0
    )
      lower.pop();
    lower.push(p);
  }
  const upper: Array<[number, number]> = [];
  for (let i = sorted.length - 1; i >= 0; i--) {
    const p = sorted[i];
    while (
      upper.length >= 2 &&
      cross(upper[upper.length - 2], upper[upper.length - 1], p) <= 0
    )
      upper.pop();
    upper.push(p);
  }
  return lower.slice(0, -1).concat(upper.slice(0, -1));
}

/** Number of sample points placed around each disc before the hull
 *  pass. 8 is enough that the hull traces the disc outline rather
 *  than the centre-to-centre polygon. */
const HULL_DISC_STEPS = 8;
/** Outward padding (in flow-space px) added to each disc before the
 *  hull pass. Gives the resulting polygon breathing room around the
 *  cluster. */
const HULL_DISC_PAD = 14;
/** Communities with fewer than this many nodes get no hull — a polygon
 *  around 1-2 points looks like noise, not structure. */
const HULL_MIN_NODES = 3;

export type HullEntry = {
  community: number;
  /** A stable, theme-friendly OKLCH hue for this community. */
  hue: number;
  polygon: Array<[number, number]>;
  centroidX: number;
};

/** Hue palette aligned with the indigo theme + sessionHue palette.
 *  Leads with cyan so single-community views read as a calm wash,
 *  not a warning. Hues spread evenly around the wheel and skip the
 *  red zone reserved for --destructive. */
const COMMUNITY_HUES = [
  195, // cyan
  145, // mint
  320, // magenta
  50, // amber  (logo hub accent)
  245, // violet
  95, // lime
  275, // indigo (theme primary)
  175, // teal
];

/** Build per-community hulls. Stable colour assignment: communities
 *  are sorted by their centroid x so the same cluster keeps the same
 *  hue across re-layouts, not by iteration order from label
 *  propagation (which can shuffle when topology changes slightly). */
export function buildClusterHulls(
  nodes: ReadonlyArray<{ key: string; x: number; y: number; r: number }>,
  communities: Map<string, number>,
): HullEntry[] {
  const groups = new Map<
    number,
    { pts: Array<[number, number]>; cxSum: number; n: number }
  >();
  for (const n of nodes) {
    const c = communities.get(n.key);
    if (c === undefined) continue;
    const g = groups.get(c) ?? { pts: [], cxSum: 0, n: 0 };
    const pad = n.r + HULL_DISC_PAD;
    for (let i = 0; i < HULL_DISC_STEPS; i++) {
      const a = (i / HULL_DISC_STEPS) * Math.PI * 2;
      g.pts.push([n.x + Math.cos(a) * pad, n.y + Math.sin(a) * pad]);
    }
    g.cxSum += n.x;
    g.n += 1;
    groups.set(c, g);
  }

  const draft: Array<{ community: number; polygon: Array<[number, number]>; centroidX: number }> =
    [];
  for (const [c, g] of groups) {
    if (g.n < HULL_MIN_NODES) continue;
    const polygon = convexHull(g.pts);
    if (polygon.length < 3) continue;
    draft.push({ community: c, polygon, centroidX: g.cxSum / g.n });
  }
  draft.sort((a, b) => a.centroidX - b.centroidX);
  return draft.map((d, i) => ({
    community: d.community,
    polygon: d.polygon,
    centroidX: d.centroidX,
    hue: COMMUNITY_HUES[i % COMMUNITY_HUES.length],
  }));
}

/** SVG path data for a polygon. Empty if too few points. */
export function polygonPath(pts: ReadonlyArray<readonly [number, number]>): string {
  if (pts.length < 3) return "";
  let d = `M${pts[0][0].toFixed(2)},${pts[0][1].toFixed(2)}`;
  for (let i = 1; i < pts.length; i++) d += `L${pts[i][0].toFixed(2)},${pts[i][1].toFixed(2)}`;
  return d + "Z";
}

/** Classify a node's bridge score against the population — returns
 *  true once it sits in the top quartile of nonzero scorers. Using a
 *  relative threshold means the halo highlights real outliers, not
 *  just "any node with a cross-community edge." */
export function bridgeThreshold(scores: Map<string, number>): number {
  const nonzero: number[] = [];
  for (const v of scores.values()) if (v > 0) nonzero.push(v);
  if (nonzero.length === 0) return Infinity;
  nonzero.sort((a, b) => a - b);
  // Top quartile cut. With small populations the index can round to
  // the max element — that's fine, only the strongest bridge lights.
  const idx = Math.floor(nonzero.length * 0.75);
  return nonzero[Math.min(idx, nonzero.length - 1)];
}
