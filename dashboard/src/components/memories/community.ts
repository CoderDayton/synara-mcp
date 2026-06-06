/**
 * Community structure for the memory map.
 *
 * Two interpretability layers on top of the session-clustered graph:
 *
 *  1. **Sessions** — `sessionCommunities` groups episodes by
 *     `session_id`, so each recall session becomes one bubble. The
 *     hulls drawn from these groups answer "which session is this" at a
 *     glance, matching how the layout packs each session apart.
 *
 *  2. **Bridges** — episodes whose edges cross session boundaries.
 *     Their cross-session weight sum is a cheap-but-honest proxy for
 *     betweenness: at 20-200 nodes it picks out the same routing
 *     abstractions a full Brandes pass would, in a fraction of the
 *     time. Bridges get a halo on the disc.
 *
 * Everything is deterministic given a sorted key order for tie-breaks.
 */

export type CommunityEdge = {
  src: string;
  dst: string;
  w: number;
};

/** Assign each episodic node a dense community id by ``session_id`` so
 *  the hull/bridge machinery carves the map into one bubble per session
 *  instead of label-propagation clusters. Non-episodic nodes (schemas)
 *  are skipped — they get no hull. Deterministic: sessions are numbered
 *  in first-seen order over the key-sorted node list. */
export function sessionCommunities(
  nodes: ReadonlyArray<{ key: string; kind: string; session_id?: string | null }>,
): Map<string, number> {
  const out = new Map<string, number>();
  const sessionIdx = new Map<string, number>();
  const eps = nodes
    .filter((n) => n.kind === "episodic")
    .slice()
    .sort((a, b) => a.key.localeCompare(b.key));
  let next = 0;
  for (const n of eps) {
    // Leading-space sentinel for null/absent session_id: groups all
    // session-less episodes into one bubble. The space can't collide
    // with a real session_id (those are never space-prefixed).
    const s = n.session_id ?? " nosession";
    let idx = sessionIdx.get(s);
    if (idx === undefined) {
      idx = next++;
      sessionIdx.set(s, idx);
    }
    out.set(n.key, idx);
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
