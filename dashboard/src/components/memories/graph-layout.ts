/**
 * Memory-map layout — deterministic clustered force embedding.
 *
 * The communities are known a priori (a node's session), so the layout
 * imposes them by construction: each session is laid out independently
 * with a deterministic Fruchterman–Reingold spring-electrical pass
 * seeded on a golden-angle spiral (hubs at the centre), which gives the
 * organic chains / leaf fans of a classic force graph *within* each
 * cluster. The settled cluster discs are then arranged by tangent
 * circle-packing-in-a-circle (Wang et al. 2006, the algorithm behind
 * d3.packSiblings) — compactness keeps the whole map roughly circular
 * while a connectivity pull places heavily-bridged sessions adjacent,
 * shortening cross-session edges.
 *
 * Schemas carry no session, so each is folded into the session its
 * consolidation sources mostly belong to (orphans share one small disc).
 * A final bounded de-overlap pass clears residual dot collisions, and
 * the map is recentred on the origin.
 *
 * Everything is deterministic given the data: the spiral seed and the
 * fixed-iteration force schedule replace RNG jitter, node order is
 * degree-then-key, disc packing is size-then-key. Same data → same map.
 */
import type { GraphData, GraphNode } from "@/lib/api";
import { clamp } from "@/lib/format";

export type Positions = Map<string, { x: number; y: number }>;

/** Dot-scale radii: the map reads as a constellation of small discs
 *  (size ∝ salience + retrievals; schemas ∝ source count), not cards. */
export function nodeRadius(n: GraphNode): number {
  if (n.kind === "semantic") return 6 + clamp(n.source_count, 0, 12) * 0.5;
  const sal = clamp(n.salience, 0, 1);
  const ret = clamp(n.retrieval_count, 0, 10);
  return (n.is_focus ? 7 : 4) + sal * 3.5 + ret * 0.35;
}

export function layoutSignature(data: GraphData | undefined): string {
  return data
    ? `${data.nodes.map((n) => n.key).join(",")}|${data.sr_edges.length}|${
        data.plasticity_edges.length
      }|${data.consolidation_edges.length}|${data.focus ?? ""}`
    : "";
}

/** Group key for an episode. Episodes without a session_id share a
 *  single fallback disc rather than scattering to the origin. */
const NO_SESSION = " nosession";
/** Group key for schemas whose consolidation sources aren't in the
 *  graph — they share one small disc instead of stranding. */
const ORPHAN = " orphan";

function sessionOf(n: GraphNode): string {
  if (n.kind !== "episodic") return NO_SESSION;
  return n.session_id ?? NO_SESSION;
}

/* ---------------------------------------------------------------- geometry */

/** Golden angle, ψ = π(3 − √5) ≈ 2.39996 rad ≈ 137.508°. */
const GOLDEN_ANGLE = Math.PI * (3 - Math.sqrt(5));
/** Gap left between session discs during circle-packing. */
const GROUP_GAP = 60;
/** How strongly disc placement is pulled toward the discs a session is
 *  most bridged to (vs. pure compactness). Keeps cross-session edges
 *  short without overriding the circular packing. */
const PACK_LINK_PULL = 0.7;
/** Padding between dot edges in the de-overlap passes. */
const NODE_PAD = 4;
/** Disc count past which tangent packing (which enumerates placed-disc
 *  pairs per insertion, ~O(D³) overall) falls back to a spiral seed +
 *  bounded de-overlap. The tangent packer is designed for the tens of
 *  session discs a normal view holds; hundreds of singleton sessions
 *  would otherwise stall the layout for seconds. */
const PACK_TANGENT_MAX_DISCS = 160;

type SimNode = { key: string; r: number; x: number; y: number };
type SimEdge = { a: number; b: number; w: number };

/* ------------------------------------------------------- per-cluster force */

/**
 * Deterministic spring-electrical settle of one session's members.
 *
 * Fruchterman–Reingold shape: pairwise repulsion ``k²/d``, spring
 * attraction toward a per-edge ideal length (stronger edges sit
 * closer), a light centring pull, and a fixed geometric cooling
 * schedule. Seeded on a golden-angle spiral with hubs (highest
 * weighted in-cluster degree) at the centre — no RNG anywhere, so the
 * settle is a pure function of the member/edge lists.
 */
function settleCluster(members: SimNode[], edges: SimEdge[]): void {
  const m = members.length;
  if (m === 0) return;
  if (m === 1) {
    members[0].x = 0;
    members[0].y = 0;
    return;
  }
  let maxR = 0;
  for (const nd of members) if (nd.r > maxR) maxR = nd.r;
  // Ideal edge length: two dots plus breathing room.
  const k = 2 * maxR + 26;

  // Seed: golden-angle spiral, dense centre outward.
  for (let i = 0; i < m; i++) {
    const rad = k * 0.55 * Math.sqrt(i + 0.5);
    const a = i * GOLDEN_ANGLE;
    members[i].x = rad * Math.cos(a);
    members[i].y = rad * Math.sin(a);
  }

  const iters = m > 400 ? 120 : 240;
  const dx = new Float64Array(m);
  const dy = new Float64Array(m);
  const k2 = k * k;
  for (let it = 0; it < iters; it++) {
    // Geometric cooling: ~k/2 initial step shrinking to ~px scale.
    const temp = (k / 2) * Math.pow(0.975, it);
    dx.fill(0);
    dy.fill(0);
    // Repulsion (all pairs — clusters are small; the budget is bounded
    // by the iteration scale-down above).
    for (let i = 0; i < m; i++) {
      for (let j = i + 1; j < m; j++) {
        let ux = members[i].x - members[j].x;
        let uy = members[i].y - members[j].y;
        let d2 = ux * ux + uy * uy;
        if (d2 < 1e-6) {
          // Deterministic tie-break for coincident points.
          ux = ((i * 7 + j) % 11) - 5 || 1;
          uy = ((i * 5 + j) % 13) - 6;
          d2 = ux * ux + uy * uy;
        }
        const d = Math.sqrt(d2);
        const f = k2 / d2;
        const fx = (ux / d) * f;
        const fy = (uy / d) * f;
        dx[i] += fx;
        dy[i] += fy;
        dx[j] -= fx;
        dy[j] -= fy;
      }
    }
    // Spring attraction along edges: stronger edges pull tighter.
    for (const e of edges) {
      const ux = members[e.a].x - members[e.b].x;
      const uy = members[e.a].y - members[e.b].y;
      const d = Math.hypot(ux, uy) || 1;
      const ideal = k * (1.5 - 0.7 * clamp(e.w, 0, 1));
      const f = (d - ideal) / ideal;
      const fx = ux * f * 0.6;
      const fy = uy * f * 0.6;
      dx[e.a] -= fx;
      dy[e.a] -= fy;
      dx[e.b] += fx;
      dy[e.b] += fy;
    }
    // Light centring keeps disconnected satellites from drifting off.
    for (let i = 0; i < m; i++) {
      dx[i] -= members[i].x * 0.03;
      dy[i] -= members[i].y * 0.03;
    }
    // Apply, displacement capped by the cooling temperature.
    for (let i = 0; i < m; i++) {
      const len = Math.hypot(dx[i], dy[i]);
      if (len < 1e-9) continue;
      const step = Math.min(len, temp);
      members[i].x += (dx[i] / len) * step;
      members[i].y += (dy[i] / len) * step;
    }
  }

  // Clear residual dot overlaps inside the cluster.
  relaxPairs(
    members.map((nd) => ({ p: nd, r: nd.r })),
    30,
  );
}

/** Recentre members on their centroid; return the disc's bounding radius. */
function recentreCluster(members: SimNode[]): number {
  const m = members.length;
  if (m === 0) return 0;
  let cx = 0;
  let cy = 0;
  for (const p of members) {
    cx += p.x;
    cy += p.y;
  }
  cx /= m;
  cy /= m;
  let R = 0;
  for (const p of members) {
    p.x -= cx;
    p.y -= cy;
    const reach = Math.hypot(p.x, p.y) + p.r;
    if (reach > R) R = reach;
  }
  return R;
}

/* ---------------------------------------------------------------- packing */

/** Centres at distance ``r1`` from c1 and ``r2`` from c2 — the 0/1/2
 *  intersection points of two circles. */
function circleIntersect(
  x1: number,
  y1: number,
  r1: number,
  x2: number,
  y2: number,
  r2: number,
): Array<[number, number]> {
  const dx = x2 - x1;
  const dy = y2 - y1;
  const d = Math.hypot(dx, dy);
  if (d === 0 || d > r1 + r2 || d < Math.abs(r1 - r2)) return [];
  const a = (r1 * r1 - r2 * r2 + d * d) / (2 * d);
  const h2 = r1 * r1 - a * a;
  const h = h2 > 0 ? Math.sqrt(h2) : 0;
  const xm = x1 + (a * dx) / d;
  const ym = y1 + (a * dy) / d;
  if (h === 0) return [[xm, ym]];
  const ox = (-dy / d) * h;
  const oy = (dx / d) * h;
  return [
    [xm + ox, ym + oy],
    [xm - ox, ym - oy],
  ];
}

/**
 * Tangent circle-packing in a circle (Wang et al. 2006 / d3.packSiblings,
 * simplified for the handful of discs this view shows). Largest disc at
 * the origin; each subsequent disc is placed tangent to two already-
 * placed discs, growing a compact, roughly circular enclosure. Among the
 * valid tangent slots we pick the one minimizing ``distance-to-centre +
 * PACK_LINK_PULL · (mean distance to the discs this session bridges to)``
 * — so compactness keeps it circular while connectivity pulls heavily-
 * bridged sessions adjacent, shortening cross-session edges. Determin-
 * istic: discs packed largest-first, ties broken by key.
 */
function packDiscs(
  discs: Array<{ key: string; r: number }>,
  links: Map<string, Map<string, number>>,
): Map<string, { x: number; y: number }> {
  const order = [...discs].sort((a, b) => b.r - a.r || a.key.localeCompare(b.key));
  const placed: Array<{ key: string; x: number; y: number; r: number }> = [];
  const out = new Map<string, { x: number; y: number }>();
  if (order.length > PACK_TANGENT_MAX_DISCS) {
    // Deterministic golden-angle spiral, radius scaled by cumulative
    // packed area (1.8× slack so the seed starts mostly clear), then the
    // bounded pairwise de-overlap clears residuals: O(D²) worst case
    // instead of the tangent packer's per-insertion pair enumeration.
    // Quality degrades gracefully (no connectivity pull) — acceptable at
    // a scale where individual disc adjacency is unreadable anyway.
    let area = 0;
    const seeded = order.map((d, i) => {
      area += d.r * d.r;
      const rad = 1.8 * Math.sqrt(area);
      const a = i * GOLDEN_ANGLE;
      return { key: d.key, pos: { x: rad * Math.cos(a), y: rad * Math.sin(a) }, r: d.r };
    });
    relaxPairs(
      seeded.map((s) => ({ p: s.pos, r: s.r })),
      40,
    );
    for (const s of seeded) out.set(s.key, s.pos);
    return out;
  }
  for (const d of order) {
    const myLinks = links.get(d.key);
    let pos: { x: number; y: number } | null = null;
    if (placed.length === 0) {
      pos = { x: 0, y: 0 };
    } else if (placed.length === 1) {
      pos = { x: placed[0].x + placed[0].r + d.r, y: placed[0].y };
    } else {
      let bestScore = Infinity;
      for (let i = 0; i < placed.length; i++) {
        for (let j = i + 1; j < placed.length; j++) {
          const cands = circleIntersect(
            placed[i].x,
            placed[i].y,
            placed[i].r + d.r,
            placed[j].x,
            placed[j].y,
            placed[j].r + d.r,
          );
          for (const [cx, cy] of cands) {
            let ok = true;
            for (const p of placed) {
              if (Math.hypot(cx - p.x, cy - p.y) < p.r + d.r - 1e-3) {
                ok = false;
                break;
              }
            }
            if (!ok) continue;
            let connPull = 0;
            let wsum = 0;
            if (myLinks) {
              for (const p of placed) {
                const w = myLinks.get(p.key) ?? 0;
                if (w > 0) {
                  connPull += w * Math.hypot(cx - p.x, cy - p.y);
                  wsum += w;
                }
              }
            }
            const score =
              Math.hypot(cx, cy) + (wsum > 0 ? PACK_LINK_PULL * (connPull / wsum) : 0);
            if (score < bestScore) {
              bestScore = score;
              pos = { x: cx, y: cy };
            }
          }
        }
      }
      if (!pos) {
        // Fallback: spiral outward from the origin until clear.
        for (let t = 0; t < 4000 && !pos; t++) {
          const ang = t * 0.5;
          const rad = 2 * ang;
          const cx = Math.cos(ang) * rad;
          const cy = Math.sin(ang) * rad;
          let ok = true;
          for (const p of placed) {
            if (Math.hypot(cx - p.x, cy - p.y) < p.r + d.r) {
              ok = false;
              break;
            }
          }
          if (ok) pos = { x: cx, y: cy };
        }
        if (!pos) pos = { x: 0, y: 0 };
      }
    }
    placed.push({ key: d.key, x: pos.x, y: pos.y, r: d.r });
    out.set(d.key, pos);
  }
  return out;
}

/* ---------------------------------------------------------------- overlap */

/** Bounded pairwise de-overlap over mutable positions. */
function relaxPairs(
  items: Array<{ p: { x: number; y: number }; r: number }>,
  passes: number,
): void {
  const count = items.length;
  if (count < 2) return;
  for (let pass = 0; pass < passes; pass++) {
    let moved = false;
    for (let i = 0; i < count; i++) {
      for (let j = i + 1; j < count; j++) {
        const a = items[i].p;
        const b = items[j].p;
        const minDist = items[i].r + items[j].r + NODE_PAD;
        let dx = b.x - a.x;
        let dy = b.y - a.y;
        let dist = Math.hypot(dx, dy);
        if (dist >= minDist) continue;
        if (dist < 1e-6) {
          // "|| 1" mirrors the settle-pass guard: both residues can be 0
          // for some (i, j), and a zero displacement vector would leave
          // the pair coincident through every pass.
          dx = (((i * 7 + j) % 11) - 5) || 1;
          dy = ((i * 5 + j) % 13) - 6;
          dist = Math.hypot(dx, dy);
        }
        const shift = (minDist - dist) / 2;
        const ux = dx / dist;
        const uy = dy / dist;
        a.x -= ux * shift;
        a.y -= uy * shift;
        b.x += ux * shift;
        b.y += uy * shift;
        moved = true;
      }
    }
    if (!moved) break;
  }
}

/** Final pass: clear residual dot collisions across the whole map.
 *  Bounded passes so a large graph can't stall the one-shot layout. */
function relaxGlobal(positions: Positions, data: GraphData): void {
  const items: Array<{ p: { x: number; y: number }; r: number }> = [];
  for (const n of data.nodes) {
    const p = positions.get(n.key);
    if (p) items.push({ p, r: nodeRadius(n) });
  }
  relaxPairs(items, items.length > 600 ? 8 : items.length > 250 ? 20 : 60);
}

/* ------------------------------------------------------------------- public */

/**
 * Lay out the memory map as one force-settled cluster per session, the
 * cluster discs circle-packed into a roughly circular whole. Returns a
 * Promise to preserve the caller's contract; the work itself is
 * synchronous and deterministic.
 */
export function computeLayout(
  data: GraphData | undefined,
): Promise<Positions> {
  if (!data || data.nodes.length === 0)
    return Promise.resolve<Positions>(new Map());

  // Episode → its session; used to route each schema to a session disc.
  const epSession = new Map<string, string>();
  for (const nd of data.nodes) {
    if (nd.kind === "episodic") epSession.set(nd.key, sessionOf(nd));
  }
  // Bucket consolidation edges by schema once: schemaGroup runs per
  // schema node, and rescanning every edge per call is O(schemas · edges).
  const schemaTallies = new Map<string, Map<string, number>>();
  for (const e of data.consolidation_edges) {
    const s = epSession.get(`ep:${e.src}`);
    if (s === undefined) continue;
    let tally = schemaTallies.get(e.dst);
    if (!tally) {
      tally = new Map();
      schemaTallies.set(e.dst, tally);
    }
    tally.set(s, (tally.get(s) ?? 0) + 1);
  }
  const schemaGroup = (key: string): string => {
    let best = ORPHAN;
    let bestN = 0;
    for (const [s, c] of schemaTallies.get(key) ?? []) {
      if (c > bestN || (c === bestN && s.localeCompare(best) < 0)) {
        best = s;
        bestN = c;
      }
    }
    return best;
  };

  // Partition nodes into groups (sessions + schema discs).
  const groups = new Map<string, SimNode[]>();
  const groupOf = new Map<string, string>();
  for (const nd of data.nodes) {
    const g = nd.kind === "episodic" ? sessionOf(nd) : schemaGroup(nd.key);
    groupOf.set(nd.key, g);
    const node: SimNode = { key: nd.key, r: nodeRadius(nd), x: 0, y: 0 };
    const arr = groups.get(g);
    if (arr) arr.push(node);
    else groups.set(g, [node]);
  }

  // One pass over every edge feeds three things: in-session weighted
  // degree (hubs sort to the cluster centre), the intra-group spring
  // list for the force settle, and an inter-disc link weight (how
  // heavily two sessions bridge) that steers the disc packing below.
  const degree = new Map<string, number>();
  const intra = new Map<string, Array<{ a: string; b: string; w: number }>>();
  const links = new Map<string, Map<string, number>>();
  const addLink = (a: string, b: string, w: number) => {
    let m = links.get(a);
    if (!m) {
      m = new Map();
      links.set(a, m);
    }
    m.set(b, (m.get(b) ?? 0) + w);
  };
  const edge = (ka: string, kb: string, w: number) => {
    const ga = groupOf.get(ka);
    const gb = groupOf.get(kb);
    if (ga === undefined || gb === undefined) return;
    if (ga === gb) {
      degree.set(ka, (degree.get(ka) ?? 0) + w);
      degree.set(kb, (degree.get(kb) ?? 0) + w);
      const arr = intra.get(ga);
      const rec = { a: ka, b: kb, w };
      if (arr) arr.push(rec);
      else intra.set(ga, [rec]);
    } else {
      addLink(ga, gb, w);
      addLink(gb, ga, w);
    }
  };
  for (const e of data.sr_edges) edge(`ep:${e.src}`, `ep:${e.dst}`, clamp(e.m, 0, 1));
  for (const e of data.plasticity_edges)
    edge(`ep:${e.src}`, `ep:${e.dst}`, clamp(e.strength, 0, 1));
  for (const e of data.consolidation_edges) edge(`ep:${e.src}`, e.dst, 0.6);

  // Force-settle each group; collect disc radii for packing.
  const discs: Array<{ key: string; r: number }> = [];
  for (const [key, members] of groups) {
    members.sort(
      (a, b) =>
        (degree.get(b.key) ?? 0) - (degree.get(a.key) ?? 0) ||
        a.key.localeCompare(b.key),
    );
    const index = new Map<string, number>();
    members.forEach((nd, i) => index.set(nd.key, i));
    const simEdges: SimEdge[] = [];
    for (const e of intra.get(key) ?? []) {
      const a = index.get(e.a);
      const b = index.get(e.b);
      if (a !== undefined && b !== undefined && a !== b)
        simEdges.push({ a, b, w: e.w });
    }
    settleCluster(members, simEdges);
    const radius = recentreCluster(members);
    discs.push({ key, r: radius + GROUP_GAP / 2 });
  }

  // Pack the discs into a circle and translate each group into place.
  const centres = packDiscs(discs, links);
  const positions: Positions = new Map();
  for (const [key, members] of groups) {
    const c = centres.get(key) ?? { x: 0, y: 0 };
    for (const nd of members) positions.set(nd.key, { x: nd.x + c.x, y: nd.y + c.y });
  }

  relaxGlobal(positions, data);

  // Recentre on the origin so the viewport framing is stable.
  const ctr = centroid(positions);
  for (const p of positions.values()) {
    p.x -= ctr.x;
    p.y -= ctr.y;
  }
  return Promise.resolve(positions);
}

function centroid(positions: Positions): { x: number; y: number } {
  let x = 0;
  let y = 0;
  let n = 0;
  for (const p of positions.values()) {
    x += p.x;
    y += p.y;
    n++;
  }
  return n > 0 ? { x: x / n, y: y / n } : { x: 0, y: 0 };
}
