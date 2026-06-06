/**
 * Memory-map layout — clustered sunflower packing.
 *
 * The communities here are known a priori (a node's session), so the
 * layout imposes them by construction rather than discovering them with
 * a force model. Noack's energy-model analysis ("Energy Models for Graph
 * Clustering", JGAA 2007) shows the classic spring-electrical model does
 * not separate clusters cleanly — dense clusters over-spread and the
 * gaps between them collapse — which is exactly why earlier force-tuned
 * attempts either sprawled or merged the sessions. Imposing the cluster
 * geometry directly gives all three properties we want, exactly:
 *
 *   1. **Dense nodes** — each session is laid out as a Vogel sunflower
 *      (``r_k = c·√(k+½)``, ``θ_k = k·ψ`` with ψ the golden angle
 *      ``π(3−√5) ≈ 137.508°``). This is the uniform-density even
 *      packing of a disc; its minimum nearest-neighbour spacing is a
 *      scale-invariant ``1.546·c`` (verified numerically), so choosing
 *      ``c = (2·ρ_max + pad)/1.546`` packs the node discs as tightly as
 *      they go without overlap. Nodes are ordered by descending in-
 *      session degree, so hubs sit at the dense centre and leaves on the
 *      rim — keeping some structure visible inside the disc.
 *   2. **Separated communities** — every session is its own disc, so the
 *      sessions never inter-mix; a gap is baked into each disc's packing
 *      radius for clear visual separation.
 *   3. **Circular overall** — the session discs are arranged by tangent
 *      circle-packing-in-a-circle (Wang et al. 2006, the algorithm
 *      behind d3.packSiblings): each disc is placed tangent to two
 *      already-placed discs at the position closest to the centre, which
 *      grows a compact, roughly circular enclosure.
 *
 * Schemas carry no session, so each is folded into the session its
 * consolidation sources mostly belong to (orphans share one small disc).
 * A final bounded de-overlap pass clears any residual disc collisions,
 * and the map is recentred on the origin. Everything is deterministic
 * given the data (golden-angle spiral, degree-then-key ordering, size-
 * then-key packing) so the layout reflects new memories, not RNG jitter.
 */
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
/** Scale-invariant min nearest-neighbour spacing of the Vogel lattice,
 *  ``min‖p_i − p_j‖ = NN_FACTOR · c`` (verified numerically for all n). */
const NN_FACTOR = 1.546;
/** Extra px folded into the sunflower spacing — small for density. */
const SUNFLOWER_PAD = 7;
/** Gap left between session discs during circle-packing. */
const GROUP_GAP = 40;
/** How strongly disc placement is pulled toward the discs a session is
 *  most bridged to (vs. pure compactness). Keeps cross-session edges
 *  short without overriding the circular packing. */
const PACK_LINK_PULL = 0.7;
/** Padding between disc edges in the residual de-overlap pass. */
const NODE_PAD = 6;

type SimNode = { key: string; r: number; x: number; y: number };

/**
 * Place ``members`` as a Vogel sunflower centred on the origin and
 * return the disc's bounding radius. ``members`` is assumed already
 * ordered (index 0 = centre). ``maxR`` is the largest node radius in the
 * group, which sets the spacing so even the two biggest neighbours clear.
 */
function placeSunflower(members: SimNode[], maxR: number): number {
  const m = members.length;
  if (m === 0) return 0;
  if (m === 1) {
    members[0].x = 0;
    members[0].y = 0;
    return members[0].r;
  }
  const c = (2 * maxR + SUNFLOWER_PAD) / NN_FACTOR;
  for (let k = 0; k < m; k++) {
    const rad = c * Math.sqrt(k + 0.5);
    const a = k * GOLDEN_ANGLE;
    members[k].x = rad * Math.cos(a);
    members[k].y = rad * Math.sin(a);
  }
  // Recentre on the lattice centroid so the disc is centred on the
  // origin (the first few Vogel points are slightly off-centre).
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

/** Final pass: clear any residual disc collisions across the whole map.
 *  Bounded passes so a large graph can't stall the one-shot layout. */
function relaxGlobal(positions: Positions, data: GraphData): void {
  const items: Array<{ p: { x: number; y: number }; r: number }> = [];
  for (const n of data.nodes) {
    const p = positions.get(n.key);
    if (p) items.push({ p, r: nodeRadius(n) });
  }
  const count = items.length;
  if (count < 2) return;
  const passes = count > 600 ? 8 : count > 250 ? 20 : 60;
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
          dx = ((i * 7 + j) % 11) - 5;
          dy = ((i * 5 + j) % 13) - 6;
          dist = Math.hypot(dx, dy) || 1;
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

/* ------------------------------------------------------------------- public */

/**
 * Lay out the memory map as one sunflower disc per session, the discs
 * circle-packed into a roughly circular whole. Returns a Promise to
 * preserve the caller's contract (the previous force/UMAP paths were
 * Promise-shaped); the work itself is synchronous and deterministic.
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
  const schemaGroup = (key: string): string => {
    const tally = new Map<string, number>();
    for (const e of data.consolidation_edges) {
      if (e.dst !== key) continue;
      const s = epSession.get(`ep:${e.src}`);
      if (s !== undefined) tally.set(s, (tally.get(s) ?? 0) + 1);
    }
    let best = ORPHAN;
    let bestN = 0;
    for (const [s, c] of tally) {
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

  // One pass over every edge feeds two things: in-session weighted
  // degree (hubs sort to the disc centre) for intra-group edges, and
  // an inter-disc link weight (how heavily two sessions bridge) for
  // cross-group edges, which steers the disc packing below.
  const degree = new Map<string, number>();
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
    } else {
      addLink(ga, gb, w);
      addLink(gb, ga, w);
    }
  };
  for (const e of data.sr_edges) edge(`ep:${e.src}`, `ep:${e.dst}`, clamp(e.m, 0, 1));
  for (const e of data.plasticity_edges)
    edge(`ep:${e.src}`, `ep:${e.dst}`, clamp(e.strength, 0, 1));
  for (const e of data.consolidation_edges) edge(`ep:${e.src}`, e.dst, 0.6);

  // Lay out each group as a sunflower; collect disc radii for packing.
  const discs: Array<{ key: string; r: number }> = [];
  for (const [key, members] of groups) {
    members.sort(
      (a, b) =>
        (degree.get(b.key) ?? 0) - (degree.get(a.key) ?? 0) ||
        a.key.localeCompare(b.key),
    );
    let maxR = 0;
    for (const nd of members) if (nd.r > maxR) maxR = nd.r;
    const radius = placeSunflower(members, maxR);
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
