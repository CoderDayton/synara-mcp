/**
 * Canvas renderer for the memory map.
 *
 * One `<canvas>` draws every node and edge — no per-node DOM, no SVG
 * paths, no framework reconciliation on pan/zoom. The scene is rebuilt
 * only when the inputs change (data, layout, selection, search); camera
 * moves redraw the same scene through a rAF-coalesced imperative loop,
 * so a 1k-node map pans at a flat O(N) per frame with zero React work.
 *
 * Look: Obsidian-style constellation — small solid dots tinted by
 * session (schemas cyan), thin translucent links, labels only past a
 * deliberate zoom-in. Selection rings the node and lights its
 * spreading-activation neighbourhood; search rings every hit.
 *
 * Interactions: drag = pan · wheel = zoom (cursor-anchored) · click =
 * inspect · double-click = recenter on episode · hover = preview
 * tooltip. Keyboard (canvas focused): arrows pan, +/- zoom, F fit,
 * Enter inspects the hovered node, Escape clears the selection.
 */
import {
  useEffect,
  useImperativeHandle,
  useRef,
  useState,
  type RefObject,
} from "react";
import type { GraphData, GraphNode } from "@/lib/api";
import { clamp, relativeTime, sessionHue, shortSession } from "@/lib/format";
import { nodeRadius, type Positions } from "./graph-layout";

export type GraphCanvasHandle = {
  fit: () => void;
  zoomBy: (factor: number) => void;
};

type SceneNode = {
  node: GraphNode;
  x: number;
  y: number;
  r: number;
  fill: string;
  /** Ring accent colour for selected/focus/hit states. */
  ringColor: string;
  dim: boolean;
  ring: "selected" | "focus" | "hit" | "bridge" | null;
};

type EdgeBucket = {
  path: Path2D;
  stroke: string;
  width: number;
  alpha: number;
  dash: number[] | null;
};

/** One edge of the lit (selected/search) neighbourhood, kept individual
 *  instead of bucketed so it can animate: a marching dash flows from
 *  src to dst (the SR transition / reinforcement direction) over a
 *  src-hue → dst-hue gradient. ``primary`` edges touch the selected
 *  node itself; the rest connect its neighbours to each other. */
type ActiveEdge = {
  x1: number;
  y1: number;
  x2: number;
  y2: number;
  c1: string;
  c2: string;
  primary: boolean;
};

/** Hoverable edge metrics, one variant per relation kind. */
export type EdgeInfo =
  | { kind: "sr"; src: number; dst: number; m: number; hits: number }
  | {
      kind: "plasticity";
      src: number;
      dst: number;
      weight: number;
      bonus: number;
      hits: number;
      is_habit: boolean;
    }
  | { kind: "consolidation"; src: number; dst: string }
  | { kind: "context"; session: string; src: number; dst: number };

type SceneEdge = {
  x1: number;
  y1: number;
  x2: number;
  y2: number;
  info: EdgeInfo;
};

/** Two edge infos describe the same relation (kind + endpoints). */
function sameEdge(a: EdgeInfo, b: EdgeInfo): boolean {
  return a.kind === b.kind && a.src === b.src && String(a.dst ?? "") === String(b.dst ?? "");
}

/** Squared distance from point p to segment (a, b). */
function segDist2(
  px: number,
  py: number,
  x1: number,
  y1: number,
  x2: number,
  y2: number,
): number {
  const dx = x2 - x1;
  const dy = y2 - y1;
  const len2 = dx * dx + dy * dy;
  let t = len2 > 0 ? ((px - x1) * dx + (py - y1) * dy) / len2 : 0;
  t = t < 0 ? 0 : t > 1 ? 1 : t;
  const qx = x1 + t * dx - px;
  const qy = y1 + t * dy - py;
  return qx * qx + qy * qy;
}

type Camera = { tx: number; ty: number; k: number };

const MIN_ZOOM = 0.1;
const MAX_ZOOM = 4;
const FIT_MAX_ZOOM = 1.4;
const FIT_PAD = 48;
const LABEL_ZOOM_MIN = 1.6;
/** Cap on hover-tooltip preview length — full text lives in the inspector. */
const PREVIEW_CHARS = 220;

/** Quantize a 0..1 strength into one of four draw buckets so edges
 *  batch into a handful of strokes instead of one per edge. */
function level(v: number): number {
  return Math.min(3, Math.floor(clamp(v, 0, 1) * 4));
}

function easeOutCubic(t: number): number {
  const u = 1 - t;
  return 1 - u * u * u;
}

/** Resolve the theme palette from CSS custom properties once per scene
 *  build — canvas can't consume `var()` directly. */
function readPalette(el: HTMLElement) {
  const css = getComputedStyle(el);
  const v = (name: string, fallback: string) =>
    css.getPropertyValue(name).trim() || fallback;
  return {
    sr: v("--color-chart-1", "oklch(0.72 0.12 250)"),
    semantic: v("--color-chart-2", "oklch(0.78 0.14 195)"),
    plasticity: v("--color-chart-3", "oklch(0.8 0.14 85)"),
    primary: v("--color-primary", "oklch(0.78 0.14 250)"),
    label: v("--color-muted-foreground", "oklch(0.72 0.02 286)"),
  };
}

export function GraphCanvas({
  ref,
  data,
  positions,
  layoutVersion,
  selectedKey,
  cameraFocus,
  focusNonce,
  lit,
  focusing,
  highlight,
  bridgeKeys,
  contextLens,
  selectedEdge,
  onSelect,
  onSelectEdge,
  onFocus,
  onZoom,
}: {
  ref?: RefObject<GraphCanvasHandle | null>;
  data: GraphData;
  positions: Positions;
  layoutVersion: number;
  selectedKey: string | null;
  cameraFocus: number | null;
  /** Bumped on every focus request so re-focusing the same episode
   *  (e.g. "Recenter" on the already-focused node) still re-centers. */
  focusNonce: number;
  /** Keys that stay bright while `focusing`; everything else dims. */
  lit: Set<string>;
  focusing: boolean;
  highlight?: Set<string>;
  bridgeKeys: Set<string>;
  contextLens: boolean;
  /** The edge currently inspected in the sidebar (matched by kind +
   *  endpoints) — kept highlighted on the canvas. */
  selectedEdge?: EdgeInfo | null;
  onSelect: (n: GraphNode | null) => void;
  /** Clicking an edge (not a dot) opens it in the sidebar. */
  onSelectEdge?: (info: EdgeInfo) => void;
  onFocus: (id: number) => void;
  onZoom?: (k: number) => void;
}) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const cameraRef = useRef<Camera>({ tx: 0, ty: 0, k: 1 });
  const sceneRef = useRef<{
    nodes: SceneNode[];
    buckets: EdgeBucket[];
    active: ActiveEdge[];
    edges: SceneEdge[];
    selected: SceneEdge | null;
  }>({
    nodes: [],
    buckets: [],
    active: [],
    edges: [],
    selected: null,
  });
  // Hovered edge, mirrored into a ref so draw() can highlight it
  // without re-running the scene build.
  const hoverEdgeRef = useRef<SceneEdge | null>(null);
  const rafRef = useRef(0);
  const animRef = useRef(0);
  // Edge-flow animation loop: runs ONLY while a lit neighbourhood has
  // edges to animate; idle maps cost zero frames.
  const flowRef = useRef(0);
  const [hover, setHover] = useState<{
    x: number;
    y: number;
    node: GraphNode;
    /** Flip the card left/up when the cursor is near the right/bottom
     *  edge so it never clips outside the canvas. */
    flipX: boolean;
    flipY: boolean;
  } | null>(null);
  const hoverRef = useRef<GraphNode | null>(null);

  /* ------------------------------------------------------------- drawing */

  const draw = () => {
    rafRef.current = 0;
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    const dpr = window.devicePixelRatio || 1;
    const { tx, ty, k } = cameraRef.current;
    const w = canvas.width / dpr;
    const h = canvas.height / dpr;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, w, h);
    ctx.setTransform(dpr * k, 0, 0, dpr * k, dpr * tx, dpr * ty);

    const scene = sceneRef.current;
    ctx.lineCap = "round";
    ctx.lineJoin = "round";
    for (const b of scene.buckets) {
      ctx.globalAlpha = b.alpha;
      ctx.strokeStyle = b.stroke;
      ctx.lineWidth = b.width;
      ctx.setLineDash(b.dash ?? []);
      ctx.stroke(b.path);
    }

    // Lit-neighbourhood edges: directional flow. Two layers per edge —
    // a faint continuous gradient base so the link reads as a whole,
    // and small marching beads (round caps on a tight dash) drifting
    // from src toward dst — the SR transition / reinforcement
    // direction. Primary edges (touching the selected node) run
    // slightly brighter and faster than neighbour↔neighbour ones.
    if (scene.active.length > 0) {
      const t = performance.now() / 1000;
      for (const e of scene.active) {
        const g = ctx.createLinearGradient(e.x1, e.y1, e.x2, e.y2);
        g.addColorStop(0, e.c1);
        g.addColorStop(1, e.c2);
        ctx.strokeStyle = g;
        // Layer 1: the base line.
        ctx.globalAlpha = e.primary ? 0.3 : 0.16;
        ctx.lineWidth = e.primary ? 0.9 : 0.6;
        ctx.setLineDash([]);
        ctx.beginPath();
        ctx.moveTo(e.x1, e.y1);
        ctx.lineTo(e.x2, e.y2);
        ctx.stroke();
        // Layer 2: drifting beads.
        ctx.globalAlpha = e.primary ? 0.85 : 0.5;
        ctx.lineWidth = e.primary ? 1.3 : 0.9;
        ctx.setLineDash([1.5, 11]);
        ctx.lineDashOffset = -t * (e.primary ? 13 : 8);
        ctx.stroke();
      }
      ctx.lineDashOffset = 0;
    }
    // Hovered / sidebar-selected edge: lift it out of the weave.
    ctx.setLineDash([]);
    for (const [e, alpha] of [
      [hoverEdgeRef.current, 0.5],
      [scene.selected, 0.65],
    ] as const) {
      if (!e) continue;
      ctx.globalAlpha = alpha;
      ctx.lineWidth = 1.5 / k;
      ctx.strokeStyle = "oklch(0.9 0.03 286)";
      ctx.beginPath();
      ctx.moveTo(e.x1, e.y1);
      ctx.lineTo(e.x2, e.y2);
      ctx.stroke();
    }

    // Viewport bounds in world space, padded a dot's worth.
    const minX = -tx / k - 16;
    const maxX = (w - tx) / k + 16;
    const minY = -ty / k - 16;
    const maxY = (h - ty) / k + 16;
    const labels = k >= LABEL_ZOOM_MIN;
    const labelFont = `${10 / k}px ui-monospace, monospace`;

    // Pass 1: faint bloom behind each bright dot — Obsidian-style depth
    // on a dark surface without shadowBlur (which would tank frame time).
    for (const sn of scene.nodes) {
      if (sn.dim) continue;
      if (sn.x < minX || sn.x > maxX || sn.y < minY || sn.y > maxY) continue;
      ctx.globalAlpha = 0.07;
      ctx.beginPath();
      ctx.arc(sn.x, sn.y, sn.r * 1.8, 0, Math.PI * 2);
      ctx.fillStyle = sn.fill;
      ctx.fill();
    }
    // Pass 2: the dots — flat solid pastel, no outline (the Obsidian
    // graph look: clean discs, colour does the grouping work).
    for (const sn of scene.nodes) {
      if (sn.x < minX || sn.x > maxX || sn.y < minY || sn.y > maxY) continue;
      ctx.globalAlpha = sn.dim ? 0.14 : 1;
      ctx.beginPath();
      ctx.arc(sn.x, sn.y, sn.r, 0, Math.PI * 2);
      ctx.fillStyle = sn.fill;
      ctx.fill();
      if (sn.ring) {
        ctx.beginPath();
        const pad = sn.ring === "bridge" ? 2.5 : 3.5;
        ctx.arc(sn.x, sn.y, sn.r + pad / k, 0, Math.PI * 2);
        ctx.lineWidth = (sn.ring === "bridge" ? 0.75 : 1.4) / k;
        ctx.strokeStyle = sn.ring === "bridge" ? sn.fill : sn.ringColor;
        ctx.globalAlpha = sn.dim ? 0.14 : sn.ring === "bridge" ? 0.35 : 0.95;
        ctx.stroke();
      }
      if (labels && !sn.dim) {
        ctx.font = labelFont;
        ctx.textAlign = "center";
        ctx.textBaseline = "top";
        ctx.globalAlpha = 0.75;
        ctx.fillStyle = "oklch(0.82 0.02 286)";
        const tag = sn.node.kind === "semantic" ? `⌬${sn.node.id}` : `#${sn.node.id}`;
        ctx.fillText(tag, sn.x, sn.y + sn.r + 3 / k);
      }
    }
    ctx.globalAlpha = 1;
  };

  const requestDraw = () => {
    if (!rafRef.current) rafRef.current = requestAnimationFrame(draw);
  };

  /* --------------------------------------------------------- scene build */

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const palette = readPalette(canvas);
    const nodes: SceneNode[] = [];
    const byKey = new Map<string, SceneNode>();
    const hueOf = new Map<string, number>();
    for (const n of data.nodes) {
      const p = positions.get(n.key);
      if (!p) continue;
      const hue = n.kind === "semantic" ? 195 : sessionHue(n.session_id);
      hueOf.set(n.key, hue);
      const isFocusNode =
        n.kind === "episodic" && cameraFocus != null && n.id === cameraFocus;
      const fill = `oklch(0.74 0.11 ${hue})`;
      const ring = isFocusNode
        ? ("focus" as const)
        : n.key === selectedKey
          ? ("selected" as const)
          : highlight?.has(n.key)
            ? ("hit" as const)
            : bridgeKeys.has(n.key)
              ? ("bridge" as const)
              : null;
      const sn: SceneNode = {
        node: n,
        x: p.x,
        y: p.y,
        r: nodeRadius(n) + (isFocusNode ? 1.5 : 0),
        fill,
        ringColor: ring === "hit" ? "oklch(0.92 0.03 286)" : palette.primary,
        dim: focusing && !lit.has(n.key),
        ring,
      };
      nodes.push(sn);
      byKey.set(n.key, sn);
    }

    // Edge buckets: group by (style, strength level, dimmed) so the
    // whole edge set strokes in a handful of Path2D draws. Edges whose
    // both ends are lit are split out for the directional-flow pass.
    const buckets = new Map<string, EdgeBucket>();
    const active: ActiveEdge[] = [];
    const edges: SceneEdge[] = [];
    const edgeInfo = (a: string, b: string, info: EdgeInfo) => {
      const pa = byKey.get(a);
      const pb = byKey.get(b);
      if (!pa || !pb) return;
      edges.push({ x1: pa.x, y1: pa.y, x2: pb.x, y2: pb.y, info });
    };
    const seg = (
      a: string,
      b: string,
      kind: string,
      stroke: string,
      lvl: number,
      width: number,
      alpha: number,
      dash: number[] | null,
    ) => {
      const pa = byKey.get(a);
      const pb = byKey.get(b);
      if (!pa || !pb) return;
      const isLit = focusing && lit.has(a) && lit.has(b);
      if (isLit) {
        // Lit edges leave the static buckets and animate individually.
        active.push({
          x1: pa.x,
          y1: pa.y,
          x2: pb.x,
          y2: pb.y,
          c1: pa.fill,
          c2: pb.fill,
          primary: a === selectedKey || b === selectedKey,
        });
        return;
      }
      const dimmed = focusing;
      const key = `${kind}:${lvl}:${dimmed ? 1 : 0}`;
      let bucket = buckets.get(key);
      if (!bucket) {
        bucket = {
          path: new Path2D(),
          stroke,
          width,
          alpha: dimmed ? 0.03 : alpha,
          dash,
        };
        buckets.set(key, bucket);
      }
      bucket.path.moveTo(pa.x, pa.y);
      bucket.path.lineTo(pb.x, pb.y);
    };

    if (contextLens) {
      const bySession = new Map<string, Array<{ id: number; t: number }>>();
      for (const n of data.nodes) {
        if (n.kind !== "episodic" || !n.session_id) continue;
        const entry = { id: n.id, t: n.created_at };
        const arr = bySession.get(n.session_id);
        if (arr) arr.push(entry);
        else bySession.set(n.session_id, [entry]);
      }
      for (const [sid, group] of bySession) {
        if (group.length < 2) continue;
        const ordered = [...group].sort((a, b) => a.t - b.t);
        const stroke = `oklch(0.72 0.14 ${sessionHue(sid)})`;
        for (let i = 1; i < ordered.length; i++) {
          seg(
            `ep:${ordered[i - 1].id}`,
            `ep:${ordered[i].id}`,
            `ctx${sessionHue(sid)}`,
            stroke,
            0,
            0.8,
            0.3,
            null,
          );
        }
      }
    }
    // SR + plasticity edges take the *source* node's session hue, so a
    // cluster's internal wiring reads in its own colour (the Obsidian
    // look) while cross-session bridges read as colour handoffs.
    // Buckets stay bounded: sessions × 4 strength levels × dim flag.
    for (const e of data.sr_edges) {
      const hue = hueOf.get(`ep:${e.src}`) ?? 286;
      const lvl = level(e.m);
      edgeInfo(`ep:${e.src}`, `ep:${e.dst}`, {
        kind: "sr",
        src: e.src,
        dst: e.dst,
        m: e.m,
        hits: e.hits,
      });
      seg(
        `ep:${e.src}`,
        `ep:${e.dst}`,
        `sr${hue}`,
        `oklch(0.78 0.05 ${hue})`,
        lvl,
        0.5 + lvl * 0.25,
        0.1 + lvl * 0.08,
        null,
      );
    }
    for (const e of data.plasticity_edges) {
      const hue = hueOf.get(`ep:${e.src}`) ?? 286;
      const lvl = level(e.strength) + (e.is_habit ? 4 : 0);
      edgeInfo(`ep:${e.src}`, `ep:${e.dst}`, {
        kind: "plasticity",
        src: e.src,
        dst: e.dst,
        weight: e.weight,
        bonus: e.bonus,
        hits: e.hits,
        is_habit: e.is_habit,
      });
      seg(
        `ep:${e.src}`,
        `ep:${e.dst}`,
        `pl${hue}`,
        `oklch(0.8 0.08 ${hue})`,
        lvl,
        e.is_habit ? 1.2 : 0.6,
        e.is_habit ? 0.42 : 0.16,
        [4, 3],
      );
    }
    for (const e of data.consolidation_edges) {
      edgeInfo(`ep:${e.src}`, e.dst, { kind: "consolidation", src: e.src, dst: e.dst });
      seg(`ep:${e.src}`, e.dst, "cs", palette.semantic, 0, 0.6, 0.25, [2, 3]);
    }

    const selected =
      selectedEdge != null
        ? (edges.find((e) => sameEdge(e.info, selectedEdge)) ?? null)
        : null;
    sceneRef.current = { nodes, buckets: [...buckets.values()], active, edges, selected };
    // Drive the flow animation only while there is something to flow.
    cancelAnimationFrame(flowRef.current);
    flowRef.current = 0;
    if (active.length > 0) {
      const tick = () => {
        draw();
        flowRef.current = requestAnimationFrame(tick);
      };
      flowRef.current = requestAnimationFrame(tick);
    } else {
      requestDraw();
    }
    return () => {
      cancelAnimationFrame(flowRef.current);
      flowRef.current = 0;
    };
  });

  /* ------------------------------------------------------ camera control */

  const setCamera = (next: Camera) => {
    cameraRef.current = next;
    onZoom?.(next.k);
    requestDraw();
  };

  const animateTo = (target: Camera, ms: number) => {
    cancelAnimationFrame(animRef.current);
    const from = { ...cameraRef.current };
    const t0 = performance.now();
    const tick = (now: number) => {
      const t = clamp((now - t0) / ms, 0, 1);
      const e = easeOutCubic(t);
      setCamera({
        tx: from.tx + (target.tx - from.tx) * e,
        ty: from.ty + (target.ty - from.ty) * e,
        k: from.k + (target.k - from.k) * e,
      });
      if (t < 1) animRef.current = requestAnimationFrame(tick);
    };
    animRef.current = requestAnimationFrame(tick);
  };

  const viewSize = () => {
    const canvas = canvasRef.current;
    const dpr = window.devicePixelRatio || 1;
    return canvas
      ? { w: canvas.width / dpr, h: canvas.height / dpr }
      : { w: 0, h: 0 };
  };

  const fit = () => {
    const { w, h } = viewSize();
    if (!w || !h || positions.size === 0) return;
    let minX = Infinity;
    let minY = Infinity;
    let maxX = -Infinity;
    let maxY = -Infinity;
    for (const p of positions.values()) {
      if (p.x < minX) minX = p.x;
      if (p.x > maxX) maxX = p.x;
      if (p.y < minY) minY = p.y;
      if (p.y > maxY) maxY = p.y;
    }
    const bw = Math.max(maxX - minX, 1);
    const bh = Math.max(maxY - minY, 1);
    const k = clamp(
      Math.min((w - FIT_PAD * 2) / bw, (h - FIT_PAD * 2) / bh),
      MIN_ZOOM,
      FIT_MAX_ZOOM,
    );
    animateTo(
      {
        k,
        tx: w / 2 - ((minX + maxX) / 2) * k,
        ty: h / 2 - ((minY + maxY) / 2) * k,
      },
      320,
    );
  };

  const centerOn = (key: string) => {
    const p = positions.get(key);
    const { w, h } = viewSize();
    if (!p || !w) return;
    // Glide in to a comfortable reading zoom — well past fit, well
    // short of max — keeping any deeper zoom the user already chose.
    const k = clamp(Math.max(cameraRef.current.k * 1.5, 2.1), 1.8, 2.6);
    animateTo({ k, tx: w / 2 - p.x * k, ty: h / 2 - p.y * k }, 600);
  };

  const zoomBy = (factor: number) => {
    const { w, h } = viewSize();
    const { tx, ty, k } = cameraRef.current;
    const nk = clamp(k * factor, MIN_ZOOM, MAX_ZOOM);
    // Keep the view centre fixed.
    const cx = w / 2;
    const cy = h / 2;
    animateTo(
      { k: nk, tx: cx - ((cx - tx) / k) * nk, ty: cy - ((cy - ty) / k) * nk },
      180,
    );
  };

  useImperativeHandle(ref, () => ({ fit, zoomBy }));

  // Frame the settled layout; pan to the focused episode instead when set.
  useEffect(() => {
    if (layoutVersion === 0) return;
    if (cameraFocus != null && positions.has(`ep:${cameraFocus}`)) {
      centerOn(`ep:${cameraFocus}`);
    } else {
      fit();
    }
     
  }, [layoutVersion, cameraFocus, focusNonce]);

  /* -------------------------------------------------------- interactions */

  const toWorld = (clientX: number, clientY: number) => {
    const canvas = canvasRef.current!;
    const rect = canvas.getBoundingClientRect();
    const { tx, ty, k } = cameraRef.current;
    return {
      x: (clientX - rect.left - tx) / k,
      y: (clientY - rect.top - ty) / k,
    };
  };

  const hitTest = (clientX: number, clientY: number): SceneNode | null => {
    const { x, y } = toWorld(clientX, clientY);
    const slack = 3 / cameraRef.current.k;
    let best: SceneNode | null = null;
    let bestD = Infinity;
    for (const sn of sceneRef.current.nodes) {
      const d = Math.hypot(sn.x - x, sn.y - y);
      if (d <= sn.r + slack && d < bestD) {
        best = sn;
        bestD = d;
      }
    }
    return best;
  };

  /** Nearest edge within a few screen pixels of the cursor — nodes
   *  take priority, so this only runs when no dot was hit. */
  const edgeTest = (clientX: number, clientY: number): SceneEdge | null => {
    const { x, y } = toWorld(clientX, clientY);
    const tol = 4 / cameraRef.current.k;
    const tol2 = tol * tol;
    let best: SceneEdge | null = null;
    let bestD = Infinity;
    for (const e of sceneRef.current.edges) {
      // Cheap bbox reject before the segment-distance math.
      if (
        x < Math.min(e.x1, e.x2) - tol ||
        x > Math.max(e.x1, e.x2) + tol ||
        y < Math.min(e.y1, e.y2) - tol ||
        y > Math.max(e.y1, e.y2) + tol
      )
        continue;
      const d2 = segDist2(x, y, e.x1, e.y1, e.x2, e.y2);
      if (d2 <= tol2 && d2 < bestD) {
        best = e;
        bestD = d2;
      }
    }
    return best;
  };

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    // Size to the container, DPR-aware.
    const resize = () => {
      const dpr = window.devicePixelRatio || 1;
      const rect = canvas.getBoundingClientRect();
      canvas.width = Math.max(1, Math.round(rect.width * dpr));
      canvas.height = Math.max(1, Math.round(rect.height * dpr));
      requestDraw();
    };
    const ro = new ResizeObserver(resize);
    ro.observe(canvas);
    resize();

    // Wheel zoom must be non-passive to preventDefault page scroll.
    const onWheel = (e: WheelEvent) => {
      e.preventDefault();
      cancelAnimationFrame(animRef.current);
      const rect = canvas.getBoundingClientRect();
      const px = e.clientX - rect.left;
      const py = e.clientY - rect.top;
      const { tx, ty, k } = cameraRef.current;
      const nk = clamp(k * Math.exp(-e.deltaY * 0.0016), MIN_ZOOM, MAX_ZOOM);
      setCamera({
        k: nk,
        tx: px - ((px - tx) / k) * nk,
        ty: py - ((py - ty) / k) * nk,
      });
    };
    canvas.addEventListener("wheel", onWheel, { passive: false });
    return () => {
      ro.disconnect();
      canvas.removeEventListener("wheel", onWheel);
    };
     
  }, []);

  useEffect(
    () => () => {
      cancelAnimationFrame(rafRef.current);
      cancelAnimationFrame(animRef.current);
      cancelAnimationFrame(flowRef.current);
    },
    [],
  );

  const dragRef = useRef<{
    id: number;
    x: number;
    y: number;
    moved: boolean;
  } | null>(null);

  const summary =
    `Memory map: ${data.nodes.length} node${data.nodes.length === 1 ? "" : "s"}, ` +
    `${data.sr_edges.length} successor edge${data.sr_edges.length === 1 ? "" : "s"}. ` +
    "Drag to pan, scroll to zoom, click to inspect, double-click to recenter. " +
    "Keyboard: arrows pan, +/- zoom, F fit, Enter inspects hovered, Escape clears.";

  return (
    <div className="relative h-full w-full">
      <canvas
        ref={canvasRef}
        role="application"
        aria-label={summary}
        tabIndex={0}
        className="h-full w-full touch-none outline-none focus-visible:ring-1 focus-visible:ring-primary"
        style={{ cursor: hover ? "pointer" : "grab" }}
        onPointerDown={(e) => {
          canvasRef.current?.setPointerCapture(e.pointerId);
          dragRef.current = { id: e.pointerId, x: e.clientX, y: e.clientY, moved: false };
          cancelAnimationFrame(animRef.current);
        }}
        onPointerMove={(e) => {
          const drag = dragRef.current;
          if (drag && drag.id === e.pointerId) {
            const dx = e.clientX - drag.x;
            const dy = e.clientY - drag.y;
            if (Math.abs(dx) + Math.abs(dy) > 3) drag.moved = true;
            drag.x = e.clientX;
            drag.y = e.clientY;
            const { tx, ty, k } = cameraRef.current;
            setCamera({ tx: tx + dx, ty: ty + dy, k });
            return;
          }
          const hit = hitTest(e.clientX, e.clientY);
          hoverRef.current = hit?.node ?? null;
          const edgeHit = hit ? null : edgeTest(e.clientX, e.clientY);
          if (hoverEdgeRef.current !== edgeHit) {
            hoverEdgeRef.current = edgeHit;
            requestDraw();
          }
          // Cursor is set imperatively: reading the hover-edge ref during
          // render trips the compiler's ref rules, and a React state just
          // for the pointer shape would re-render per mousemove.
          e.currentTarget.style.cursor = hit || edgeHit ? "pointer" : "grab";
          const rect = canvasRef.current!.getBoundingClientRect();
          const x = e.clientX - rect.left;
          const y = e.clientY - rect.top;
          setHover(
            hit
              ? {
                  x,
                  y,
                  node: hit.node,
                  flipX: x > rect.width - 320,
                  flipY: y > rect.height - 220,
                }
              : null,
          );
        }}
        onPointerUp={(e) => {
          const drag = dragRef.current;
          dragRef.current = null;
          if (drag && drag.moved) return;
          const hit = hitTest(e.clientX, e.clientY);
          if (hit) {
            onSelect(hit.node);
            return;
          }
          const edgeHit = edgeTest(e.clientX, e.clientY);
          if (edgeHit && onSelectEdge) {
            onSelectEdge(edgeHit.info);
            return;
          }
          onSelect(null);
        }}
        onPointerLeave={() => {
          hoverRef.current = null;
          if (hoverEdgeRef.current) {
            hoverEdgeRef.current = null;
            requestDraw();
          }
          setHover(null);
        }}
        onDoubleClick={(e) => {
          const hit = hitTest(e.clientX, e.clientY);
          if (hit && hit.node.kind === "episodic") onFocus(hit.node.id);
        }}
        onKeyDown={(e) => {
          const { tx, ty, k } = cameraRef.current;
          const step = 60;
          if (e.key === "ArrowLeft") setCamera({ tx: tx + step, ty, k });
          else if (e.key === "ArrowRight") setCamera({ tx: tx - step, ty, k });
          else if (e.key === "ArrowUp") setCamera({ tx, ty: ty + step, k });
          else if (e.key === "ArrowDown") setCamera({ tx, ty: ty - step, k });
          else if (e.key === "+" || e.key === "=") zoomBy(1.25);
          else if (e.key === "-") zoomBy(0.8);
          else if (e.key === "f" || e.key === "F") fit();
          else if (e.key === "Enter" && hoverRef.current) onSelect(hoverRef.current);
          else if (e.key === "Escape") onSelect(null);
          else return;
          e.preventDefault();
        }}
      />
      {hover && (
        <div
          className="pointer-events-none absolute z-30 w-60 overflow-hidden rounded-lg border border-border/70 bg-surface-overlay/95 font-mono text-[0.62rem] leading-snug shadow-card backdrop-blur duration-150 animate-in fade-in-0 zoom-in-95 sm:w-72 sm:text-[0.65rem]"
          style={{
            left: hover.flipX ? undefined : hover.x + 14,
            right: hover.flipX ? `calc(100% - ${hover.x - 14}px)` : undefined,
            top: hover.flipY ? undefined : hover.y + 14,
            bottom: hover.flipY ? `calc(100% - ${hover.y - 14}px)` : undefined,
          }}
        >
          {(() => {
              const node = hover.node;
              return (
                <>
                  {/* Header: hue swatch + id + kind tag */}
                  <div className="flex items-center gap-1.5 border-b border-border/60 px-2.5 py-1.5">
                    <span
                      aria-hidden
                      className="size-2 shrink-0 rounded-full"
                      style={{
                        background: `oklch(0.74 0.11 ${
                          node.kind === "semantic" ? 195 : sessionHue(node.session_id)
                        })`,
                      }}
                    />
                    <span className="font-semibold text-foreground">
                      {node.kind === "semantic" ? `⌬${node.id}` : `#${node.id}`}
                    </span>
                    {node.kind === "episodic" && node.session_id && (
                      <span className="truncate text-muted-foreground">
                        {shortSession(node.session_id)}
                      </span>
                    )}
                    <span className="ml-auto shrink-0 rounded bg-foreground/10 px-1.5 py-0.5 text-[0.55rem] tracking-wider text-muted-foreground uppercase">
                      {node.kind === "semantic"
                        ? node.user_asserted
                          ? "asserted"
                          : "schema"
                        : "episode"}
                    </span>
                  </div>
                  {/* Stats strip */}
                  <div className="flex flex-wrap gap-x-3 gap-y-0.5 px-2.5 py-1.5 text-muted-foreground">
                    {node.kind === "episodic" ? (
                      <>
                        <span>
                          sal{" "}
                          <span className="tabular-nums text-foreground/90">
                            {node.salience.toFixed(2)}
                          </span>
                        </span>
                        <span>
                          recalls{" "}
                          <span className="tabular-nums text-foreground/90">
                            {node.retrieval_count}
                          </span>
                        </span>
                        <span>
                          age{" "}
                          <span className="tabular-nums text-foreground/90">
                            {relativeTime(node.created_at)}
                          </span>
                        </span>
                        {node.consolidated_into > 0 && (
                          <span>
                            schema{" "}
                            <span className="tabular-nums text-foreground/90">
                              ⌬{node.consolidated_into}
                            </span>
                          </span>
                        )}
                      </>
                    ) : (
                      <>
                        <span>
                          conf{" "}
                          <span className="tabular-nums text-foreground/90">
                            {clamp(node.confidence, 0, 1).toFixed(2)}
                          </span>
                        </span>
                        <span>
                          sources{" "}
                          <span className="tabular-nums text-foreground/90">
                            {node.source_count}
                          </span>
                        </span>
                      </>
                    )}
                  </div>
                  {node.preview && (
                    <div className="line-clamp-4 border-t border-border/60 px-2.5 py-1.5 text-muted-foreground">
                      {node.preview.slice(0, PREVIEW_CHARS)}
                    </div>
                  )}
                  <div className="border-t border-border/60 bg-foreground/[0.03] px-2.5 py-1 text-[0.55rem] tracking-wide text-muted-foreground/70">
                    click — inspect
                    {node.kind === "episodic" ? " · double-click — recenter" : ""}
                  </div>
                </>
              );
            })()}
        </div>
      )}
    </div>
  );
}
