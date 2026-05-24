/**
 * Typed client for the Synara dashboard API.
 *
 * The backend sets `openapi_url=None` (internal admin API, not a public
 * contract), so these types are hand-mirrored from the FastAPI route
 * signatures in `src/synara/features/dashboard/routes/*`. Keep in sync
 * with that surface — it is the source of truth.
 *
 * Auth: the server's bearer token is optional (loopback) but required
 * off-loopback. When present it is read from `localStorage` and sent as
 * `Authorization: Bearer <token>`; a 401 surfaces as `ApiError` so the
 * UI can prompt for it.
 */

const TOKEN_KEY = "synara.dashboard.token";

export function getToken(): string | null {
  try {
    return localStorage.getItem(TOKEN_KEY);
  } catch {
    return null;
  }
}

export function setToken(token: string | null): void {
  try {
    if (token) localStorage.setItem(TOKEN_KEY, token);
    else localStorage.removeItem(TOKEN_KEY);
  } catch {
    /* storage unavailable (private mode) — requests just go unauthenticated */
  }
}

export class ApiError extends Error {
  readonly status: number;
  readonly detail: string;
  constructor(status: number, detail: string) {
    super(`API ${status}: ${detail}`);
    this.name = "ApiError";
    this.status = status;
    this.detail = detail;
  }
}

async function request<T>(
  path: string,
  init?: RequestInit & { json?: unknown; validate?: (raw: unknown) => T },
): Promise<T> {
  const headers = new Headers(init?.headers);
  const token = getToken();
  if (token) headers.set("Authorization", `Bearer ${token}`);
  let body = init?.body;
  if (init?.json !== undefined) {
    headers.set("Content-Type", "application/json");
    body = JSON.stringify(init.json);
  }

  const res = await fetch(`/api${path}`, { ...init, headers, body });
  if (!res.ok) {
    let detail = res.statusText;
    try {
      const data = (await res.json()) as { detail?: unknown };
      if (typeof data.detail === "string") detail = data.detail;
    } catch {
      /* non-JSON error body — keep statusText */
    }
    throw new ApiError(res.status, detail);
  }
  if (res.status === 204) return undefined as T;
  const raw: unknown = await res.json();
  // Optional runtime validation at the boundary: when supplied, the
  // validator either returns a well-typed value or throws an ApiError
  // carrying a 502 (Bad Gateway) — the server gave us a shape we
  // cannot trust to render. This converts silent "undefined in the
  // UI" failures into a loud, retryable error.
  if (init?.validate) {
    try {
      return init.validate(raw);
    } catch (e) {
      const msg = e instanceof Error ? e.message : "invalid response shape";
      throw new ApiError(502, `Schema mismatch on ${path}: ${msg}`);
    }
  }
  return raw as T;
}

/* ---------------------------------------------------- runtime type guards
 *
 * Minimal hand-rolled validators for the endpoints whose shape is most
 * likely to silently break the UI (numeric counts, enum strings). No
 * external schema library so the dashboard stays dependency-light;
 * normalizeGraph already covers the graph endpoint's tolerance path.
 */
function fail(msg: string): never {
  throw new Error(msg);
}
function isObj(v: unknown): v is Record<string, unknown> {
  return typeof v === "object" && v !== null && !Array.isArray(v);
}
function asStr(v: unknown, field: string): string {
  return typeof v === "string" ? v : fail(`expected string at ${field}`);
}
function asNum(v: unknown, field: string): number {
  return typeof v === "number" && Number.isFinite(v)
    ? v
    : fail(`expected number at ${field}`);
}

function validateHealth(raw: unknown): Health {
  if (!isObj(raw)) fail("expected object");
  const backend = raw.embedding_backend;
  if (backend !== "local" && backend !== "remote")
    fail(`embedding_backend must be "local" | "remote"`);
  return {
    status: asStr(raw.status, "status"),
    version: asStr(raw.version, "version"),
    transport: asStr(raw.transport, "transport"),
    db_path: asStr(raw.db_path, "db_path"),
    embedding_backend: backend,
    embedding_model: asStr(raw.embedding_model, "embedding_model"),
    uptime_seconds: asNum(raw.uptime_seconds, "uptime_seconds"),
  };
}

function validateStats(raw: unknown): Stats {
  if (!isObj(raw)) fail("expected object");
  return {
    episodic_count: asNum(raw.episodic_count, "episodic_count"),
    semantic_count: asNum(raw.semantic_count, "semantic_count"),
  };
}

function validateMemoryList(raw: unknown): MemoryList {
  if (!isObj(raw)) fail("expected object");
  if (!Array.isArray(raw.items)) fail("expected items: array");
  return raw as unknown as MemoryList;
}

/* ----------------------------------------------------------- response types */

export interface Health {
  status: string;
  version: string;
  transport: string;
  db_path: string;
  embedding_backend: "local" | "remote";
  embedding_model: string;
  uptime_seconds: number;
}

export interface Stats {
  episodic_count: number;
  semantic_count: number;
}

export type Params = Record<string, unknown>;

export interface MemoryListItem {
  id: number;
  content: string;
  metadata: Record<string, unknown>;
}

export interface RecallHit {
  id?: number;
  content?: string;
  score?: number;
  [k: string]: unknown;
}

export type MemoryList =
  | { kind: string; items: MemoryListItem[]; count: number; offset: number }
  | { kind: string; query: string; items: RecallHit[]; count: number };

export interface MemoryDetail {
  id: number;
  group_id: number;
  segments: Array<Record<string, unknown>>;
  sr_transitions: Array<{ dst: number; count: number }>;
  plasticity_edges: Array<{
    src: number;
    dst: number;
    weight: number;
    bonus: number;
    hits: number;
  }>;
}

export interface DeleteResult {
  deleted_ids: number[];
  count: number;
}

export interface EpisodicGraphNode {
  id: number;
  key: string;
  kind: "episodic";
  label: string;
  salience: number;
  retrieval_count: number;
  session_id: string | null;
  encoded_at: number;
  last_accessed: number;
  consolidated_into: number;
  group_id: number;
  segment_count: number;
  preview: string;
  is_focus: boolean;
}

export interface SemanticGraphNode {
  id: number;
  key: string;
  kind: "semantic";
  label: string;
  confidence: number;
  source_count: number;
  user_asserted: boolean;
  preview: string;
}

export type GraphNode = EpisodicGraphNode | SemanticGraphNode;

export interface SrEdge {
  src: number;
  dst: number;
  hits: number;
  m: number;
}

export interface PlasticityEdge {
  src: number;
  dst: number;
  hits: number;
  weight: number;
  bonus: number;
  strength: number;
  is_habit: boolean;
}

export interface GraphData {
  nodes: GraphNode[];
  sr_edges: SrEdge[];
  plasticity_edges: PlasticityEdge[];
  consolidation_edges: Array<{ src: number; dst: string }>;
  omega: number;
  episode_count: number;
  focus: number | null;
  truncated: boolean;
  /** True when the response came from a pre-enrichment server (the
   *  process was not restarted after a deploy): nodes arrive as bare
   *  ids with no salience/SR/plasticity metadata. The UI surfaces this
   *  instead of silently showing an empty-looking map. */
  legacy: boolean;
}

export interface ForgetResult {
  candidate_ids: number[];
  removed: number;
  dry_run: boolean;
  scanned: number;
}

export interface ConsolidateResult {
  schemas_formed: number;
  schemas: Array<Record<string, unknown>>;
}

export type ReflectResult = Record<string, unknown>;

/* ------------------------------------------------------- graph normalisation
 *
 * The served SPA and the running Python process can be version-skewed
 * (static assets are refreshed on disk independently of the live
 * server). A pre-enrichment server returns `nodes: number[]`, no
 * `consolidation_edges`, no `omega`. Normalising here — at the single
 * data-entry boundary — means every consumer always gets a well-formed
 * `GraphData`, so a not-yet-restarted server degrades gracefully
 * instead of white-screening the map. Full fidelity returns once the
 * server is restarted onto the enriched route.
 */
function num(v: unknown, d = 0): number {
  return typeof v === "number" && Number.isFinite(v) ? v : d;
}

function normalizeNode(raw: unknown, i: number): GraphNode {
  if (typeof raw === "number") {
    return {
      id: raw,
      key: `ep:${raw}`,
      kind: "episodic",
      label: `#${raw}`,
      salience: 0,
      retrieval_count: 0,
      session_id: null,
      encoded_at: 0,
      last_accessed: 0,
      consolidated_into: 0,
      group_id: raw,
      segment_count: 1,
      preview: "",
      is_focus: false,
    };
  }
  const o = (raw ?? {}) as Record<string, unknown>;
  const id = num(o.id, i);
  if (o.kind === "semantic") {
    return {
      id,
      key: typeof o.key === "string" ? o.key : `sem:${id}`,
      kind: "semantic",
      label: typeof o.label === "string" ? o.label : `schema #${id}`,
      confidence: num(o.confidence),
      source_count: num(o.source_count),
      user_asserted: o.user_asserted === true,
      preview: typeof o.preview === "string" ? o.preview : "",
    };
  }
  return {
    id,
    key: typeof o.key === "string" ? o.key : `ep:${id}`,
    kind: "episodic",
    label: typeof o.label === "string" ? o.label : `#${id}`,
    salience: num(o.salience),
    retrieval_count: num(o.retrieval_count),
    session_id: typeof o.session_id === "string" ? o.session_id : null,
    encoded_at: num(o.encoded_at),
    last_accessed: num(o.last_accessed),
    consolidated_into: num(o.consolidated_into),
    group_id: num(o.group_id, id),
    segment_count: num(o.segment_count, 1),
    preview: typeof o.preview === "string" ? o.preview : "",
    is_focus: o.is_focus === true,
  };
}

export function normalizeGraph(raw: unknown): GraphData {
  const r = (raw ?? {}) as Record<string, unknown>;
  const legacy =
    !("omega" in r) ||
    !("consolidation_edges" in r) ||
    (Array.isArray(r.nodes) && r.nodes.some((n) => typeof n === "number"));
  const nodes = Array.isArray(r.nodes) ? r.nodes.map(normalizeNode) : [];
  const srEdges = (Array.isArray(r.sr_edges) ? r.sr_edges : []).map((e) => {
    const o = e as Record<string, unknown>;
    return {
      src: num(o.src),
      dst: num(o.dst),
      hits: num(o.hits),
      m: num(o.m),
    };
  });
  const plEdges = (
    Array.isArray(r.plasticity_edges) ? r.plasticity_edges : []
  ).map((e) => {
    const o = e as Record<string, unknown>;
    const weight = num(o.weight);
    const bonus = num(o.bonus);
    return {
      src: num(o.src),
      dst: num(o.dst),
      hits: num(o.hits),
      weight,
      bonus,
      strength: num(o.strength, weight + bonus),
      is_habit: o.is_habit === true,
    };
  });
  const csEdges = (
    Array.isArray(r.consolidation_edges) ? r.consolidation_edges : []
  )
    .map((e) => {
      const o = e as Record<string, unknown>;
      // Schema ids arrive as `sem:N` strings. Guard against the
      // legacy/garbled case explicitly — `String(unknown)` may render
      // as `[object Object]`, which would silently corrupt the graph.
      const dst = typeof o.dst === "string" ? o.dst : "";
      return { src: num(o.src), dst };
    })
    .filter((e) => e.dst !== "");
  return {
    nodes,
    sr_edges: srEdges,
    plasticity_edges: plEdges,
    consolidation_edges: csEdges,
    omega: num(r.omega),
    episode_count: num(r.episode_count, nodes.length),
    focus: typeof r.focus === "number" ? r.focus : null,
    truncated: r.truncated === true,
    legacy,
  };
}

/* --------------------------------------------------------------- operations */

export const api = {
  health: () => request<Health>("/health", { validate: validateHealth }),
  stats: () => request<Stats>("/stats", { validate: validateStats }),
  params: () => request<Params>("/params"),

  memories: (q: {
    kind: "episodic" | "semantic";
    q?: string;
    limit?: number;
    offset?: number;
  }) => {
    const p = new URLSearchParams({ kind: q.kind });
    if (q.q) p.set("q", q.q);
    if (q.limit != null) p.set("limit", String(q.limit));
    if (q.offset != null) p.set("offset", String(q.offset));
    return request<MemoryList>(`/memories?${p.toString()}`, {
      validate: validateMemoryList,
    });
  },
  memoryDetail: (id: number) => request<MemoryDetail>(`/memories/${id}`),
  deleteMemory: (id: number) =>
    request<DeleteResult>(`/memories/${id}`, { method: "DELETE" }),

  graph: async (q: { focus?: number; depth?: number; max_nodes?: number }) => {
    const p = new URLSearchParams();
    if (q.focus != null) p.set("focus", String(q.focus));
    if (q.depth != null) p.set("depth", String(q.depth));
    if (q.max_nodes != null) p.set("max_nodes", String(q.max_nodes));
    const qs = p.toString();
    const raw = await request<unknown>(`/graph${qs ? `?${qs}` : ""}`);
    return normalizeGraph(raw);
  },

  consolidate: (body: {
    session_id?: string | null;
    n_clusters?: number | null;
    min_cluster_size?: number | null;
  }) => request<ConsolidateResult>("/admin/consolidate", { method: "POST", json: body }),
  forget: (body: {
    strength_floor?: number;
    decay_tau_seconds?: number | null;
    dry_run?: boolean;
    max_scan?: number;
  }) => request<ForgetResult>("/admin/forget", { method: "POST", json: body }),
  reflect: (body: { session_id: string; query?: string | null; k?: number }) =>
    request<ReflectResult>("/admin/reflect", { method: "POST", json: body }),
};
