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
  init?: RequestInit & { json?: unknown },
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
  return (await res.json()) as T;
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

export interface GraphData {
  nodes: number[];
  sr_edges: Array<{ src: number; dst: number; hits: number }>;
  plasticity_edges: Array<{
    src: number;
    dst: number;
    hits: number;
    weight: number;
    bonus: number;
  }>;
  truncated: boolean;
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

/* --------------------------------------------------------------- operations */

export const api = {
  health: () => request<Health>("/health"),
  stats: () => request<Stats>("/stats"),
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
    return request<MemoryList>(`/memories?${p.toString()}`);
  },
  memoryDetail: (id: number) => request<MemoryDetail>(`/memories/${id}`),
  deleteMemory: (id: number) =>
    request<DeleteResult>(`/memories/${id}`, { method: "DELETE" }),

  graph: (q: { focus?: number; depth?: number; max_nodes?: number }) => {
    const p = new URLSearchParams();
    if (q.focus != null) p.set("focus", String(q.focus));
    if (q.depth != null) p.set("depth", String(q.depth));
    if (q.max_nodes != null) p.set("max_nodes", String(q.max_nodes));
    const qs = p.toString();
    return request<GraphData>(`/graph${qs ? `?${qs}` : ""}`);
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
