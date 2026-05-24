import { afterEach, describe, expect, it, vi } from "vitest";
import { ApiError, api, setToken } from "@/lib/api";

// Minimal well-formed Health body — the api client now validates the
// response shape at the boundary (api.ts), so bearer-token tests need
// a payload that passes validation.
const HEALTH_OK = {
  status: "ok",
  version: "0.0.0",
  transport: "stdio",
  db_path: ":memory:",
  embedding_backend: "local",
  embedding_model: "test",
  uptime_seconds: 0,
};

function mockFetch(status: number, body: unknown) {
  const fn = vi.fn().mockResolvedValue({
    ok: status >= 200 && status < 300,
    status,
    statusText: "x",
    json: () => Promise.resolve(body),
  });
  vi.stubGlobal("fetch", fn);
  return fn;
}

afterEach(() => {
  setToken(null);
  vi.unstubAllGlobals();
});

describe("api bearer-token injection", () => {
  it("omits Authorization when no token is set", async () => {
    const f = mockFetch(200, HEALTH_OK);
    await api.health();
    const headers = (f.mock.calls[0][1] as RequestInit).headers as Headers;
    expect(headers.get("Authorization")).toBeNull();
  });

  it("sends Authorization: Bearer <token> when a token is set", async () => {
    setToken("s3cr3t");
    const f = mockFetch(200, HEALTH_OK);
    await api.health();
    const headers = (f.mock.calls[0][1] as RequestInit).headers as Headers;
    expect(headers.get("Authorization")).toBe("Bearer s3cr3t");
  });

  it("raises ApiError with detail on 401", async () => {
    mockFetch(401, { detail: "missing bearer token" });
    await expect(api.health()).rejects.toMatchObject({
      name: "ApiError",
      status: 401,
      detail: "missing bearer token",
    });
    await expect(api.health()).rejects.toBeInstanceOf(ApiError);
  });
});

describe("api boundary validation", () => {
  it("rejects malformed health payload with ApiError 502", async () => {
    mockFetch(200, { status: "ok" /* missing version, backend, etc. */ });
    await expect(api.health()).rejects.toMatchObject({
      name: "ApiError",
      status: 502,
    });
  });

  it("rejects an unknown embedding_backend value", async () => {
    mockFetch(200, { ...HEALTH_OK, embedding_backend: "magic" });
    await expect(api.health()).rejects.toBeInstanceOf(ApiError);
  });

  it("accepts a well-formed health payload", async () => {
    mockFetch(200, HEALTH_OK);
    const h = await api.health();
    expect(h.embedding_backend).toBe("local");
    expect(h.uptime_seconds).toBe(0);
  });
});

describe("api.graph normalisation (version-skew tolerance)", () => {
  it("coerces a legacy payload (numeric nodes, missing arrays) into a safe shape", async () => {
    // A server that has not been restarted onto the enriched route:
    // nodes are bare ids, sr_edges lack `m`, and the new arrays/scalars
    // are absent entirely. The client must not let this crash render.
    mockFetch(200, {
      nodes: [1, 2],
      sr_edges: [{ src: 1, dst: 2, hits: 3 }],
      plasticity_edges: [{ src: 1, dst: 2, hits: 4, weight: 0.5, bonus: 0.1 }],
      truncated: false,
    });
    const g = await api.graph({});
    expect(Array.isArray(g.consolidation_edges)).toBe(true);
    expect(g.consolidation_edges).toHaveLength(0);
    expect(typeof g.omega).toBe("number");
    expect(g.nodes[0]).toMatchObject({ id: 1, key: "ep:1", kind: "episodic" });
    expect(g.sr_edges[0].m).toBe(0);
    expect(g.plasticity_edges[0].strength).toBeCloseTo(0.6);
    expect(g.plasticity_edges[0].is_habit).toBe(false);
    expect(g.focus).toBeNull();
    expect(g.legacy).toBe(true);
  });

  it("passes an enriched payload through unchanged", async () => {
    const node = {
      id: 7,
      key: "ep:7",
      kind: "episodic",
      label: "#7",
      salience: 0.8,
      retrieval_count: 2,
      session_id: "s1",
      encoded_at: 1,
      last_accessed: 2,
      consolidated_into: 0,
      group_id: 7,
      segment_count: 1,
      preview: "hi",
      is_focus: true,
    };
    mockFetch(200, {
      nodes: [node],
      sr_edges: [{ src: 7, dst: 8, hits: 1, m: 0.42 }],
      plasticity_edges: [],
      consolidation_edges: [{ src: 7, dst: "sem:3" }],
      omega: 0.21,
      episode_count: 99,
      focus: 7,
      truncated: false,
    });
    const g = await api.graph({});
    expect(g.nodes[0]).toMatchObject({ id: 7, salience: 0.8 });
    expect(g.sr_edges[0].m).toBe(0.42);
    expect(g.consolidation_edges[0]).toEqual({ src: 7, dst: "sem:3" });
    expect(g.omega).toBe(0.21);
    expect(g.focus).toBe(7);
    expect(g.legacy).toBe(false);
  });
});
