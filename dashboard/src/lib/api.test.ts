import { afterEach, describe, expect, it, vi } from "vitest";
import { ApiError, api, setToken } from "@/lib/api";

function mockFetch(status: number, body: unknown) {
  const fn = vi.fn().mockResolvedValue({
    ok: status >= 200 && status < 300,
    status,
    statusText: "x",
    json: async () => body,
  } as Response);
  vi.stubGlobal("fetch", fn);
  return fn;
}

afterEach(() => {
  setToken(null);
  vi.unstubAllGlobals();
});

describe("api bearer-token injection", () => {
  it("omits Authorization when no token is set", async () => {
    const f = mockFetch(200, { status: "ok" });
    await api.health();
    const headers = (f.mock.calls[0][1] as RequestInit).headers as Headers;
    expect(headers.get("Authorization")).toBeNull();
  });

  it("sends Authorization: Bearer <token> when a token is set", async () => {
    setToken("s3cr3t");
    const f = mockFetch(200, { status: "ok" });
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
