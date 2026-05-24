import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { renderHook, waitFor } from "@testing-library/react";
import type { ReactNode } from "react";
import { afterEach, describe, expect, it, vi } from "vitest";
import { useConsolidate, useDeleteMemory, useForget } from "@/lib/queries";

function mockFetch(body: unknown, status = 200) {
  const fn = vi.fn().mockResolvedValue({
    ok: status >= 200 && status < 300,
    status,
    statusText: "x",
    json: () => Promise.resolve(body),
  });
  vi.stubGlobal("fetch", fn);
  return fn;
}

function wrapper(qc: QueryClient) {
  return function Wrapper({ children }: { children: ReactNode }) {
    return <QueryClientProvider client={qc}>{children}</QueryClientProvider>;
  };
}

afterEach(() => {
  vi.unstubAllGlobals();
});

describe("useDeleteMemory", () => {
  it("invalidates list/stats/graph and evicts the detail cache for the id", async () => {
    const qc = new QueryClient({
      defaultOptions: { queries: { retry: false, gcTime: 0 } },
    });
    // Seed the detail cache so we can prove it was removed.
    qc.setQueryData(["memory", 42], { id: 42 });
    expect(qc.getQueryData(["memory", 42])).toBeDefined();

    mockFetch({ deleted_ids: [42], count: 1 });

    const { result } = renderHook(() => useDeleteMemory(), {
      wrapper: wrapper(qc),
    });

    const invalidate = vi.spyOn(qc, "invalidateQueries");
    const remove = vi.spyOn(qc, "removeQueries");

    result.current.mutate(42);
    await waitFor(() => expect(result.current.isSuccess).toBe(true));

    const invalidatedKeys = invalidate.mock.calls.map(
      (c) => (c[0] as { queryKey: unknown[] }).queryKey,
    );
    expect(invalidatedKeys).toContainEqual(["memories"]);
    expect(invalidatedKeys).toContainEqual(["stats"]);
    expect(invalidatedKeys).toContainEqual(["graph"]);

    expect(remove).toHaveBeenCalledWith({
      queryKey: ["memory", 42],
      exact: true,
    });
    expect(qc.getQueryData(["memory", 42])).toBeUndefined();
  });
});

describe("useForget", () => {
  it("skips list/stats/graph invalidation when dry_run is true", async () => {
    const qc = new QueryClient({
      defaultOptions: { queries: { retry: false, gcTime: 0 } },
    });
    mockFetch({ candidate_ids: [1, 2], removed: 0, dry_run: true, scanned: 5 });

    const { result } = renderHook(() => useForget(), { wrapper: wrapper(qc) });
    const invalidate = vi.spyOn(qc, "invalidateQueries");

    result.current.mutate({ dry_run: true });
    await waitFor(() => expect(result.current.isSuccess).toBe(true));

    expect(invalidate).not.toHaveBeenCalled();
  });

  it("invalidates list/stats/graph when the run is real", async () => {
    const qc = new QueryClient({
      defaultOptions: { queries: { retry: false, gcTime: 0 } },
    });
    mockFetch({ candidate_ids: [], removed: 3, dry_run: false, scanned: 9 });

    const { result } = renderHook(() => useForget(), { wrapper: wrapper(qc) });
    const invalidate = vi.spyOn(qc, "invalidateQueries");

    result.current.mutate({ dry_run: false });
    await waitFor(() => expect(result.current.isSuccess).toBe(true));

    const keys = invalidate.mock.calls.map(
      (c) => (c[0] as { queryKey: unknown[] }).queryKey,
    );
    expect(keys).toContainEqual(["memories"]);
    expect(keys).toContainEqual(["stats"]);
    expect(keys).toContainEqual(["graph"]);
  });
});

describe("useConsolidate", () => {
  it("invalidates only stats on success", async () => {
    const qc = new QueryClient({
      defaultOptions: { queries: { retry: false, gcTime: 0 } },
    });
    mockFetch({ schemas_formed: 2, schemas: [] });

    const { result } = renderHook(() => useConsolidate(), {
      wrapper: wrapper(qc),
    });
    const invalidate = vi.spyOn(qc, "invalidateQueries");

    result.current.mutate({});
    await waitFor(() => expect(result.current.isSuccess).toBe(true));

    expect(invalidate).toHaveBeenCalledTimes(1);
    expect(invalidate).toHaveBeenCalledWith({ queryKey: ["stats"] });
  });
});
