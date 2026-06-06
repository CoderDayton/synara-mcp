import {
  keepPreviousData,
  useMutation,
  useQuery,
  useQueryClient,
  QueryClient,
} from "@tanstack/react-query";
import { api } from "@/lib/api";

export const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      staleTime: 10_000,
      retry: (count, err) =>
        // Never retry auth failures — the token is wrong, not flaky.
        !(err instanceof Error && err.name === "ApiError" && /^API 401/.test(err.message)) &&
        count < 2,
    },
  },
});

export const useHealth = () =>
  useQuery({ queryKey: ["health"], queryFn: api.health, refetchInterval: 15_000 });

export const useStats = () =>
  useQuery({ queryKey: ["stats"], queryFn: api.stats, refetchInterval: 15_000 });

export const useParams = () =>
  useQuery({ queryKey: ["params"], queryFn: api.params });

export const useToolMetrics = () =>
  useQuery({
    queryKey: ["tool-metrics"],
    queryFn: api.toolMetrics,
    refetchInterval: 5_000,
  });

export const useMemories = (q: {
  kind: "episodic" | "semantic";
  q?: string;
  limit?: number;
  offset?: number;
}) =>
  useQuery({
    queryKey: ["memories", q],
    queryFn: () => api.memories(q),
    // Keep prior page visible while pagination/search refetches so the
    // table doesn't blank-flash between keystrokes or page flips.
    placeholderData: keepPreviousData,
  });

export const useMemoryDetail = (id: number | null) =>
  useQuery({
    queryKey: ["memory", id],
    queryFn: () => api.memoryDetail(id as number),
    enabled: id != null,
  });

export const useSemanticDetail = (id: number | null) =>
  useQuery({
    queryKey: ["semantic", id],
    queryFn: () => api.semanticDetail(id as number),
    enabled: id != null,
  });

export const useGraph = (q: {
  focus?: number;
  depth?: number;
  max_nodes?: number;
}) =>
  useQuery({
    queryKey: ["graph", q],
    queryFn: () => api.graph(q),
    // Keep the prior graph data while refetching on focus/depth change
    // so the d3-force layout doesn't unmount + reset node positions.
    placeholderData: keepPreviousData,
  });

export function useDeleteMemory() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (id: number) => api.deleteMemory(id),
    onSuccess: (_data, id) => {
      void qc.invalidateQueries({ queryKey: ["memories"] });
      void qc.invalidateQueries({ queryKey: ["stats"] });
      void qc.invalidateQueries({ queryKey: ["graph"] });
      // Evict the detail cache for the deleted id so any still-mounted
      // inspector unmounts cleanly instead of showing stale content.
      qc.removeQueries({ queryKey: ["memory", id], exact: true });
    },
  });
}

export function useDeleteSemantic() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (id: number) => api.deleteSemantic(id),
    onSuccess: (_data, id) => {
      void qc.invalidateQueries({ queryKey: ["memories"] });
      void qc.invalidateQueries({ queryKey: ["stats"] });
      void qc.invalidateQueries({ queryKey: ["graph"] });
      // Evict the semantic detail cache so a still-mounted inspector
      // unmounts cleanly instead of showing a deleted schema.
      qc.removeQueries({ queryKey: ["semantic", id], exact: true });
    },
  });
}

export function useForget() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: api.forget,
    onSuccess: (r) => {
      if (!r.dry_run) {
        void qc.invalidateQueries({ queryKey: ["memories"] });
        void qc.invalidateQueries({ queryKey: ["stats"] });
        void qc.invalidateQueries({ queryKey: ["graph"] });
      }
    },
  });
}

export function useConsolidate() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: api.consolidate,
    onSuccess: () => void qc.invalidateQueries({ queryKey: ["stats"] }),
  });
}

export function useReflect() {
  return useMutation({ mutationFn: api.reflect });
}

export function useRestart() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: api.restart,
    // The server is re-execing; drop cached health/stats so the UI
    // reflects the brief outage and refetches once it's back up.
    onSuccess: () => {
      void qc.invalidateQueries({ queryKey: ["health"] });
      void qc.invalidateQueries({ queryKey: ["stats"] });
    },
  });
}
