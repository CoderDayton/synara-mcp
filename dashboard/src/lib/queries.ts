import {
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

export const useMemories = (q: {
  kind: "episodic" | "semantic";
  q?: string;
  limit?: number;
  offset?: number;
}) => useQuery({ queryKey: ["memories", q], queryFn: () => api.memories(q) });

export const useMemoryDetail = (id: number | null) =>
  useQuery({
    queryKey: ["memory", id],
    queryFn: () => api.memoryDetail(id as number),
    enabled: id != null,
  });

export const useGraph = (q: {
  focus?: number;
  depth?: number;
  max_nodes?: number;
}) => useQuery({ queryKey: ["graph", q], queryFn: () => api.graph(q) });

export function useDeleteMemory() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (id: number) => api.deleteMemory(id),
    onSuccess: () => {
      void qc.invalidateQueries({ queryKey: ["memories"] });
      void qc.invalidateQueries({ queryKey: ["stats"] });
      void qc.invalidateQueries({ queryKey: ["graph"] });
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
