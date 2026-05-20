import { useHealth } from "@/lib/queries";
import { cn } from "@/lib/utils";

/** Live connection chrome — single source of truth for the green/red
 *  dot + label pair shown in the top bar and the sidebar footer. */

export function StatusPill({ className }: { className?: string }) {
  const { data, isError } = useHealth();
  const ok = !!data && !isError;
  return (
    <div
      className={cn(
        "hidden items-center gap-2 rounded-full border border-border/60 bg-surface-overlay px-3 py-1.5 text-xs shadow-card backdrop-blur sm:flex",
        className,
      )}
    >
      <span
        className={cn(
          "size-1.5 rounded-full",
          ok ? "animate-pulse bg-success" : "bg-destructive",
        )}
        aria-hidden
      />
      <span className="font-medium">{ok ? "Connected" : "Offline"}</span>
      {data && (
        <span className="font-mono text-muted-foreground">
          {data.transport} · v{data.version} ·{" "}
          {Math.round(data.uptime_seconds)}s
        </span>
      )}
    </div>
  );
}

export function StatusPanel() {
  const { data, isError } = useHealth();
  const ok = !!data && !isError;
  return (
    <div className="rounded-xl border border-sidebar-border/70 bg-surface-canvas p-3 text-xs shadow-card sm:p-4">
      <div className="flex items-center gap-2 font-medium">
        <span
          className={cn(
            "size-2 rounded-full ring-4",
            ok
              ? "bg-success ring-success/20"
              : "bg-destructive ring-destructive/20",
          )}
          aria-hidden
        />
        <span className={ok ? "text-success" : "text-destructive"}>
          {ok ? "Connected" : "Offline"}
        </span>
      </div>
      {data && (
        <dl className="mt-2 space-y-1 text-muted-foreground">
          <div className="flex justify-between gap-2">
            <dt>Transport</dt>
            <dd className="font-mono">{data.transport}</dd>
          </div>
          <div className="flex justify-between gap-2">
            <dt>Embedding</dt>
            <dd className="font-mono">{data.embedding_backend}</dd>
          </div>
          <div className="flex justify-between gap-2">
            <dt>Uptime</dt>
            <dd className="font-mono">{Math.round(data.uptime_seconds)}s</dd>
          </div>
        </dl>
      )}
    </div>
  );
}
