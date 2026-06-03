import { formatDuration } from "@/lib/format";
import { useHealth } from "@/lib/queries";
import { cn } from "@/lib/utils";

/** Single source of truth for live MCP connection state.
 *
 *  Renders inline in the dock; no sidebar duplicate. Composed of a
 *  pulsing phosphor dot, an `OK / OFF` literal, and a tabular row of
 *  transport/version/uptime/embedding fields so the user always sees
 *  *what kind* of connection is live, not just whether one exists. */
export function StatusReadout({ className }: { className?: string }) {
  const { data, isError } = useHealth();
  const ok = !!data && !isError;
  return (
    <div
      className={cn(
        "flex items-center gap-3 font-mono text-[0.7rem] uppercase tracking-wider",
        className,
      )}
      role="status"
      aria-live="polite"
    >
      <span className="flex items-center gap-2">
        <span className="pulse-dot" data-state={ok ? "ok" : "off"} aria-hidden />
        <span className={ok ? "text-success" : "text-destructive"}>
          {ok ? "ok" : "off"}
        </span>
      </span>
      {data && (
        <>
          <span className="hidden h-3 w-px bg-border sm:inline-block" aria-hidden />
          <span className="hidden items-center gap-3 text-muted-foreground sm:flex">
            <span>
              <span className="text-foreground/60">tx</span>{" "}
              <span className="text-foreground">{data.transport}</span>
            </span>
            <span>
              <span className="text-foreground/60">ver</span>{" "}
              <span className="text-foreground">v{data.version}</span>
            </span>
            <span>
              <span className="text-foreground/60">up</span>{" "}
              <span className="text-foreground tabular-nums">
                {formatDuration(data.uptime_seconds)}
              </span>
            </span>
            <span className="hidden md:inline">
              <span className="text-foreground/60">emb</span>{" "}
              <span className="text-foreground">{data.embedding_backend}</span>
            </span>
            <span className="hidden lg:inline">
              <span className="text-foreground/60">client</span>{" "}
              <span className="text-foreground">{data.mcp_client}</span>
            </span>
          </span>
        </>
      )}
    </div>
  );
}
