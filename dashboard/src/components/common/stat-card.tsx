import type { LucideIcon } from "lucide-react";
import { TrendingDown, TrendingUp } from "lucide-react";
import { cn } from "@/lib/utils";

type Trend = { value: string; dir: "up" | "down" | "flat" };

/** Terminal-style metric tile. No card chrome — a hairline strip,
 *  mono label above a tabular figure, with a tiny iconographic
 *  glyph on the right edge as a visual anchor. */
export function StatCard({
  label,
  value,
  hint,
  icon: Icon,
  trend,
}: {
  label: string;
  value: string | number;
  hint?: string;
  icon: LucideIcon;
  trend?: Trend;
}) {
  return (
    <div className="panel interactive group flex flex-col gap-2 p-4">
      <div className="flex items-center justify-between gap-2">
        <div className="eyebrow flex items-center gap-1.5">
          <span className="inline-block h-1 w-1 bg-primary" aria-hidden />
          {label}
        </div>
        <Icon
          className="size-4 text-muted-foreground transition-colors group-hover:text-primary"
          aria-hidden
        />
      </div>
      <div className="metric text-3xl leading-none text-foreground sm:text-4xl">
        {value}
      </div>
      {(hint || trend) && (
        <div className="flex items-center gap-2 text-[0.7rem]">
          {trend && (
            <span
              className={cn(
                "inline-flex items-center gap-1 border border-border bg-muted/40 px-1.5 py-0.5 font-mono",
                trend.dir === "up" && "border-primary/30 text-primary",
                trend.dir === "down" && "border-destructive/30 text-destructive",
              )}
            >
              {trend.dir === "down" ? (
                <TrendingDown className="size-3" aria-hidden />
              ) : (
                <TrendingUp className="size-3" aria-hidden />
              )}
              {trend.value}
            </span>
          )}
          {hint && <span className="truncate text-muted-foreground">{hint}</span>}
        </div>
      )}
    </div>
  );
}
