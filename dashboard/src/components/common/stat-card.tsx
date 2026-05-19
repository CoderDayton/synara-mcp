import type { LucideIcon } from "lucide-react";
import { TrendingDown, TrendingUp } from "lucide-react";
import { Badge } from "@/components/ui/badge";
import { Card, CardContent } from "@/components/ui/card";
import { cn } from "@/lib/utils";

type Trend = { value: string; dir: "up" | "down" | "flat" };

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
    <Card className="group transition-colors hover:border-primary/30">
      <CardContent className="flex flex-col gap-4 p-4 sm:p-6">
        <div className="flex items-start justify-between gap-4">
          <div className="eyebrow">{label}</div>
          <div className="grid size-9 shrink-0 place-items-center rounded-xl bg-primary/10 text-primary ring-1 ring-inset ring-primary/15 transition-colors group-hover:bg-primary/15">
            <Icon className="size-5" aria-hidden />
          </div>
        </div>
        <div className="min-w-0">
          <div className="metric text-3xl tracking-tight sm:text-4xl">
            {value}
          </div>
          {(hint || trend) && (
            <div className="mt-2 flex items-center gap-2 text-xs text-muted-foreground">
              {trend && (
                <Badge
                  variant="outline"
                  className={cn(
                    "gap-1 px-1.5 font-mono tabular-nums",
                    trend.dir === "up" && "text-success",
                    trend.dir === "down" && "text-destructive",
                  )}
                >
                  {trend.dir === "down" ? (
                    <TrendingDown className="size-3" aria-hidden />
                  ) : (
                    <TrendingUp className="size-3" aria-hidden />
                  )}
                  {trend.value}
                </Badge>
              )}
              {hint && <span className="truncate">{hint}</span>}
            </div>
          )}
        </div>
      </CardContent>
    </Card>
  );
}
