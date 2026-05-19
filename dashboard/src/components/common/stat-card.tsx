import type { LucideIcon } from "lucide-react";
import { Card, CardContent } from "@/components/ui/card";

export function StatCard({
  label,
  value,
  hint,
  icon: Icon,
}: {
  label: string;
  value: string | number;
  hint?: string;
  icon: LucideIcon;
}) {
  return (
    <Card className="group transition-colors hover:border-primary/30">
      <CardContent className="flex items-start justify-between gap-4 p-4 sm:p-6">
        <div className="min-w-0">
          <div className="text-[0.7rem] font-semibold uppercase tracking-[0.12em] text-muted-foreground">
            {label}
          </div>
          <div className="mt-2.5 font-mono text-3xl font-semibold tracking-tight tabular-nums sm:text-4xl">
            {value}
          </div>
          {hint && (
            <div className="mt-1 truncate text-xs text-muted-foreground">
              {hint}
            </div>
          )}
        </div>
        <div className="grid size-10 shrink-0 place-items-center rounded-xl bg-primary/10 text-primary ring-1 ring-inset ring-primary/15 transition-colors group-hover:bg-primary/15">
          <Icon className="size-5" aria-hidden />
        </div>
      </CardContent>
    </Card>
  );
}
