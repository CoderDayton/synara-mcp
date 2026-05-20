import type { ReactNode } from "react";
import { cn } from "@/lib/utils";

/**
 * Flat console panel — hairline border, eyebrow + optional title row,
 * no drop shadow. The single Section primitive across Overview, Admin,
 * and Config; replaces the prior mix of Card/CardHeader and an inline
 * Panel in overview.tsx.
 *
 * - `eyebrow`     mono micro-label (the console identity cue)
 * - `title`       optional heading text (paired with `icon`/`subtitle`)
 * - `aside`       right-side controls (status pill, count, action)
 * - `bodyClassName` lets the caller drop the default padding when the
 *                  body holds a chart or a flush grid
 */
export function Panel({
  eyebrow,
  title,
  subtitle,
  icon,
  aside,
  children,
  className,
  bodyClassName,
}: {
  eyebrow?: ReactNode;
  title?: ReactNode;
  subtitle?: ReactNode;
  icon?: ReactNode;
  aside?: ReactNode;
  children: ReactNode;
  className?: string;
  bodyClassName?: string;
}) {
  const hasHeader = !!(eyebrow || title || aside);
  return (
    <section
      className={cn(
        "flex flex-col border border-border/70 bg-card",
        className,
      )}
    >
      {hasHeader && (
        <header className="flex items-start justify-between gap-3 border-b border-border/70 px-5 py-3">
          <div className="min-w-0 space-y-1">
            {eyebrow && <div className="eyebrow">{eyebrow}</div>}
            {title && (
              <h2 className="flex items-center gap-2 text-sm font-semibold tracking-tight text-foreground">
                {icon}
                {title}
              </h2>
            )}
            {subtitle && (
              <p className="text-xs text-muted-foreground">{subtitle}</p>
            )}
          </div>
          {aside && <div className="shrink-0">{aside}</div>}
        </header>
      )}
      <div className={cn("flex-1 p-5", bodyClassName)}>{children}</div>
    </section>
  );
}
