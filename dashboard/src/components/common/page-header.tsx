import type { ReactNode } from "react";

/** Page header — a single mono command line, not a hero band.
 *  The dock already names the route; this row owns the in-page
 *  description and any page-scoped actions. */
export function PageHeader({
  eyebrow,
  title,
  subtitle,
  actions,
}: {
  eyebrow?: string;
  title: string;
  subtitle?: string;
  actions?: ReactNode;
}) {
  return (
    <header className="mb-4 flex flex-col gap-3 border-b border-border/70 pb-4 sm:mb-5 sm:flex-row sm:items-end sm:justify-between">
      <div className="min-w-0">
        {eyebrow && (
          <div className="eyebrow flex items-center gap-2">
            <span className="inline-block h-1.5 w-1.5 bg-primary" aria-hidden />
            {eyebrow}
          </div>
        )}
        <h1 className="display mt-2 font-mono">
          <span className="text-primary">$ </span>
          <span className="text-foreground">{title}</span>
        </h1>
        {subtitle && (
          <p className="mt-2 max-w-prose text-xs leading-relaxed text-muted-foreground">
            {subtitle}
          </p>
        )}
      </div>
      {actions && (
        <div className="flex shrink-0 flex-wrap items-center gap-2">
          {actions}
        </div>
      )}
    </header>
  );
}
