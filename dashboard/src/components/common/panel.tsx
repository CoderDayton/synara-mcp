import type { ReactNode } from "react";
import { cn } from "@/lib/utils";

/**
 * Terminal panel — the single Section primitive. A hairline phosphor
 * border, near-black fill, a compact mono header bar with a prompt
 * mark, and an optional right-aligned aside for inline controls /
 * live tags.
 *
 * Variants:
 *  - default: solid card surface
 *  - canvas:  sunken substrate (graphs, code blocks)
 *  - raised:  one notch above canvas, for stacked sub-panels
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
  variant = "default",
}: {
  eyebrow?: ReactNode;
  title?: ReactNode;
  subtitle?: ReactNode;
  icon?: ReactNode;
  aside?: ReactNode;
  children: ReactNode;
  className?: string;
  bodyClassName?: string;
  variant?: "default" | "canvas" | "raised";
}) {
  const hasHeader = !!(eyebrow || title || aside);
  const surface =
    variant === "canvas"
      ? "bg-surface-canvas"
      : variant === "raised"
        ? "bg-surface-floating"
        : "bg-card";
  return (
    <section
      className={cn(
        "panel relative flex flex-col",
        surface,
        className,
      )}
    >
      {hasHeader && (
        <header
          className={cn(
            "relative z-10 flex items-center justify-between gap-3 border-b border-border px-4 py-2",
            "before:absolute before:inset-x-0 before:bottom-0 before:h-px before:bg-gradient-to-r before:from-transparent before:via-primary/20 before:to-transparent",
          )}
        >
          <div className="flex min-w-0 items-center gap-2.5">
            {icon && (
              <span className="grid size-5 shrink-0 place-items-center text-primary">
                {icon}
              </span>
            )}
            <div className="min-w-0 leading-tight">
              {title && (
                <h2 className="prompt truncate font-mono text-[0.72rem] font-medium uppercase tracking-wider text-foreground">
                  {title}
                </h2>
              )}
              {eyebrow && !title && (
                <div className="eyebrow">{eyebrow}</div>
              )}
              {subtitle && (
                <p className="mt-0.5 truncate text-[0.7rem] text-muted-foreground">
                  {subtitle}
                </p>
              )}
            </div>
          </div>
          {aside && (
            <div className="flex shrink-0 items-center gap-2 font-mono text-[0.65rem] uppercase tracking-wider text-muted-foreground">
              {aside}
            </div>
          )}
        </header>
      )}
      <div className={cn("relative z-10 flex-1 p-4", bodyClassName)}>
        {children}
      </div>
    </section>
  );
}
