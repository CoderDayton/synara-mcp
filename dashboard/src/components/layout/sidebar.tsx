import { NavLink } from "react-router-dom";
import { Brain, LayoutDashboard, Settings2, Wrench } from "lucide-react";
import type { LucideIcon } from "lucide-react";
import { cn } from "@/lib/utils";

type Item = { to: string; label: string; icon: LucideIcon; end: boolean };

export const NAV_ITEMS: Item[] = [
  { to: "/", label: "Overview", icon: LayoutDashboard, end: true },
  { to: "/memories", label: "Memories", icon: Brain, end: false },
  { to: "/admin", label: "Admin", icon: Wrench, end: false },
  { to: "/config", label: "Config", icon: Settings2, end: false },
];

/** Path → label. Kept exported so the dock breadcrumb can resolve the
 *  current route to a human name without duplicating the source. */
export const ROUTE_TITLES: Record<string, string> = Object.fromEntries(
  NAV_ITEMS.map((i) => [i.to, i.label] as const),
);

/** Inline top-nav: 4 mono labels with an active phosphor underline,
 *  hover glow, and a filled glyph when current. Lives inside the dock. */
export function TopNav() {
  return (
    <nav
      aria-label="Primary"
      className="hidden items-center gap-0.5 md:flex"
    >
      {NAV_ITEMS.map((item) => (
        <NavLink
          key={item.to}
          to={item.to}
          end={item.end}
          className={({ isActive }) =>
            cn(
              "group relative flex h-9 items-center gap-2 rounded-sm px-2.5 font-mono text-[0.72rem] uppercase tracking-wider transition-colors",
              "focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-ring",
              isActive
                ? "text-primary"
                : "text-muted-foreground hover:text-foreground",
            )
          }
        >
          {({ isActive }) => (
            <>
              <item.icon
                className={cn(
                  "size-[14px] transition-all",
                  isActive
                    ? "text-primary drop-shadow-[0_0_4px_var(--primary)]"
                    : "text-muted-foreground/70 group-hover:text-foreground",
                )}
                aria-hidden
              />
              <span>{item.label.toLowerCase()}</span>
              {/* Underline — animates in from center on hover, fully lit on active */}
              <span
                aria-hidden
                className={cn(
                  "pointer-events-none absolute inset-x-2 -bottom-px h-px origin-center bg-primary transition-transform duration-200 ease-out",
                  isActive
                    ? "scale-x-100 shadow-[0_0_6px_var(--primary),0_0_12px_var(--primary)]"
                    : "scale-x-0 group-hover:scale-x-75",
                )}
              />
            </>
          )}
        </NavLink>
      ))}
    </nav>
  );
}

/** Mobile drawer — labelled list, same NAV source. Used by the
 *  hamburger toggle under the md breakpoint. */
export function MobileNav({ onNavigate }: { onNavigate?: () => void }) {
  return (
    <div className="flex h-full flex-col gap-5 p-4">
      <div className="flex items-center gap-3">
        <img
          src="/synara-logo.svg"
          alt=""
          aria-hidden
          className="size-9"
        />
        <div className="leading-tight">
          <div className="font-mono text-sm font-medium tracking-wide">
            synara
          </div>
          <div className="font-mono text-[0.65rem] uppercase tracking-wider text-muted-foreground">
            memory console
          </div>
        </div>
      </div>

      <nav className="flex-1 space-y-1" aria-label="Primary">
        {NAV_ITEMS.map((item) => (
          <NavLink
            key={item.to}
            to={item.to}
            end={item.end}
            onClick={onNavigate}
            className={({ isActive }) =>
              cn(
                "flex items-center gap-3 rounded px-3 py-2 text-sm transition-colors",
                "focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-sidebar-ring",
                isActive
                  ? "bg-sidebar-accent text-sidebar-accent-foreground"
                  : "text-sidebar-foreground/70 hover:bg-sidebar-accent/40 hover:text-sidebar-foreground",
              )
            }
          >
            <item.icon className="size-4 shrink-0" aria-hidden />
            <span className="font-mono text-xs uppercase tracking-wider">
              {item.label}
            </span>
          </NavLink>
        ))}
      </nav>
    </div>
  );
}
