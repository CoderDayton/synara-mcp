import { NavLink } from "react-router-dom";
import {
  Activity as ActivityIcon,
  Brain,
  LayoutDashboard,
  Network,
  Settings2,
  Wrench,
} from "lucide-react";
import { useHealth } from "@/lib/queries";
import { cn } from "@/lib/utils";

const NAV = [
  {
    group: "Main",
    items: [
      { to: "/", label: "Overview", icon: LayoutDashboard, end: true },
      { to: "/memories", label: "Memories", icon: Brain, end: false },
      { to: "/graph", label: "Graph", icon: Network, end: false },
    ],
  },
  {
    group: "Operations",
    items: [{ to: "/admin", label: "Admin", icon: Wrench, end: false }],
  },
  {
    group: "System",
    items: [{ to: "/config", label: "Config", icon: Settings2, end: false }],
  },
];

function StatusFooter() {
  const { data, isError } = useHealth();
  const ok = !!data && !isError;
  return (
    <div className="rounded-xl border border-sidebar-border/70 bg-card/50 p-3 text-xs shadow-card sm:p-4">
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

export function SidebarContent({ onNavigate }: { onNavigate?: () => void }) {
  const { data } = useHealth();
  return (
    <div className="flex h-full flex-col gap-6 p-4 sm:p-5">
      <div className="flex items-center gap-3">
        <div className="grid size-9 place-items-center rounded-xl bg-gradient-to-br from-primary to-primary/65 text-white shadow-glow ring-1 ring-inset ring-white/15">
          <ActivityIcon className="size-5" aria-hidden />
        </div>
        <div className="leading-tight">
          <div className="text-[0.95rem] font-semibold tracking-tight">
            Synara
          </div>
          <div className="text-xs text-muted-foreground">
            {data ? `v${data.version}` : "memory console"}
          </div>
        </div>
      </div>

      <nav className="flex-1 space-y-5" aria-label="Primary">
        {NAV.map((section) => (
          <div key={section.group}>
            <div className="px-3 pb-2 text-[0.65rem] font-semibold uppercase tracking-[0.14em] text-muted-foreground/70">
              {section.group}
            </div>
            <ul className="space-y-1">
              {section.items.map((item) => (
                <li key={item.to}>
                  <NavLink
                    to={item.to}
                    end={item.end}
                    onClick={onNavigate}
                    className={({ isActive }) =>
                      cn(
                        "group relative flex items-center gap-3 rounded-lg px-3 py-2 text-sm font-medium transition-colors",
                        "focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-sidebar-ring",
                        isActive
                          ? "bg-sidebar-accent text-sidebar-accent-foreground before:absolute before:inset-y-1.5 before:left-0 before:w-1 before:rounded-r-full before:bg-primary"
                          : "text-sidebar-foreground/70 hover:bg-sidebar-accent/50 hover:text-sidebar-foreground",
                      )
                    }
                  >
                    {({ isActive }) => (
                      <>
                        <item.icon
                          className={cn(
                            "size-4 shrink-0 transition-colors",
                            isActive
                              ? "text-primary"
                              : "text-muted-foreground group-hover:text-sidebar-foreground",
                          )}
                          aria-hidden
                        />
                        {item.label}
                      </>
                    )}
                  </NavLink>
                </li>
              ))}
            </ul>
          </div>
        ))}
      </nav>

      <StatusFooter />
    </div>
  );
}
