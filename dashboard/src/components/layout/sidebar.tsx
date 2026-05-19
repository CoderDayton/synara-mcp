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
    <div className="rounded-lg border border-sidebar-border bg-sidebar-accent/40 p-3 text-xs sm:p-4">
      <div className="flex items-center gap-2 font-medium">
        <span
          className={cn(
            "size-2 rounded-full",
            ok ? "bg-success" : "bg-destructive",
          )}
          aria-hidden
        />
        {ok ? "Connected" : "Offline"}
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
      <div className="flex items-center gap-2.5">
        <div className="grid size-9 place-items-center rounded-lg bg-primary text-primary-foreground">
          <ActivityIcon className="size-5" aria-hidden />
        </div>
        <div className="leading-tight">
          <div className="font-semibold tracking-tight">Synara</div>
          <div className="text-xs text-muted-foreground">
            {data ? `v${data.version}` : "memory console"}
          </div>
        </div>
      </div>

      <nav className="flex-1 space-y-5" aria-label="Primary">
        {NAV.map((section) => (
          <div key={section.group}>
            <div className="px-2 pb-2 text-[0.7rem] font-semibold uppercase tracking-wider text-muted-foreground">
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
                        "flex items-center gap-3 rounded-md px-3 py-2 text-sm font-medium transition-colors",
                        "hover:bg-sidebar-accent hover:text-sidebar-accent-foreground",
                        "focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-sidebar-ring",
                        isActive
                          ? "bg-sidebar-primary text-sidebar-primary-foreground"
                          : "text-sidebar-foreground/80",
                      )
                    }
                  >
                    <item.icon className="size-4 shrink-0" aria-hidden />
                    {item.label}
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
