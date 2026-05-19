import { Suspense, useState } from "react";
import { Outlet } from "react-router-dom";
import { Menu, X } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Toaster } from "@/components/ui/sonner";
import { Loading } from "@/components/common/states";
import { ThemeToggle } from "@/components/common/theme-toggle";
import { TokenDialog } from "@/components/common/token-dialog";
import { SidebarContent } from "@/components/layout/sidebar";
import { useHealth } from "@/lib/queries";
import { cn } from "@/lib/utils";

function ConnectionPill() {
  const { data, isError } = useHealth();
  const ok = !!data && !isError;
  return (
    <div className="hidden items-center gap-2 rounded-full border border-border/60 bg-card/60 px-3 py-1.5 text-xs shadow-card sm:flex">
      <span
        className={cn(
          "size-1.5 rounded-full",
          ok ? "animate-pulse bg-success" : "bg-destructive",
        )}
        aria-hidden
      />
      <span className="font-medium">{ok ? "Connected" : "Offline"}</span>
      {data && (
        <span className="font-mono text-muted-foreground">
          {data.transport} · v{data.version} ·{" "}
          {Math.round(data.uptime_seconds)}s
        </span>
      )}
    </div>
  );
}

export function AppShell() {
  const [open, setOpen] = useState(false);

  return (
    <div className="min-h-svh lg:grid lg:grid-cols-[16rem_1fr]">
      {/* Desktop sidebar */}
      <aside className="sticky top-0 hidden h-svh border-r border-sidebar-border/70 bg-sidebar lg:block">
        <SidebarContent />
      </aside>

      {/* Mobile off-canvas */}
      {open && (
        <div className="fixed inset-0 z-50 lg:hidden">
          <button
            type="button"
            aria-label="Close menu"
            className="absolute inset-0 bg-black/60"
            onClick={() => setOpen(false)}
          />
          <div className="absolute inset-y-0 left-0 w-64 max-w-[80vw] border-r border-sidebar-border/70 bg-sidebar shadow-pop">
            <SidebarContent onNavigate={() => setOpen(false)} />
          </div>
        </div>
      )}

      <div className="flex min-w-0 flex-col">
        <header className="sticky top-0 z-30 flex h-14 items-center gap-3 border-b border-border/60 bg-background/70 px-4 backdrop-blur-xl sm:h-16 sm:px-6">
          <Button
            variant="ghost"
            size="icon"
            className="lg:hidden"
            aria-label={open ? "Close menu" : "Open menu"}
            aria-expanded={open}
            onClick={() => setOpen((v) => !v)}
          >
            {open ? <X className="size-5" /> : <Menu className="size-5" />}
          </Button>
          <ConnectionPill />
          <div className="flex-1" />
          <TokenDialog />
          <ThemeToggle />
        </header>

        <main
          className={cn(
            "mx-auto w-full max-w-7xl flex-1 px-4 py-5 sm:px-6 sm:py-8 lg:px-8",
          )}
        >
          <Suspense fallback={<Loading label="Loading view" />}>
            <Outlet />
          </Suspense>
        </main>
      </div>

      <Toaster theme="dark" position="bottom-right" richColors />
    </div>
  );
}
