import { Suspense, useEffect, useState } from "react";
import { Outlet, useLocation } from "react-router-dom";
import { Menu, X } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Toaster } from "@/components/ui/sonner";
import { Loading } from "@/components/common/states";
import { StatusPill } from "@/components/common/status-indicator";
import { ThemeToggle } from "@/components/common/theme-toggle";
import { TokenDialog } from "@/components/common/token-dialog";
import { ROUTE_TITLES, SidebarContent } from "@/components/layout/sidebar";
import { cn } from "@/lib/utils";

export function AppShell() {
  const [open, setOpen] = useState(false);
  const { pathname } = useLocation();
  const pageTitle = ROUTE_TITLES[pathname] ?? "";

  // Drawer side-effects: lock body scroll while open and close on Esc,
  // so the off-canvas menu behaves like a real modal layer.
  useEffect(() => {
    if (!open) return;
    const prev = document.body.style.overflow;
    document.body.style.overflow = "hidden";
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") setOpen(false);
    };
    window.addEventListener("keydown", onKey);
    return () => {
      document.body.style.overflow = prev;
      window.removeEventListener("keydown", onKey);
    };
  }, [open]);

  return (
    <div className="min-h-svh lg:grid lg:grid-cols-[16rem_1fr]">
      {/* Desktop sidebar */}
      <aside className="sticky top-0 hidden h-svh border-r border-sidebar-border/70 bg-sidebar lg:block">
        <SidebarContent />
      </aside>

      {/* Mobile off-canvas — always mounted so it can transition */}
      <div
        className={cn(
          "fixed inset-0 z-50 lg:hidden",
          open ? "pointer-events-auto" : "pointer-events-none",
        )}
        aria-hidden={!open}
      >
        <button
          type="button"
          aria-label="Close menu"
          tabIndex={open ? 0 : -1}
          className={cn(
            "absolute inset-0 bg-black/60 transition-opacity duration-200",
            open ? "opacity-100" : "opacity-0",
          )}
          onClick={() => setOpen(false)}
        />
        <div
          className={cn(
            "absolute inset-y-0 left-0 w-64 max-w-[80vw] border-r border-sidebar-border/70 bg-sidebar shadow-pop transition-transform duration-200 ease-out",
            open ? "translate-x-0" : "-translate-x-full",
          )}
        >
          <SidebarContent onNavigate={() => setOpen(false)} />
        </div>
      </div>

      <div className="grid min-h-svh min-w-0 grid-rows-[auto_1fr]">
        <header className="sticky top-0 z-30 flex h-14 items-center gap-3 border-b border-border/60 bg-surface-overlay px-4 backdrop-blur-xl sm:h-16 sm:px-6">
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
          {pageTitle && (
            <h1 className="truncate text-sm font-semibold tracking-tight sm:hidden">
              {pageTitle}
            </h1>
          )}
          <StatusPill />
          <div className="flex-1" />
          <TokenDialog />
          <ThemeToggle />
        </header>

        <main
          className={cn(
            "mx-auto flex w-full min-h-0 max-w-7xl flex-col px-4 py-5 sm:px-6 sm:py-8 lg:px-8",
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
