import { Suspense, useEffect, useState } from "react";
import { Link, Outlet } from "react-router-dom";
import { Activity as ActivityIcon, Menu, X } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Toaster } from "@/components/ui/sonner";
import { Loading } from "@/components/common/states";
import { StatusReadout } from "@/components/common/status-indicator";
import { ThemeToggle } from "@/components/common/theme-toggle";
import { TokenDialog } from "@/components/common/token-dialog";
import { useDocumentTheme } from "@/lib/use-document-theme";
import { MobileNav, TopNav } from "@/components/layout/sidebar";
import {
  CommandPalette,
  CommandPaletteTrigger,
} from "@/components/layout/command-palette";
import { cn } from "@/lib/utils";

/** Brand wordmark — phosphor activity glyph in a hairline cell + the
 *  mono "synara" wordmark. Always links to /. */
function Brand() {
  return (
    <Link
      to="/"
      aria-label="Synara — go to overview"
      className="group flex items-center gap-2 focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-ring"
    >
      <span className="relative grid size-7 place-items-center border border-primary/50 bg-primary/10 text-primary transition-all group-hover:border-primary group-hover:shadow-[0_0_12px_var(--primary)]">
        <ActivityIcon className="size-4" aria-hidden />
        <span
          aria-hidden
          className="absolute -inset-px border border-primary/0 transition-all group-hover:-inset-1 group-hover:border-primary/30"
        />
      </span>
      <span className="hidden font-mono text-sm font-medium uppercase tracking-[0.18em] text-foreground group-hover:text-primary sm:inline-block">
        synara
      </span>
    </Link>
  );
}

/** Top dock — brand · primary nav · status · ⌘K · token · theme. */
function Dock({
  onOpenMenu,
  menuOpen,
  onOpenPalette,
  tokenOpen,
  onTokenOpenChange,
}: {
  onOpenMenu: () => void;
  menuOpen: boolean;
  onOpenPalette: () => void;
  tokenOpen: boolean;
  onTokenOpenChange: (next: boolean) => void;
}) {
  return (
    <header
      className={cn(
        "sticky top-0 z-30 flex h-14 items-center gap-3 border-b border-border bg-surface-overlay px-3 backdrop-blur-md sm:px-5",
        // Bottom-edge phosphor wash that bleeds into the canvas — the
        // dock reads as "lit" without a heavy shadow.
        "after:pointer-events-none after:absolute after:inset-x-0 after:-bottom-px after:h-px after:bg-gradient-to-r after:from-transparent after:via-primary/30 after:to-transparent",
      )}
    >
      <Button
        variant="ghost"
        size="icon"
        className="md:hidden"
        aria-label={menuOpen ? "Close menu" : "Open menu"}
        aria-expanded={menuOpen}
        onClick={onOpenMenu}
      >
        {menuOpen ? <X className="size-4" /> : <Menu className="size-4" />}
      </Button>

      <Brand />

      <span
        className="hidden h-5 w-px bg-border md:inline-block"
        aria-hidden
      />

      <TopNav />

      <div className="flex-1" />

      <StatusReadout className="hidden lg:flex" />

      <span
        className="hidden h-5 w-px bg-border lg:inline-block"
        aria-hidden
      />

      <CommandPaletteTrigger onOpen={onOpenPalette} />

      <TokenDialog open={tokenOpen} onOpenChange={onTokenOpenChange} />

      <ThemeToggle />
    </header>
  );
}

export function AppShell() {
  const [open, setOpen] = useState(false);
  const [paletteOpen, setPaletteOpen] = useState(false);
  const [tokenOpen, setTokenOpen] = useState(false);
  const theme = useDocumentTheme();

  // ⌘K / Ctrl+K opens the palette anywhere. Don't intercept in any
  // other modifier combo so browser shortcuts (Ctrl+Shift+K, etc.)
  // remain available.
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      const isCmdK =
        (e.key === "k" || e.key === "K") &&
        (e.metaKey || e.ctrlKey) &&
        !e.shiftKey &&
        !e.altKey;
      if (isCmdK) {
        e.preventDefault();
        setPaletteOpen((v) => !v);
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, []);

  // Drawer side-effects: lock body scroll and close on Esc.
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
    <div className="min-h-svh">
      {/* Mobile drawer */}
      <div
        className={cn(
          "fixed inset-0 z-50 md:hidden",
          open ? "pointer-events-auto" : "pointer-events-none",
        )}
        aria-hidden={!open}
      >
        <button
          type="button"
          aria-label="Close menu"
          tabIndex={open ? 0 : -1}
          className={cn(
            "absolute inset-0 bg-black/70 transition-opacity duration-200",
            open ? "opacity-100" : "opacity-0",
          )}
          onClick={() => setOpen(false)}
        />
        <div
          className={cn(
            "absolute inset-y-0 left-0 w-64 max-w-[80vw] border-r border-sidebar-border bg-sidebar shadow-pop transition-transform duration-200 ease-out",
            open ? "translate-x-0" : "-translate-x-full",
          )}
        >
          <MobileNav onNavigate={() => setOpen(false)} />
        </div>
      </div>

      <div className="grid min-h-svh min-w-0 grid-rows-[auto_1fr]">
        <Dock
          menuOpen={open}
          onOpenMenu={() => setOpen((v) => !v)}
          onOpenPalette={() => setPaletteOpen(true)}
          tokenOpen={tokenOpen}
          onTokenOpenChange={setTokenOpen}
        />

        <main className="flex min-h-0 min-w-0 flex-col">
          <div
            className={cn(
              "mx-auto flex min-h-0 w-full max-w-[1480px] flex-1 flex-col",
              "px-3 py-4 sm:px-5 sm:py-6 lg:px-8 lg:py-7",
            )}
          >
            <Suspense fallback={<Loading label="Loading view" />}>
              <Outlet />
            </Suspense>
          </div>
        </main>
      </div>

      <CommandPalette
        open={paletteOpen}
        onOpenChange={setPaletteOpen}
        onOpenTokenDialog={() => setTokenOpen(true)}
      />

      <Toaster theme={theme} position="bottom-right" richColors />
    </div>
  );
}
