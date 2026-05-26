import {
  type ReactNode,
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState,
} from "react";
import { useNavigate } from "react-router-dom";
import {
  CornerDownLeft,
  KeyRound,
  Moon,
  Sun,
  Terminal,
} from "lucide-react";
import type { LucideIcon } from "lucide-react";
import { NAV_ITEMS } from "@/components/layout/sidebar";
import { useDocumentTheme } from "@/lib/use-document-theme";
import {
  Dialog,
  DialogContent,
  DialogTitle,
} from "@/components/ui/dialog";
import { cn } from "@/lib/utils";

type Action = {
  id: string;
  label: string;
  hint?: string;
  group: "Navigate" | "Actions";
  icon: LucideIcon;
  run: () => void;
  keywords?: string[];
};

const THEME_KEY = "synara.dashboard.theme";

function setTheme(next: "dark" | "light") {
  document.documentElement.classList.toggle("dark", next === "dark");
  document.documentElement.style.colorScheme = next;
  try {
    localStorage.setItem(THEME_KEY, next);
  } catch {
    /* storage unavailable — change applies for the session */
  }
}

/** Score a candidate against the query. Higher = better.
 *  - exact substring match in label: 100
 *  - prefix match in label: 80
 *  - all chars in order anywhere (fuzzy): 40 + contiguous bonus
 *  - keyword match: 20
 *  - 0 = filter it out
 */
function fuzzyScore(query: string, candidate: Action): number {
  const q = query.toLowerCase().trim();
  if (!q) return 1;
  const label = candidate.label.toLowerCase();
  const hint = (candidate.hint ?? "").toLowerCase();
  const keys = (candidate.keywords ?? []).join(" ").toLowerCase();

  if (label.startsWith(q)) return 100;
  if (label.includes(q)) return 80;
  if (hint.includes(q) || keys.includes(q)) return 40;

  // Fuzzy: every char of q appears in label in order.
  let i = 0;
  let lastIdx = -1;
  let contig = 0;
  for (const ch of q) {
    const idx = label.indexOf(ch, lastIdx + 1);
    if (idx < 0) return 0;
    if (idx === lastIdx + 1) contig++;
    lastIdx = idx;
    i++;
  }
  return 20 + contig * 2 + Math.max(0, 10 - i);
}

export function CommandPalette({
  open,
  onOpenChange,
  onOpenTokenDialog,
}: {
  open: boolean;
  onOpenChange: (next: boolean) => void;
  onOpenTokenDialog: () => void;
}) {
  const navigate = useNavigate();
  const theme = useDocumentTheme();
  const [query, setQuery] = useState("");
  const [active, setActive] = useState(0);
  const listRef = useRef<HTMLDivElement | null>(null);

  // Reset query/cursor on every open so the previous search doesn't
  // leak. Auto-focus is handled by Radix on the first focusable input.
  useEffect(() => {
    if (open) {
      // eslint-disable-next-line react-hooks/set-state-in-effect
      setQuery("");
      setActive(0);
    }
  }, [open]);

  const actions: Action[] = useMemo(
    () => [
      ...NAV_ITEMS.map<Action>((n) => ({
        id: `nav:${n.to}`,
        label: n.label,
        hint: n.to,
        group: "Navigate",
        icon: n.icon,
        run: () => {
          void navigate(n.to);
          onOpenChange(false);
        },
        keywords: [n.to.replace(/^\//, ""), "go", "open"],
      })),
      {
        id: "action:token",
        label: "Set API token",
        hint: "Bearer token for off-loopback servers",
        group: "Actions",
        icon: KeyRound,
        keywords: ["auth", "bearer", "secret"],
        run: () => {
          onOpenChange(false);
          // Defer so the palette close animation can run first.
          setTimeout(onOpenTokenDialog, 60);
        },
      },
      {
        id: "action:theme",
        label: theme === "dark" ? "Switch to light theme" : "Switch to dark theme",
        hint: theme === "dark" ? "current: dark" : "current: light",
        group: "Actions",
        icon: theme === "dark" ? Sun : Moon,
        keywords: ["theme", "dark", "light", "mode", "appearance"],
        run: () => {
          setTheme(theme === "dark" ? "light" : "dark");
          onOpenChange(false);
        },
      },
    ],
    [navigate, onOpenChange, onOpenTokenDialog, theme],
  );

  const filtered = useMemo(() => {
    const scored = actions
      .map((a) => ({ a, s: fuzzyScore(query, a) }))
      .filter((x) => x.s > 0)
      .sort((x, y) => y.s - x.s);
    return scored.map((x) => x.a);
  }, [actions, query]);

  const grouped = useMemo(() => {
    const out: Array<[Action["group"], Action[]]> = [];
    const order: Action["group"][] = ["Navigate", "Actions"];
    for (const g of order) {
      const items = filtered.filter((a) => a.group === g);
      if (items.length) out.push([g, items]);
    }
    return out;
  }, [filtered]);

  // Clamp active when the filter changes
  useEffect(() => {
    if (active >= filtered.length) {
      // eslint-disable-next-line react-hooks/set-state-in-effect
      setActive(Math.max(0, filtered.length - 1));
    }
  }, [filtered.length, active]);

  // Scroll the active item into view as the cursor moves
  useEffect(() => {
    if (!listRef.current) return;
    const node = listRef.current.querySelector<HTMLElement>(
      `[data-cmd-index="${active}"]`,
    );
    node?.scrollIntoView({ block: "nearest" });
  }, [active]);

  const onKeyDown = useCallback(
    (e: React.KeyboardEvent) => {
      if (e.key === "ArrowDown") {
        e.preventDefault();
        setActive((i) => Math.min(filtered.length - 1, i + 1));
      } else if (e.key === "ArrowUp") {
        e.preventDefault();
        setActive((i) => Math.max(0, i - 1));
      } else if (e.key === "Enter") {
        e.preventDefault();
        filtered[active]?.run();
      } else if (e.key === "Home") {
        e.preventDefault();
        setActive(0);
      } else if (e.key === "End") {
        e.preventDefault();
        setActive(Math.max(0, filtered.length - 1));
      }
    },
    [active, filtered],
  );

  // Build a flat index map so each row knows its global position for
  // keyboard cursor highlighting.
  let cursor = -1;

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent
        showCloseButton={false}
        onKeyDown={onKeyDown}
        className={cn(
          // Override the default centered modal: float near the top,
          // wider, square-edged, phosphor frame, mono surface.
          "top-[22%] left-1/2 grid w-[min(640px,calc(100%-2rem))] max-w-[640px] -translate-x-1/2 translate-y-0",
          "gap-0 overflow-hidden rounded-none border border-primary/40 bg-card p-0 font-mono shadow-glow",
          "data-[state=open]:animate-in data-[state=open]:fade-in-0 data-[state=open]:zoom-in-95",
          "data-[state=closed]:animate-out data-[state=closed]:fade-out-0 data-[state=closed]:zoom-out-95",
        )}
      >
        <DialogTitle className="sr-only">Command palette</DialogTitle>

        {/* Phosphor edge: a subtle scanline gradient on the top border */}
        <div
          aria-hidden
          className="pointer-events-none absolute inset-x-0 top-0 h-px bg-gradient-to-r from-transparent via-primary to-transparent"
        />

        {/* Query input — prompt prefix + blinking cursor when empty */}
        <div className="flex items-center gap-2 border-b border-border bg-surface-canvas px-4 py-3">
          <Terminal className="size-4 text-primary" aria-hidden />
          <span className="text-primary">›</span>
          <input
            type="text"
            autoFocus
            value={query}
            onChange={(e) => {
              setQuery(e.target.value);
              setActive(0);
            }}
            placeholder="type a command or page…"
            aria-label="Command palette query"
            className="flex-1 bg-transparent text-sm text-foreground outline-none placeholder:text-muted-foreground/70"
          />
          {!query && (
            <span
              className="inline-block h-4 w-1.5 animate-pulse bg-primary/70"
              aria-hidden
            />
          )}
          <kbd className="hidden border border-border bg-muted/40 px-1.5 py-0.5 text-[0.6rem] uppercase tracking-wider text-muted-foreground sm:inline-block">
            esc
          </kbd>
        </div>

        {/* Results list */}
        <div
          ref={listRef}
          className="max-h-[min(440px,60vh)] overflow-y-auto"
          role="listbox"
          aria-label="Commands"
        >
          {grouped.length === 0 ? (
            <div className="px-4 py-10 text-center text-xs text-muted-foreground">
              <span className="text-primary">›</span> no match for{" "}
              <span className="text-foreground">"{query}"</span>
            </div>
          ) : (
            grouped.map(([group, items]) => (
              <div key={group}>
                <div className="sticky top-0 z-10 border-b border-border bg-card px-4 py-1.5 text-[0.6rem] uppercase tracking-[0.22em] text-muted-foreground">
                  {group}
                </div>
                <ul role="presentation">
                  {items.map((a) => {
                    cursor++;
                    const idx = cursor;
                    const isActive = idx === active;
                    return (
                      <li key={a.id}>
                        <button
                          type="button"
                          role="option"
                          aria-selected={isActive}
                          data-cmd-index={idx}
                          onMouseMove={() => setActive(idx)}
                          onClick={() => a.run()}
                          className={cn(
                            "group/row relative flex w-full items-center gap-3 px-4 py-2.5 text-left transition-colors",
                            isActive
                              ? "bg-primary/10 text-foreground"
                              : "text-foreground/80 hover:bg-muted/30",
                          )}
                        >
                          {/* Cursor mark */}
                          <span
                            aria-hidden
                            className={cn(
                              "absolute left-0 top-1/2 h-5 w-0.5 -translate-y-1/2 transition-colors",
                              isActive
                                ? "bg-primary shadow-[0_0_8px_var(--primary)]"
                                : "bg-transparent",
                            )}
                          />
                          <a.icon
                            className={cn(
                              "size-4 shrink-0 transition-colors",
                              isActive
                                ? "text-primary drop-shadow-[0_0_5px_var(--primary)]"
                                : "text-muted-foreground group-hover/row:text-foreground",
                            )}
                            aria-hidden
                          />
                          <div className="min-w-0 flex-1">
                            <div className="truncate text-[0.78rem] uppercase tracking-wider">
                              {a.label}
                            </div>
                            {a.hint && (
                              <div className="truncate text-[0.65rem] text-muted-foreground">
                                {a.hint}
                              </div>
                            )}
                          </div>
                          {isActive && (
                            <span className="flex items-center gap-1 text-[0.6rem] uppercase tracking-wider text-primary">
                              run
                              <CornerDownLeft className="size-3" aria-hidden />
                            </span>
                          )}
                        </button>
                      </li>
                    );
                  })}
                </ul>
              </div>
            ))
          )}
        </div>

        {/* Footer hint strip */}
        <FooterHints />
      </DialogContent>
    </Dialog>
  );
}

function FooterHints() {
  return (
    <div className="flex items-center justify-between gap-3 border-t border-border bg-surface-canvas px-4 py-2 text-[0.6rem] uppercase tracking-wider text-muted-foreground">
      <div className="flex items-center gap-3">
        <Hint k="↑↓">navigate</Hint>
        <Hint k="↵">select</Hint>
        <Hint k="esc">close</Hint>
      </div>
      <span className="text-primary/80">synara · command palette</span>
    </div>
  );
}

function Hint({ k, children }: { k: string; children: ReactNode }) {
  return (
    <span className="flex items-center gap-1.5">
      <kbd className="border border-border bg-muted/40 px-1 py-0.5 text-[0.55rem] text-foreground/80">
        {k}
      </kbd>
      <span>{children}</span>
    </span>
  );
}

/** Compact ⌘K trigger chip — sits in the dock and opens the palette. */
export function CommandPaletteTrigger({ onOpen }: { onOpen: () => void }) {
  const [isMac, setIsMac] = useState(false);
  useEffect(() => {
    // eslint-disable-next-line react-hooks/set-state-in-effect
    setIsMac(/Mac|iPhone|iPad/.test(navigator.platform));
  }, []);
  const mod = isMac ? "⌘" : "Ctrl";
  return (
    <button
      type="button"
      onClick={onOpen}
      aria-label="Open command palette"
      title={`Command palette (${mod}K)`}
      className={cn(
        "group hidden h-8 items-center gap-2 border border-border bg-card/60 px-2 font-mono text-[0.65rem] uppercase tracking-wider text-muted-foreground transition-all hover:border-primary/40 hover:text-foreground sm:flex",
      )}
    >
      <Terminal className="size-3.5 text-primary/80 group-hover:text-primary group-hover:drop-shadow-[0_0_5px_var(--primary)]" />
      <span>command</span>
      <kbd className="border border-border bg-muted/30 px-1 py-0.5 text-[0.55rem] text-foreground/80">
        {mod}K
      </kbd>
    </button>
  );
}
