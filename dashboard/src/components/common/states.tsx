import type { LucideIcon } from "lucide-react";
import { AlertCircle, Inbox, Loader2 } from "lucide-react";
import type { ReactNode } from "react";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";

export function Loading({ label = "loading" }: { label?: string }) {
  return (
    <div
      role="status"
      aria-live="polite"
      className="flex items-center justify-center gap-2 py-10 font-mono text-[0.7rem] uppercase tracking-wider text-muted-foreground"
    >
      <Loader2 className="size-3.5 animate-spin text-primary" aria-hidden />
      <span className="text-primary">›</span>
      <span>{label}…</span>
    </div>
  );
}

export function ErrorState({ error }: { error: unknown }) {
  const msg = error instanceof Error ? error.message : String(error);
  const isAuth = /^API 401/.test(msg);
  return (
    <Alert variant="destructive" className="my-3 font-mono text-xs">
      <AlertCircle className="size-4" aria-hidden />
      <AlertTitle className="uppercase tracking-wider">
        {isAuth ? "auth required" : "request failed"}
      </AlertTitle>
      <AlertDescription>
        {isAuth
          ? "This server requires a bearer token. Set it from the rail."
          : msg}
      </AlertDescription>
    </Alert>
  );
}

/** Framed, branded empty state — terminal frame with phosphor glyph. */
export function Empty({
  label,
  hint,
  icon: Icon = Inbox,
  children,
}: {
  label: string;
  hint?: string;
  icon?: LucideIcon;
  children?: ReactNode;
}) {
  return (
    <div className="flex flex-col items-center justify-center gap-4 border border-dashed border-border bg-surface-canvas px-6 py-12 text-center sm:py-16">
      <div className="grid size-10 place-items-center border border-primary/40 bg-primary/10 text-primary">
        <Icon className="size-5" aria-hidden />
      </div>
      <div className="space-y-1.5">
        <p className="font-mono text-xs uppercase tracking-wider text-foreground">
          {label}
        </p>
        {hint && (
          <p className="mx-auto max-w-sm text-xs leading-relaxed text-muted-foreground">
            {hint}
          </p>
        )}
      </div>
      {children}
    </div>
  );
}
