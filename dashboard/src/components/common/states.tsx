import type { LucideIcon } from "lucide-react";
import { AlertCircle, Inbox, Loader2 } from "lucide-react";
import type { ReactNode } from "react";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";

export function Loading({ label = "Loading" }: { label?: string }) {
  return (
    <div
      role="status"
      aria-live="polite"
      className="flex items-center justify-center gap-2 py-12 text-sm text-muted-foreground sm:py-16"
    >
      <Loader2 className="size-4 animate-spin" aria-hidden />
      {label}…
    </div>
  );
}

export function ErrorState({ error }: { error: unknown }) {
  const msg = error instanceof Error ? error.message : String(error);
  const isAuth = /^API 401/.test(msg);
  return (
    <Alert variant="destructive" className="my-4 sm:my-6">
      <AlertCircle className="size-4" aria-hidden />
      <AlertTitle>{isAuth ? "Authentication required" : "Request failed"}</AlertTitle>
      <AlertDescription>
        {isAuth
          ? "This server requires a bearer token. Set it from the top bar."
          : msg}
      </AlertDescription>
    </Alert>
  );
}

/** Framed, branded empty state — never a lone line of text in a void. */
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
    <div className="flex flex-col items-center justify-center gap-4 rounded-xl border border-dashed border-border/70 bg-surface-canvas px-6 py-16 text-center shadow-card sm:py-20">
      <div className="grid size-12 place-items-center rounded-2xl bg-primary/10 text-primary ring-1 ring-inset ring-primary/20">
        <Icon className="size-5" aria-hidden />
      </div>
      <div className="space-y-1.5">
        <p className="text-sm font-medium text-foreground">{label}</p>
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
