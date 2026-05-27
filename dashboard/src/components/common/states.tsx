import type { LucideIcon } from "lucide-react";
import { AlertCircle, Inbox, Loader2 } from "lucide-react";
import type { ReactNode } from "react";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import {
  Empty as UiEmpty,
  EmptyContent,
  EmptyDescription,
  EmptyHeader,
  EmptyMedia,
  EmptyTitle,
} from "@/components/ui/empty";

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

/** Framed, branded empty state — terminal phosphor glyph composed on
 *  shadcn's Empty primitives so spacing/structure stays canonical.
 *
 *  - `dense`: drop the dashed frame and shrink padding for use inside a
 *    Panel that already provides its own border (avoids double chrome). */
export function Empty({
  label,
  hint,
  icon: Icon = Inbox,
  dense = false,
  children,
}: {
  label: string;
  hint?: ReactNode;
  icon?: LucideIcon;
  dense?: boolean;
  children?: ReactNode;
}) {
  return (
    <UiEmpty
      className={
        dense
          ? "gap-3 rounded-none border-0 bg-transparent p-6 md:p-6"
          : "gap-4 rounded-none border border-dashed border-border bg-surface-canvas px-6 py-12 md:py-16"
      }
    >
      <EmptyHeader className="gap-3">
        <EmptyMedia
          variant="icon"
          className="size-10 rounded-none border border-primary/40 bg-primary/10 text-primary [&_svg:not([class*='size-'])]:size-5"
        >
          <Icon aria-hidden />
        </EmptyMedia>
        <EmptyTitle className="font-mono text-xs font-normal uppercase tracking-wider text-foreground">
          {label}
        </EmptyTitle>
        {hint && (
          <EmptyDescription className="text-xs leading-relaxed">
            {hint}
          </EmptyDescription>
        )}
      </EmptyHeader>
      {children && <EmptyContent>{children}</EmptyContent>}
    </UiEmpty>
  );
}
