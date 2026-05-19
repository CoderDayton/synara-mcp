import { AlertCircle, Inbox, Loader2 } from "lucide-react";
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

export function Empty({ label }: { label: string }) {
  return (
    <div className="flex flex-col items-center justify-center gap-2 py-12 text-center text-sm text-muted-foreground sm:py-16">
      <Inbox className="size-6 opacity-60" aria-hidden />
      {label}
    </div>
  );
}
