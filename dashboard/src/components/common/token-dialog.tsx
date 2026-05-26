import { useEffect, useState } from "react";
import { Eye, EyeOff, KeyRound } from "lucide-react";
import { useQueryClient } from "@tanstack/react-query";
import { getToken, setToken } from "@/lib/api";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import {
  Dialog,
  DialogClose,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
  DialogTrigger,
} from "@/components/ui/dialog";

/** Token dialog. Always controllable from outside (so the command
 *  palette can open it) while still exposing its own icon trigger
 *  inside the dock. Pass `open`/`onOpenChange` to drive it externally;
 *  omit them for fully self-contained behaviour. */
export function TokenDialog({
  open: controlledOpen,
  onOpenChange: controlledOnOpenChange,
}: {
  open?: boolean;
  onOpenChange?: (next: boolean) => void;
} = {}) {
  const qc = useQueryClient();
  const [internalOpen, setInternalOpen] = useState(false);
  const isControlled = controlledOpen !== undefined;
  const openState = isControlled ? controlledOpen : internalOpen;

  const [value, setValue] = useState("");
  const [reveal, setReveal] = useState(false);
  const has = (getToken() ?? "") !== "";

  function onOpenChange(next: boolean) {
    if (next) {
      // Re-read on every open: another tab/component may have changed
      // the stored token since the dialog was last shown.
      setValue(getToken() ?? "");
      setReveal(false);
    }
    if (isControlled) controlledOnOpenChange?.(next);
    else setInternalOpen(next);
  }

  // Keep the form in sync when the *parent* opens us externally.
  // (Internal opens go through onOpenChange which sets these directly.)
  useEffect(() => {
    if (isControlled && controlledOpen) {
      // eslint-disable-next-line react-hooks/set-state-in-effect
      setValue(getToken() ?? "");
      setReveal(false);
    }
  }, [isControlled, controlledOpen]);

  function save() {
    setToken(value.trim() || null);
    void qc.invalidateQueries();
    onOpenChange(false);
  }

  return (
    <Dialog open={openState} onOpenChange={onOpenChange}>
      <DialogTrigger asChild>
        <Button
          variant="ghost"
          size="icon"
          className="relative size-9"
          aria-label={has ? "API token set — click to change" : "Set API bearer token"}
          title={has ? "Token set" : "Set token"}
        >
          <KeyRound className="size-[18px]" aria-hidden />
          {has && (
            <span
              className="absolute right-1.5 top-1.5 size-1.5 rounded-full bg-primary shadow-[0_0_6px_var(--primary)]"
              aria-hidden
            />
          )}
        </Button>
      </DialogTrigger>
      <DialogContent>
        <DialogHeader>
          <DialogTitle>API bearer token</DialogTitle>
          <DialogDescription>
            Only required when the server binds off-loopback or
            <code className="mx-1 rounded bg-muted px-1 py-0.5 text-xs">
              SYNARA_DASHBOARD_TOKEN
            </code>
            is set. Stored in this browser only.
          </DialogDescription>
        </DialogHeader>
        <div className="space-y-2">
          <Label htmlFor="token">Token</Label>
          <div className="relative">
            <Input
              id="token"
              type={reveal ? "text" : "password"}
              autoComplete="off"
              value={value}
              onChange={(e) => setValue(e.target.value)}
              placeholder="paste token…"
              className="pr-10"
            />
            <Button
              type="button"
              variant="ghost"
              size="icon"
              aria-label={reveal ? "Hide token" : "Show token"}
              aria-pressed={reveal}
              onClick={() => setReveal((v) => !v)}
              className="absolute right-1 top-1/2 size-7 -translate-y-1/2"
            >
              {reveal ? (
                <EyeOff className="size-4" aria-hidden />
              ) : (
                <Eye className="size-4" aria-hidden />
              )}
            </Button>
          </div>
        </div>
        <DialogFooter>
          <DialogClose asChild>
            <Button variant="ghost">Cancel</Button>
          </DialogClose>
          <Button onClick={save}>Save</Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}
