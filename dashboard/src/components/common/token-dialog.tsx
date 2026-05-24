import { useState } from "react";
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

export function TokenDialog() {
  const qc = useQueryClient();
  // Dialog is controlled so we can reset `value` from storage every
  // time it opens — otherwise an unsaved draft persists across
  // open/close cycles and silently overrides a token saved elsewhere.
  const [openState, setOpenState] = useState(false);
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
    setOpenState(next);
  }

  function save() {
    setToken(value.trim() || null);
    void qc.invalidateQueries();
    setOpenState(false);
  }

  return (
    <Dialog open={openState} onOpenChange={onOpenChange}>
      <DialogTrigger asChild>
        <Button
          variant={has ? "ghost" : "outline"}
          size="sm"
          className="gap-2"
          aria-label="Set API bearer token"
        >
          <KeyRound className="size-4" aria-hidden />
          <span className="hidden sm:inline">{has ? "Token set" : "Set token"}</span>
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
