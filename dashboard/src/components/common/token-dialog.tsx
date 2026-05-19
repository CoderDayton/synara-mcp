import { useState } from "react";
import { KeyRound } from "lucide-react";
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
  const [value, setValue] = useState(() => getToken() ?? "");
  const has = (getToken() ?? "") !== "";

  function save() {
    setToken(value.trim() || null);
    void qc.invalidateQueries();
  }

  return (
    <Dialog>
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
          <Input
            id="token"
            type="password"
            autoComplete="off"
            value={value}
            onChange={(e) => setValue(e.target.value)}
            placeholder="paste token…"
          />
        </div>
        <DialogFooter>
          <DialogClose asChild>
            <Button variant="ghost">Cancel</Button>
          </DialogClose>
          <DialogClose asChild>
            <Button onClick={save}>Save</Button>
          </DialogClose>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}
