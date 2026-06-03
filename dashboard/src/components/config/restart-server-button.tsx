import { useState } from "react";
import { RotateCw } from "lucide-react";
import { toast } from "sonner";
import { useRestart } from "@/lib/queries";
import { Button } from "@/components/ui/button";
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

/**
 * Server-control action gated behind an explicit confirm dialog.
 * Calls `POST /api/admin/restart` → the server re-execs itself in place
 * (reloads on-disk code, same stdio pipe; `SYNARA_*` env is preserved
 * unchanged across the re-exec). Disruptive: every
 * connected MCP client — including this dashboard — briefly drops while
 * the new process re-handshakes, so it sits behind a confirm modal.
 */
export function RestartServerButton() {
  const restart = useRestart();
  // Controlled so a failed restart keeps the dialog open (toast shows
  // the error); a successful one closes it and the status dock flips to
  // "off" until the server is back.
  const [open, setOpen] = useState(false);

  function confirm() {
    restart.mutate(undefined, {
      onSuccess: () => {
        toast.success("Server restarting — reconnecting in a moment.");
        setOpen(false);
      },
      onError: (e) =>
        toast.error(e instanceof Error ? e.message : "Restart failed"),
    });
  }

  return (
    <Dialog open={open} onOpenChange={setOpen}>
      <DialogTrigger asChild>
        <Button
          variant="outline"
          size="sm"
          className="text-warning hover:text-warning"
        >
          <RotateCw className="size-3.5" aria-hidden />
          Restart server
        </Button>
      </DialogTrigger>
      <DialogContent>
        <DialogHeader>
          <DialogTitle>Restart the server?</DialogTitle>
          <DialogDescription>
            The server re-executes in place, reloading its code from disk.
            Changes to <code>SYNARA_*</code> environment variables are not
            applied — that needs a full restart of the MCP client itself.
            Every connected MCP client — including this dashboard — briefly
            disconnects and reconnects, and any in-flight requests are dropped.
          </DialogDescription>
        </DialogHeader>
        <DialogFooter>
          <DialogClose asChild>
            <Button variant="ghost">Cancel</Button>
          </DialogClose>
          <Button
            variant="destructive"
            onClick={confirm}
            disabled={restart.isPending}
          >
            {restart.isPending ? "Restarting…" : "Restart"}
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}
