import { Trash2 } from "lucide-react";
import { toast } from "sonner";
import { useDeleteMemory } from "@/lib/queries";
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
 * Destructive action gated behind an explicit confirm dialog.
 * Delegates to `DELETE /api/memories/{id}` → `MemoryService.delete_episode`
 * (FK-safe, evicts SR + removes the whole theta-segment group).
 */
export function DeleteMemoryButton({
  id,
  onDeleted,
}: {
  id: number;
  onDeleted?: () => void;
}) {
  const del = useDeleteMemory();

  function confirm() {
    del.mutate(id, {
      onSuccess: (r) => {
        toast.success(
          `Deleted ${r.count} record${r.count === 1 ? "" : "s"} (group of #${id}).`,
        );
        onDeleted?.();
      },
      onError: (e) =>
        toast.error(e instanceof Error ? e.message : "Delete failed"),
    });
  }

  return (
    <Dialog>
      <DialogTrigger asChild>
        <Button
          variant="ghost"
          size="icon"
          aria-label={`Delete episode ${id}`}
          className="text-muted-foreground hover:text-destructive"
        >
          <Trash2 className="size-4" aria-hidden />
        </Button>
      </DialogTrigger>
      <DialogContent>
        <DialogHeader>
          <DialogTitle>Delete episode #{id}?</DialogTitle>
          <DialogDescription>
            This removes the episode and its entire theta-segment group,
            and evicts it from the Successor Representation. This cannot be
            undone.
          </DialogDescription>
        </DialogHeader>
        <DialogFooter>
          <DialogClose asChild>
            <Button variant="ghost">Cancel</Button>
          </DialogClose>
          <DialogClose asChild>
            <Button
              variant="destructive"
              onClick={confirm}
              disabled={del.isPending}
            >
              {del.isPending ? "Deleting…" : "Delete"}
            </Button>
          </DialogClose>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}
