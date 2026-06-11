import { Trash2 } from "lucide-react";
import { toast } from "sonner";
import { useDeleteMemory, useDeleteSemantic } from "@/lib/queries";
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

type DeleteKind = "episodic" | "semantic";

/** Per-kind copy + delete semantics. Episodic deletes route through
 * `DELETE /api/memories/{id}` → `MemoryService.delete_episode` (FK-safe,
 * evicts SR + the whole theta-segment group); semantic through
 * `DELETE /api/semantic/{id}` → `MemoryService.delete_semantic` (cold-schema
 * GC path; source episodes are kept). */
const COPY: Record<
  DeleteKind,
  {
    aria: (id: number) => string;
    title: (id: number) => string;
    description: string;
    success: (count: number, id: number) => string;
  }
> = {
  episodic: {
    aria: (id) => `Delete episode ${id}`,
    title: (id) => `Delete episode #${id}?`,
    description:
      "This removes the episode and its entire theta-segment group, and " +
      "evicts it from the Successor Representation. This cannot be undone.",
    success: (count, id) =>
      `Deleted ${count} record${count === 1 ? "" : "s"} (group of #${id}).`,
  },
  semantic: {
    aria: (id) => `Delete semantic memory ${id}`,
    title: (id) => `Delete semantic memory #${id}?`,
    description:
      "This removes the semantic memory — a consolidated schema or a " +
      "user-asserted entry — from the store. Source episodes are kept. " +
      "This cannot be undone.",
    success: (_count, id) => `Deleted semantic memory #${id}.`,
  },
};

/**
 * Destructive action gated behind an explicit confirm dialog. Handles both
 * episodic and semantic deletes; defaults to episodic.
 */
export function DeleteMemoryButton({
  id,
  kind = "episodic",
  onDeleted,
}: {
  id: number;
  kind?: DeleteKind;
  onDeleted?: () => void;
}) {
  const delEpisode = useDeleteMemory();
  const delSemantic = useDeleteSemantic();
  const del = kind === "semantic" ? delSemantic : delEpisode;
  const copy = COPY[kind];

  function confirm() {
    del.mutate(id, {
      onSuccess: (r) => {
        toast.success(copy.success(r.count, id));
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
          size="icon-sm"
          aria-label={copy.aria(id)}
          className="text-muted-foreground hover:text-destructive"
        >
          <Trash2 className="size-4" aria-hidden />
        </Button>
      </DialogTrigger>
      <DialogContent>
        <DialogHeader>
          <DialogTitle>{copy.title(id)}</DialogTitle>
          <DialogDescription>{copy.description}</DialogDescription>
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
