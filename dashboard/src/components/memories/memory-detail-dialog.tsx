import { useMemoryDetail } from "@/lib/queries";
import { ErrorState, Loading } from "@/components/common/states";
import { Badge } from "@/components/ui/badge";
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { ScrollArea } from "@/components/ui/scroll-area";

export function MemoryDetailDialog({
  id,
  onOpenChange,
}: {
  id: number | null;
  onOpenChange: (open: boolean) => void;
}) {
  const { data, isLoading, error } = useMemoryDetail(id);
  return (
    <Dialog open={id != null} onOpenChange={onOpenChange}>
      <DialogContent className="max-w-[95vw] sm:max-w-2xl">
        <DialogHeader>
          <DialogTitle className="text-base sm:text-lg">
            Episode #{id}
          </DialogTitle>
        </DialogHeader>
        {isLoading && <Loading />}
        {error && <ErrorState error={error} />}
        {data && (
          <ScrollArea className="max-h-[60vh] pr-4">
            <div className="space-y-4 text-sm">
              <section>
                <h3 className="mb-2 font-medium">
                  Segments
                  <Badge variant="secondary" className="ml-2">
                    {data.segments.length}
                  </Badge>
                </h3>
                <div className="space-y-2">
                  {data.segments.map((seg, i) => (
                    <pre
                      key={i}
                      className="overflow-x-auto rounded-md border border-border bg-muted/40 p-3 text-xs"
                    >
                      {JSON.stringify(seg, null, 2)}
                    </pre>
                  ))}
                </div>
              </section>
              <section>
                <h3 className="mb-2 font-medium">
                  SR transitions
                  <Badge variant="secondary" className="ml-2">
                    {data.sr_transitions.length}
                  </Badge>
                </h3>
                <div className="flex flex-wrap gap-2">
                  {data.sr_transitions.map((t) => (
                    <Badge key={t.dst} variant="outline" className="font-mono">
                      → #{t.dst} ×{t.count}
                    </Badge>
                  ))}
                  {data.sr_transitions.length === 0 && (
                    <span className="text-muted-foreground">none</span>
                  )}
                </div>
              </section>
              <section>
                <h3 className="mb-2 font-medium">
                  Plasticity edges
                  <Badge variant="secondary" className="ml-2">
                    {data.plasticity_edges.length}
                  </Badge>
                </h3>
                <div className="space-y-1 font-mono text-xs">
                  {data.plasticity_edges.map((e, i) => (
                    <div key={i}>
                      #{e.src} → #{e.dst} · w={e.weight.toFixed(3)} · b=
                      {e.bonus.toFixed(3)} · hits={e.hits}
                    </div>
                  ))}
                  {data.plasticity_edges.length === 0 && (
                    <span className="text-muted-foreground">none</span>
                  )}
                </div>
              </section>
            </div>
          </ScrollArea>
        )}
      </DialogContent>
    </Dialog>
  );
}
