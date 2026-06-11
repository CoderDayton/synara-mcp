/**
 * Full-content modal for a memory node. The inspector aside shows
 * relational structure (successors, plasticity, schema lineage); this
 * dialog shows the *content* — full text plus authoritative metadata.
 *
 * View-only. Editing affordance is intentionally deferred.
 */
import { useState } from "react";
import { useMemoryDetail, useSemanticDetail } from "@/lib/queries";
import type { GraphNode } from "@/lib/api";
import { relativeTime, shortSession } from "@/lib/format";
import { ErrorState } from "@/components/common/states";
import { MemoryContent } from "@/components/memories/memory-content";
import { Badge } from "@/components/ui/badge";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Separator } from "@/components/ui/separator";
import { Skeleton } from "@/components/ui/skeleton";

function MetaRow({ label, value }: { label: string; value: string }) {
  return (
    <div className="flex items-baseline gap-3">
      <span className="eyebrow w-24 shrink-0">{label}</span>
      <span className="metric text-xs text-foreground">{value}</span>
    </div>
  );
}

function EpisodicContent({
  node,
}: {
  node: Extract<GraphNode, { kind: "episodic" }>;
}) {
  const { data, isLoading, error } = useMemoryDetail(node.id);
  // Theta segmentation is a storage detail: segments are char-budget
  // slices of ONE episode and can split mid-word. Reassemble the
  // original text exactly like `MemoryService.get_episode` does —
  // ordered by position, joined with no separator.
  const segments = data?.segments ?? [];
  const ordered = [...segments].sort(
    (a, b) =>
      (typeof a.position === "number" ? a.position : 0) -
      (typeof b.position === "number" ? b.position : 0),
  );
  const textParts = ordered
    .map((seg) => (typeof seg.content === "string" ? seg.content : null))
    .filter((s): s is string => s !== null);
  const fullText = textParts.join("");
  return (
    <div className="space-y-5">
      <section className="space-y-2">
        <h3 className="eyebrow">
          Content
          {ordered.length > 1 && (
            <span className="ml-2 normal-case tracking-normal text-muted-foreground/70">
              · {ordered.length} segments, reassembled
            </span>
          )}
        </h3>
        {isLoading && (
          <div className="space-y-2" role="status" aria-busy="true">
            <Skeleton className="h-3 w-full" />
            <Skeleton className="h-3 w-5/6" />
            <Skeleton className="h-3 w-2/3" />
          </div>
        )}
        {error && <ErrorState error={error} />}
        {data &&
          (fullText ? (
            <MemoryContent text={fullText} />
          ) : ordered.length > 0 ? (
            // Non-string segment payloads (defensive): show raw.
            <pre className="rounded-md border border-border bg-muted/40 p-3 text-xs leading-relaxed break-words whitespace-pre-wrap text-foreground">
              {JSON.stringify(ordered, null, 2)}
            </pre>
          ) : (
            <p className="text-xs text-muted-foreground">no content</p>
          ))}
      </section>

      <Separator />

      <section className="space-y-2">
        <h3 className="eyebrow">Metadata</h3>
        <div className="space-y-1.5">
          <MetaRow label="Session" value={shortSession(node.session_id)} />
          <MetaRow label="Salience" value={node.salience.toFixed(2)} />
          <MetaRow
            label="Retrievals"
            value={String(node.retrieval_count)}
          />
          <MetaRow label="Encoded" value={relativeTime(node.created_at)} />
          <MetaRow
            label="Last recall"
            value={relativeTime(node.last_accessed)}
          />
          <MetaRow
            label="Schema"
            value={
              node.consolidated_into > 0
                ? `#${node.consolidated_into}`
                : "none"
            }
          />
        </div>
      </section>
    </div>
  );
}

function SemanticContent({
  node,
}: {
  node: Extract<GraphNode, { kind: "semantic" }>;
}) {
  const { data, isLoading, error } = useSemanticDetail(node.id);
  return (
    <div className="space-y-5">
      <section className="space-y-2">
        <h3 className="eyebrow">Content</h3>
        {isLoading && (
          <div className="space-y-2" role="status" aria-busy="true">
            <Skeleton className="h-3 w-full" />
            <Skeleton className="h-3 w-5/6" />
            <Skeleton className="h-3 w-3/4" />
          </div>
        )}
        {error && <ErrorState error={error} />}
        {data &&
          (data.content ? (
            <MemoryContent text={data.content} />
          ) : (
            <p className="text-xs text-muted-foreground">(empty)</p>
          ))}
      </section>

      <Separator />

      <section className="space-y-2">
        <h3 className="eyebrow">Metadata</h3>
        <div className="space-y-1.5">
          {data && <MetaRow label="Kind" value={data.kind} />}
          <MetaRow label="Confidence" value={node.confidence.toFixed(2)} />
          <MetaRow label="Sources" value={String(node.source_count)} />
          <MetaRow
            label="Origin"
            value={node.user_asserted ? "user-asserted" : "consolidated"}
          />
          {data && (
            <MetaRow label="Created" value={relativeTime(data.created_at)} />
          )}
          {data && data.updated_at !== data.created_at && (
            <MetaRow label="Updated" value={relativeTime(data.updated_at)} />
          )}
        </div>
        {data && data.tags.length > 0 && (
          <div className="flex flex-wrap gap-1.5 pt-1">
            {data.tags.map((t) => (
              <Badge key={t} variant="outline" className="font-mono">
                {t}
              </Badge>
            ))}
          </div>
        )}
      </section>
    </div>
  );
}

export function MemoryDialog({
  node,
  open,
  onOpenChange,
}: {
  node: GraphNode | null;
  open: boolean;
  onOpenChange: (open: boolean) => void;
}) {
  // Pin the node while the close animation plays — otherwise the title and
  // body flash to empty when `selected` is cleared upstream. Sync the last
  // non-null node during render (React's "information from previous renders"
  // idiom) — guarded so it can't loop, and avoiding an effect that would
  // cascade an extra render.
  const [view, setView] = useState<GraphNode | null>(node);
  if (node && node !== view) setView(node);
  const titleKind =
    view?.kind === "semantic"
      ? view.user_asserted
        ? "Memory"
        : "Schema"
      : "Episode";
  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="max-w-full sm:max-w-2xl lg:max-w-3xl">
        <DialogHeader>
          <DialogTitle className="font-mono text-sm uppercase tracking-wider">
            {titleKind}
            {view ? ` · ${view.label}` : ""}
          </DialogTitle>
          <DialogDescription className="text-xs">
            View-only. Editing isn't wired yet.
          </DialogDescription>
        </DialogHeader>
        <ScrollArea className="max-h-[70vh] pr-2">
          {view?.kind === "episodic" && <EpisodicContent node={view} />}
          {view?.kind === "semantic" && <SemanticContent node={view} />}
        </ScrollArea>
      </DialogContent>
    </Dialog>
  );
}
