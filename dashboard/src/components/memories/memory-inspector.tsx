/**
 * Node inspector — the focus+context detail surface that replaces the
 * old modal. For an episodic node it pulls the full trace
 * (`GET /memories/{id}`: segments, SR transitions, plasticity edges)
 * and overlays the live ranking signals the map encodes. For a schema
 * node it shows the consolidation summary.
 */
import { CircleHelp, Crosshair, Maximize2, X } from "lucide-react";
import { useMemoryDetail } from "@/lib/queries";
import type { GraphNode } from "@/lib/api";
import { relativeTime, shortSession } from "@/lib/format";
import { ErrorState } from "@/components/common/states";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Separator } from "@/components/ui/separator";
import { Skeleton } from "@/components/ui/skeleton";
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import { DeleteMemoryButton } from "@/components/memories/delete-memory-button";

/** Layout-shaped placeholder for the trace section while the detail
 *  query resolves. Mirrors the rendered structure (SR transitions /
 *  plasticity / segments headings) so the inspector doesn't flash from
 *  a single centred spinner to a dense panel.
 *  `aria-busy + role=status + visually-hidden label` gives assistive
 *  tech the loading cue without showing visible text. */
function TraceSkeleton() {
  return (
    <div
      className="space-y-5 text-sm"
      role="status"
      aria-busy="true"
      aria-live="polite"
    >
      <span className="sr-only">Loading memory trace…</span>
      <section>
        <h3 className="eyebrow mb-2">Successor transitions</h3>
        <div className="flex flex-wrap gap-1.5">
          <Skeleton className="h-5 w-16" />
          <Skeleton className="h-5 w-20" />
          <Skeleton className="h-5 w-14" />
        </div>
      </section>
      <section>
        <h3 className="eyebrow mb-2">Plasticity</h3>
        <div className="space-y-1.5">
          <Skeleton className="h-3 w-full" />
          <Skeleton className="h-3 w-5/6" />
          <Skeleton className="h-3 w-3/4" />
        </div>
      </section>
      <section>
        <h3 className="eyebrow mb-2">Segments</h3>
        <Skeleton className="h-20 w-full" />
      </section>
    </div>
  );
}

function Stat({
  label,
  value,
  hint,
}: {
  label: string;
  value: string;
  hint?: string;
}) {
  return (
    <div className="space-y-1">
      <div className="eyebrow flex items-center gap-1">
        <span>{label}</span>
        {hint && (
          <Tooltip>
            <TooltipTrigger asChild>
              <button
                type="button"
                aria-label={`What is ${label.toLowerCase()}?`}
                className="inline-flex size-3 items-center justify-center rounded-full text-muted-foreground/60 transition-colors hover:text-foreground focus-visible:text-foreground focus-visible:outline-none"
              >
                <CircleHelp className="size-3" aria-hidden />
              </button>
            </TooltipTrigger>
            <TooltipContent
              side="top"
              className="max-w-xs border border-border/70 bg-popover/95 text-xs leading-snug text-popover-foreground shadow-card backdrop-blur"
            >
              {hint}
            </TooltipContent>
          </Tooltip>
        )}
      </div>
      <div className="metric text-sm text-foreground">{value}</div>
    </div>
  );
}

function EpisodicBody({
  node,
  onFocus,
  onClose,
  onOpenFull,
}: {
  node: Extract<GraphNode, { kind: "episodic" }>;
  onFocus: (id: number) => void;
  onClose: () => void;
  onOpenFull: () => void;
}) {
  const { data, isLoading, error } = useMemoryDetail(node.id);
  return (
    <div className="space-y-5">
      <div className="grid grid-cols-1 gap-4 sm:grid-cols-2">
        <Stat
          label="Salience"
          value={node.salience.toFixed(2)}
          hint="How important this memory is. Drives retention (low-salience traces are forgotten first) and gets a small bonus during recall."
        />
        <Stat
          label="Retrievals"
          value={String(node.retrieval_count)}
          hint="How many times this episode has been recalled. Each retrieval reinforces the trace and refreshes its last-access time."
        />
        <Stat
          label="Context"
          value={shortSession(node.session_id)}
          hint="The context the memory was encoded in — modelling state-dependent retrieval. Episodes from the same context get a small ranking bonus when recalled in that same context."
        />
        <Stat
          label="Encoded"
          value={relativeTime(node.encoded_at)}
          hint="When this episode was first stored."
        />
        <Stat
          label="Last recall"
          value={relativeTime(node.last_accessed)}
          hint="When this episode was most recently retrieved. Recency is one input to the forgetting policy."
        />
        <Stat
          label="Schema"
          value={node.consolidated_into > 0 ? `#${node.consolidated_into}` : "none"}
          hint="The semantic schema (gist) this episode has been consolidated into, if any. Consolidation distils repeated episodes into reusable semantic memory."
        />
      </div>

      {node.preview && (
        <p className="rounded-md border border-border/70 bg-muted/40 p-3 text-xs leading-relaxed text-muted-foreground">
          {node.preview}
        </p>
      )}

      <Separator />

      {isLoading && <TraceSkeleton />}
      {error && <ErrorState error={error} />}
      {data && (
        <div className="space-y-5 text-sm">
          <section>
            <h3 className="eyebrow mb-2">
              Successor transitions · {data.sr_transitions.length}
            </h3>
            <div className="flex flex-wrap gap-1.5">
              {data.sr_transitions.length === 0 && (
                <span className="text-xs text-muted-foreground">none</span>
              )}
              {data.sr_transitions.map((t) => (
                <Badge key={t.dst} variant="outline" className="font-mono">
                  → #{t.dst} ×{t.count}
                </Badge>
              ))}
            </div>
          </section>

          <section>
            <h3 className="eyebrow mb-2">
              Plasticity · {data.plasticity_edges.length}
            </h3>
            <div className="space-y-1 font-mono text-xs text-muted-foreground">
              {data.plasticity_edges.length === 0 && <span>none</span>}
              {data.plasticity_edges.map((e, i) => (
                <div key={i} className="flex justify-between gap-2">
                  <span className="text-foreground">
                    #{e.src} → #{e.dst}
                  </span>
                  <span>
                    w={e.weight.toFixed(2)} · b={e.bonus.toFixed(2)} · ×{e.hits}
                  </span>
                </div>
              ))}
            </div>
          </section>

          <section>
            <h3 className="eyebrow mb-2">
              Segments · {data.segments.length}
            </h3>
            <div className="space-y-2">
              {data.segments.map((seg, i) => (
                <pre
                  key={i}
                  className="overflow-x-auto rounded-md border border-border bg-muted/40 p-3 text-[0.7rem] leading-relaxed"
                >
                  {JSON.stringify(seg, null, 2)}
                </pre>
              ))}
            </div>
          </section>
        </div>
      )}

      <Separator />
      <div className="flex flex-wrap items-center justify-between gap-2">
        <div className="flex items-center gap-2">
          <Button
            variant="secondary"
            size="sm"
            onClick={() => onFocus(node.id)}
          >
            <Crosshair className="size-4" aria-hidden />
            Recenter map
          </Button>
          <Button variant="ghost" size="sm" onClick={onOpenFull}>
            <Maximize2 className="size-4" aria-hidden />
            View full
          </Button>
        </div>
        <DeleteMemoryButton id={node.id} onDeleted={onClose} />
      </div>
    </div>
  );
}

function SemanticBody({
  node,
  onOpenFull,
}: {
  node: Extract<GraphNode, { kind: "semantic" }>;
  onOpenFull: () => void;
}) {
  return (
    <div className="space-y-5">
      <div className="grid grid-cols-1 gap-4 sm:grid-cols-2">
        <Stat
          label="Confidence"
          value={node.confidence.toFixed(2)}
          hint="How well-supported this schema is. Grows as more episodes consolidate into it; saturates at the confidence ceiling."
        />
        <Stat
          label="Sources"
          value={String(node.source_count)}
          hint="How many episodes were absorbed into this schema during consolidation."
        />
        <Stat
          label="Origin"
          value={node.user_asserted ? "user-asserted" : "consolidated"}
          hint="Whether the schema was distilled automatically from episodes (consolidated) or written directly by the caller (user-asserted)."
        />
      </div>
      {node.preview && (
        <p className="rounded-md border border-border/70 bg-muted/40 p-3 text-xs leading-relaxed text-muted-foreground">
          {node.preview}
        </p>
      )}
      <p className="text-xs leading-relaxed text-muted-foreground">
        {node.source_count > 0 ? (
          <>
            A semantic schema distilled from{" "}
            <span className="font-mono text-foreground">{node.source_count}</span>{" "}
            episode{node.source_count === 1 ? "" : "s"}. Episodes that fed it
            are drawn with a cyan link on the map.
          </>
        ) : (
          <>
            A user-asserted semantic memory — written directly via{" "}
            <span className="font-mono text-foreground">
              store_semantic_memory
            </span>
            , with no consolidating episodes.
          </>
        )}
      </p>
      <Separator />
      <div>
        <Button variant="ghost" size="sm" onClick={onOpenFull}>
          <Maximize2 className="size-4" aria-hidden />
          View full
        </Button>
      </div>
    </div>
  );
}

export function MemoryInspector({
  node,
  onFocus,
  onClose,
  onOpenFull,
}: {
  node: GraphNode | null;
  onFocus: (id: number) => void;
  onClose: () => void;
  onOpenFull: () => void;
}) {
  if (!node) {
    return (
      <div className="flex h-full flex-col items-center justify-center gap-3 px-6 text-center">
        <div className="eyebrow">Inspector</div>
        <p className="max-w-[16rem] text-xs leading-relaxed text-muted-foreground">
          Select a node to trace its successor transitions, plasticity
          associations, and consolidation lineage.
        </p>
      </div>
    );
  }
  return (
    <TooltipProvider delayDuration={120}>
    <div className="flex h-full flex-col">
      <div className="flex items-start justify-between gap-3 border-b border-border/60 p-4">
        <div className="min-w-0">
          <div className="eyebrow mb-1.5">
            {node.kind === "semantic" ? "Schema" : "Episode"}
          </div>
          <div className="metric truncate text-lg text-foreground">
            {node.label}
          </div>
        </div>
        <Button
          variant="ghost"
          size="icon-sm"
          aria-label="Close inspector"
          onClick={onClose}
        >
          <X className="size-4" aria-hidden />
        </Button>
      </div>
      <ScrollArea className="min-h-0 flex-1">
        <div className="p-4 sm:p-5">
          {node.kind === "episodic" ? (
            <EpisodicBody
              node={node}
              onFocus={onFocus}
              onClose={onClose}
              onOpenFull={onOpenFull}
            />
          ) : (
            <SemanticBody node={node} onOpenFull={onOpenFull} />
          )}
        </div>
      </ScrollArea>
    </div>
    </TooltipProvider>
  );
}
