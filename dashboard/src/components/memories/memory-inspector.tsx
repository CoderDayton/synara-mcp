/**
 * Node inspector — the focus+context detail surface that replaces the
 * old modal. For an episodic node it pulls the full trace
 * (`GET /memories/{id}`: segments, SR transitions, plasticity edges)
 * and overlays the live ranking signals the map encodes. For a schema
 * node it shows the consolidation summary.
 */
import { Crosshair, X } from "lucide-react";
import { useMemoryDetail } from "@/lib/queries";
import type { GraphNode } from "@/lib/api";
import { relativeTime, shortSession } from "@/lib/format";
import { ErrorState, Loading } from "@/components/common/states";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Separator } from "@/components/ui/separator";
import { DeleteMemoryButton } from "@/components/memories/delete-memory-button";

function Stat({ label, value }: { label: string; value: string }) {
  return (
    <div className="space-y-1">
      <div className="eyebrow">{label}</div>
      <div className="metric text-sm text-foreground">{value}</div>
    </div>
  );
}

function EpisodicBody({
  node,
  onFocus,
  onClose,
}: {
  node: Extract<GraphNode, { kind: "episodic" }>;
  onFocus: (id: number) => void;
  onClose: () => void;
}) {
  const { data, isLoading, error } = useMemoryDetail(node.id);
  return (
    <div className="space-y-5">
      <div className="grid grid-cols-1 gap-4 sm:grid-cols-2">
        <Stat label="Salience" value={node.salience.toFixed(2)} />
        <Stat label="Retrievals" value={String(node.retrieval_count)} />
        <Stat label="Session" value={shortSession(node.session_id)} />
        <Stat label="Encoded" value={relativeTime(node.encoded_at)} />
        <Stat label="Last recall" value={relativeTime(node.last_accessed)} />
        <Stat
          label="Schema"
          value={node.consolidated_into > 0 ? `#${node.consolidated_into}` : "none"}
        />
      </div>

      {node.preview && (
        <p className="rounded-md border border-border/70 bg-muted/40 p-3 text-xs leading-relaxed text-muted-foreground">
          {node.preview}
        </p>
      )}

      <Separator />

      {isLoading && <Loading label="Loading trace" />}
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
      <div className="flex items-center justify-between">
        <Button
          variant="secondary"
          size="sm"
          onClick={() => onFocus(node.id)}
        >
          <Crosshair className="size-4" aria-hidden />
          Recenter map
        </Button>
        <DeleteMemoryButton id={node.id} onDeleted={onClose} />
      </div>
    </div>
  );
}

function SemanticBody({
  node,
}: {
  node: Extract<GraphNode, { kind: "semantic" }>;
}) {
  return (
    <div className="space-y-5">
      <div className="grid grid-cols-1 gap-4 sm:grid-cols-2">
        <Stat label="Confidence" value={node.confidence.toFixed(2)} />
        <Stat label="Sources" value={String(node.source_count)} />
        <Stat
          label="Origin"
          value={node.user_asserted ? "user-asserted" : "consolidated"}
        />
      </div>
      {node.preview && (
        <p className="rounded-md border border-border/70 bg-muted/40 p-3 text-xs leading-relaxed text-muted-foreground">
          {node.preview}
        </p>
      )}
      <p className="text-xs leading-relaxed text-muted-foreground">
        A semantic schema is the gist distilled from{" "}
        <span className="font-mono text-foreground">{node.source_count}</span>{" "}
        episode{node.source_count === 1 ? "" : "s"}. Episodes that fed it are
        drawn with a cyan link on the map.
      </p>
    </div>
  );
}

export function MemoryInspector({
  node,
  onFocus,
  onClose,
}: {
  node: GraphNode | null;
  onFocus: (id: number) => void;
  onClose: () => void;
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
            <EpisodicBody node={node} onFocus={onFocus} onClose={onClose} />
          ) : (
            <SemanticBody node={node} />
          )}
        </div>
      </ScrollArea>
    </div>
  );
}
