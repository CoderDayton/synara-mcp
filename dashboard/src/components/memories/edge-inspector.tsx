/**
 * Edge inspector — the sidebar detail surface for a clicked edge.
 *
 * Where the node inspector shows a memory's content and lineage, this
 * explains a *relation*: the metric values plus what each metric means
 * in the memory model, so the map doubles as documentation of its own
 * mechanics. Endpoint chips jump to the connected memories.
 */
import { ArrowRight, X } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Separator } from "@/components/ui/separator";
import type { EdgeInfo } from "./graph-canvas";

function Metric({
  label,
  value,
  explain,
}: {
  label: string;
  value: string;
  explain: string;
}) {
  return (
    <div className="space-y-1">
      <div className="flex items-baseline justify-between gap-3">
        <span className="eyebrow">{label}</span>
        <span className="metric text-sm text-foreground">{value}</span>
      </div>
      <p className="text-xs leading-relaxed text-muted-foreground">{explain}</p>
    </div>
  );
}

const KIND_META: Record<
  EdgeInfo["kind"],
  { title: string; blurb: string }
> = {
  sr: {
    title: "Successor edge",
    blurb:
      "A temporal transition in the successor representation: these two " +
      "memories were accessed within the same ~60-second window, so " +
      "recalling the source predicts the destination. The discounted " +
      "closure of this graph is blended into recall ranking (weight ω).",
  },
  plasticity: {
    title: "Plasticity edge",
    blurb:
      "A Hebbian association reinforced every time the two memories are " +
      "co-recalled. Repeated reinforcement inside the early-LTP window " +
      "folds the transient bonus into durable weight (late-LTP); idle " +
      "edges decay and are eventually pruned.",
  },
  consolidation: {
    title: "Consolidation link",
    blurb:
      "This episode was absorbed into the schema during consolidation — " +
      "its gist is preserved upstream, so the raw episode can be " +
      "forgotten safely without losing the knowledge.",
  },
  context: {
    title: "Context link",
    blurb:
      "Same-session timeline ordering (the context lens): these episodes " +
      "were encoded back-to-back in one session.",
  },
};

export function EdgeInspector({
  info,
  onInspectEndpoint,
  onClose,
}: {
  info: EdgeInfo;
  /** Jump to one of the connected memories ("ep:12" / "sem:3" key). */
  onInspectEndpoint: (key: string) => void;
  onClose: () => void;
}) {
  const meta = KIND_META[info.kind];
  const srcKey = `ep:${info.src}`;
  const dstKey =
    info.kind === "consolidation" ? info.dst : `ep:${info.dst}`;
  const dstLabel =
    info.kind === "consolidation"
      ? `⌬${info.dst.replace(/^sem:/, "")}`
      : `#${info.dst}`;

  return (
    <div className="flex h-full flex-col">
      <div className="flex items-start justify-between gap-3 border-b border-border/60 p-4">
        <div className="min-w-0">
          <div className="eyebrow mb-1.5">{meta.title}</div>
          <div className="metric flex items-center gap-1.5 text-lg text-foreground">
            <button
              type="button"
              onClick={() => onInspectEndpoint(srcKey)}
              className="rounded underline-offset-4 transition-colors hover:text-primary hover:underline focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-primary"
            >
              #{info.src}
            </button>
            <ArrowRight className="size-4 text-muted-foreground" aria-hidden />
            <button
              type="button"
              onClick={() => onInspectEndpoint(dstKey)}
              className="rounded underline-offset-4 transition-colors hover:text-primary hover:underline focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-primary"
            >
              {dstLabel}
            </button>
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

      <div className="min-h-0 flex-1 space-y-5 overflow-y-auto p-4 sm:p-5">
        <p className="text-xs leading-relaxed text-muted-foreground">{meta.blurb}</p>
        <Separator />

        {info.kind === "sr" && (
          <>
            <Metric
              label="M (closure)"
              value={info.m.toFixed(3)}
              explain="The discounted successor value M[src, dst] — how strongly recalling the source predicts the destination, including multi-hop paths. This is the prior recall ranking blends in once enough edges exist."
            />
            <Metric
              label="Transitions"
              value={`×${info.hits}`}
              explain="How many times the destination was accessed shortly after the source (stores and recalls both count). The raw tally that M is rebuilt from."
            />
          </>
        )}

        {info.kind === "plasticity" && (
          <>
            <Metric
              label="Weight"
              value={info.weight.toFixed(3)}
              explain="Durable (late-LTP) strength. Only forms when the edge is reinforced repeatedly within the early-LTP window; decays slowly with disuse and powers spreading activation."
            />
            <Metric
              label="Bonus"
              value={info.bonus.toFixed(3)}
              explain="Transient (early-LTP) potentiation from recent co-recalls. Decays over ~2 hours unless reinforced enough to fold into durable weight."
            />
            <Metric
              label="Reinforcements"
              value={`×${info.hits}`}
              explain="Lifetime reinforcement count. Past the habit threshold the edge decays far slower and relearns faster (savings)."
            />
            <Metric
              label="Habit"
              value={info.is_habit ? "yes" : "no"}
              explain="A habit edge has been reinforced enough times to persist through long disuse — it fades over months, not weeks."
            />
          </>
        )}

        {info.kind === "consolidation" && (
          <Metric
            label="Direction"
            value={`#${info.src} → ${dstLabel}`}
            explain="Episode → schema. The episode's consolidated_into marker points at this schema; the forgetting pass treats such episodes as safe to prune at the normal threshold."
          />
        )}

        {info.kind === "context" && (
          <Metric
            label="Session"
            value={info.session}
            explain="Both episodes belong to this session; the link shows encoded-at order, not a learned association."
          />
        )}
      </div>
    </div>
  );
}
