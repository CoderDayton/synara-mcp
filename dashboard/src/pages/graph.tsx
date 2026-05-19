import { Activity, useMemo, useState } from "react";
import cytoscape from "cytoscape";
import fcose from "cytoscape-fcose";
import CytoscapeComponent from "react-cytoscapejs";
import type { ElementDefinition, StylesheetJson } from "cytoscape";
import { useGraph } from "@/lib/queries";
import { PageHeader } from "@/components/common/page-header";
import { Empty, ErrorState, Loading } from "@/components/common/states";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Badge } from "@/components/ui/badge";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Card, CardContent } from "@/components/ui/card";

cytoscape.use(fcose);

// Canvas needs concrete colors (no oklch); these track the design tokens.
const C = {
  node: "#26272f",
  nodeBorder: "#7c5cff",
  focus: "#a78bfa",
  label: "#d8d8e0",
  sr: "#8b5cf6",
  plastic: "#e0a93b",
};

const STYLESHEET = [
  {
    selector: "node",
    style: {
      "background-color": C.node,
      "border-color": C.nodeBorder,
      "border-width": 1.5,
      label: "data(label)",
      color: C.label,
      "font-size": 10,
      "text-valign": "center",
      "text-halign": "center",
      width: 30,
      height: 30,
    },
  },
  {
    selector: "node.focus",
    style: { "background-color": C.focus, width: 42, height: 42 },
  },
  {
    selector: "edge.sr",
    style: {
      width: "mapData(hits, 1, 20, 1, 6)",
      "line-color": C.sr,
      "target-arrow-color": C.sr,
      "target-arrow-shape": "triangle",
      "curve-style": "bezier",
      opacity: 0.75,
    },
  },
  {
    selector: "edge.plastic",
    style: {
      width: 1.5,
      "line-color": C.plastic,
      "line-style": "dashed",
      "curve-style": "bezier",
      opacity: 0.6,
    },
  },
  // Cytoscape's stylesheet css unions are intentionally stringly; a
  // single cast keeps the styling readable without per-property noise.
] as unknown as StylesheetJson;

const LAYOUT = {
  name: "fcose",
  quality: "default",
  animate: false,
  randomize: true,
  nodeSeparation: 80,
} as unknown as cytoscape.LayoutOptions;

export default function Graph() {
  const [focusInput, setFocusInput] = useState("");
  const [depth, setDepth] = useState("1");
  const [maxNodes, setMaxNodes] = useState("200");
  const [params, setParams] = useState<{
    focus?: number;
    depth: number;
    max_nodes: number;
  }>({ depth: 1, max_nodes: 200 });

  const { data, isLoading, error } = useGraph(params);

  const elements = useMemo<ElementDefinition[]>(() => {
    if (!data) return [];
    const els: ElementDefinition[] = data.nodes.map((id) => ({
      data: { id: String(id), label: `#${id}` },
      classes: params.focus === id ? "focus" : undefined,
    }));
    data.sr_edges.forEach((e, i) =>
      els.push({
        data: {
          id: `s${i}`,
          source: String(e.src),
          target: String(e.dst),
          hits: e.hits,
        },
        classes: "sr",
      }),
    );
    data.plasticity_edges.forEach((e, i) =>
      els.push({
        data: { id: `p${i}`, source: String(e.src), target: String(e.dst) },
        classes: "plastic",
      }),
    );
    return els;
  }, [data, params.focus]);

  function apply(e: React.FormEvent) {
    e.preventDefault();
    const f = focusInput.trim();
    setParams({
      focus: f ? Number(f) : undefined,
      depth: Number(depth),
      max_nodes: Number(maxNodes),
    });
  }

  return (
    <div className="space-y-6">
      <PageHeader
        title="Successor graph"
        subtitle="Durable SR transitions (solid) and plasticity edges (dashed). Bounded server-side."
        actions={
          data && (
            <div className="flex flex-wrap items-center gap-2 text-xs">
              <Badge variant="outline">{data.nodes.length} nodes</Badge>
              <Badge variant="outline">{data.sr_edges.length} SR</Badge>
              <Badge variant="outline">
                {data.plasticity_edges.length} plastic
              </Badge>
              {data.truncated && (
                <Badge variant="destructive">truncated</Badge>
              )}
            </div>
          )
        }
      />

      <Card>
        <CardContent className="p-4 sm:p-6">
          <form
            onSubmit={apply}
            className="grid grid-cols-1 gap-3 sm:grid-cols-2 lg:grid-cols-4"
          >
            <div className="space-y-1.5">
              <Label htmlFor="focus">Focus episode ID</Label>
              <Input
                id="focus"
                inputMode="numeric"
                placeholder="all (global)"
                value={focusInput}
                onChange={(e) => setFocusInput(e.target.value)}
              />
            </div>
            <div className="space-y-1.5">
              <Label htmlFor="depth">BFS depth</Label>
              <Select value={depth} onValueChange={setDepth}>
                <SelectTrigger id="depth">
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="1">1</SelectItem>
                  <SelectItem value="2">2</SelectItem>
                  <SelectItem value="3">3</SelectItem>
                </SelectContent>
              </Select>
            </div>
            <div className="space-y-1.5">
              <Label htmlFor="max">Max nodes</Label>
              <Input
                id="max"
                inputMode="numeric"
                value={maxNodes}
                onChange={(e) => setMaxNodes(e.target.value)}
              />
            </div>
            <div className="flex items-end">
              <Button type="submit" className="w-full">
                Render
              </Button>
            </div>
          </form>
        </CardContent>
      </Card>

      {isLoading && <Loading label="Loading graph" />}
      {error && <ErrorState error={error} />}
      {data && elements.length === 0 && (
        <Empty
          label="No successor edges yet"
          hint="The transition graph is built from episode co-occurrence within a session window. Encode a few episodes and recall them, then render."
        />
      )}

      <Activity mode={data && elements.length > 0 ? "visible" : "hidden"}>
        <div className="h-[60vh] overflow-hidden rounded-xl border border-border/60 bg-card shadow-card sm:h-[65vh] lg:h-[70vh]">
          <CytoscapeComponent
            elements={elements}
            stylesheet={STYLESHEET}
            layout={LAYOUT}
            style={{ width: "100%", height: "100%" }}
            minZoom={0.2}
            maxZoom={2.5}
          />
        </div>
      </Activity>
    </div>
  );
}
