import { describe, expect, it } from "vitest";
import { normalizeGraph } from "@/lib/api";
import { computeLayout, nodeRadius } from "./graph-layout";

/** Two sessions (a, b) of three SR-linked episodes each, plus one
 *  schema consolidated from session a. */
function twoSessionGraph() {
  const ep = (id: number, session: string) => ({
    id,
    key: `ep:${id}`,
    kind: "episodic",
    session_id: session,
    salience: 0.5,
    retrieval_count: 0,
    embedding: null,
  });
  return normalizeGraph({
    omega: 0.3,
    consolidation_edges: [
      { src: 1, dst: "sem:100" },
      { src: 2, dst: "sem:100" },
    ],
    nodes: [
      ep(1, "a"),
      ep(2, "a"),
      ep(3, "a"),
      ep(4, "b"),
      ep(5, "b"),
      ep(6, "b"),
      { id: 100, key: "sem:100", kind: "semantic", source_count: 2, embedding: null },
    ],
    sr_edges: [
      { src: 1, dst: 2, m: 0.8, hits: 3 },
      { src: 2, dst: 3, m: 0.8, hits: 3 },
      { src: 4, dst: 5, m: 0.8, hits: 3 },
      { src: 5, dst: 6, m: 0.8, hits: 3 },
    ],
    plasticity_edges: [],
  });
}

const dist = (
  a: { x: number; y: number },
  b: { x: number; y: number },
) => Math.hypot(a.x - b.x, a.y - b.y);

function centroidOf(
  pos: Map<string, { x: number; y: number }>,
  keys: string[],
) {
  let x = 0;
  let y = 0;
  for (const k of keys) {
    const p = pos.get(k)!;
    x += p.x;
    y += p.y;
  }
  return { x: x / keys.length, y: y / keys.length };
}

describe("computeLayout (session-clustered)", () => {
  it("positions every node", async () => {
    const data = twoSessionGraph();
    const pos = await computeLayout(data);
    for (const n of data.nodes) expect(pos.has(n.key)).toBe(true);
  });

  it("separates session bubbles further than they spread internally", async () => {
    const data = twoSessionGraph();
    const pos = await computeLayout(data);
    const aKeys = ["ep:1", "ep:2", "ep:3"];
    const bKeys = ["ep:4", "ep:5", "ep:6"];
    const ca = centroidOf(pos, aKeys);
    const cb = centroidOf(pos, bKeys);
    // The two bubbles are distinct regions.
    const between = dist(ca, cb);
    const spreadA = Math.max(...aKeys.map((k) => dist(pos.get(k)!, ca)));
    const spreadB = Math.max(...bKeys.map((k) => dist(pos.get(k)!, cb)));
    expect(between).toBeGreaterThan(spreadA + spreadB);
  });

  it("drops a schema beside the session it consolidated from", async () => {
    const data = twoSessionGraph();
    const pos = await computeLayout(data);
    const schema = pos.get("sem:100")!;
    const ca = centroidOf(pos, ["ep:1", "ep:2", "ep:3"]);
    const cb = centroidOf(pos, ["ep:4", "ep:5", "ep:6"]);
    expect(dist(schema, ca)).toBeLessThan(dist(schema, cb));
  });

  it("leaves no overlapping discs", async () => {
    const data = twoSessionGraph();
    const pos = await computeLayout(data);
    const items = data.nodes.map((n) => ({ p: pos.get(n.key)!, r: nodeRadius(n) }));
    for (let i = 0; i < items.length; i++) {
      for (let j = i + 1; j < items.length; j++) {
        // Allow a 1px slack for float settling; discs must not interpenetrate.
        expect(dist(items[i].p, items[j].p)).toBeGreaterThan(
          items[i].r + items[j].r - 1,
        );
      }
    }
  });

  it("returns an empty layout for empty data", async () => {
    expect((await computeLayout(undefined)).size).toBe(0);
  });
});
