import { describe, expect, it } from "vitest";
import {
  bridgeThreshold,
  buildClusterHulls,
  computeBridgeScores,
  convexHull,
  detectCommunities,
  polygonPath,
  type CommunityEdge,
} from "./community";

describe("detectCommunities", () => {
  it("returns an empty map for an empty node list", () => {
    const out = detectCommunities([], []);
    expect(out.size).toBe(0);
  });

  it("puts an isolated node in its own community", () => {
    const out = detectCommunities(["a", "b"], []);
    // Both nodes have no neighbours → labels never change → after
    // compaction they land in 0 and 1 (in sorted-key order).
    expect(out.get("a")).toBe(0);
    expect(out.get("b")).toBe(1);
  });

  it("merges two densely-connected nodes into one community", () => {
    const edges: CommunityEdge[] = [{ src: "a", dst: "b", w: 1 }];
    const out = detectCommunities(["a", "b"], edges);
    expect(out.get("a")).toBe(out.get("b"));
  });

  it("ignores edges with non-positive weight", () => {
    const edges: CommunityEdge[] = [
      { src: "a", dst: "b", w: 0 },
      { src: "a", dst: "b", w: -0.5 },
    ];
    const out = detectCommunities(["a", "b"], edges);
    expect(out.get("a")).not.toBe(out.get("b"));
  });

  it("ignores edges referencing unknown nodes", () => {
    const edges: CommunityEdge[] = [
      { src: "a", dst: "ghost", w: 1 },
      { src: "ghost", dst: "b", w: 1 },
    ];
    const out = detectCommunities(["a", "b"], edges);
    expect(out.get("a")).not.toBe(out.get("b"));
  });

  it("is deterministic across runs", () => {
    const nodes = ["a", "b", "c", "d", "e"];
    const edges: CommunityEdge[] = [
      { src: "a", dst: "b", w: 1 },
      { src: "b", dst: "c", w: 1 },
      { src: "d", dst: "e", w: 1 },
    ];
    const a = detectCommunities(nodes, edges);
    const b = detectCommunities(nodes, edges);
    for (const k of nodes) expect(a.get(k)).toBe(b.get(k));
  });

  it("breaks ties by lowest label id (sorted-key traversal)", () => {
    // Triangle: a-b-c all equal weight. Every node sees the same total
    // neighbour weight per community; ties must resolve to the lowest
    // label, so all three converge into one cluster — stable.
    const nodes = ["a", "b", "c"];
    const edges: CommunityEdge[] = [
      { src: "a", dst: "b", w: 1 },
      { src: "b", dst: "c", w: 1 },
      { src: "a", dst: "c", w: 1 },
    ];
    const out = detectCommunities(nodes, edges);
    expect(out.get("a")).toBe(out.get("b"));
    expect(out.get("b")).toBe(out.get("c"));
  });

  it("produces a dense [0..k) label space", () => {
    const nodes = ["a", "b", "c", "d"];
    const edges: CommunityEdge[] = [
      { src: "a", dst: "b", w: 1 },
      { src: "c", dst: "d", w: 1 },
    ];
    const out = detectCommunities(nodes, edges);
    const labels = new Set(out.values());
    const max = Math.max(...labels);
    expect(max).toBe(labels.size - 1);
  });
});

describe("computeBridgeScores", () => {
  it("returns zero for every node when no edges cross communities", () => {
    const nodes = ["a", "b"];
    const edges: CommunityEdge[] = [{ src: "a", dst: "b", w: 0.5 }];
    const comms = new Map<string, number>([
      ["a", 0],
      ["b", 0],
    ]);
    const scores = computeBridgeScores(nodes, edges, comms);
    expect(scores.get("a")).toBe(0);
    expect(scores.get("b")).toBe(0);
  });

  it("sums cross-community edge weight on both endpoints", () => {
    const nodes = ["a", "b"];
    const edges: CommunityEdge[] = [{ src: "a", dst: "b", w: 0.7 }];
    const comms = new Map<string, number>([
      ["a", 0],
      ["b", 1],
    ]);
    const scores = computeBridgeScores(nodes, edges, comms);
    expect(scores.get("a")).toBeCloseTo(0.7);
    expect(scores.get("b")).toBeCloseTo(0.7);
  });

  it("skips edges whose endpoints have no community", () => {
    const nodes = ["a", "b"];
    const edges: CommunityEdge[] = [{ src: "a", dst: "ghost", w: 1 }];
    const comms = new Map<string, number>([
      ["a", 0],
      ["b", 1],
    ]);
    const scores = computeBridgeScores(nodes, edges, comms);
    expect(scores.get("a")).toBe(0);
  });
});

describe("bridgeThreshold", () => {
  it("returns Infinity when no nonzero scores exist", () => {
    const scores = new Map<string, number>([
      ["a", 0],
      ["b", 0],
    ]);
    expect(bridgeThreshold(scores)).toBe(Infinity);
  });

  it("returns the single value when only one nonzero scorer", () => {
    const scores = new Map<string, number>([
      ["a", 0],
      ["b", 0.4],
    ]);
    expect(bridgeThreshold(scores)).toBe(0.4);
  });

  it("returns the top-quartile cut for a populated set", () => {
    // Sorted nonzero values: [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8].
    // floor(8 * 0.75) = 6, so index 6 → 0.7.
    const scores = new Map<string, number>([
      ["a", 0.1],
      ["b", 0.2],
      ["c", 0.3],
      ["d", 0.4],
      ["e", 0.5],
      ["f", 0.6],
      ["g", 0.7],
      ["h", 0.8],
    ]);
    expect(bridgeThreshold(scores)).toBeCloseTo(0.7);
  });
});

describe("convexHull", () => {
  it("returns empty for empty input", () => {
    expect(convexHull([])).toEqual([]);
  });

  it("returns the single point for a single-point input", () => {
    expect(convexHull([[1, 2]])).toEqual([[1, 2]]);
  });

  it("traces the outer corners of a square (drops the interior point)", () => {
    const hull = convexHull([
      [0, 0],
      [10, 0],
      [10, 10],
      [0, 10],
      [5, 5], // interior — must not appear in the output
    ]);
    expect(hull).toHaveLength(4);
    expect(hull).not.toContainEqual([5, 5]);
  });

  it("is degenerate (no polygon) when all points are collinear", () => {
    const hull = convexHull([
      [0, 0],
      [1, 1],
      [2, 2],
    ]);
    // Collinear → no enclosing area; downstream `polygonPath` returns
    // "" for fewer than 3 vertices, which is the gate the renderer
    // relies on.
    expect(hull.length).toBeLessThan(3);
  });
});

describe("buildClusterHulls", () => {
  it("skips communities below the minimum size (default 3)", () => {
    const nodes = [
      { key: "a", x: 0, y: 0, r: 10 },
      { key: "b", x: 5, y: 0, r: 10 },
    ];
    const comms = new Map<string, number>([
      ["a", 0],
      ["b", 0],
    ]);
    expect(buildClusterHulls(nodes, comms)).toEqual([]);
  });

  it("emits a hull for a community of >= 3 nodes", () => {
    const nodes = [
      { key: "a", x: 0, y: 0, r: 8 },
      { key: "b", x: 40, y: 0, r: 8 },
      { key: "c", x: 20, y: 30, r: 8 },
    ];
    const comms = new Map<string, number>([
      ["a", 0],
      ["b", 0],
      ["c", 0],
    ]);
    const hulls = buildClusterHulls(nodes, comms);
    expect(hulls).toHaveLength(1);
    expect(hulls[0].polygon.length).toBeGreaterThanOrEqual(3);
  });

  it("assigns hues by centroid-x order so colour is layout-stable", () => {
    const nodes = [
      // Left cluster (centroid x ≈ 0)
      { key: "a", x: -100, y: 0, r: 8 },
      { key: "b", x: -90, y: 10, r: 8 },
      { key: "c", x: -80, y: -10, r: 8 },
      // Right cluster (centroid x ≈ 100)
      { key: "d", x: 80, y: 0, r: 8 },
      { key: "e", x: 90, y: 10, r: 8 },
      { key: "f", x: 100, y: -10, r: 8 },
    ];
    const comms = new Map<string, number>([
      ["a", 7],
      ["b", 7],
      ["c", 7],
      ["d", 2],
      ["e", 2],
      ["f", 2],
    ]);
    const hulls = buildClusterHulls(nodes, comms);
    expect(hulls).toHaveLength(2);
    // Sorted by centroidX, so the LEFT community (7) lands at index 0
    // regardless of its raw label id (2 vs 7). That stability is the
    // whole point of the centroid-sort.
    expect(hulls[0].community).toBe(7);
    expect(hulls[1].community).toBe(2);
  });
});

describe("polygonPath", () => {
  it("returns the empty string for fewer than 3 points", () => {
    expect(polygonPath([])).toBe("");
    expect(polygonPath([[0, 0]])).toBe("");
    expect(polygonPath([[0, 0], [1, 1]])).toBe("");
  });

  it("emits an SVG path that opens with M, joins with L, and closes with Z", () => {
    const d = polygonPath([
      [0, 0],
      [10, 0],
      [5, 10],
    ]);
    expect(d.startsWith("M")).toBe(true);
    expect(d).toContain("L");
    expect(d.endsWith("Z")).toBe(true);
  });
});
