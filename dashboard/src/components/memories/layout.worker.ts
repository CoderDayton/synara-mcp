/**
 * Layout Worker — settles the d3-force simulation off the main thread.
 *
 * For graphs in the dozens of nodes the synchronous tick(320) blocks
 * the main thread for noticeable frames; moving it to a Worker keeps
 * scroll, hover, and ReactFlow's pan/zoom responsive while the next
 * settled layout is computed in the background.
 *
 * Protocol:
 *   in : { id, data: GraphData, prev: [string, {x,y}][] }
 *   out: { id, positions: [string, {x,y}][] }
 *
 * Maps are sent as entry arrays (Worker postMessage uses structured
 * clone — Maps are supported in modern browsers but arrays are
 * cheaper and unambiguous across runtimes).
 */
/// <reference lib="webworker" />
import { computeLayout } from "./graph-layout";
import type { GraphData } from "@/lib/api";

interface InMessage {
  id: number;
  data: GraphData | undefined;
  prev: Array<[string, { x: number; y: number }]>;
}

interface OutMessage {
  id: number;
  positions: Array<[string, { x: number; y: number }]>;
}

self.onmessage = (e: MessageEvent<InMessage>) => {
  const { id, data, prev } = e.data;
  const settled = computeLayout(data, new Map(prev));
  const msg: OutMessage = { id, positions: Array.from(settled.entries()) };
  (self as unknown as DedicatedWorkerGlobalScope).postMessage(msg);
};

export {};
