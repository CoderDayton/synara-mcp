import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import {
  clamp,
  relativeTime,
  sessionHue,
  shortSession,
} from "@/lib/format";

describe("clamp", () => {
  it("returns the value when inside the range", () => {
    expect(clamp(5, 0, 10)).toBe(5);
  });
  it("clamps to the lower bound", () => {
    expect(clamp(-3, 0, 10)).toBe(0);
  });
  it("clamps to the upper bound", () => {
    expect(clamp(99, 0, 10)).toBe(10);
  });
});

describe("relativeTime", () => {
  beforeEach(() => {
    // Fixed clock so the boundaries are deterministic.
    vi.useFakeTimers();
    vi.setSystemTime(new Date("2026-05-24T12:00:00Z"));
  });
  afterEach(() => {
    vi.useRealTimers();
  });

  const NOW = Math.floor(new Date("2026-05-24T12:00:00Z").getTime() / 1000);

  it("returns em-dash for falsy / non-positive epoch", () => {
    expect(relativeTime(0)).toBe("—");
    expect(relativeTime(-1)).toBe("—");
  });
  it("returns 'now' for future timestamps (clock skew)", () => {
    expect(relativeTime(NOW + 60)).toBe("now");
  });
  it("returns seconds for very recent past", () => {
    expect(relativeTime(NOW - 10)).toBe("10s");
  });
  it("rolls over to minutes at 45s", () => {
    expect(relativeTime(NOW - 45)).toBe("0m");
    expect(relativeTime(NOW - 60 * 5)).toBe("5m");
  });
  it("rolls over to hours at 60m", () => {
    expect(relativeTime(NOW - 60 * 60 * 3)).toBe("3h");
  });
  it("rolls over to days at 24h", () => {
    expect(relativeTime(NOW - 60 * 60 * 24 * 4)).toBe("4d");
  });
  it("rolls over to months at 30d", () => {
    expect(relativeTime(NOW - 60 * 60 * 24 * 90)).toBe("3mo");
  });
  it("rolls over to years at 12mo", () => {
    expect(relativeTime(NOW - 60 * 60 * 24 * 365 * 2)).toBe("2y");
  });
});

describe("sessionHue", () => {
  it("falls back to a neutral hue for null / undefined / empty", () => {
    expect(sessionHue(null)).toBe(286);
    expect(sessionHue(undefined)).toBe(286);
    expect(sessionHue("")).toBe(286);
  });
  it("is deterministic for the same session id", () => {
    expect(sessionHue("alpha")).toBe(sessionHue("alpha"));
  });
  it("disperses across the 0–359 hue circle", () => {
    const hues = new Set(
      ["a", "b", "c", "d", "e", "f", "g", "h"].map((s) => sessionHue(s)),
    );
    expect(hues.size).toBeGreaterThan(1);
    for (const h of hues) {
      expect(h).toBeGreaterThanOrEqual(0);
      expect(h).toBeLessThan(360);
    }
  });
});

describe("shortSession", () => {
  it("returns em-dash for falsy input", () => {
    expect(shortSession(null)).toBe("—");
    expect(shortSession(undefined)).toBe("—");
    expect(shortSession("")).toBe("—");
  });
  it("returns the original when short", () => {
    expect(shortSession("abc")).toBe("abc");
    expect(shortSession("123456789012")).toBe("123456789012");
  });
  it("truncates with ellipsis when longer than 12 chars", () => {
    expect(shortSession("1234567890123")).toBe("12345678901…");
  });
});
