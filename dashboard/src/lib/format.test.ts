import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import {
  clamp,
  formatDuration,
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

describe("formatDuration", () => {
  it("returns em-dash for negative / NaN", () => {
    expect(formatDuration(-1)).toBe("—");
    expect(formatDuration(Number.NaN)).toBe("—");
  });
  it("renders seconds under a minute", () => {
    expect(formatDuration(0)).toBe("0s");
    expect(formatDuration(42)).toBe("42s");
  });
  it("renders minutes under an hour", () => {
    expect(formatDuration(60)).toBe("1m");
    expect(formatDuration(59 * 60)).toBe("59m");
  });
  it("renders hours with minutes when present, hours-only when even", () => {
    expect(formatDuration(60 * 60)).toBe("1h");
    expect(formatDuration(60 * 60 + 60 * 7)).toBe("1h 07m");
    expect(formatDuration(21474)).toBe("5h 57m");
  });
  it("renders days with hours when present, days-only when even", () => {
    expect(formatDuration(60 * 60 * 24)).toBe("1d");
    expect(formatDuration(60 * 60 * 24 * 2 + 60 * 60 * 4)).toBe("2d 04h");
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
  it("falls back to the neutral indigo for null / undefined / empty", () => {
    expect(sessionHue(null)).toBe(275);
    expect(sessionHue(undefined)).toBe(275);
    expect(sessionHue("")).toBe(275);
  });
  it("is deterministic for the same session id", () => {
    expect(sessionHue("alpha")).toBe(sessionHue("alpha"));
  });
  it("snaps to the curated palette", () => {
    const hues = new Set(
      ["a", "b", "c", "d", "e", "f", "g", "h"].map((s) => sessionHue(s)),
    );
    expect(hues.size).toBeGreaterThan(1);
    const palette = new Set([275, 200, 175, 145, 95, 50, 320, 245]);
    for (const h of hues) {
      expect(palette.has(h)).toBe(true);
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
