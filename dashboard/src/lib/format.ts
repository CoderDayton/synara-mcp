/** Small presentational helpers shared by the memory views. */

export function clamp(n: number, lo: number, hi: number): number {
  return Math.max(lo, Math.min(hi, n));
}

/** Compact, human-readable duration from a seconds value.
 *  Examples: `42s`, `7m`, `1h 02m`, `2d 04h`, `3d`. Uses the two largest
 *  non-zero units so a quick glance gives both magnitude and precision. */
export function formatDuration(seconds: number): string {
  if (!Number.isFinite(seconds) || seconds < 0) return "—";
  const s = Math.floor(seconds);
  if (s < 60) return `${s}s`;
  const m = Math.floor(s / 60);
  if (m < 60) return `${m}m`;
  const h = Math.floor(m / 60);
  const rm = m % 60;
  if (h < 24) return rm > 0 ? `${h}h ${String(rm).padStart(2, "0")}m` : `${h}h`;
  const d = Math.floor(h / 24);
  const rh = h % 24;
  return rh > 0 ? `${d}d ${String(rh).padStart(2, "0")}h` : `${d}d`;
}

/** Compact relative time from a unix-seconds timestamp ("3m", "2h", "5d"). */
export function relativeTime(epochSeconds: number): string {
  if (!epochSeconds || epochSeconds <= 0) return "—";
  const deltaMs = Date.now() - epochSeconds * 1000;
  if (deltaMs < 0) return "now";
  const s = Math.floor(deltaMs / 1000);
  if (s < 45) return `${s}s`;
  const m = Math.floor(s / 60);
  if (m < 60) return `${m}m`;
  const h = Math.floor(m / 60);
  if (h < 24) return `${h}h`;
  const d = Math.floor(h / 24);
  if (d < 30) return `${d}d`;
  const mo = Math.floor(d / 30);
  if (mo < 12) return `${mo}mo`;
  return `${Math.floor(mo / 12)}y`;
}

/**
 * Deterministic hue for a session id. Sessions are a *context hint* in
 * the memory model, not a hard partition — colour-coding them lets the
 * map show state-dependent clustering without implying isolation.
 */
export function sessionHue(sessionId: string | null | undefined): number {
  if (!sessionId) return 286; // neutral violet-grey, matches the theme base
  let h = 2166136261;
  for (let i = 0; i < sessionId.length; i++) {
    h ^= sessionId.charCodeAt(i);
    h = Math.imul(h, 16777619);
  }
  return Math.abs(h) % 360;
}

export function shortSession(sessionId: string | null | undefined): string {
  if (!sessionId) return "—";
  return sessionId.length > 12 ? `${sessionId.slice(0, 11)}…` : sessionId;
}
