"""Per-tool runtime metrics.

In-process, in-memory counters + a small rolling latency window for the
MCP tool surface. Used by the dashboard to turn the static tool list
into live operational telemetry (call counts, last-called age, p50/p95).

Scope is deliberately tiny: no persistence, no histograms, no labels
beyond the tool name. The collector lives for the lifetime of the
server process and is shared between the FastMCP tool handlers (which
call :meth:`record`) and the dashboard's ``/api/tool-metrics`` route
(which calls :meth:`snapshot`). Both sides receive the same instance —
created in :mod:`synara.server` — so divergence is impossible.

Thread-safety: today both writers (tool wrappers) and the reader
(dashboard route) live on the same asyncio loop, so a lock is not
strictly required. The :class:`threading.Lock` is kept as cheap
future-proofing in case a tool ever offloads work to a thread pool
(``asyncio.to_thread``) and records from there — contention stays
negligible (one record per tool call, one snapshot per dashboard poll).
"""

from __future__ import annotations

import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any

__all__ = ["ToolMetrics", "ToolSnapshot"]


_DEFAULT_WINDOW = 200


@dataclass(slots=True)
class _ToolStats:
    headline: str
    count: int = 0
    error_count: int = 0
    last_called_at: float | None = None
    last_duration_seconds: float | None = None
    durations: deque[float] = field(default_factory=lambda: deque(maxlen=_DEFAULT_WINDOW))


@dataclass(frozen=True, slots=True)
class ToolSnapshot:
    """One tool's serialisable telemetry row."""

    name: str
    headline: str
    count: int
    error_count: int
    last_called_at: float | None
    last_duration_seconds: float | None
    p50_ms: float | None
    p95_ms: float | None

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "headline": self.headline,
            "count": self.count,
            "error_count": self.error_count,
            "last_called_at": self.last_called_at,
            "last_duration_seconds": self.last_duration_seconds,
            "p50_ms": self.p50_ms,
            "p95_ms": self.p95_ms,
        }


def _quantile(sorted_samples: list[float], q: float) -> float:
    """Nearest-rank quantile in seconds. ``sorted_samples`` must be non-empty."""
    if not sorted_samples:
        raise ValueError("quantile of empty sample")
    n = len(sorted_samples)
    # Nearest-rank: rank = ceil(q * n); clamp to [1, n]; index = rank - 1.
    rank = max(1, min(n, int(-(-q * n // 1))))
    return sorted_samples[rank - 1]


class ToolMetrics:
    """Per-tool counter + rolling latency window.

    A tool must be :meth:`declare`'d before snapshot includes it, so the
    UI can render the full surface (including never-called tools) on the
    very first poll instead of an empty list.
    """

    __slots__ = ("_lock", "_tools", "_window")

    def __init__(self, *, window_size: int = _DEFAULT_WINDOW) -> None:
        if window_size <= 0:
            raise ValueError("window_size must be positive")
        self._window = window_size
        self._lock = threading.Lock()
        self._tools: dict[str, _ToolStats] = {}

    def declare(self, name: str, *, headline: str) -> None:
        """Register a tool's display headline. Idempotent: re-declare updates the headline."""
        with self._lock:
            stats = self._tools.get(name)
            if stats is None:
                self._tools[name] = _ToolStats(
                    headline=headline,
                    durations=deque(maxlen=self._window),
                )
            else:
                stats.headline = headline

    def record(self, name: str, duration_seconds: float, *, ok: bool) -> None:
        """Record one tool invocation. Late-declare on first call if needed."""
        with self._lock:
            stats = self._tools.get(name)
            if stats is None:
                stats = _ToolStats(
                    headline=name,
                    durations=deque(maxlen=self._window),
                )
                self._tools[name] = stats
            stats.count += 1
            if not ok:
                stats.error_count += 1
            stats.last_called_at = time.time()
            stats.last_duration_seconds = duration_seconds
            stats.durations.append(duration_seconds)

    def snapshot(self) -> list[ToolSnapshot]:
        """Point-in-time snapshot, sorted by declared order then name."""
        with self._lock:
            rows: list[ToolSnapshot] = []
            for name, s in self._tools.items():
                if s.durations:
                    samples = sorted(s.durations)
                    p50 = _quantile(samples, 0.50) * 1000.0
                    p95 = _quantile(samples, 0.95) * 1000.0
                else:
                    p50 = None
                    p95 = None
                rows.append(
                    ToolSnapshot(
                        name=name,
                        headline=s.headline,
                        count=s.count,
                        error_count=s.error_count,
                        last_called_at=s.last_called_at,
                        last_duration_seconds=s.last_duration_seconds,
                        p50_ms=p50,
                        p95_ms=p95,
                    ),
                )
            return rows
