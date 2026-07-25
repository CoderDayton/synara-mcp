"""Miss diagnostics for the recall tools.

A recall that finds nothing used to return a bare ``[]``. Measured over
three days of real transcripts, 5 of 17 recalls (29%) ended that way —
``recall_semantic_memory`` worst at 3 of 8 — and an empty array says
nothing about *why*: whether the store is empty, whether the session
scope excluded otherwise-matching records, whether a tag or kind filter
did, or whether the query simply has no near neighbours. Each of those
has a different fix, and the caller could not tell them apart, so the
usual next move was to stop using memory.

This module carries the counters the recall pipeline observes on its way
to an empty result (:class:`RecallDiagnostics`) and turns them into a
report naming the dominant cause and the specific retry that would
address it (:func:`build_miss_report`).

Layering: the dataclass is written by ``hippocampus.recall`` and read by
``tools``, so it lives here at the plumbing level (alongside
``memory_types`` / ``timestamps``) rather than in either — the tool
surface must not import a brain region.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

# Cosine distances are reported to this many decimals. The raw float
# carries ~16 significant digits of noise that no caller can act on.
_DISTANCE_PRECISION = 4


@dataclass(slots=True)
class RecallDiagnostics:
    """Counters recorded by the recall pipeline as candidates are filtered.

    Passed down as an optional out-parameter so the happy path pays
    nothing: when a caller does not ask for diagnostics the pipeline
    never allocates or updates one, and when it does, every field is
    already computed for other reasons (store counts come from the
    fetch's own guard, drop counts from list lengths either side of a
    filter that already runs).
    """

    #: Documents searchable under this call's ``mode`` — the sum of the
    #: legs actually queried, not the whole database.
    stored: int = 0
    #: Raw neighbours the vector store(s) returned, before any filter.
    candidates_scanned: int = 0
    #: Best (smallest) cosine distance among those raw neighbours.
    nearest_distance: float | None = None
    dropped_by_scope: int = 0
    dropped_by_tags: int = 0
    dropped_by_kind: int = 0
    dropped_by_gate: int = 0

    def note_candidates(self, distances: list[float]) -> None:
        """Record the raw candidate set produced by the vector fetch."""
        self.candidates_scanned = len(distances)
        # A degenerate stored vector yields a NaN distance, and NaN
        # poisons min() by comparing false against everything — filter
        # before reducing so the reported nearest is a real distance.
        finite = [d for d in distances if not math.isnan(d)]
        self.nearest_distance = min(finite) if finite else None

    @property
    def dropped_total(self) -> int:
        return (
            self.dropped_by_scope
            + self.dropped_by_tags
            + self.dropped_by_kind
            + self.dropped_by_gate
        )


@dataclass(frozen=True, slots=True)
class RecallRequest:
    """The call parameters echoed back in a miss report.

    Echoing them is not padding: an agent that reads ``scope_session:
    true`` in the report can see the constraint it did not realise it
    had set, which is precisely the state that produces a confusing
    empty result.
    """

    query: str
    k: int
    mode: str | None = None
    session_id: str | None = None
    scope_session: bool | None = None
    tags: list[str] | None = None
    kind: str | None = None

    def as_scope(self) -> dict[str, Any]:
        scope: dict[str, Any] = {"k": self.k, "session_id": self.session_id}
        if self.mode is not None:
            scope["mode"] = self.mode
        scope["scope_session"] = self.scope_session
        if self.tags:
            scope["tags"] = self.tags
        if self.kind is not None:
            scope["kind"] = self.kind
        return scope


def _round_distance(value: float | None) -> float | None:
    return None if value is None else round(value, _DISTANCE_PRECISION)


def _empty_store_reason(*, semantic_only: bool) -> str:
    if semantic_only:
        return (
            "The semantic store is empty — no schemas have been consolidated and no "
            "semantic memory has been written yet."
        )
    return "The memory store is empty — nothing has been stored in this database yet."


def _suggestions(diag: RecallDiagnostics, request: RecallRequest) -> list[str]:
    """Retry advice derived from the counters, most actionable first.

    Generated from the observed numbers rather than emitted as a fixed
    list, so a suggestion only ever appears when the state it addresses
    actually occurred.
    """
    out: list[str] = []
    if diag.dropped_by_scope:
        out.append(
            f"{diag.dropped_by_scope} candidate(s) matched but belong to other sessions. "
            "Retry with scope_session=false to search across sessions."
        )
    if diag.dropped_by_tags:
        out.append(
            f"{diag.dropped_by_tags} candidate(s) matched but do not carry every tag in "
            f"{request.tags!r}. Retry without tags, or with fewer of them."
        )
    if diag.dropped_by_kind:
        out.append(
            f"{diag.dropped_by_kind} candidate(s) matched but are not kind={request.kind!r}. "
            "Retry without kind to search all kinds."
        )
    if diag.dropped_by_gate:
        out.append(
            f"{diag.dropped_by_gate} candidate(s) ranked but fell below the relevance gate "
            f"(nearest distance {_round_distance(diag.nearest_distance)}). Rephrase with more "
            "distinctive terms — raising k will not recover them."
        )
    if not out:
        out.append(
            "No near neighbours were found. Try fewer, more distinctive keywords, or a "
            "phrase closer to how the memory would have been written."
        )
    return out


def _reason(diag: RecallDiagnostics, request: RecallRequest, *, semantic_only: bool) -> str:
    """One-sentence dominant cause, so a caller can act without parsing counters."""
    if request.k <= 0:
        # The pipeline short-circuits on k <= 0 before it searches
        # anything, so every counter is still zero. Say that, rather than
        # let the all-zero state read as "the store is empty".
        return f"k={request.k} requested no results, so no search was performed."
    if diag.stored == 0:
        return _empty_store_reason(semantic_only=semantic_only)
    causes = (
        (diag.dropped_by_scope, f"session scoping (session_id={request.session_id!r})"),
        (diag.dropped_by_tags, f"the tags filter ({request.tags!r})"),
        (diag.dropped_by_kind, f"the kind filter ({request.kind!r})"),
        (diag.dropped_by_gate, "the relevance gate"),
    )
    count, label = max(causes, key=lambda c: c[0])
    if count == 0:
        return (
            f"No candidate was close enough to the query among {diag.stored} stored "
            "record(s); the vector search returned no neighbours."
        )
    return (
        f"{count} of {diag.candidates_scanned} candidate(s) were excluded by {label}, "
        "leaving nothing to return."
    )


def build_miss_report(
    diag: RecallDiagnostics,
    request: RecallRequest,
    *,
    semantic_only: bool = False,
    episodic_fallback: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Shape an empty recall into an actionable report.

    Returned in place of the empty list, so the two outcomes are
    unmistakable: a hit set is a JSON array, a miss is an object. The
    object still carries ``results: []`` so a caller that normalises with
    ``r if isinstance(r, list) else r["results"]`` needs no other branch.

    ``episodic_fallback`` carries raw episodic traces found for the same
    query when the semantic store had nothing. They are kept under their
    own key rather than merged into ``results``: a raw trace is not a
    distilled fact, and a caller asking for semantic memory must be able
    to tell which it received.
    """
    miss: dict[str, Any] = {
        "reason": _reason(diag, request, semantic_only=semantic_only),
        "scope": request.as_scope(),
        "searched": {
            "stored": diag.stored,
            "candidates_scanned": diag.candidates_scanned,
            "nearest_distance": _round_distance(diag.nearest_distance),
            "dropped_by_scope": diag.dropped_by_scope,
            "dropped_by_tags": diag.dropped_by_tags,
            "dropped_by_kind": diag.dropped_by_kind,
            "dropped_by_gate": diag.dropped_by_gate,
        },
        # No retry is worth suggesting when nothing was searched: an
        # empty store has nothing to relax, and k <= 0 asked for nothing.
        "suggestions": _suggestions(diag, request) if diag.stored and request.k > 0 else [],
    }
    report: dict[str, Any] = {"results": [], "miss": miss}
    if episodic_fallback:
        miss["suggestions"] = [
            f"The semantic store had no match, but {len(episodic_fallback)} raw episodic "
            "trace(s) did — see episodic_fallback. Use recall_episodes for the full set.",
            *miss["suggestions"],
        ]
        report["episodic_fallback"] = episodic_fallback
    return report


__all__ = [
    "RecallDiagnostics",
    "RecallRequest",
    "build_miss_report",
]
