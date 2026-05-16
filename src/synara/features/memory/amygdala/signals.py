"""Content-shape signal extraction for auto-filling episode metadata.

Pure function: given a raw episode content string, returns a dict of
co-acting structural signal flags plus extracted references.

The librarian model
-------------------
Callers (LLM agents) dump unstructured content into the memory store.
The memory store extracts what the *shape* of the content already
implies — diff markers, traceback patterns, decision verbs, code
density, file paths, symbol mentions — without making the caller
declare an event ``kind``. Signals are deliberately **overlapping**
rather than mutually exclusive categories: a single "fix" event can
fire both ``has_traceback`` and ``has_diff_markers``, which is exactly
the most learnable shape (failure + resolution in one record).

Downstream uses
---------------
* ``encode_episode`` merges the signal dict into episode metadata so
  recall filters can pivot on ``has_traceback`` or ``references``.
* Auto-salience (gated behind a config flag) sums weighted flags into
  a default salience when the caller omits one — failures and decision
  records float to the top without per-call hyperparameter tuning.
* Consolidation can label semantic schemas by the *modal* signal
  profile of their member episodes, so "strategy" schemas derived from
  clusters of failures emerge automatically.

This module is intentionally regex-only: no embedder, no LLM, no
network, no state. It is fast enough to run on every encode call and
its behavior is fully covered by tests.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any, TypedDict


class SignalDict(TypedDict):
    """Structural shape of the dict returned by :func:`derive_signals`."""

    has_diff_markers: bool
    has_traceback: bool
    has_decision_verbs: bool
    has_tool_call: bool
    has_question: bool
    has_structured_doc: bool
    code_block_count: int
    references: list[str]
    length_class: str
    length_chars: int


# --- Shape signals -----------------------------------------------------

# Unified-diff markers and ``diff --git`` headers — anchored to line
# start to avoid false positives inside prose mentioning ``---``.
_DIFF_RE = re.compile(r"^(?:---\s|\+\+\+\s|@@|diff --git)", re.MULTILINE)

# Common Python/JS/Rust/Go exception headers. Conservative on purpose —
# free-form prose like "this is an error" must not match.
_TRACEBACK_RE = re.compile(
    r"\bTraceback \(most recent call last\)"
    r"|\b[A-Z][A-Za-z]*Error:"
    r"|\b[A-Z][A-Za-z]*Exception:"
    r"|^panic:"
    r"|\bpanicked at\b",
    re.MULTILINE,
)

# Modal/imperative verbs that mark a decision being recorded. Matched
# case-insensitively but only on word boundaries.
_DECISION_RE = re.compile(
    r"\b(?:decided to|going with|chose to|switching to|instead of|"
    r"opted for|will use|let's use|we'll use)\b",
    re.IGNORECASE,
)

# JSON-shaped tool-call payload, or an explicit ``tool_call`` marker.
_TOOL_CALL_RE = re.compile(
    r'"(?:name|tool|function)"\s*:\s*"[A-Za-z_][\w.-]*"'
    r"|\btool[_ ]call\b",
    re.IGNORECASE,
)

# Fenced code block openers/closers; count // 2 yields full blocks.
_CODE_FENCE_RE = re.compile(r"^```[\w+-]*\s*$", re.MULTILINE)

# Sentence-ending question mark — ``?`` followed by whitespace or end
# of string. Conservative to avoid matching nullable-type syntax like
# ``String?`` or shell globs.
_QUESTION_RE = re.compile(r"\?(?=\s|$)", re.MULTILINE)

# Markdown-ish structural cues: ATX headers, bullet lists, or ordered
# lists. The required `[ \t]+\S` tail means unified-diff lines like
# ``--- a/foo.py`` (second char is ``-``, not whitespace) cannot match.
_STRUCTURED_DOC_RE = re.compile(
    r"^(?:#{1,6}[ \t]+\S|[-*+][ \t]+\S|\d+\.[ \t]+\S)",
    re.MULTILINE,
)


# --- Reference extraction ----------------------------------------------

# File-path-like tokens with a recognised source/data extension. The
# extension list is deliberately narrow to keep the recall filter
# useful — adding "txt"/"log" turns this into a noise generator.
_FILE_EXT = (
    "py|pyi|ts|tsx|js|jsx|rs|go|java|kt|swift|c|cc|cpp|cxx|h|hpp"
    "|md|rst|toml|yaml|yml|json|jsonl|sh|bash|zsh|sql|proto|graphql"
)
_FILE_PATH_RE = re.compile(rf"\b[\w./\\-]+\.(?:{_FILE_EXT})\b")

# Identifiers inside backticks — the conventional "this is a symbol"
# marker in markdown-ish content. Captures up to dots/slashes so
# qualified names like ``foo.bar.baz`` survive intact.
_BACKTICK_IDENT_RE = re.compile(r"`([A-Za-z_][A-Za-z0-9_./-]{0,127})`")

# CamelCase identifiers outside backticks. Requires at least one
# lower-then-upper transition so all-caps acronyms (``HTTP``) and
# single-word capitalised English (``The``) are excluded.
_CAMEL_CASE_RE = re.compile(r"\b([A-Z][a-z]+(?:[A-Z][A-Za-z0-9]*)+)\b")

# CamelCase-shaped words that show up frequently in prose and are not
# actually project symbols. Keep narrow — caller-specific noise belongs
# in config, not here.
_CAMEL_NOISE = frozenset(
    {
        "JavaScript",
        "TypeScript",
        "NodeJs",
        "NodeJS",
        "MacOS",
        "MacOs",
        "PostgreSQL",
        "MySQL",
        "GitHub",
        "GitLab",
        "OpenAI",
    }
)

# Bare URLs. Greedy match through the body; sentence-trailing
# punctuation is stripped by :func:`_clean_url`.
_URL_RE = re.compile(r"https?://[^\s<>'\"`()]+", re.IGNORECASE)

# Issue / PR references — ``#123`` or project-prefixed tracker IDs
# like ``GH-12`` or ``JIRA-1234``. Two-digit minimum on the numeric
# tail filters out incidental tokens like ``UTF-8``.
_ISSUE_REF_RE = re.compile(r"(?<![A-Za-z0-9_])(?:#\d+|[A-Z]{2,}-\d{2,})(?![A-Za-z0-9_])")

# Trailing punctuation we strip off matched URLs so ``see https://x.com.``
# yields ``https://x.com`` not ``https://x.com.``.
_URL_TRAIL = ".,;:!?)]}"


def _clean_url(url: str) -> str:
    while url and url[-1] in _URL_TRAIL:
        url = url[:-1]
    return url


# --- Length classification ---------------------------------------------


def _strip_diff_prefix(path: str) -> str:
    """Drop the universal ``a/`` or ``b/`` prefix from unified-diff paths."""
    if path.startswith(("a/", "b/")):
        return path[2:]
    return path


# Length-class bucket boundaries (inclusive lower, exclusive upper).
# Short ~ conversational; medium ~ paragraph/tool-call; long ~ diff,
# stack trace, or multi-paragraph note.
_SHORT_MAX_CHARS = 200
_MEDIUM_MAX_CHARS = 1500


def _length_class(n: int) -> str:
    if n < _SHORT_MAX_CHARS:
        return "short"
    if n < _MEDIUM_MAX_CHARS:
        return "medium"
    return "long"


# --- Public surface ----------------------------------------------------


def derive_signals(content: str) -> SignalDict:
    """Extract structural signals from raw episode content.

    Returns a dict suitable for direct merge into episode metadata. All
    boolean flags default to False, ``references`` to ``[]``, so the
    return value is always populated even for empty content.
    """
    text = content or ""
    text_len = len(text)
    file_paths = {_strip_diff_prefix(m.group(0)) for m in _FILE_PATH_RE.finditer(text)}
    bt_idents = {m.group(1) for m in _BACKTICK_IDENT_RE.finditer(text)}
    camel_idents = {
        m.group(1) for m in _CAMEL_CASE_RE.finditer(text) if m.group(1) not in _CAMEL_NOISE
    }
    urls = {_clean_url(m.group(0)) for m in _URL_RE.finditer(text)}
    issue_refs = {m.group(0) for m in _ISSUE_REF_RE.finditer(text)}
    references = sorted(file_paths | bt_idents | camel_idents | urls | issue_refs)

    return {
        "has_diff_markers": bool(_DIFF_RE.search(text)),
        "has_traceback": bool(_TRACEBACK_RE.search(text)),
        "has_decision_verbs": bool(_DECISION_RE.search(text)),
        "has_tool_call": bool(_TOOL_CALL_RE.search(text)),
        "has_question": bool(_QUESTION_RE.search(text)),
        "has_structured_doc": bool(_STRUCTURED_DOC_RE.search(text)),
        "code_block_count": len(_CODE_FENCE_RE.findall(text)) // 2,
        "references": references,
        "length_class": _length_class(text_len),
        "length_chars": text_len,
    }


# Weights for derived salience. Each fires additively on top of a base
# salience supplied by ``derive_salience``. Tuned so that a typical
# turn (no flags, short, no refs) sits near the base while a failure-
# plus-diff record saturates at the ceiling. Values are exposed so
# callers / tests can introspect rather than mining magic numbers.
SALIENCE_WEIGHTS: dict[str, float] = {
    "has_traceback": 0.30,
    "has_diff_markers": 0.20,
    "has_decision_verbs": 0.15,
    "has_tool_call": 0.05,
    "has_question": 0.05,
    "reference_density": 0.20,  # applied if len(references) >= 3
    "long_content": 0.05,
}

# Hot-path lookup: pre-bound (key, weight) pairs so the per-call loop
# avoids repeated dict indirection. Adding a flag means adding a key
# here and to ``SALIENCE_WEIGHTS`` — kept side-by-side on purpose.
_FLAG_CONTRIB: tuple[tuple[str, float], ...] = tuple(
    (k, SALIENCE_WEIGHTS[k])
    for k in (
        "has_traceback",
        "has_diff_markers",
        "has_decision_verbs",
        "has_tool_call",
        "has_question",
    )
)
_REF_DENSITY_W = SALIENCE_WEIGHTS["reference_density"]
_LONG_W = SALIENCE_WEIGHTS["long_content"]

# Threshold at which an episode's reference list contributes the
# reference-density weight to derived salience.
_REF_DENSITY_THRESHOLD = 3


def derive_salience(signals: Mapping[str, Any], *, base: float = 0.3) -> float:
    """Compose a default salience from a signal dict.

    The function is intentionally additive and pure — composing it
    with a session-relative z-score (the recommended next step) is a
    *separate* concern that lives in the encode pipeline, not here.
    """
    s = float(base)
    for key, weight in _FLAG_CONTRIB:
        if signals.get(key):
            s += weight
    refs = signals.get("references") or []
    if isinstance(refs, list) and len(refs) >= _REF_DENSITY_THRESHOLD:
        s += _REF_DENSITY_W
    if signals.get("length_class") == "long":
        s += _LONG_W
    if s < 0.0:
        return 0.0
    if s > 1.0:
        return 1.0
    return s


# --- Signal registry (extensibility) -----------------------------------
#
# The hardcoded ``derive_signals`` / ``SALIENCE_WEIGHTS`` pair stays the
# default. Callers that want to add a signal without forking encode.py
# can plug a :class:`SignalRegistry` into ``MemoryConfig`` —
# encode.py routes through it when set, otherwise the legacy path runs
# unchanged.


@dataclass(frozen=True, slots=True)
class SignalSpec:
    """One extensibility entry.

    name: stable key written into episode metadata when ``compute``
        returns a truthy value (booleans stored as-is, numbers stored
        when non-zero).
    weight: contribution to derived salience when the signal "fires"
        (boolean True, or numeric > 0).
    compute: callable producing the signal value from raw content. Must
        be a *pure* synchronous function — runs on every encode.
    """

    name: str
    weight: float
    compute: Callable[[str], Any]


@dataclass(frozen=True, slots=True)
class SignalRegistry:
    """Ordered, immutable bundle of :class:`SignalSpec` entries.

    Construct via :func:`default_signal_registry` or directly. Adding a
    custom signal is one line at the call site, not a five-file edit.
    """

    specs: tuple[SignalSpec, ...] = ()
    base_salience: float = 0.3
    reference_density_threshold: int = _REF_DENSITY_THRESHOLD
    reference_density_weight: float = _REF_DENSITY_W
    long_content_weight: float = _LONG_W
    include_legacy_structural: bool = True

    def derive(self, content: str) -> dict[str, Any]:
        """Run every registered signal + the legacy structural pass.

        Always returns a dict that is safe to merge into episode
        metadata. The legacy fields (``has_diff_markers``, ``references``,
        ...) are included when ``include_legacy_structural`` is True so
        downstream filters keep working.
        """
        out: dict[str, Any] = {}
        if self.include_legacy_structural:
            out.update(derive_signals(content))
        for spec in self.specs:
            value = spec.compute(content)
            if isinstance(value, bool) or value:
                out[spec.name] = value
        return out

    def salience(self, signals: Mapping[str, Any]) -> float:
        """Compose a salience score from a derived-signals dict."""
        s = float(self.base_salience)
        if self.include_legacy_structural:
            for key, weight in _FLAG_CONTRIB:
                if signals.get(key):
                    s += weight
            refs = signals.get("references") or []
            if isinstance(refs, list) and len(refs) >= self.reference_density_threshold:
                s += self.reference_density_weight
            if signals.get("length_class") == "long":
                s += self.long_content_weight
        for spec in self.specs:
            value = signals.get(spec.name)
            if isinstance(value, bool):
                if value:
                    s += spec.weight
            elif isinstance(value, (int, float)) and value > 0:
                s += spec.weight
        return max(0.0, min(1.0, s))


def default_signal_registry() -> SignalRegistry:
    """Registry that reproduces the legacy hardcoded behaviour exactly."""
    return SignalRegistry(specs=())


__all__ = [
    "SALIENCE_WEIGHTS",
    "SignalDict",
    "SignalRegistry",
    "SignalSpec",
    "default_signal_registry",
    "derive_salience",
    "derive_signals",
]
