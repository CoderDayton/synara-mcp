"""Theta-segmented intra-episode encoding (Lisman & Jensen 2013).

Theta-gamma phase coding packs ~7 ordinal items per theta cycle in the
hippocampus. For long content we mirror that capacity by splitting input
into ordered sub-records sharing an ``episode_group_id``. Each sub-record
is a normal episode that carries its position; recall returns
sub-records as usual and callers that want a contiguous walk can group
results by ``episode_group_id`` and sort by ``position_in_episode``.

The splitter is deliberately dependency-free. Sentence boundaries take
precedence; a single sentence longer than the per-segment budget is
character-windowed; the segment count is capped at ``max_items`` by
folding overflow into the final bucket so no content is dropped.
"""

from __future__ import annotations

import re

# Sentence-end heuristic: punctuation followed by whitespace and a
# capital/quote/paren start. Imperfect on abbreviations, but the
# downstream embedder treats sub-records as semantic units so the
# occasional split inside "U.S. Navy" is harmless.
_SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?])\s+(?=[A-Z(\"'])")


def split_into_segments(
    content: str,
    *,
    max_chars: int,
    max_items: int,
) -> list[str]:
    """Split content into ordered segments, respecting sentence boundaries.

    Returns [content] if splitting is disabled or content fits budget.
    Otherwise returns 2..max_items segments. Concatenation preserves
    original content (modulo whitespace). List is never empty.
    """
    if max_chars <= 0 or max_items <= 1:
        return [content]
    stripped = content.strip()
    if not stripped or len(stripped) <= max_chars:
        return [content]

    sentences = [s.strip() for s in _SENTENCE_BOUNDARY.split(stripped) if s.strip()]
    if not sentences:
        return [content]

    segments: list[str] = []
    buf = ""
    for sent in sentences:
        if len(sent) > max_chars:
            if buf:
                segments.append(buf)
                buf = ""
            segments.extend(_window(sent, max_chars))
            continue
        candidate = f"{buf} {sent}".strip() if buf else sent
        if len(candidate) > max_chars and buf:
            segments.append(buf)
            buf = sent
        else:
            buf = candidate
    if buf:
        segments.append(buf)

    if not segments:
        return [content]
    if len(segments) > max_items:
        head = segments[: max_items - 1]
        tail = " ".join(segments[max_items - 1 :])
        head.append(tail)
        segments = head
    return segments


def _window(text: str, size: int) -> list[str]:
    return [text[i : i + size] for i in range(0, len(text), size)]
