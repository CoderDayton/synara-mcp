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

Segments are exact contiguous slices of the input — each sentence keeps
its trailing separator — so whenever a split happens,
``"".join(segments) == content`` byte-for-byte. ``get_episode`` relies on
this invariant to reassemble the original text losslessly.
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
    Otherwise returns 2..max_items exact contiguous slices of ``content``
    whose concatenation reproduces it unchanged. List is never empty.
    """
    if max_chars <= 0 or max_items <= 1:
        return [content]
    stripped = content.strip()
    if not stripped or len(stripped) <= max_chars:
        return [content]

    pieces = _sentence_pieces(content)
    if not pieces:
        return [content]

    segments: list[str] = []
    buf = ""
    for piece in pieces:
        if len(piece) > max_chars:
            if buf:
                segments.append(buf)
                buf = ""
            segments.extend(_window(piece, max_chars))
            continue
        if buf and len(buf) + len(piece) > max_chars:
            segments.append(buf)
            buf = piece
        else:
            buf += piece
    if buf:
        segments.append(buf)

    if not segments:
        return [content]
    if len(segments) > max_items:
        head = segments[: max_items - 1]
        head.append("".join(segments[max_items - 1 :]))
        segments = head
    return segments


def _sentence_pieces(content: str) -> list[str]:
    """Sentence pieces as exact slices: each piece runs from the end of
    the previous boundary match through its own trailing separator, so
    no character is dropped or rewritten and ``"".join(pieces)`` always
    reproduces ``content``."""
    pieces: list[str] = []
    start = 0
    for m in _SENTENCE_BOUNDARY.finditer(content):
        pieces.append(content[start : m.end()])
        start = m.end()
    if start < len(content):
        pieces.append(content[start:])
    return pieces


def _window(text: str, size: int) -> list[str]:
    return [text[i : i + size] for i in range(0, len(text), size)]
