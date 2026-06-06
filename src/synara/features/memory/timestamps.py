"""Canonical record-timestamp accessors.

Episodic and semantic records historically grew separate metadata keys
for the same two concepts:

    creation       episodic ``encoded_at``    semantic ``created_at``
    last retrieval episodic ``last_accessed``  semantic ``last_hit_at``

Writes now use the canonical names (``created_at`` / ``last_accessed``)
for *both* stores; these readers fall back to the legacy per-store names
so records written before the unification keep their timestamps -- no
data migration required. Genuinely distinct stamps (``updated_at``
mutation time, ``last_reconsolidated_at``, the ``access_history`` list)
are not handled here; they are not duplicates.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

CREATED_AT = "created_at"
LAST_ACCESSED = "last_accessed"
# Legacy per-store aliases, read-only fallback for pre-unification records.
_CREATED_LEGACY = "encoded_at"  # episodic
_LAST_ACCESSED_LEGACY = "last_hit_at"  # semantic


def _first_float(md: Mapping[str, Any] | None, *keys: str) -> float | None:
    md = md or {}
    for k in keys:
        v = md.get(k)
        if isinstance(v, (int, float)):
            return float(v)
    return None


def created_at(md: Mapping[str, Any] | None) -> float | None:
    """Creation time of a record (canonical ``created_at``, legacy ``encoded_at``)."""
    return _first_float(md, CREATED_AT, _CREATED_LEGACY)


def last_accessed(md: Mapping[str, Any] | None) -> float | None:
    """Last-retrieval time of a record (canonical ``last_accessed``, legacy ``last_hit_at``)."""
    return _first_float(md, LAST_ACCESSED, _LAST_ACCESSED_LEGACY)
