"""Single source of truth for how the vector store is opened.

Every process that touches the database — the MCP server, the dashboard,
and the offline maintenance scripts — must open it with *identical*
index parameters. They are not cosmetic: ``quantization`` is baked into
the on-disk usearch index, and ``store_embeddings`` decides whether the
canonical float32 copy exists in SQLite at all.

This was learned the expensive way. ``scripts/reembed.py`` originally
opened the store with the library defaults instead of the server's
``Quantization.INT8``, which silently rewrote the episodic index from
~0.9 KB/vector to ~3.2 KB/vector and left it inconsistent with the
server's expectations. Constants that must agree across processes belong
in one place, so opening the store goes through here.
"""

from __future__ import annotations

from pathlib import Path

from simplevecdb import AsyncVectorDB, Quantization

# Scalar quantization of the on-disk usearch index. INT8 costs ~1 byte
# per dimension instead of 4 and is lossy: reopening an INT8 index
# without this flag does not merely change new writes, it changes how
# the existing file is interpreted.
VECTOR_QUANTIZATION = Quantization.INT8

# Keep the full-precision vectors in SQLite alongside the index. The
# index is a *derived* artefact: ``rebuild_index()`` reconstructs it from
# this column, and ``cluster()`` reads it directly. Without it, a
# corrupted or dimension-changed index is unrecoverable.
STORE_EMBEDDINGS = True

# ``:memory:`` is simplevecdb's ephemeral-store sentinel, not a path.
MEMORY_DB_PATH = ":memory:"


def open_database(db_path: str) -> AsyncVectorDB:
    """Open the vector store with the parameters every process shares.

    Creates the parent directory for a file-backed store so a fresh
    install does not fail on a missing ``$XDG_DATA_HOME`` subdirectory.
    """
    if db_path != MEMORY_DB_PATH:
        Path(db_path).resolve().parent.mkdir(parents=True, exist_ok=True)
    return AsyncVectorDB(db_path, quantization=VECTOR_QUANTIZATION)


def sidecar_paths(db_path: Path) -> list[Path]:
    """Every file that makes up the store alongside the SQLite database.

    The usearch index lives in its own file per collection, so copying
    only the ``.db`` produces a backup that restores to an index out of
    sync with its catalog. SQLite's own ``-wal``/``-shm`` companions are
    included for the same reason: a copy taken mid-transaction without
    them loses the tail of the write-ahead log.
    """
    parent = db_path.parent
    names = [f"{db_path.name}{suffix}" for suffix in ("-wal", "-shm")]
    found = [parent / name for name in names]
    return [p for p in found if p.exists()] + sorted(parent.glob(f"{db_path.name}.*.usearch"))
