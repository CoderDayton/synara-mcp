#!/usr/bin/env python3
"""Re-embed a synara store after an embedding-model change.

Vectors produced by one model are meaningless to another, so switching
models silently breaks recall until every stored record is re-encoded.
The stored text is intact, so this rewrites the vectors from it.

Design constraints, all load-bearing:

* **Update in place, keyed by id.** The successor-representation and
  plasticity graphs reference episode ids. Re-creating the collections
  would renumber them and orphan the whole relational structure, which
  is the part of the store that cannot be rebuilt from text.

* **Reuse the production embedder.** Retrieval models are trained with
  asymmetric task prefixes; this must apply exactly the same document
  prefix that ``store_episode`` will, or every stored vector sits in a
  slightly different space from the queries that search it.

* **Write SQLite, then rebuild the index from it.** This is the subtle
  one, and getting it wrong is silent. simplevecdb keeps vectors twice:
  the canonical float32 copy in the catalog's ``embedding`` column, and
  the usearch HNSW index derived from it. ``update_embedding()`` writes
  *neither* — it appends to a pending buffer that ``pending.flush()``
  later promotes into the index alone, leaving the canonical column
  holding the pre-migration vectors. Recall looks correct immediately
  afterwards, because recall reads the index. Then the first
  ``rebuild_if_needed()`` (auto-consolidation calls it) rebuilds the
  index *from the canonical column* and the store reverts to the old
  model, at which point recall returns confident nonsense with no error
  anywhere. So: write the canonical column first, drop any stale pending
  buffer, and let ``rebuild_index()`` derive the index from it. The two
  copies are then identical by construction rather than by luck.

* **Open the store exactly as the server does.** Index parameters come
  from ``synara.storage``; quantization is baked into the on-disk index,
  so opening with library defaults rewrites it into a shape the server
  did not intend.

Usage::

    uv run --frozen python scripts/reembed.py --dry-run
    uv run --frozen python scripts/reembed.py
    uv run --frozen python scripts/reembed.py --verify-only

The run is idempotent and resumable: re-running rewrites the same rows
to the same values. A timestamped backup of the database and every
sidecar is taken before the first write unless ``--no-backup`` is given.
"""

from __future__ import annotations

import argparse
import asyncio
import math
import random
import shutil
import sqlite3
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from synara.config import Settings
from synara.coordination.election import probe_leader_dead
from synara.features.embedding import build_embedder
from synara.features.memory.config import MemoryConfig
from synara.features.memory.memory_types import default_registry
from synara.storage import STORE_EMBEDDINGS, open_database, sidecar_paths

# Rewriting one collection at a time in modest batches keeps peak memory
# bounded and makes an interrupted run resumable (already-rewritten rows
# are simply re-written on the next pass).
_BATCH = 32

# An embedder batch response must be a (rows, dim) matrix.
_MATRIX_NDIM = 2

# How many documents the post-run verification re-embeds and re-searches.
# Every row is checked for storage/index agreement; this smaller sample
# additionally pays for a fresh forward pass to prove the stored vector
# is what *this* model produces for that text.
_VERIFY_SAMPLE = 24

# A round-trip search for a document's own stored vector must return that
# document at essentially zero distance. INT8 quantization of the index
# makes it not exactly zero, so allow a small margin — still two orders
# of magnitude tighter than any real near-neighbour distance.
_SELF_DISTANCE_TOLERANCE = 0.02

# Cosine distance between the stored vector and a fresh embedding of the
# same text. A model mismatch shows up as ~0.5, so this only has to sit
# well below that.
#
# It cannot sit at float32 noise. The ONNX backend is deterministic for
# an identical call but *not* across batch shapes: a batch is padded to
# its longest member, so the same text embedded among 70 others and
# embedded alone differ by up to ~0.013 cosine (measured on
# ``AllMiniLML6V2Q``). The rewrite embeds in ``_BATCH``-sized pages and
# the verification re-embeds a 24-row random sample — deliberately a
# different batch composition — so a tolerance at noise level fails the
# check on a migration that is in fact correct, *after* the store has
# already been rewritten and reindexed. 0.05 clears the padding effect
# with an order of magnitude to spare and still catches a wrong model.
_REEMBED_DRIFT_TOLERANCE = 0.05


class MigrationError(RuntimeError):
    """A condition that must abort the run rather than half-apply it."""


def _require_exclusive_access() -> None:
    """Refuse to run while a server still holds the store.

    Two processes writing here is not merely racy, it silently destroys
    the migration. SQLite is single-writer, and worse, a running server
    holds its usearch index open: this script replaces that file, the
    server keeps serving from its in-memory copy, and its next save
    writes the pre-migration vectors back over the new ones. The result
    looks fine until the next restart.

    The check reuses the server's own leader election rather than
    guessing from process names, so it agrees with whatever the
    coordinator actually thinks is running. The election lock is global
    rather than per-path, so migrating a *copy* while a server serves the
    real store trips it too; that is what ``--force`` is for.
    """
    if not probe_leader_dead():
        raise MigrationError(
            "a synara server is running and holds the database. Stop it first "
            "(quit the MCP client, or kill the synara-mcp leader process), then "
            "re-run. Migrating underneath a live server silently reverts itself. "
            "Pass --force only if this database is not the one that server has open."
        )


def _backup(db_path: Path) -> Path | None:
    """Snapshot the database and every sidecar, then verify the result.

    The database goes through SQLite's own backup API rather than
    ``shutil.copy2``: a byte copy of a live database can be torn
    mid-transaction and silently restores to a corrupt store, which is
    the one failure a backup exists to prevent. The API produces a
    consistent snapshot with the write-ahead log already folded in, so
    the copy stands alone.

    The usearch sidecars are plain files with no such API and are copied
    directly. That is sound here because the migration requires no other
    process to hold the store, and it is checked below.

    An unverified backup is worse than none — it invites the destructive
    step to proceed on a false promise — so the snapshot is reopened and
    its row counts compared against the source before the caller is told
    it succeeded.
    """
    if not db_path.exists():
        return None
    stamp = time.strftime("%Y%m%d-%H%M%S")

    def target_for(path: Path) -> Path:
        return path.with_name(f"{path.name}.bak-{stamp}")

    target = target_for(db_path)
    written: list[Path] = [target]
    try:
        source = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
        try:
            destination = sqlite3.connect(target)
            try:
                source.backup(destination)
            finally:
                destination.close()
            tables = _user_tables(source)
            counts = {name: _row_count(source, name) for name in tables}
        finally:
            source.close()

        for src in sidecar_paths(db_path):
            dst = target_for(src)
            shutil.copy2(src, dst)
            written.append(dst)
            if dst.stat().st_size != src.stat().st_size:
                raise MigrationError(f"backup of {src.name} is truncated — aborting")

        check = sqlite3.connect(f"file:{target}?mode=ro", uri=True)
        try:
            restored = {name: _row_count(check, name) for name in tables}
        finally:
            check.close()
        differing = {n: (counts[n], restored[n]) for n in tables if counts[n] != restored[n]}
        if differing:
            raise MigrationError(f"backup does not match the source: {differing}")
    except Exception:
        for path in written:
            path.unlink(missing_ok=True)
        raise
    return target


def _user_tables(conn: sqlite3.Connection) -> list[str]:
    rows = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
    ).fetchall()
    return sorted(str(name) for (name,) in rows)


def _row_count(conn: sqlite3.Connection, table: str) -> int:
    # ``table`` comes from sqlite_master, never from caller input.
    (count,) = conn.execute(f"SELECT COUNT(*) FROM {quote_identifier(table)}").fetchone()
    return int(count)


def quote_identifier(name: str) -> str:
    """SQLite identifier quoting for a name taken from ``sqlite_master``."""
    escaped = name.replace('"', '""')
    return f'"{escaped}"'


def _as_matrix(vectors: object, *, expected_rows: int, label: str) -> np.ndarray:
    """Validate an embedder batch response into a finite (rows, dim) array.

    The embedder is the one component here that can fail *plausibly* — a
    remote endpoint truncating a batch, a model returning ragged output.
    Every such failure must stop the run, because a short or non-finite
    batch written to storage is indistinguishable afterwards from a
    correctly migrated one.
    """
    arr = np.asarray(vectors, dtype=np.float32)
    if arr.ndim != _MATRIX_NDIM:
        raise MigrationError(f"{label}: embedder returned shape {arr.shape}, expected 2-D")
    if arr.shape[0] != expected_rows:
        raise MigrationError(
            f"{label}: embedder returned {arr.shape[0]} vectors for {expected_rows} texts"
        )
    if not np.all(np.isfinite(arr)):
        raise MigrationError(f"{label}: embedder produced non-finite values")
    zero_rows = int(np.count_nonzero(np.linalg.norm(arr, axis=1) == 0.0))
    if zero_rows:
        raise MigrationError(f"{label}: embedder produced {zero_rows} zero vector(s)")
    return arr


def _write_canonical(sync_collection: Any, ids: list[int], matrix: np.ndarray) -> None:
    """Write the catalog's ``embedding`` column for a batch of ids.

    simplevecdb exposes no public setter for the canonical column —
    ``update_embedding`` targets the pending buffer instead (see the
    module docstring) — so this writes it directly, in the same float32
    little-endian layout ``add_texts`` uses, and under the catalog's own
    lock so it serialises against any other catalog access.
    """
    catalog = sync_collection._catalog
    stride = matrix.shape[1] * 4
    raw = matrix.tobytes()
    rows = [(raw[i * stride : (i + 1) * stride], doc_id) for i, doc_id in enumerate(ids)]
    table = sync_collection._table_name
    with catalog._lock:
        with catalog.conn:
            catalog.conn.executemany(
                # nosec B608 - ``table`` is simplevecdb's own identifier for
                # this collection, never caller input.
                f"UPDATE {table} SET embedding = ? WHERE id = ?",
                rows,
            )
        # A pending entry outlives this write and would be promoted into
        # the index by the next flush, re-introducing a pre-migration
        # vector for a row we just rewrote.
        catalog.delete_pending_vectors(ids)


async def _reembed_collection(
    collection: Any, embedder: Any, *, label: str, dry_run: bool
) -> tuple[int, int]:
    """Return ``(rows, rewritten)`` for one collection.

    Paged rather than loaded whole: ``get_documents()`` with no limit
    pulls every document's text and metadata into memory at once, which
    is fine at a few hundred episodes and ruinous at a hundred thousand.
    Offset paging is stable here because the migration requires that no
    other process is writing.
    """
    total = int(await collection.count())
    if not total:
        print(f"  {label:18} 0 rows")
        return 0, 0
    sync = collection._collection
    rewritten = 0
    for start in range(0, total, _BATCH):
        chunk = await collection.get_documents(limit=_BATCH, offset=start)
        if not chunk:
            break
        ids = [int(doc_id) for doc_id, _text, _md in chunk]
        texts = [text for _id, text, _md in chunk]
        # embed_documents applies the backend's document prefix — the
        # same call the store path uses.
        matrix = _as_matrix(
            await embedder.embed_documents(texts),
            expected_rows=len(chunk),
            label=f"{label}[{start}:{start + len(chunk)}]",
        )
        if not dry_run:
            await collection._run(_write_canonical, sync, ids, matrix)
        rewritten += len(chunk)
        print(f"  {label:18} {rewritten}/{total}", end="\r", flush=True)
    print(f"  {label:18} {rewritten}/{total} rewritten")
    return total, rewritten


async def _rebuild(collection: Any, *, label: str) -> int:
    """Derive the usearch index from the canonical column just written."""
    count = await collection.rebuild_index()
    await collection.save()
    print(f"  {label:18} index rebuilt from storage ({count} vectors)")
    return int(count)


def _unit(vector: np.ndarray) -> np.ndarray:
    arr = np.asarray(vector, dtype=np.float32)
    norm = float(np.linalg.norm(arr))
    if norm == 0.0 or not math.isfinite(norm):
        raise MigrationError("encountered a zero or non-finite stored vector")
    return arr / norm


async def _check_every_row(collection: Any, *, label: str, dim: int, total: int) -> list[int]:
    """Every row has a stored vector, at ``dim``. Returns the row ids.

    Paged for the same reason as the rewrite: only row ids are kept for
    the whole collection, and vectors only one page at a time.
    """
    ids: list[int] = []
    missing: list[int] = []
    wrong_dim: list[int] = []
    for start in range(0, total, _BATCH):
        page = await collection.get_documents(limit=_BATCH, offset=start)
        if not page:
            break
        page_ids = [int(doc_id) for doc_id, _text, _md in page]
        ids.extend(page_ids)
        page_vectors = await collection.get_embeddings_by_ids(page_ids)
        for i in page_ids:
            vector = page_vectors.get(i)
            if vector is None:
                missing.append(i)
            elif np.asarray(vector).shape[0] != dim:
                wrong_dim.append(i)
    if missing:
        raise MigrationError(
            f"{label}: {len(missing)} row(s) have no stored vector "
            f"(first: {missing[:5]}) — the index cannot be rebuilt from storage"
        )
    if wrong_dim:
        raise MigrationError(
            f"{label}: {len(wrong_dim)} row(s) still at the old dimension "
            f"(first: {wrong_dim[:5]}) — the migration did not complete"
        )
    return ids


async def _verify_collection(
    collection: Any, embedder: Any, *, label: str, dim: int, rng: random.Random
) -> None:
    """Prove storage and index agree, and that both match this model.

    Three independent checks, because they fail for different reasons:
    every row is checked for the right dimension (catches a partial
    migration), a sample is searched for its own stored vector (catches
    storage and index having diverged), and the same sample is
    re-embedded (catches storage holding another model's vectors).
    """
    total = int(await collection.count())
    if not total:
        print(f"  {label:18} empty — nothing to verify")
        return
    ids = await _check_every_row(collection, label=label, dim=dim, total=total)

    sample = rng.sample(ids, min(_VERIFY_SAMPLE, len(ids)))
    stored = await collection.get_embeddings_by_ids(sample)
    # Second pass for the sampled texts only. Deliberately not a metadata
    # filter on ``id``: that field is written by the memory feature, so a
    # legacy row without it would silently drop out of the check that
    # exists to catch exactly that kind of partial state.
    wanted = set(sample)
    by_id: dict[int, str] = {}
    for start in range(0, total, _BATCH):
        page = await collection.get_documents(limit=_BATCH, offset=start)
        if not page:
            break
        by_id.update({int(doc_id): text for doc_id, text, _md in page if int(doc_id) in wanted})

    worst_self = 0.0
    for doc_id in sample:
        hits = await collection.similarity_search(_unit(stored[doc_id]).tolist(), k=1)
        if not hits:
            raise MigrationError(f"{label}: id {doc_id} is absent from the index")
        top_doc, distance = hits[0]
        # simplevecdb carries the row id in Document.metadata, not as an
        # attribute; a getattr() probe here silently returns None and
        # skips the check that matters most.
        top_id = top_doc.metadata.get("id")
        if top_id is None:
            raise MigrationError(
                f"{label}: search result for id {doc_id} carries no row id; "
                "cannot confirm storage and index agree"
            )
        if int(top_id) != doc_id:
            raise MigrationError(
                f"{label}: searching id {doc_id}'s own vector returned id {top_id} — "
                "storage and index disagree"
            )
        worst_self = max(worst_self, float(distance))
    if worst_self > _SELF_DISTANCE_TOLERANCE:
        raise MigrationError(
            f"{label}: self-search distance {worst_self:.4f} exceeds "
            f"{_SELF_DISTANCE_TOLERANCE} — the index does not match storage"
        )

    fresh = _as_matrix(
        await embedder.embed_documents([by_id[i] for i in sample]),
        expected_rows=len(sample),
        label=f"{label} verify",
    )
    worst_drift = 0.0
    for row, doc_id in zip(fresh, sample, strict=True):
        drift = 1.0 - float(_unit(row) @ _unit(stored[doc_id]))
        worst_drift = max(worst_drift, drift)
    if worst_drift > _REEMBED_DRIFT_TOLERANCE:
        raise MigrationError(
            f"{label}: stored vectors differ from a fresh embedding by {worst_drift:.4f} — "
            "storage holds another model's vectors"
        )

    print(
        f"  {label:18} ok — {len(ids)} rows at {dim}-d, "
        f"self-search {worst_self:.4f}, re-embed drift {worst_drift:.4f}"
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true", help="embed but do not write")
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="check an already-migrated store without rewriting anything",
    )
    parser.add_argument("--no-backup", action="store_true", help="skip the database backup")
    parser.add_argument(
        "--force",
        action="store_true",
        help="run even though a synara server holds the leader lock (only when "
        "this database is not the one it has open, e.g. migrating a copy)",
    )
    parser.add_argument(
        "--seed", type=int, default=0, help="seed for the verification sample (default: 0)"
    )
    args = parser.parse_args()
    if args.dry_run and args.verify_only:
        parser.error("--dry-run and --verify-only are mutually exclusive")
    return args


def _report(args: argparse.Namespace, *, written: int, rows: int, dim: int) -> None:
    if args.verify_only:
        print("\nverified — storage and index agree and match the configured model")
    elif args.dry_run:
        print(f"\nwould rewrite {written} of {rows} record(s) at {dim}-d")
        print("dry run — nothing written. Re-run without --dry-run to apply.")
    else:
        print(f"\nrewrote {written} of {rows} record(s) at {dim}-d")


async def main() -> int:
    args = _parse_args()

    settings = Settings.from_env()
    db_path = Path(settings.db_path)
    embedder = build_embedder(settings.embedding)

    print(f"database : {db_path}")
    dim = await embedder.dim()
    print(f"model    : {settings.embedding.model or 'default'} ({dim}-d)")

    writing = not (args.dry_run or args.verify_only)
    if writing and not args.force:
        _require_exclusive_access()
    if writing and not args.no_backup:
        backup = _backup(db_path)
        print(f"backup   : {backup or '(no existing database)'}")

    memory = MemoryConfig()
    registry = default_registry(
        episodic_collection=memory.episodic_collection,
        semantic_collection=memory.semantic_collection,
        schema_candidate_collection=memory.schema_candidate_collection,
    )
    db = open_database(str(db_path))
    rng = random.Random(args.seed)
    grand_rows = grand_written = 0
    try:
        collections = {
            spec.type.value: db.collection(spec.collection, store_embeddings=STORE_EMBEDDINGS)
            for spec in registry
        }
        if not args.verify_only:
            print("\nrewriting stored vectors")
            for label, collection in collections.items():
                rows, written = await _reembed_collection(
                    collection, embedder, label=label, dry_run=args.dry_run
                )
                grand_rows += rows
                grand_written += written
            if writing:
                # Only after *every* collection is written, so an abort
                # mid-run leaves the old index intact rather than a mix.
                print("\nrebuilding indexes")
                for label, collection in collections.items():
                    if await collection.count():
                        await _rebuild(collection, label=label)
        if not args.dry_run:
            print("\nverifying")
            for label, collection in collections.items():
                await _verify_collection(collection, embedder, label=label, dim=dim, rng=rng)
    finally:
        await db.close()
        await embedder.aclose()

    _report(args, written=grand_written, rows=grand_rows, dim=dim)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(asyncio.run(main()))
    except MigrationError as exc:
        print(f"\nERROR: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
