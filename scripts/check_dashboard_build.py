#!/usr/bin/env python3
"""Freshness guard for the committed dashboard SPA build.

The Vite build is committed into ``src/synara/features/dashboard/static``
and shipped in the wheel. If a dev edits ``dashboard/`` sources but
forgets to rebuild, the wheel would ship a stale UI. This script hashes
the build *inputs* and compares them to a manifest written at build time
(``static/.assets-hash``):

* ``--write``  — recompute and store the hash (run as ``postbuild``).
* (default)    — recompute and fail (exit 2) on drift; used by lefthook.

Stdlib only; hashes inputs (cheap) and never rebuilds, so it is fast
enough for every pre-commit. Test files are excluded — they are not in
the production bundle.
"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
_DASH = _REPO / "dashboard"
_HASH_FILE = _REPO / "src/synara/features/dashboard/static/.assets-hash"

# Top-level files whose contents change the build output.
_ROOT_INPUTS = (
    "index.html",
    "package.json",
    "bun.lock",
    "bun.lockb",
    "vite.config.ts",
    "tsconfig.json",
    "tsconfig.app.json",
    "tsconfig.node.json",
    "components.json",
)


def _is_test(p: Path) -> bool:
    name = p.name
    return (
        name.endswith((".test.ts", ".test.tsx"))
        or "/test/" in p.as_posix()
        or "/__tests__/" in p.as_posix()
    )


def _iter_inputs() -> list[Path]:
    files: list[Path] = []
    src = _DASH / "src"
    if src.is_dir():
        files += [p for p in src.rglob("*") if p.is_file() and not _is_test(p)]
    for name in _ROOT_INPUTS:
        p = _DASH / name
        if p.is_file():
            files.append(p)
    return sorted(files, key=lambda p: p.relative_to(_DASH).as_posix())


def _digest() -> str:
    h = hashlib.sha256()
    for p in _iter_inputs():
        h.update(p.relative_to(_DASH).as_posix().encode())
        h.update(b"\0")
        h.update(p.read_bytes())
        h.update(b"\0")
    return h.hexdigest()


def main(argv: list[str]) -> int:
    write = "--write" in argv[1:]
    current = _digest()

    if write:
        _HASH_FILE.parent.mkdir(parents=True, exist_ok=True)
        _HASH_FILE.write_text(current + "\n", encoding="utf-8")
        print(f"dashboard build hash written: {current[:12]}")
        return 0

    if not _HASH_FILE.is_file():
        print(
            "ERROR: dashboard build manifest missing "
            f"({_HASH_FILE.relative_to(_REPO)}). "
            "Run `bun run build` in dashboard/ and commit static/.",
            file=sys.stderr,
        )
        return 2

    recorded = _HASH_FILE.read_text(encoding="utf-8").strip()
    if recorded != current:
        print(
            "ERROR: dashboard sources changed but the committed build is "
            "stale.\n  expected "
            f"{current[:12]}, committed {recorded[:12]}.\n"
            "  Fix: cd dashboard && bun run build, then stage "
            "src/synara/features/dashboard/static/.",
            file=sys.stderr,
        )
        return 2

    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
