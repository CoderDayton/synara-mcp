#!/usr/bin/env python
"""Write the dashboard OpenAPI schema to a file (for TypeScript codegen).

Thin CLI over ``synara.features.dashboard.openapi_export.export_schema``; the
schema source of truth lives in the package so it is import-testable. Output is
stable (sorted keys) so the committed ``dashboard/openapi.json`` only changes
when the API actually changes — which is what the codegen drift guard checks.

Usage (from repo root or dashboard/, via the project venv):
    uv run --no-sync python scripts/export_openapi.py --out dashboard/openapi.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from synara.features.dashboard.openapi_export import export_schema


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, type=Path, help="output JSON path")
    args = parser.parse_args(argv)
    schema = export_schema()
    args.out.write_text(json.dumps(schema, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote OpenAPI schema: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
