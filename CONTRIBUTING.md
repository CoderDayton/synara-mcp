# Contributing to Synara MCP

Thanks for your interest in improving Synara MCP. This project is pre-1.0
and ships from `main`.

## Ground rules

- By participating you agree to the [Code of Conduct](CODE_OF_CONDUCT.md).
- For security issues, **do not** open a public issue — follow
  [SECURITY.md](SECURITY.md).
- Keep changes minimal and traceable to a specific issue or request.
  Prefer targeted repair over broad refactors.

## Development setup

Requires Python 3.13 and [uv](https://docs.astral.sh/uv/).

```bash
uv sync
uv run --no-sync synara-mcp        # run the server (stdio)
```

## Before you open a pull request

Run the full local gate (these also run via `lefthook` on
pre-commit / pre-push — install hooks with `lefthook install`):

```bash
uv run --no-sync ruff format .
uv run --no-sync ruff check --fix src tests
uv run --no-sync mypy
uv run --no-sync bandit -q -r src
uv run --no-sync pytest -q
```

Network / model-download tests are marked `slow`:

```bash
uv run --no-sync pytest -m slow
```

## Conventions

- ruff: line length 100, double quotes, selects `E,F,W,I,B,UP,SIM,RUF,PL,PT`.
- mypy: strict, `python_version = 3.13`.
- `from __future__ import annotations` at the top of every module.
- Frozen `@dataclass(frozen=True, slots=True)` for value objects.
- Tests mirror `src/` under `tests/`; `asyncio_mode = auto`.
- New features live in `src/synara/features/<name>/` with their own
  `register_tools` and follow the `register(mcp, db, ...)` convention.

## Pull request checklist

- [ ] The change traces to a clear motivation (issue, bug, or request).
- [ ] Tests added or updated; full suite green locally.
- [ ] `ruff`, `mypy`, and `bandit` pass.
- [ ] No secrets, credentials, or `.env` content committed.
- [ ] Commit messages describe *why*, not just *what*.
