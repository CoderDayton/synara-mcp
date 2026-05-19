# Dashboard Server — Implementation Plan

**Date:** 2026-05-18
**Companion:** `2026-05-18-dashboard-server-design.md`
**Status:** Not started

Each task: **Target** (exact file/symbol), **Boundary** (what must not
change), **Pre-verification** (prove the assumption before editing),
**Change**, **Proof** (`VERIFICATION: <command> → <result>`). No task
begins until its pre-verification passes. Phases are sequential; a phase
is not started until the prior phase's proofs are green.

---

## Phase 0 — Foundational verification (no code) — ✅ COMPLETE 2026-05-18

**Result summary:**
- **0.1 PASS** — source (transport.py:91 `anyio.run(run_async)`;
  transport.py:206-222 `async with self._lifespan_manager(): async with
  stdio_server(): await self._mcp_server.run(...)`; lifespan.py:160
  user lifespan entered via `stack.enter_async_context`; lifespan.py:177
  teardown under `anyio.CancelScope(shield=True)`) + empirical
  (`/tmp/phase0_loop_proof.py` → `ALL_SAME_LOOP=True`,
  `task_cancelled_cleanly=True`, exit 0). Single anyio loop; setup,
  spawned task, teardown, serving all share it. **Bonus:** lifespan
  teardown is shielded from Ctrl-C cancellation and runs last (after
  providers/docket) — strengthens the shutdown-drain design.
- **0.2 PASS** — `MemoryService.__init__(db, config=None, *, embed_fn=
  None)` is pure logic, no MCP `Context`. **Key finding:**
  `memory.register()` already constructs exactly one `MemoryService`
  and **returns it**, but `server.py:62` discards the return value.
  The dashboard reuses the *identical* instance (single SR/plasticity
  state, zero divergence) by capturing that return in `build_server` —
  **no memory-feature refactor, no second instance, no new seam.**
- **0.3** — deferred to Phase 2.3 by design (asserted by stdio-safety
  test).

**Plan adjustment (Phase 3.1):** `build_server` must capture
`service = memory.register(...)` (currently discarded at `server.py:62`)
and pass that same object into `build_dashboard_app`. This is the only
change needed to satisfy the "reuse the live service" requirement.

### Original Phase 0 tasks (for traceability)

**0.1 — Confirm FastMCP `@lifespan` runs inside the serving event loop**
- Pre-verification: locate `fastmcp.server.lifespan` source; confirm the
  lifespan async body is entered within the same loop that drives
  `mcp.run(transport="stdio")` (not a separate bootstrap loop).
- Proof: `VERIFICATION: minimal repro — create_task in lifespan,
  assert id(asyncio.get_running_loop()) matches the task's loop → equal`.
- If false: redesign (fallback = dedicated loop + thread-safe DB proxy);
  STOP and report before proceeding.

**0.2 — Confirm `MemoryService` is constructible/usable outside an MCP
tool context**
- Pre-verification: read `features/memory/service.py` + `port.py`;
  confirm the service can be built from `(db, embedder, settings)`
  without a FastMCP request/Context object, or identify the exact
  dependency.
- Proof: `VERIFICATION: pytest -q tests/.../test_service_standalone.py
  → pass` (tiny throwaway test instantiating the service on :memory:).

**0.3 — Confirm uvicorn log routing can fully suppress stdout**
- Pre-verification: identify uvicorn `Config(log_config=...)` shape that
  sends startup + access logs to stderr/stdlib only.
- Proof: deferred to 2.3 (asserted by test there).

---

## Phase 1 — Configuration — ✅ COMPLETE 2026-05-18

**Result:** `DashboardConfig` (frozen/slots, token `repr=False`) +
`_bool_env`/`_port_env`/`_host_is_loopback` helpers in
`features/dashboard/config.py`; `Settings` gained
`dashboard: DashboardConfig` + `DashboardConfig.from_env()` wiring.
Package `__init__` is stdlib-only (lazy `build_dashboard_app` export
deferred to Phase 2.1). Docs updated: `.env.example`, `CLAUDE.md`
(Layout + Config), `README.md` table, `docs/env_variables.md` (quick
ref + Dashboard section + recipe). Proof: `ruff` clean; `mypy` clean
(3 files); `pytest tests/test_config.py` 17 passed; full non-slow
suite 317 passed (no regressions); import-safety verified
(`fastapi` not imported by `Settings.from_env()`).

### Original Phase 1 tasks (for traceability)

**1.1 — `DashboardConfig` value object + env parsing**
- Target: `src/synara/features/dashboard/config.py` (new);
  `src/synara/config.py` (`Settings`, `from_env`).
- Boundary: no change to existing env var parsing, defaults, or
  `EmbeddingConfig`; `Settings` stays frozen/slots.
- Pre-verification: re-read current `config.py` (force fresh ranged
  read); confirm `Settings` field/`from_env` structure unchanged from
  design assumptions.
- Change: frozen `DashboardConfig(enabled, host, port, token)`; parse
  `SYNARA_DASHBOARD*`; loopback detection helper; non-loopback + no
  token → `ValueError`.
- Proof: `VERIFICATION: pytest -q tests/test_config.py → pass`
  (cases: default disabled; enabled loopback; non-loopback+token ok;
  non-loopback+no-token raises; bad port raises).
- Gate: `mypy` strict on the two files.

**1.2 — Document env vars**
- Target: `.env.example`, `CLAUDE.md` Config section, `README` env table.
- Boundary: documentation only; no behavior change.
- Pre-verification: grep existing `SYNARA_` doc blocks to match format.
- Proof: `VERIFICATION: ruff/lint n/a; manual diff review → consistent`.

---

## Phase 2 — Backend (API, no UI)

**2.1 — Package skeleton + `build_dashboard_app`**
- Target: `features/dashboard/__init__.py`, `app.py`, `auth.py`.
- Boundary: no MCP tool registration changes; no import of dashboard
  from default code paths (optional extra).
- Pre-verification: confirm `[dashboard]` extra resolves
  (`uv pip install -e ".[dashboard]"` dry-run / lock check) and that
  `fastapi`/`uvicorn`/`sse-starlette` are not pulled into the default
  dependency set.
- Change: `build_dashboard_app(settings, db, embedder, service)`;
  bearer dependency; SPA static mount stub. Add the lazy
  :pep:`562` ``__getattr__`` in `dashboard/__init__.py` exposing
  `build_dashboard_app` (deferred from Phase 1.1 — must not eagerly
  import FastAPI; `Settings` imports this package transitively).
- Proof: `VERIFICATION: pytest -q tests/.../test_app_build.py → pass`
  (app builds; `/api/health` 200; default install does NOT import
  fastapi — assert via import-isolation test).

**2.2 — Routes (delegating to `MemoryService`)**
- Target: `routes/health.py stats.py memories.py graph.py admin.py
  events.py`.
- Boundary: **zero memory/SR/forget logic** in routes — delegate only.
- Pre-verification: for each route, read the `MemoryService` method it
  will call; confirm signature + return shape (don't assume).
- Change: implement endpoints per design API table; graph endpoint
  enforces `depth` bound + node cap + pagination.
- Proof: `VERIFICATION: pytest -q tests/.../test_routes.py → pass` —
  per route: status + schema + **delegation asserted via patched
  `MemoryService`** (spy: route calls service, no inline logic).

**2.3 — Auth matrix + stdio-safety tests**
- Target: `tests/.../test_auth.py`, `test_stdio_safety.py`.
- Pre-verification: confirm loopback-detection helper handles the exact
  `host` strings used (`127.0.0.1`, `localhost`, `::1`, `0.0.0.0`).
- Proof: `VERIFICATION: pytest -q → pass` — loopback+no-token allowed;
  token → 401 w/o bearer, 200 with (constant-time path); non-loopback
  +no-token construction raises; uvicorn `log_config` emits nothing to
  captured stdout.

---

## Phase 3 — Lifecycle integration

**3.1 — Wire dashboard task into `@lifespan`**
- Target: `src/synara/server.py` (`build_server`, `app_lifespan`).
- Boundary: behavior identical when `SYNARA_DASHBOARD` unset; existing
  `yield {db, embedder, settings}` contract and teardown of
  embedder/db unchanged in order *relative to each other*.
- Pre-verification: re-confirm Phase 0.1 result still holds against the
  installed FastMCP version (version pin check); read current
  `app_lifespan` fresh.
- Change: (a) capture `service = memory.register(mcp, db, embedder=...)`
  at `server.py:62` (return value currently discarded — confirmed
  Phase 0.2); (b) conditional `asyncio.create_task(Server.serve())`
  passing that same `service`; (c) `finally` ordering: `should_exit` →
  bounded await → cancel → then `embedder.aclose()` → `db.close()`.
  Note (Phase 0.1): this `finally` already runs inside FastMCP's
  shielded teardown (`anyio.CancelScope(shield=True)`), so the drain
  survives Ctrl-C.
- Proof: `VERIFICATION: pytest -q tests/.../test_lifecycle.py → pass`:
  (a) disabled → no task created; (b) enabled → task runs, `/api/health`
  reachable via ASGI/loopback; (c) lifespan exit → task awaited then
  cancelled, no "Task was destroyed" / no pending tasks; (d) DB.close
  called strictly after task drain (ordering asserted via mock call
  sequence).
- Gate: `pytest -q` full suite (regression: existing server tests
  green), `mypy` strict, `bandit -r src` (auth/bind).

---

## Phase 4 — Frontend

**4.1 — Scaffold `dashboard/` (Bun + Vite + React 19.2 + TS + Tailwind
v4 + shadcn)**
- Target: `dashboard/` (new), repo root.
- Boundary: no Python changes; nothing added to default wheel yet.
- Pre-verification: `bun --version` available; confirm Vite `outDir`
  can target `../src/synara/features/dashboard/static`.
- Proof: `VERIFICATION: bun run build → exit 0, static/ populated`.

**4.2 — API client, routing, pages, design system**
- Target: `dashboard/src/**`.
- Boundary: consumes only the documented `/api` surface; no new
  backend endpoints invented here (if needed → amend Phase 2 first).
- Pre-verification: dev proxy hits a running
  `SYNARA_DASHBOARD=true synara-mcp`; `/api/health` 200 through proxy.
- Change: TanStack Query + SSE client; Overview/Memories/Graph/Admin/
  Config pages; shadcn design system, dark mode, WCAG passes; Graph
  route lazy + `<Activity>`.
- Proof: `VERIFICATION: bun run typecheck → 0 errors;
  bun run build → exit 0; minimal Vitest (delete-confirm dialog +
  bearer header injection) → pass`.

**4.3 — Build artifact commit + freshness guard**
- Target: committed `src/synara/features/dashboard/static/`;
  `lefthook.yml`; CI workflow.
- Boundary: do not modify `.claude/hooks/`; lefthook only.
- Pre-verification: read current `lefthook.yml` structure.
- Proof: `VERIFICATION: clean `git diff --stat static/` after
  `bun run build`; lefthook pre-commit hash-check fails on a simulated
  stale-source commit; CI step asserts clean static/ diff`.

---

## Phase 5 — Final gates & docs

- Full toolchain: `ruff format`, `ruff check --fix src tests`,
  `mypy`, `bandit -q -r src`, `pytest -q` (+ `-m slow` excluded).
- `CLAUDE.md` Layout + MCP/Config sections updated; `.env.example`
  finalized; `README` install note for `[dashboard]` extra + dashboard
  usage.
- Proof: `VERIFICATION: full lefthook pre-push set → all green`.

---

## Sequencing & stop conditions

- Phase 0 is blocking. If 0.1 fails, **stop and report** — the
  same-loop architecture is invalidated; do not proceed to a workaround
  without re-approval (it would change the design's risk profile).
- Phases 1→2→3 strictly sequential. Phase 4 may start after Phase 2
  (API frozen); Phase 4.3/5 require Phase 3 green.
- Any task whose pre-verification fails halts that phase; report
  evidence, do not improvise around a failed assumption.
- File I/O via `mcp__semantic-cache-mcp__*` only; commits authored by
  the user (`git -c commit.gpgsign=false commit`, no AI attribution).
