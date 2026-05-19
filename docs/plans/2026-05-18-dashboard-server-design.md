# Dashboard Server Design

**Date:** 2026-05-18
**Status:** Approved design (brainstorm complete; not yet implemented)

## Goal

Run a standalone FastAPI admin console **in parallel with the FastMCP
server, on the same lifecycle**, gated by an env variable. The console
provides full read/write administration and observability of the memory
system and MCP server state, viewable in a browser independent of the
MCP client.

## Decisions (locked)

| Axis | Decision |
|---|---|
| Scope | Full admin console (view, edit, delete, trigger ops, live param tuning) |
| Process model | uvicorn launched as an `asyncio` task inside FastMCP's `@lifespan` — same event loop, shared live `db`/`embedder`/`MemoryService` |
| Security posture | Localhost-only by default, token optional; non-loopback bind without token is a hard startup error |
| Frontend | React 19.2 SPA · TypeScript · Tailwind v4 · shadcn/ui · Bun · Vite |
| Build artifacts | Vite build committed to `src/synara/features/dashboard/static/`, shipped in the wheel; no Bun needed at install |
| Delivery | Standalone browser console (FastMCPApp in-conversation variant explicitly deferred as a possible future additive feature) |

## 1. Configuration & lifecycle integration

### Env vars (all optional; dashboard off by default)

| Var | Default | Meaning |
|---|---|---|
| `SYNARA_DASHBOARD` | `false` | Master gate |
| `SYNARA_DASHBOARD_HOST` | `127.0.0.1` | Bind address |
| `SYNARA_DASHBOARD_PORT` | `8765` | Bind port |
| `SYNARA_DASHBOARD_TOKEN` | _unset_ | Bearer token; **required** before any non-loopback bind is permitted |

- Add a frozen `DashboardConfig(enabled, host, port, token)` value object
  (mirrors `EmbeddingConfig`), parsed in `Settings.from_env`.
- Posture validation at startup: non-loopback `host` + unset `token`
  → raise `ValueError` (fail fast; never silently expose a write console).

### Lifecycle (`server.py`)

Inside the existing `@lifespan` async generator, when
`settings.dashboard.enabled`:

1. Build the FastAPI app, injecting the **live** `db`, `embedder`, and
   the `MemoryService` already used by the MCP tools.
2. `uvicorn.Server(uvicorn.Config(app, host, port, log_config=...))`
   launched via `asyncio.create_task(server.serve())` — **same event
   loop** as FastMCP; one `AsyncVectorDB` connection, one loop, no
   cross-thread/cross-loop hazard.
3. `yield` as today.
4. In `finally`, **before** `embedder.aclose()`/`db.close()`: set
   `server.should_exit = True`, await the task (bounded timeout, then
   cancel). Dashboard drains first, then resources close.

**Critical stdio gotcha:** uvicorn/access logs must never touch stdout
(would corrupt MCP stdio JSON framing). `log_config` routes everything
through stdlib logging → stderr only. Token never logged.

## 2. Backend structure & API surface

New feature package `src/synara/features/dashboard/`, exposing
`build_dashboard_app(settings, db, embedder, service) -> FastAPI`:

```
features/dashboard/
  __init__.py         # build_dashboard_app, DashboardConfig re-export
  config.py           # DashboardConfig + env parsing
  app.py              # FastAPI assembly, static mount, SPA fallback
  auth.py             # bearer dependency (no-op on loopback w/o token)
  routes/
    health.py  stats.py  memories.py  graph.py  admin.py  events.py
  static/             # committed Vite build output
```

**Hard rule — zero logic duplication:** every route delegates to the
existing `MemoryService`. No reimplemented recall/forget/SR math.

### JSON API

| Method · Path | Purpose |
|---|---|
| `GET /api/health` | version, transport, db path, embedder backend, uptime |
| `GET /api/stats` | `memory_stats()` passthrough |
| `GET /api/memories?type=&q=&limit=&offset=` | list/search episodic + semantic |
| `GET /api/memories/{id}` | detail + SR neighbors + plasticity + salience |
| `PATCH /api/memories/{id}` | edit content |
| `DELETE /api/memories/{id}` | delete |
| `GET /api/graph?focus=&depth=` | SR subgraph: nodes=episodes, edges=`T`/`M`, plasticity overlay |
| `GET /api/params` · `PATCH /api/params` | live SR/plasticity tunables |
| `POST /api/admin/{consolidate\|forget\|reflect\|dream}` | trigger ops |
| `GET /api/events` | SSE: periodic stats/health push |

**Auth:** one FastAPI dependency on all `/api/*`. Loopback + no token →
allowed. Token set → bearer required, constant-time compare, never
logged. Optional dependency group `[dashboard]`
(`fastapi`, `uvicorn`, `sse-starlette`), out of the default install.

## 3. Frontend & build pipeline

- **Location:** `dashboard/` at repo root, isolated from `src/`.
- **Build:** `bun run build` → `src/synara/features/dashboard/static/`,
  committed to git, shipped in the wheel. FastAPI serves it via
  `StaticFiles` with SPA fallback (non-`/api` → `index.html`).
- **Dev:** `bun run dev` (Vite dev server) proxies `/api` → FastAPI;
  developer runs `SYNARA_DASHBOARD=true synara-mcp` (stdio fine).
- **Freshness guard:** lefthook pre-commit hash check fails if
  `dashboard/src/**` changed without a matching `static/` rebuild;
  CI runs `bun install --frozen-lockfile && bun run build` and asserts
  a clean `git diff` on `static/`.
- **Data layer:** TanStack Query + a thin SSE client for live Overview
  tiles. Token from localStorage → `Authorization: Bearer`.

### Pages

- **Overview** — live health/stats tiles (SSE).
- **Memories** — shadcn DataTable, server-side search/filter/paginate,
  detail drawer with inline edit + delete.
- **Graph** — interactive SR/plasticity graph (Cytoscape.js / React
  Flow), bounded by `depth`/node cap.
- **Admin** — trigger consolidate/forget/reflect/dream; zod-validated
  forms for live SR/plasticity param tuning (behind "Advanced").
- **Config** — read-only effective settings (token redacted).

## 4. UX & visual design system

Dense internal admin tool — optimize for scanning, trust, low cognitive
load (no persuasion/dopamine trends).

- **Hierarchy & proximity:** fixed left nav; one primary data region +
  grouped secondary controls; destructive controls subordinate.
- **Consistency:** shadcn/ui as the only component vocabulary except the
  graph canvas.
- **Contrast (semantic):** neutral slate base, one accent; green=
  healthy/consolidated, amber=decaying/low-salience, red=destructive
  only. Destructive actions require a confirm dialog naming the target.
- **Progressive disclosure:** summary rows → detail drawer; live param
  tuning behind an "Advanced" disclosure.
- **Dark mode:** standard (Tailwind v4 class toggle, localStorage).
- **Accessibility (non-negotiable):** WCAG 2.2 — 4.5:1 text contrast
  both themes, ≥24×24px targets, focus never obscured by sticky header,
  full keyboard nav of table + graph, no auth cognitive tests.
- **Performance:** lean bundle, tree-shaken shadcn, Graph route code-
  split (lazy + `<Activity>` prefetch), system variable font stack.

## 5. Testing & verification

Every implementation task is bracketed by **pre-verification** (prove
the assumption before touching code) and **post-change proof** (narrowest
meaningful check). Reasoning is never accepted as proof; report as
`VERIFICATION: <command> → <result>`.

- **Backend (pytest + httpx ASGI):** `:memory:` DB + fake embedder.
  Per route: status, schema, delegation to `MemoryService` (spy/patch —
  guards zero-duplication). Auth matrix: loopback+no-token allowed;
  token → 401/200; non-loopback+no-token → construction raises.
- **Lifecycle:** disabled → no task; enabled → task starts and is
  awaited-then-cancelled cleanly (no leaked/destroyed task); DB closes
  after task drains.
- **stdio safety:** assert uvicorn `log_config` emits nothing to stdout.
- **Gates:** `ruff format`, `ruff check`, `mypy` (strict — import-path
  gate after new package), `bandit -r src`, `pytest -q`. Frontend:
  `bun run typecheck` + `bun run build` in CI; minimal component tests
  (delete-confirm, token header injection).

## 6. Risks & mitigations

| Risk | Mitigation |
|---|---|
| stdout corruption (stdio) | uvicorn `log_config` → stderr only; test asserts empty stdout |
| Wrong event loop | only `asyncio.create_task(Server.serve())` in lifespan; pre-verified lifespan runs in serving loop |
| Shutdown race | `finally` order: `should_exit` → await/cancel task → close embedder/db |
| Accidental network exposure | non-loopback + no token → hard `ValueError` at startup |
| Logic drift vs MCP tools | routes delegate to `MemoryService`; test asserts delegation |
| Committed asset drift | lefthook hash check + CI clean-`git diff` on `static/` |
| Graph endpoint blowup (CPU-bound) | bounded `depth` + node cap + pagination; lazy route |
| Token leakage | constant-time compare; never logged; redacted in `/api/config` |
| Dependency weight | `[dashboard]` optional extra, not default install |

**Out of scope (YAGNI):** multi-user/RBAC, HTTPS termination,
historical metrics time-series, websockets, i18n, the FastMCPApp
in-conversation variant (deferred future option).
