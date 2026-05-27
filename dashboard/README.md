<p align="center">
  <img src="../assets/synara-logo.svg" width="128" height="128" alt="Synara MCP Logo" />
</p>

<h1 align="center">Synara Dashboard</h1>

<p align="center">
  <a href="https://react.dev/">
    <img src="https://img.shields.io/badge/React-19-149ECA?style=for-the-badge&logo=react&logoColor=white" alt="React 19" />
  </a>
  <a href="https://vite.dev/">
    <img src="https://img.shields.io/badge/Vite-8-646CFF?style=for-the-badge&logo=vite&logoColor=white" alt="Vite 8" />
  </a>
  <a href="https://bun.sh/">
    <img src="https://img.shields.io/badge/Bun-fbf0df?style=for-the-badge&logo=bun&logoColor=000" alt="Bun" />
  </a>
  <a href="https://tailwindcss.com/">
    <img src="https://img.shields.io/badge/Tailwind-4-38BDF8?style=for-the-badge&logo=tailwindcss&logoColor=white" alt="Tailwind v4" />
  </a>
  <a href="../LICENSE">
    <img src="https://img.shields.io/badge/License-MIT-D4A017?style=for-the-badge" alt="License: MIT" />
  </a>
</p>

---

**A terminal-styled admin console for the Synara MCP memory server — browse traces, watch the brain consolidate, prune what's stale.**

The dashboard is a single-page React app served from the same Python process as the MCP server (see [Synara MCP](../README.md)). It talks to a small read/write JSON API mounted under `/api`, renders the successor + plasticity graph with `@xyflow/react`, projects embeddings with `umap-js`, and is built with Bun + Vite. The compiled bundle is committed to `src/synara/features/dashboard/static/` so the wheel ships with no Node-at-install-time step.

---

## Why this exists

In order of impact:

**1. The memory store is otherwise opaque.** Episodes, semantic schemas, the successor-representation graph, and plasticity edges are real objects that change every time the agent encodes or recalls. The dashboard makes that state directly observable — you can see which clusters are forming, which traces are about to be pruned, and which tools are actually being called.

**2. Maintenance is one click, not one shell incantation.** `consolidate_episodes`, `forget_episodes`, and `reflect_session` are all gated behind dry-run previews in the Admin page — you see the projected impact (clusters formed, episodes pruned) before anything is written.

**3. It's gated, scoped, and parallel.** The console is opt-in (`SYNARA_DASHBOARD=true` + `[dashboard]` extra), shares the MCP server's event loop (no second process), and is loopback-only by default. Non-loopback binds require a bearer token or startup fails.

---

## Pages

| Page | Purpose |
| --- | --- |
| **Overview** | Live store composition, hippocampus pressure (raw → schema ratio), recent episodes feed, per-tool telemetry (call counts, last-called, p50/p95) |
| **Memories** | UMAP-projected memory map with SR + plasticity edges, community hulls, selection inspector, full-trace viewer |
| **Admin** | Dry-run-first consolidate / forget / reflect / replay with impact preview and a session-scoped transcript |
| **Config** | Read-only view of the server's resolved `Settings` (transport, DB path, embedding model, dashboard bind) |

---

## Development

Bun is the package manager and dev runner; the repo's [`lefthook`](../lefthook.yml) hooks (`dashboard-build-fresh`, `dashboard-typecheck`) gate commits on a fresh build and a clean `tsc` pass, so a stale or broken UI cannot ship.

```bash
cd dashboard
bun install
bun run dev          # http://127.0.0.1:5173 — API proxied to :8765
```

In a second terminal, run the Python server with the dashboard extra so the proxy target is live:

```bash
SYNARA_DASHBOARD=true uv run --no-sync synara-mcp
```

Other scripts:

```bash
bun run typecheck    # tsc -b --noEmit
bun run test         # vitest run
bun run lint         # eslint .
bun run build        # tsc -b && vite build → ../src/synara/features/dashboard/static/
```

`bun run build` invokes `postbuild` which writes `.assets-hash` so the freshness hook (pre-commit, pre-push, CI) detects whether `dashboard/` sources changed without a matching rebuild.

---

## Good to know

**The bundle is committed.** The compiled SPA lives under [`src/synara/features/dashboard/static/`](../src/synara/features/dashboard/static/) and is part of the wheel. Editing `dashboard/src/**` means rebuilding before commit — the `dashboard-build-fresh` hook will refuse otherwise.

**It runs on the MCP loop.** When `SYNARA_DASHBOARD=true`, [`runner.py`](../src/synara/features/dashboard/runner.py) attaches a uvicorn task to the MCP server's lifespan via an `AsyncExitStack`. There's no second process, no extra port management — when MCP shuts down, the dashboard drains on the same signal.

**Auth is loopback-first.** With no `SYNARA_DASHBOARD_TOKEN` and the default `127.0.0.1` bind, the console is reachable only from your machine — the auth middleware no-ops. Set a non-loopback `SYNARA_DASHBOARD_HOST` without a token and the server refuses to start.

**The API surface is tiny.** Routes live in [`src/synara/features/dashboard/routes/`](../src/synara/features/dashboard/routes/): `health`, `stats`, `memories`, `graph`, `admin`, `tools`. Every response is shape-validated client-side (see `dashboard/src/lib/api.ts`) so a server contract drift fails loudly instead of rendering blank panels.

---

## Configuration

The dashboard reads no env vars of its own — it inherits the Python server's `Settings`. The variables that change its behaviour are:

| Variable | Default | Purpose |
| --- | --- | --- |
| `SYNARA_DASHBOARD` | `false` | Master gate. Nothing dashboard-related starts unless truthy. Requires `[dashboard]` extra |
| `SYNARA_DASHBOARD_HOST` | `127.0.0.1` | Bind address. Non-loopback values require a token or startup fails |
| `SYNARA_DASHBOARD_PORT` | `8765` | Bind port |
| `SYNARA_DASHBOARD_TOKEN` | _(unset)_ | Bearer token; optional on loopback, required for any non-loopback bind |

Full reference and bounds: [`docs/env_variables.md`](../docs/env_variables.md). The annotated [`.env.example`](../.env.example) is a ready-to-edit template.

---

## How it works

The SPA is intentionally thin. Each page is a TanStack Query consumer of one or two `/api/*` routes; the rest is presentation:

- **`pages/`** — one route per page (`overview`, `memories`, `admin`, `config`). React Router is the only routing primitive; no nested layouts beyond the app shell.
- **`components/common/`** — the design vocabulary (`Panel`, `StatCard`, `Loading`, `Empty`, `ErrorState`, `PageHeader`). New pages compose these; ad-hoc empty/loading strings are not the convention.
- **`components/ui/`** — shadcn primitives (new-york style). Add via `bunx shadcn@latest add @shadcn/<name>`. The project's `Empty` wrapper in `common/states.tsx` is built on `ui/empty`.
- **`components/layout/`** — `AppShell` (rail + status bar), `Sidebar`, `CommandPalette` (⌘K).
- **`components/memories/`** — the memory map: ELK-layouted graph, UMAP projection, community detection, per-node inspector.
- **`lib/api.ts`** — single `fetch` wrapper with a `validate` hook. All response shapes are runtime-checked; an unrecognised field surfaces as a loud error, not a `undefined.foo` crash.
- **`lib/queries.ts`** — TanStack Query hooks. Polling intervals (5s for tool telemetry, etc.) live here, not scattered through components.

The React Compiler (`target: "19"`) auto-memoises components at build time, so the memory map can re-render at interactive frame rates without manual `useMemo` / `React.memo` noise.

---

## Tech

- **React 19** + **Vite 8** + **Bun** (package manager + runtime)
- **TypeScript** strict, **Tailwind v4** (CSS-first config), **shadcn/ui** (new-york)
- **TanStack Query** for server state, **React Router 7** for routing
- **@xyflow/react** + **elkjs** for the SR/plasticity graph, **umap-js** for embedding projection, **recharts** for store-composition charts
- **Vitest** + **@testing-library/react** for component tests; **eslint** + **typescript-eslint** for lint

---

## License

MIT — see [LICENSE](../LICENSE).
