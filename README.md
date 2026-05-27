<p align="center">
  <img src="https://files.catbox.moe/54z916.svg" width="128" height="128" alt="Synara MCP Logo" />
</p>

<h1 align="center">Synara MCP</h1>

<p align="center">
  <a href="https://www.python.org/downloads/">
    <img src="https://img.shields.io/badge/Python-3.13%2B-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54" alt="Python 3.13+" />
  </a>
  <a href="https://github.com/modelcontextprotocol/python-sdk">
    <img src="https://img.shields.io/badge/FastMCP-3.2%2B-00A67E?style=for-the-badge" alt="FastMCP 3.2+" />
  </a>
  <a href="LICENSE">
    <img src="https://img.shields.io/badge/License-MIT-D4A017?style=for-the-badge" alt="License: MIT" />
  </a>
</p>

---

**Give your MCP agent a memory that remembers across sessions — and forgets like a brain.**

Synara MCP is a [Model Context Protocol](https://modelcontextprotocol.io) server that stores episodic and semantic memories in a local vector store, embeds text via a local sentence-transformer or any OpenAI-compatible endpoint, and layers a successor-representation graph on top so recall is ranked by relational structure, not just cosine similarity. Eight tools (`store_episode`, `recall_episodes`, `consolidate_episodes`, `forget_episodes`, `reflect_session`, `store_semantic_memory`, `recall_semantic_memory`, `memory_stats`) expose it to any MCP-capable agent.

---

## What you get

**1. Recall is cross-session by default.** `session_id` is a context hint, not a filter — in-session episodes get a soft ranking bonus (state-dependent retrieval), but memories from every prior session are still returned and ranked by cosine + successor representation + spreading activation.

**2. Memory consolidates and decays on its own.** A self-triggering reactor runs consolidation, power-law forgetting, and off-policy dream replay in the background, so salient, recent, and frequently-retrieved traces survive while noise fades — no manual cleanup.

**3. Storage stays local and swappable.** Memories live in `simplevecdb`; embedding is a local sentence-transformer by default, or any OpenAI-compatible base URL (Ollama, OpenAI, simplevecdb-server) by setting one env var.

---

## Installation

Run it with [uvx](https://docs.astral.sh/uv/), no install step:

```bash
uvx synara-mcp
```

`uvx` builds a throwaway environment and launches the server — nothing lands in your global Python. For development, install from source into a managed venv instead:

```bash
uv sync
uv run --no-sync synara-mcp
```

Wire it into your MCP client (e.g. Claude Code `~/.claude.json`):

```json
{
  "mcpServers": {
    "synara": {
      "command": "uvx",
      "args": ["synara-mcp"]
    }
  }
}
```

Restart the client. The first run downloads the local embedding model (once) unless a remote endpoint is configured.

---

## Web dashboard

An optional read/write admin console (memory browser, SR/plasticity graph, forget/consolidate/reflect, live stats, per-tool telemetry — call counts, last-called, p50/p95) runs **in-process on the same event loop** as the MCP server — gated behind the `[dashboard]` extra and an env flag, so a default install never pulls in FastAPI/uvicorn.

```bash
SYNARA_DASHBOARD=true uvx 'synara-mcp[dashboard]'
```

It binds `127.0.0.1:8765` by default — open <http://127.0.0.1:8765>. On loopback the bearer token is optional; **any non-loopback bind requires `SYNARA_DASHBOARD_TOKEN`** or startup fails. See the `SYNARA_DASHBOARD*` rows under [Configuration](#configuration). The console works under any transport, including `stdio` (it never writes to stdout).

---

## Tools

| Tool | Purpose |
| --- | --- |
| `store_episode` | Encode an episodic memory (auto-salience, theta-segmented) |
| `recall_episodes` | Cross-session recall ranked by cosine + SR + spreading activation |
| `consolidate_episodes` | Cluster episodes into semantic schemas |
| `forget_episodes` | Power-law decay + selective pruning (dry-run by default) |
| `reflect_session` | Summarise a session's episodes |
| `store_semantic_memory` | Store a durable semantic fact |
| `recall_semantic_memory` | Retrieve semantic facts |
| `memory_stats` | Episodic and semantic memory counts |

---

## Telling your agent to use it

Most agents won't call memory tools on their own. Drop a few lines into your `CLAUDE.md` / `AGENTS.md` so it knows when to reach for them:

```md
## Memory (synara-mcp)

- Recall before responding when prior context could matter.
- Store after a task, decision, or durable fact.
- Reflect at end of session. Consolidation and forgetting are automatic — don't call them.
```

---

## Good to know

**Maintenance is automatic.** A background reactor handles consolidation, power-law forgetting, and idle "dream" replay on its own schedule. Recall is cross-session by default; `session_id` biases ranking rather than restricting visibility.

**Storage is local.** Memory lives in a `simplevecdb` file under your cache directory and embeddings run on-device by default. If `SYNARA_EMBEDDING_URL` is set, that endpoint is the only external service that sees your text; the API key is read from the environment and never logged. Set `SYNARA_DB_PATH=:memory:` for an ephemeral store.

**Defaults are safe.** Invalid configuration is rejected at startup rather than silently ignored. `forget_episodes` is dry-run by default, so deletions are previewed before they apply. Tool inputs (`content`, `tags`, `k`, `session_id`) and remote embedding responses are size-capped to prevent memory exhaustion.

---

## Configuration

Every variable is optional and read from the process environment at startup — Synara runs with none set. Full reference, bounds, and copy-paste recipes: **[docs/env_variables.md](docs/env_variables.md)**. The annotated [`.env.example`](.env.example) is a ready-to-edit template.

| Variable | Default | Purpose |
| --- | --- | --- |
| `SYNARA_TRANSPORT` | `stdio` | `stdio` \| `http` \| `sse` \| `streamable-http` |
| `SYNARA_LOG_LEVEL` | `INFO` | Log verbosity |
| `SYNARA_DB_PATH` | `$XDG_CACHE_HOME/synara-mcp/synara.db` | Store path; `:memory:` for ephemeral |
| `SYNARA_EMBEDDING_MODEL` | local HF model | Embedding model id |
| `SYNARA_EMBEDDING_URL` | _(unset)_ | Set to an OpenAI-compatible base URL for remote embeddings |
| `SYNARA_DASHBOARD` | `false` | Enable the parallel web admin console (`[dashboard]` extra) |
| `SYNARA_DASHBOARD_HOST` | `127.0.0.1` | Dashboard bind address (loopback-only by default) |
| `SYNARA_DASHBOARD_PORT` | `8765` | Dashboard bind port |
| `SYNARA_DASHBOARD_TOKEN` | _(unset)_ | Bearer token; optional on loopback, required for any non-loopback bind |

---

## How it works

The memory feature is organised by brain region:

- **Hippocampus** — encoding, pattern separation (DG), pattern completion (CA3), cross-session recall, the successor representation, and off-policy dream replay.
- **Neocortex** — consolidation into semantic schemas, power-law forgetting, session reflection.
- **Basal ganglia** — a self-triggering reactor that schedules consolidation and dream cycles from an event feed.
- **Amygdala** — salience tagging that modulates what consolidates fastest.

The successor representation is a discounted transition graph over episode IDs (built from co-occurrence within a session window) used as a recall-ranking prior; its transition tally is durable and the closure is rebuilt at load.

---

## Development

See [CONTRIBUTING.md](CONTRIBUTING.md) for development setup, the pre-commit gate, the dashboard SPA build, and PR requirements.

---

## License

MIT — see [LICENSE](LICENSE).
