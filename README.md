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

Synara MCP is a [Model Context Protocol](https://modelcontextprotocol.io) server that stores episodic and semantic memories in a local vector store, embeds text via a local sentence-transformer or any OpenAI-compatible endpoint, and layers a successor-representation graph on top so recall is ranked by relational structure, not just cosine similarity. Ten tools (`store_episode`, `recall_episodes`, `get_episode`, `remove_episode`, `consolidate_episodes`, `forget_episodes`, `reflect_session`, `store_semantic_memory`, `recall_semantic_memory`, `memory_stats`) expose it to any MCP-capable agent.

---

## Why Synara

**1. Recall is context-aware, not just similar.** Pass a `session_id` and recall returns that session's memories plus anything stored as global, with in-session hits ranked higher. Pass `scope_session=false` and recall spans every prior session — ranked by relational structure, so linked memories surface even when their wording doesn't match the query.

**2. Memory maintains itself.** Consolidation, forgetting, and replay run automatically in the background. Salient, recent, and frequently-retrieved memories survive; noise fades. You never schedule cleanup.

**3. Storage stays local and swappable.** Memories live in a single local file; embedding runs on-device by default, or against any OpenAI-compatible base URL (Ollama, OpenAI, simplevecdb-server) by setting one env var.

---

## Quick start

Run it with [uvx](https://docs.astral.sh/uv/) — no install step, nothing lands in your global Python:

```bash
uvx synara-mcp
```

Wire it into your MCP client (e.g. Claude Code `~/.claude.json`) and restart the client:

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

The first run downloads the local embedding model (once) unless a remote endpoint is configured.

Finally, tell your agent when to reach for memory — most agents won't on their own. Drop this into your `CLAUDE.md` / `AGENTS.md`:

```md
## Memory (synara-mcp)

- Recall before responding when prior context could matter.
- Store after a task, decision, or durable fact.
- Reflect at end of session. Consolidation and forgetting are automatic — don't call them.
```

---

## Tools

| Tool | Purpose |
| --- | --- |
| `store_episode` | Encode an episodic memory |
| `recall_episodes` | Session-scoped recall (plus global records); `scope_session=false` for cross-session |
| `get_episode` | Fetch one episode's full content by id (recall returns bounded snippets) |
| `remove_episode` | Delete one episode by id (dry-run by default) |
| `consolidate_episodes` | Cluster episodes into semantic schemas |
| `forget_episodes` | Prune weak memories (dry-run by default) |
| `reflect_session` | Summarise a session's episodes |
| `store_semantic_memory` | Store a durable fact (session-scoped or global; `supersedes` retires a stale entry) |
| `recall_semantic_memory` | Retrieve semantic facts |
| `memory_stats` | Store counts and health telemetry |

---

## Web dashboard

An optional admin console (memory browser, memory graph, maintenance triggers, live stats, per-tool telemetry) runs in the same process as the MCP server — gated behind the `[dashboard]` extra and an env flag, so a default install pulls in no web dependencies.

```bash
SYNARA_DASHBOARD=true uvx 'synara-mcp[dashboard]'
```

It binds `127.0.0.1:8765` by default — open <http://127.0.0.1:8765>. On loopback a token is optional; **any non-loopback bind requires `SYNARA_DASHBOARD_TOKEN`** or startup fails. Works under any transport, including `stdio`.

---

## Configuration

Every variable is optional and read from the process environment at startup — Synara runs with none set. Full reference, bounds, and copy-paste recipes: **[docs/env_variables.md](docs/env_variables.md)**. The annotated [`.env.example`](.env.example) is a ready-to-edit template.

| Variable | Default | Purpose |
| --- | --- | --- |
| `SYNARA_TRANSPORT` | `stdio` | `stdio` \| `http` \| `sse` \| `streamable-http` |
| `SYNARA_LOG_LEVEL` | `INFO` | Log verbosity |
| `SYNARA_DB_PATH` | `$XDG_DATA_HOME/synara-mcp/synara.db` | Store path; `:memory:` for ephemeral |
| `SYNARA_EMBEDDING_MODEL` | local HF model | Embedding model id |
| `SYNARA_EMBEDDING_URL` | _(unset)_ | Set to an OpenAI-compatible base URL for remote embeddings |
| `SYNARA_DASHBOARD` | `false` | Enable the parallel web admin console (`[dashboard]` extra) |
| `SYNARA_DASHBOARD_HOST` | `127.0.0.1` | Dashboard bind address (loopback-only by default) |
| `SYNARA_DASHBOARD_PORT` | `8765` | Dashboard bind port |
| `SYNARA_DASHBOARD_TOKEN` | _(unset)_ | Bearer token; optional on loopback, required for any non-loopback bind |

---

## How it works

The memory feature is organised by brain region:

- **Hippocampus** — encoding, pattern separation and completion, ranked recall, and dream replay.
- **Neocortex** — consolidation into semantic schemas, forgetting, session reflection.
- **Basal ganglia** — the reactor that decides when consolidation and dream cycles fire.
- **Amygdala** — salience tagging that decides what consolidates fastest.

Recall is ranked by cosine similarity plus a successor representation — a transition graph built from which memories are accessed together — and spreading activation over learned associations. Every record carries a scope, `session` or `global`: episodes always belong to their session, while semantic memories are session-scoped when stored with a `session_id` and global otherwise.

---

## Good to know

**Storage is local.** Memory lives in a `simplevecdb` file under your XDG data directory (not the cache directory — this store is durable and shouldn't be swept by a cache clean) and embeddings run on-device by default. If `SYNARA_EMBEDDING_URL` is set, that endpoint is the only external service that sees your text; the API key is read from the environment and never logged. Set `SYNARA_DB_PATH=:memory:` for an ephemeral store.

**Defaults are safe.** Invalid configuration fails at startup rather than silently degrading. `forget_episodes` is dry-run by default, so deletions are previewed before they apply. Tool inputs and remote embedding responses are size-capped, and invalid input is rejected with an actionable message the agent can act on.

**Results are bounded.** Recall returns a few snippet-length hits by default so results never blow your context budget; truncated hits say so, and `get_episode` fetches the complete text on demand.

---

## Development

Install from source into a managed venv:

```bash
uv sync
uv run --no-sync synara-mcp
```

See [CONTRIBUTING.md](CONTRIBUTING.md) for development setup, the pre-commit gate, the dashboard SPA build, and PR requirements.

---

## License

MIT — see [LICENSE](LICENSE).
