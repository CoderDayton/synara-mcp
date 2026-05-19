<p align="center">
  <img src="assets/synara-logo.svg" width="128" height="128" alt="Synara MCP Logo" />
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

## Why this exists

In order of impact:

**1. Recall is cross-session by default.** `session_id` is a context hint, not a filter — in-session episodes get a soft ranking bonus (state-dependent retrieval), but memories from every prior session are still returned and ranked by cosine + successor representation + spreading activation.

**2. Memory consolidates and decays on its own.** A self-triggering reactor runs consolidation, power-law forgetting, and off-policy dream replay in the background, so salient, recent, and frequently-retrieved traces survive while noise fades — no manual cleanup.

**3. Storage stays local and swappable.** Memories live in `simplevecdb`; embedding is a local sentence-transformer by default, or any OpenAI-compatible base URL (Ollama, OpenAI, simplevecdb-server) by setting one env var.

---

## Installation

Run from source with [uv](https://docs.astral.sh/uv/):

```bash
uv sync
uv run --no-sync synara-mcp
```

Add to your MCP client config (e.g. Claude Code `~/.claude.json`):

```json
{
  "mcpServers": {
    "synara": {
      "command": "uv",
      "args": ["run", "--no-sync", "synara-mcp"],
      "cwd": "/absolute/path/to/synara-mcp"
    }
  }
}
```

Restart the client. The first run downloads the local embedding model unless a remote endpoint is configured.

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

## Configuration

All optional, read from the environment at startup. See `.env.example` for the full list.

| Variable | Default | Purpose |
| --- | --- | --- |
| `SYNARA_TRANSPORT` | `stdio` | `stdio` \| `http` \| `sse` \| `streamable-http` |
| `SYNARA_LOG_LEVEL` | `INFO` | Log verbosity |
| `SYNARA_DB_PATH` | `$XDG_CACHE_HOME/synara-mcp/synara.db` | Store path; `:memory:` for ephemeral |
| `SYNARA_EMBEDDING_MODEL` | local HF model | Embedding model id |
| `SYNARA_EMBEDDING_URL` | _(unset)_ | Set to an OpenAI-compatible base URL for remote embeddings |

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

```bash
uv run --no-sync pytest -q          # tests
uv run --no-sync ruff format .      # format
uv run --no-sync ruff check --fix src tests
uv run --no-sync mypy               # strict types
uv run --no-sync bandit -q -r src   # security lint
```

`lefthook install` wires the full set into pre-commit / pre-push.

---

## License

MIT — see [LICENSE](LICENSE).
