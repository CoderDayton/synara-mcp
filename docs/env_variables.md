# Environment variables

Every Synara setting is an environment variable. All are optional — with
nothing set, Synara runs a local embedding model and stores memories in
your user cache directory.

Variables are read **once, from the process environment, at startup**.
Synara does not auto-load a `.env` file: export the variables in your
shell, or pass them through your MCP client's `env` block. The tracked
[`.env.example`](../.env.example) is an annotated template you copy and
source yourself.

Invalid values fail loudly. A malformed number, an out-of-range bound,
or an unknown transport raises at startup instead of silently degrading.

---

## Quick reference

| Variable | Default | Accepted | Applies to |
| --- | --- | --- | --- |
| `SYNARA_TRANSPORT` | `stdio` | `stdio` \| `http` \| `sse` \| `streamable-http` | always |
| `SYNARA_LOG_LEVEL` | `INFO` | `DEBUG` `INFO` `WARNING` `ERROR` `CRITICAL` | always |
| `SYNARA_DB_PATH` | `$XDG_CACHE_HOME/synara-mcp/synara.db` | filesystem path or `:memory:` | always |
| `SYNARA_EMBEDDING_MODEL` | `jinaai/jina-embeddings-v5-text-nano` | HF model id (local) or remote model name | always |
| `SYNARA_EMBEDDING_URL` | _(unset)_ | OpenAI-compatible base URL | switches to remote |
| `SYNARA_EMBEDDING_API_KEY` | _(unset)_ | bearer token string | remote only |
| `SYNARA_EMBEDDING_TIMEOUT` | `30` | finite number, `0 < t ≤ 86400` (s) | remote only |
| `SYNARA_EMBEDDING_DIM` | _(probed)_ | integer, `1 … 1000000` | always |
| `SYNARA_EMBEDDING_BATCH_SIZE` | `64` | integer, `1 … 1000000` | always |
| `SYNARA_EMBEDDING_MAX_SEQ_LENGTH` | _(model default)_ | integer, `1 … 1000000` | local only |
| `SYNARA_DASHBOARD` | `false` | boolean (`true/false/1/0/yes/no/on/off`) | always |
| `SYNARA_DASHBOARD_HOST` | `127.0.0.1` | bind address | dashboard only |
| `SYNARA_DASHBOARD_PORT` | `8765` | integer, `1 … 65535` | dashboard only |
| `SYNARA_DASHBOARD_TOKEN` | _(unset)_ | bearer token string | dashboard only |

---

## Server

### `SYNARA_TRANSPORT`

How the server talks to its client. `stdio` (default) is what local MCP
clients like Claude Code expect — the client launches Synara as a
subprocess and speaks over stdin/stdout. Pick `http`, `sse`, or
`streamable-http` only when connecting over a network. Any other value
is rejected at startup.

```bash
SYNARA_TRANSPORT=stdio
```

### `SYNARA_LOG_LEVEL`

Log verbosity, using standard Python logging level names. Lower-case is
accepted (it is upper-cased internally). `DEBUG` traces every tool call
and the reactor's decisions; `INFO` is the sensible default.

```bash
SYNARA_LOG_LEVEL=DEBUG
```

### `SYNARA_DB_PATH`

Where memories are persisted. Defaults to a per-user cache file:

- `$XDG_CACHE_HOME/synara-mcp/synara.db` if `XDG_CACHE_HOME` is set,
- otherwise `~/.cache/synara-mcp/synara.db`.

The parent directory is created automatically. The path is resolved
(symlinks and `..` segments are normalized) before the directory is
created, so a relative or symlinked path lands where you expect.

Set `:memory:` for an ephemeral store that lives in RAM and is discarded
when the server stops — useful for tests, demos, and throwaway sessions.

```bash
SYNARA_DB_PATH=/data/synara/memory.db
SYNARA_DB_PATH=:memory:
```

---

## Embeddings

Synara turns text into vectors so it can search memory by meaning. It
runs a small model locally by default. Setting `SYNARA_EMBEDDING_URL`
switches the whole embedding path to a remote OpenAI-compatible service;
the other `SYNARA_EMBEDDING_*` variables tune whichever mode is active.

### `SYNARA_EMBEDDING_MODEL`

The embedding model. In **local mode** this is a Hugging Face model id,
downloaded automatically on first use and cached thereafter. In **remote
mode** it is the model name sent in each request. Default:
`jinaai/jina-embeddings-v5-text-nano` — fast, small, no setup.

```bash
SYNARA_EMBEDDING_MODEL=jinaai/jina-embeddings-v5-text-nano   # local
SYNARA_EMBEDDING_MODEL=text-embedding-3-small                # remote (OpenAI)
```

### `SYNARA_EMBEDDING_URL`

Leave unset to use the local model. Set it to an OpenAI-compatible base
URL to delegate embedding to a remote service; Synara calls
`<url>/v1/embeddings`. The remote endpoint receives your episode text —
**treat it as part of your trust boundary** (see [SECURITY.md](../SECURITY.md)).
Responses are size-capped internally so a hostile or misconfigured
endpoint cannot exhaust memory.

```bash
SYNARA_EMBEDDING_URL=http://localhost:11434     # Ollama
SYNARA_EMBEDDING_URL=http://localhost:8080      # simplevecdb-server
SYNARA_EMBEDDING_URL=https://api.openai.com     # OpenAI
```

### `SYNARA_EMBEDDING_API_KEY`

Bearer token for the remote service, if it requires one. Sent as
`Authorization: Bearer <key>`. Ignored in local mode. Never logged and
never echoed back; read from the environment only — do not commit it.

```bash
SYNARA_EMBEDDING_API_KEY=sk-...
```

### `SYNARA_EMBEDDING_TIMEOUT`

Seconds to wait for the remote embedding service before giving up.
Default `30`. Must be a finite number greater than `0` and at most
`86400` (24 hours); anything else is rejected at startup. Raise it on a
slow link or a cold remote model.

```bash
SYNARA_EMBEDDING_TIMEOUT=120
```

### `SYNARA_EMBEDDING_DIM`

Output vector dimensionality. Unset (default), Synara probes the
embedder once and caches the observed dimension. Set it to pin the
dimension explicitly: the value is validated against the real output on
first use, so a swapped-out model that no longer matches fails fast
instead of silently corrupting stored vectors. With OpenAI-compatible
remotes that support it (e.g. `text-embedding-3-*`), the value is also
sent as the `dimensions` request field to truncate the vector. Integer,
`1` to `1000000`.

```bash
SYNARA_EMBEDDING_DIM=768
```

### `SYNARA_EMBEDDING_BATCH_SIZE`

How many texts are embedded per request/forward pass. Default `64`.
Larger batches are faster but use more memory (and may exceed a remote
provider's per-request limit). Integer, `1` to `1000000`.

```bash
SYNARA_EMBEDDING_BATCH_SIZE=32
```

### `SYNARA_EMBEDDING_MAX_SEQ_LENGTH`

Maximum tokens the local model processes per text; longer inputs are
truncated before embedding. Unset uses the model's own default. Has no
effect in remote mode (the remote service controls truncation). Integer,
`1` to `1000000`.

```bash
SYNARA_EMBEDDING_MAX_SEQ_LENGTH=8192
```

---

## Dashboard

An optional web admin console that runs in the **same process and on the
same lifecycle** as the MCP server: it starts when the server starts and
shuts down with it. It exposes memory browsing, the successor/plasticity
graph, live stats, and maintenance triggers. Off by default. Requires the
optional extra:

```bash
pip install "synara-mcp[dashboard]"
```

### `SYNARA_DASHBOARD`

Master gate. Nothing dashboard-related starts unless this is truthy.
Accepts `true/false`, `1/0`, `yes/no`, `on/off` (case-insensitive); any
other value is rejected at startup. Default `false`.

```bash
SYNARA_DASHBOARD=true
```

### `SYNARA_DASHBOARD_HOST`

Bind address. Defaults to `127.0.0.1` — reachable only from the local
machine, which is the safe choice for a write-capable console. Loopback
hosts (`127.0.0.1`, `localhost`, `::1`) need no token. Binding any other
address **requires** `SYNARA_DASHBOARD_TOKEN`; without it the server
refuses to start rather than silently expose admin endpoints.

```bash
SYNARA_DASHBOARD_HOST=127.0.0.1
```

### `SYNARA_DASHBOARD_PORT`

TCP port the console listens on. Integer, `1` to `65535`. Default `8765`.

```bash
SYNARA_DASHBOARD_PORT=8765
```

### `SYNARA_DASHBOARD_TOKEN`

Bearer secret required to use the console. Optional when bound to
loopback; **mandatory** before any non-loopback bind is permitted. Sent
by the client as `Authorization: Bearer <token>` and compared in
constant time. Never written to the logs; read from the environment
only — do not commit it.

```bash
SYNARA_DASHBOARD_TOKEN=$(openssl rand -hex 32)
```

---

## Recipes

**Default — zero config, fully local.** Set nothing. Synara downloads
the Jina nano model on first run and stores memory under `~/.cache`.

**Local Ollama for embeddings:**

```bash
SYNARA_EMBEDDING_URL=http://localhost:11434
SYNARA_EMBEDDING_MODEL=nomic-embed-text
```

**OpenAI embeddings:**

```bash
SYNARA_EMBEDDING_URL=https://api.openai.com
SYNARA_EMBEDDING_MODEL=text-embedding-3-small
SYNARA_EMBEDDING_API_KEY=sk-...
SYNARA_EMBEDDING_DIM=1536
```

**Ephemeral — nothing touches disk** (tests, demos):

```bash
SYNARA_DB_PATH=:memory:
```

**Networked transport with debug logs:**

```bash
SYNARA_TRANSPORT=streamable-http
SYNARA_LOG_LEVEL=DEBUG
```

**Local admin dashboard** (loopback, no token needed):

```bash
SYNARA_DASHBOARD=true
```

---

See the [README](../README.md) for installation and an overview, and
[`.env.example`](../.env.example) for a template you can copy to `.env`.
