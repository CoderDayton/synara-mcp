# Changelog

All notable changes to this project are documented here.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).
While the version is below `1.0.0`, minor releases may still break
compatibility; the notes below call out anything that does.

The release workflow reads the section matching the pushed tag and uses it
as the GitHub Release body, so every tagged version needs a heading of the
form `## [X.Y.Z] - YYYY-MM-DD`.

## [Unreleased]

## [0.1.0] - 2026-07-25

First public release.

### Added

- **MCP memory server.** Ten tools over FastMCP — `store_episode`,
  `recall_episodes`, `get_episode`, `remove_episode`,
  `consolidate_episodes`, `forget_episodes`, `reflect_session`,
  `store_semantic_memory`, `recall_semantic_memory`, `memory_stats` —
  plus a `memory://recall/{query}` resource for ambient, non-reinforcing
  recall.
- **Episodic + semantic stores** backed by `simplevecdb`, with
  consolidation of episode clusters into semantic schemas and power-law
  forgetting of traces that stop being retrieved.
- **Successor representation.** A durable transition graph between
  episodes, built from co-occurrence inside a session window and used as
  a relational prior when ranking recall, so results reflect structure
  rather than cosine distance alone.
- **Hippocampal mechanics.** Theta-segmented encoding, DG-style pattern
  separation, CA3 pattern completion, reconsolidation write-back,
  off-policy dream replay, and a persistent plasticity layer.
- **Scope axis.** Every record is `session` or `global`. Recall is scoped
  to the caller's session by default and still returns global records;
  `scope_session` overrides it in both directions. Global semantic
  memories are opt-in, never a silent default.
- **Recall miss reports.** A recall that finds nothing explains why —
  empty store, session scoping, tag filter, or nothing close enough —
  with per-reason drop counts, and probes the episodic store when a
  semantic lookup misses.
- **Tolerant tool arguments.** Pre-validation middleware repairs common
  near-miss call shapes (`tags="a,b"`, `limit`/`n`/`top_k` for `k`,
  `sessionId` for `session_id`) while the declared schema stays narrow.
- **Local-first embedding.** On-device ONNX inference via
  `embed-anything` (default `nomic-embed-text-v1`), with asymmetric
  query/document prefixes handled per model, or any OpenAI-compatible
  HTTP endpoint via `SYNARA_EMBEDDING_URL`.
- **Multi-client coordination.** Concurrent stdio clients elect a single
  leader that owns the database; the rest proxy onto it and re-elect if
  the leader dies.
- **Optional admin dashboard** (`synara-mcp[dashboard]`): a FastAPI +
  SPA console running in parallel on the MCP lifecycle, requiring a token
  before any non-loopback bind.

[Unreleased]: https://github.com/CoderDayton/synara-mcp/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/CoderDayton/synara-mcp/releases/tag/v0.1.0
