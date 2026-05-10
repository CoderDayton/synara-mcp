"""Hippocampus feature.

Hippocampal-neocortical inspired memory framework backed by simplevecdb.

Two collections form the substrate:

* ``hippocampus_episodic`` - raw event traces with rich relational metadata
  (session, tags, salience, encoded_at, retrieval_count, consolidated_into).
  This is the "inside-trail" / aHPC analogue.
* ``hippocampus_semantic`` - consolidated schemas distilled from clusters of
  related episodes. This is the "cross-trail" / vmPFC-schema analogue.

The transformation episodic -> semantic is implemented by
``HippocampusService.consolidate`` (clustering + summarisation). Pattern
separation lives in ``encode_episode`` (near-duplicate refusal). Pattern
completion lives in ``recall``. Forgetting follows an Ebbinghaus decay law
in ``forget``.

Parameter-internalisation (synaptic weight modification via SFT/RL on
historical trajectories) is *out of scope* for an MCP server: it would
require a training loop with access to the host model's weights, which
this server does not have. The framework here owns only the explicit,
externalised half of the loop.
"""

from __future__ import annotations

from fastmcp import FastMCP
from simplevecdb import AsyncVectorDB

from .service import EmbedFn, HippocampusConfig, HippocampusService
from .tools import register_tools

__all__ = [
    "EmbedFn",
    "HippocampusConfig",
    "HippocampusService",
    "register",
]


def register(
    mcp: FastMCP,
    db: AsyncVectorDB,
    *,
    config: HippocampusConfig | None = None,
    embed_fn: EmbedFn | None = None,
) -> HippocampusService:
    """Wire the hippocampus feature into the FastMCP server.

    ``embed_fn`` is optional. When ``None``, simplevecdb's bundled local
    embedder is used (downloads a SentenceTransformer on first call).
    Tests pass a deterministic embedder to avoid network access.
    """
    service = HippocampusService(db, config=config, embed_fn=embed_fn)
    register_tools(mcp, service)
    return service
