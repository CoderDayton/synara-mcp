"""Memory feature.

Hippocampal-neocortical inspired memory framework backed by simplevecdb.

Two collections form the substrate:

* ``memory_episodic`` - raw event traces with rich relational metadata
  (session, tags, salience, encoded_at, retrieval_count, consolidated_into).
  This is the "inside-trail" / aHPC analogue.
* ``memory_semantic`` - consolidated schemas distilled from clusters of
  related episodes. This is the "cross-trail" / vmPFC-schema analogue.

The transformation episodic -> semantic is implemented by
``MemoryService.consolidate`` (clustering + summarisation). Pattern
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

from synara.features.embedding import Embedder

from .primitives.memory_types import (
    MemoryType,
    MemoryTypeRegistry,
    MemoryTypeSpec,
    default_registry,
)
from .primitives.port import MemoryServicePort
from .primitives.signals import SignalRegistry, SignalSpec
from .primitives.tracing import RequestContext, start_request
from .service import EmbedFn, MemoryConfig, MemoryService
from .tools import register_tools

__all__ = [
    "EmbedFn",
    "MemoryConfig",
    "MemoryService",
    "MemoryServicePort",
    "MemoryType",
    "MemoryTypeRegistry",
    "MemoryTypeSpec",
    "RequestContext",
    "SignalRegistry",
    "SignalSpec",
    "default_registry",
    "register",
    "start_request",
]


def register(
    mcp: FastMCP,
    db: AsyncVectorDB,
    *,
    config: MemoryConfig | None = None,
    embedder: Embedder | None = None,
    embed_fn: EmbedFn | None = None,
) -> MemoryService:
    """Wire the memory feature into the FastMCP server.

    Pass ``embedder`` (preferred) to enable lazy warmup with progress
    reporting through the MCP context. ``embed_fn`` remains supported for
    tests that drive the service with a deterministic in-process embedder
    and don't need progress/log wiring.
    """
    resolved_embed_fn = embed_fn if embed_fn is not None else (embedder.embed if embedder else None)
    service = MemoryService(db, config=config, embed_fn=resolved_embed_fn)
    register_tools(mcp, service, embedder=embedder)
    return service
