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

from .amygdala.signals import SignalRegistry, SignalSpec
from .memory_types import (
    MemoryType,
    MemoryTypeRegistry,
    MemoryTypeSpec,
    default_registry,
)
from .metrics import ToolMetrics, ToolSnapshot
from .port import MemoryServicePort
from .resources import register_resources
from .service import EmbedBatchFn, EmbedFn, MemoryConfig, MemoryService
from .tools import register_tools
from .tracing import RequestContext, start_request

__all__ = [
    "EmbedBatchFn",
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
    "ToolMetrics",
    "ToolSnapshot",
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
    metrics: ToolMetrics | None = None,
) -> MemoryService:
    """Wire the memory feature into the FastMCP server.

    Pass ``embedder`` (preferred) to enable lazy warmup with progress
    reporting through the MCP context. ``embed_fn`` remains supported for
    tests that drive the service with a deterministic in-process embedder
    and don't need progress/log wiring.
    """
    # Queries and documents take different paths on purpose: retrieval
    # models are commonly trained with asymmetric task prefixes, so
    # ``query_arg`` (search) must not embed text the same way
    # ``vectorise`` (storage) does. ``embed_query``/``embed_documents``
    # apply the right prefix for the active backend.
    resolved_embed_fn = (
        embed_fn if embed_fn is not None else (embedder.embed_query if embedder else None)
    )
    # Wire the batch hook only when the production embedder is in play.
    # An explicit ``embed_fn`` override (tests) keeps the per-text path —
    # those callers expect to count single-text invocations and would be
    # surprised by a coalesced batch call. Test embedders are symmetric,
    # so the shared per-text path is safe for them.
    embed_batch_fn = (
        embedder.embed_documents if (embedder is not None and embed_fn is None) else None
    )
    service = MemoryService(
        db,
        config=config,
        embed_fn=resolved_embed_fn,
        embed_batch_fn=embed_batch_fn,
        # Only the production embedder can be asymmetric: an ``embed_fn``
        # override drives both sides through the same callable, so query
        # and document encodings are identical by construction.
        embed_asymmetric=embed_batch_fn is not None
        and embedder is not None
        and embedder.asymmetric,
    )
    register_tools(mcp, service, embedder=embedder, metrics=metrics)
    register_resources(mcp, service, embedder=embedder, metrics=metrics)
    return service
