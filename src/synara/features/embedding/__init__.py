"""Embedding feature: internal capability, no MCP tools.

Other features import ``Embedder`` and call ``await
embedder.embed_documents(texts)`` for text going into storage and ``await
embedder.embed_query(text)`` for a search. Use those two, not the
symmetric ``embed``/``embed_batch``: retrieval models are commonly
trained with asymmetric task prefixes, and embedding a stored document as
though it were a query silently degrades recall. The factory builds the
right backend (local ONNX vs remote OpenAI-compatible HTTP) from the
resolved ``EmbeddingConfig``.
"""

from __future__ import annotations

from .service import (
    Embedder,
    EmbeddingConfig,
    EmbeddingError,
    LocalBackend,
    RemoteBackend,
    build_embedder,
)

__all__ = [
    "Embedder",
    "EmbeddingConfig",
    "EmbeddingError",
    "LocalBackend",
    "RemoteBackend",
    "build_embedder",
]
