"""Embedding feature: internal capability, no MCP tools.

Other features import ``Embedder`` and call ``await embedder.embed(text)``
or ``await embedder.embed_batch(texts)``. The factory builds the right
backend (local sentence-transformer vs remote OpenAI-compatible HTTP)
from the resolved ``EmbeddingConfig``.
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
