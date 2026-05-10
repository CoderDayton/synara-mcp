"""Embedding service.

Internal capability — not an MCP tool surface. Other features (hippocampus
today, future memory/reasoning modules tomorrow) consume an ``Embedder``
to vectorise text without caring whether vectors come from a locally
loaded SentenceTransformer or a remote OpenAI-compatible HTTP endpoint.

Two backends, one interface:

* ``LocalBackend``  - loads a SentenceTransformer directly (the bundled
  simplevecdb loader force-disables ``trust_remote_code``, which Jina v5
  requires). torch encode is sync; we offload to a thread so the event
  loop stays free. Default model: ``jinaai/jina-embeddings-v5-text-nano``,
  loaded with ``dtype=bfloat16`` and (when ``flash_attn`` is importable)
  ``_attn_implementation="flash_attention_2"`` on CUDA.
* ``RemoteBackend`` - POSTs to ``{base_url}/v1/embeddings`` (OpenAI shape).
  Covers ollama, the bundled ``simplevecdb-server``, OpenAI proper, and
  any other compatible provider with one configuration switch.

The selector is environment-driven: setting ``SYNARA_EMBEDDING_URL``
flips the factory to remote; otherwise local. No second adapter class
per provider.
"""

from __future__ import annotations

import asyncio
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, ClassVar, Protocol

import httpx

_HTTP_ERROR_FLOOR = 400


class EmbeddingError(RuntimeError):
    """Raised when an embedding backend fails to return a usable vector."""


class _Backend(Protocol):
    async def embed_batch(self, texts: Sequence[str]) -> list[list[float]]: ...

    def warmup(self) -> None: ...

    async def aclose(self) -> None: ...


@dataclass(frozen=True, slots=True)
class EmbeddingConfig:
    """Resolved embedding backend selection.

    ``url`` switches to the remote backend when truthy. ``model`` is the
    repo-id (local) or model alias/name sent to the remote endpoint.
    ``api_key`` is sent as ``Authorization: Bearer ...`` when set.
    ``timeout_seconds`` bounds remote HTTP calls.
    """

    model: str | None = None
    url: str | None = None
    api_key: str | None = None
    timeout_seconds: float = 30.0


class LocalBackend:
    """Run a SentenceTransformer (default: Jina v5 nano) off the event loop.

    The model is loaded directly rather than through simplevecdb's bundled
    embedder because Jina v5 ships custom modeling code and requires
    ``trust_remote_code=True`` — the bundled loader force-disables that flag.

    On CUDA we additionally request ``dtype=bfloat16`` and (if the
    ``flash_attn`` extension is importable) ``_attn_implementation =
    "flash_attention_2"``. flash_attn is a heavy compiled dep; missing
    it falls back silently to the default attention kernel.
    """

    DEFAULT_MODEL = "jinaai/jina-embeddings-v5-text-nano"
    _ENCODE_BATCH_SIZE = 64

    # Process-level cache keyed by model id. Re-instantiating LocalBackend
    # for an already-loaded model reuses the existing SentenceTransformer.
    # Two reasons:
    #
    # 1. Avoid re-paying the multi-GB VRAM cost on hot reloads / tests.
    # 2. Side-step a transformers ↔ Jina v5 dynamic-module-registration
    #    bug. After the first ``trust_remote_code=True`` load, transformers
    #    registers the dynamic ``JinaEmbeddingsV5Model`` into
    #    ``MODEL_MAPPING``. The next AutoModel load follows the
    #    ``has_local_code`` branch (``auto_factory.py:395``) which reads
    #    ``model_class.config_class`` — an attribute the Jina class never
    #    exposes — and crashes with ``AttributeError``. Reusing the
    #    already-loaded instance bypasses the second AutoModel call
    #    entirely.
    _CACHE: ClassVar[dict[str, Any]] = {}

    def __init__(self, model: str | None = None) -> None:
        self._model_id = model or self.DEFAULT_MODEL
        # ``Any``-typed because sentence_transformers ships no public stubs;
        # the runtime contract is just ``model.encode(...)``.
        self._model: Any = None

    def warmup(self) -> None:
        """Construct the SentenceTransformer; downloads on first run."""
        if self._model is not None:
            return
        cached = self._CACHE.get(self._model_id)
        if cached is not None:
            self._model = cached
            return
        # Lazy imports keep the module importable on machines without torch
        # (e.g. lint-only CI); the ImportError surfaces here instead.
        import torch  # noqa: PLC0415
        from sentence_transformers import SentenceTransformer  # noqa: PLC0415

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # ``default_task='retrieval'`` is required by Jina v5 — the model has
        # multiple task heads (retrieval / separation / classification /
        # similarity) and refuses to encode without one. Memory store →
        # retrieval is the natural fit.
        model_kwargs: dict[str, object] = {"default_task": "retrieval"}
        if device.type == "cuda":
            model_kwargs["dtype"] = torch.bfloat16
        load_kwargs: dict[str, object] = {
            "trust_remote_code": True,
            "device": device,
            "model_kwargs": model_kwargs,
        }
        if device.type == "cuda":
            try:
                import flash_attn  # noqa: F401, PLC0415
            except ImportError:
                pass
            else:
                load_kwargs["config_kwargs"] = {"_attn_implementation": "flash_attention_2"}
        self._model = SentenceTransformer(self._model_id, **load_kwargs)
        self._CACHE[self._model_id] = self._model

    async def aclose(self) -> None:
        """No-op: the loaded model lives for the process."""

    async def embed_batch(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []
        if self._model is None:
            self.warmup()
        return await asyncio.to_thread(self._encode, list(texts))

    def _encode(self, texts: list[str]) -> list[list[float]]:
        arr = self._model.encode(
            texts,
            normalize_embeddings=True,
            batch_size=self._ENCODE_BATCH_SIZE,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        return [[float(x) for x in row] for row in arr]


class RemoteBackend:
    """OpenAI-compatible /v1/embeddings client."""

    def __init__(
        self,
        url: str,
        *,
        model: str | None = None,
        api_key: str | None = None,
        timeout_seconds: float = 30.0,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self._endpoint = url.rstrip("/") + "/v1/embeddings"
        self._model = model or "default"
        self._headers: dict[str, str] = {"Content-Type": "application/json"}
        if api_key:
            self._headers["Authorization"] = f"Bearer {api_key}"
        self._timeout = timeout_seconds
        # Tests inject a client wired to MockTransport; production builds
        # one lazily and reuses it so HTTP keepalive amortises TLS setup
        # across the many encode/recall calls a memory feature triggers.
        self._client = client
        self._owned_client = client is None

    def warmup(self) -> None:
        """No-op: remote model lifecycle is owned by the remote service."""

    def _ensure_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=self._timeout)
        return self._client

    async def aclose(self) -> None:
        """Close the owned httpx client. Injected clients are caller-owned."""
        if self._owned_client and self._client is not None:
            await self._client.aclose()
            self._client = None

    async def _post(self, payload: dict[str, object]) -> dict[str, object]:
        client = self._ensure_client()
        response = await client.post(self._endpoint, json=payload, headers=self._headers)
        if response.status_code >= _HTTP_ERROR_FLOOR:
            raise EmbeddingError(
                f"embedding endpoint {self._endpoint} returned "
                f"{response.status_code}: {response.text[:200]}"
            )
        body = response.json()
        if not isinstance(body, dict):
            raise EmbeddingError("embedding response was not a JSON object")
        return body

    async def embed_batch(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []
        body = await self._post({"input": list(texts), "model": self._model})
        data = body.get("data")
        if not isinstance(data, list) or len(data) != len(texts):
            raise EmbeddingError(f"expected {len(texts)} embeddings, got {data!r:.80}")
        out: list[list[float]] = []
        # OpenAI guarantees an ``index`` field — sort to be safe.
        for item in sorted(data, key=lambda r: int(r.get("index", 0))):
            vector = item.get("embedding")
            if not isinstance(vector, list):
                raise EmbeddingError("embedding entry missing 'embedding' list")
            out.append([float(x) for x in vector])
        return out


class Embedder:
    """Async-first embedder with single + batch convenience methods."""

    def __init__(self, backend: _Backend) -> None:
        self._backend = backend

    def warmup(self) -> None:
        """Eagerly initialise the backend (e.g. download/load a local model)."""
        self._backend.warmup()

    async def aclose(self) -> None:
        """Release backend resources (e.g. an httpx connection pool)."""
        await self._backend.aclose()

    async def embed(self, text: str) -> list[float]:
        out = await self._backend.embed_batch([text])
        if not out:
            raise EmbeddingError("backend returned no embeddings for non-empty input")
        return out[0]

    async def embed_batch(self, texts: Sequence[str]) -> list[list[float]]:
        return await self._backend.embed_batch(texts)


def build_embedder(config: EmbeddingConfig) -> Embedder:
    """Pick a backend from config: remote if URL set, else local."""
    backend: _Backend
    if config.url:
        backend = RemoteBackend(
            config.url,
            model=config.model,
            api_key=config.api_key,
            timeout_seconds=config.timeout_seconds,
        )
    else:
        backend = LocalBackend(model=config.model)
    return Embedder(backend)
