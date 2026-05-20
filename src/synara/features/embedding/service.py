"""Embedding service.

Internal capability, not an MCP tool surface. Other features (memory
today, future memory/reasoning modules later) use an ``Embedder`` to
turn text into vectors without caring where the vectors came from.

``LocalBackend`` loads a SentenceTransformer directly. We don't go
through the bundled simplevecdb loader because it disables
``trust_remote_code``, which Jina v5 needs. Encode is sync, so we run
it on a worker thread to keep the event loop free. Default model is
``jinaai/jina-embeddings-v5-text-nano``, loaded as bf16 on CUDA with
SDPA.

``RemoteBackend`` POSTs to ``{base_url}/v1/embeddings``. That shape
covers ollama, the bundled ``simplevecdb-server``, OpenAI, and most
other providers without a per-provider adapter class.

The factory picks based on ``SYNARA_EMBEDDING_URL``: set it for
remote, leave it unset for local.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, ClassVar, Protocol

import httpx

_HTTP_ERROR_FLOOR = 400
# Hard ceiling on a single embedding HTTP response body. httpx buffers
# the whole body in memory, so without a cap a hostile or misconfigured
# endpoint (or a proxy error page) can drive an unbounded allocation. A
# normal batch of embeddings is well under this.
_MAX_RESPONSE_BYTES = 32 * 1024 * 1024


class EmbeddingError(RuntimeError):
    """The backend gave us back something we can't use."""


class _Backend(Protocol):
    async def embed_batch(self, texts: Sequence[str]) -> list[list[float]]: ...

    def warmup(self) -> None: ...

    def is_ready(self) -> bool: ...

    async def aclose(self) -> None: ...

    async def dim(self) -> int: ...


_DIM_PROBE_TEXT = "."


async def _probe_dim(backend: _Backend) -> int:
    """Encode a single character to find out how wide the output vectors are."""
    out = await backend.embed_batch([_DIM_PROBE_TEXT])
    if not out or not out[0]:
        raise EmbeddingError("dimension probe returned an empty embedding")
    return len(out[0])


@dataclass(frozen=True, slots=True)
class EmbeddingConfig:
    """Backend config. Either a local model repo-id or a remote URL.

    url: if set, we use the remote backend.
    model: repo-id when local, server alias when remote.
    api_key: sent as a Bearer token on remote requests.
    timeout_seconds: HTTP timeout.
    dim: pin the output width. Locally this becomes ``truncate_dim`` on
        the SentenceTransformer call, which works for Matryoshka models
        (Jina v3/v5, BGE-M3, Nomic). Remotely it goes in the request as
        ``dimensions`` (OpenAI text-embedding-3 and friends). Leave it
        as ``None`` to keep the model's native dimension.
    batch_size: encode chunk size locally, per-request size remotely.
    max_seq_length: token cap for the local model. Lower it to trade
        context for throughput. The remote backend ignores this; the
        server decides truncation.
    """

    model: str | None = None
    url: str | None = None
    # repr=False: this dataclass is yielded into the FastMCP lifespan
    # context and may land in tracebacks/log dumps — keep the secret
    # out of its auto-generated repr.
    api_key: str | None = field(default=None, repr=False)
    timeout_seconds: float = 30.0
    dim: int | None = None
    batch_size: int = 64
    max_seq_length: int | None = None


class LocalBackend:
    """SentenceTransformer encode, run on a worker thread.

    We load the model ourselves rather than through simplevecdb because
    Jina v5 requires ``trust_remote_code=True`` and the bundled loader
    forces it off. On CUDA we use bf16 with SDPA.
    """

    DEFAULT_MODEL = "jinaai/jina-embeddings-v5-text-nano"
    DEFAULT_BATCH_SIZE = 64

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

    def __init__(
        self,
        model: str | None = None,
        *,
        dim: int | None = None,
        batch_size: int | None = None,
        max_seq_length: int | None = None,
    ) -> None:
        if dim is not None and dim <= 0:
            raise ValueError("dim must be positive when set")
        if batch_size is not None and batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if max_seq_length is not None and max_seq_length <= 0:
            raise ValueError("max_seq_length must be positive")
        self._model_id = model or self.DEFAULT_MODEL
        # ``Any``-typed because sentence_transformers ships no public stubs;
        # the runtime contract is just ``model.encode(...)``.
        self._model: Any = None
        self._configured_dim = dim
        self._dim: int | None = dim
        self._batch_size = batch_size or self.DEFAULT_BATCH_SIZE
        self._max_seq_length = max_seq_length

    def is_ready(self) -> bool:
        return self._model is not None

    async def dim(self) -> int:
        """Output dimension. Cached once we've found it.

        We try the configured value first, then the model's
        ``get_sentence_embedding_dimension`` if it has one, then fall
        back to a one-shot encode probe.
        """
        if self._dim is not None:
            return self._dim
        if self._model is None:
            await asyncio.to_thread(self.warmup)
        getter = getattr(self._model, "get_sentence_embedding_dimension", None)
        if callable(getter):
            value = getter()
            if isinstance(value, int) and value > 0:
                self._dim = value
                return self._dim
        self._dim = await _probe_dim(self)
        return self._dim

    def warmup(self) -> None:
        """Load the SentenceTransformer. The first call downloads weights."""
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
            load_kwargs["config_kwargs"] = {"_attn_implementation": "sdpa"}
        self._model = SentenceTransformer(self._model_id, **load_kwargs)
        if self._max_seq_length is not None:
            # Direct attribute write is the public knob (see
            # SentenceTransformer.max_seq_length).
            self._model.max_seq_length = self._max_seq_length
        self._CACHE[self._model_id] = self._model

    async def aclose(self) -> None:
        """Nothing to release; the model stays loaded for the process."""

    async def embed_batch(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []
        if self._model is None:
            # warmup() loads (and on first run downloads) the model — a
            # multi-second, multi-GB blocking call. Off-load it so the
            # event loop is not frozen on the first embed request.
            await asyncio.to_thread(self.warmup)
        return await asyncio.to_thread(self._encode_to_lists, list(texts))

    def _is_on_cuda(self) -> bool:
        device = getattr(self._model, "device", None)
        return getattr(device, "type", None) == "cuda"

    def _encode(self, texts: list[str]) -> Any:
        """Encode and return the raw model output.

        On CUDA you get a ``torch.Tensor`` (bf16 preserved). On CPU you
        get a numpy array. ``_encode_to_lists`` does the materialisation
        to Python lists; callers that want to stay on the GPU (e.g. to
        chain another bf16 op) can call this method directly.
        """
        encode_kwargs: dict[str, Any] = {
            "normalize_embeddings": True,
            "batch_size": self._batch_size,
            "show_progress_bar": False,
        }
        # Matryoshka truncation: only pass when the user pinned a dim,
        # so models that don't support it keep their native shape.
        if self._configured_dim is not None:
            encode_kwargs["truncate_dim"] = self._configured_dim
        if self._is_on_cuda():
            encode_kwargs["convert_to_tensor"] = True
        else:
            encode_kwargs["convert_to_numpy"] = True
        return self._model.encode(texts, **encode_kwargs)

    def _encode_to_lists(self, texts: list[str]) -> list[list[float]]:
        out = self._encode(texts)
        # Tensor (CUDA bf16): widen to float32 on CPU before materialising;
        # Python lists don't support bf16, so f32 is the standard intermediate.
        # .tolist() runs in C and yields native Python floats — far faster than
        # a per-element float() comprehension.
        if hasattr(out, "float") and hasattr(out, "cpu"):
            out = out.float().cpu()
        return out.tolist() if hasattr(out, "tolist") else [list(row) for row in out]


class RemoteBackend:
    """Client for any server speaking the OpenAI ``/v1/embeddings`` shape."""

    DEFAULT_BATCH_SIZE = 64

    def __init__(
        self,
        url: str,
        *,
        model: str | None = None,
        api_key: str | None = None,
        timeout_seconds: float = 30.0,
        client: httpx.AsyncClient | None = None,
        dim: int | None = None,
        batch_size: int | None = None,
    ) -> None:
        if dim is not None and dim <= 0:
            raise ValueError("dim must be positive when set")
        if batch_size is not None and batch_size <= 0:
            raise ValueError("batch_size must be positive")
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
        self._configured_dim = dim
        self._dim: int | None = dim
        self._batch_size = batch_size or self.DEFAULT_BATCH_SIZE

    def is_ready(self) -> bool:
        return True

    def warmup(self) -> None:
        """Nothing to do; the remote server handles its own model loading."""

    async def dim(self) -> int:
        """Output dimension. Cached after the first probe."""
        if self._dim is None:
            self._dim = await _probe_dim(self)
        return self._dim

    def _ensure_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=self._timeout)
        return self._client

    async def aclose(self) -> None:
        """Close the httpx client if we built it.

        If the caller injected one, it stays theirs to close.
        """
        if self._owned_client and self._client is not None:
            await self._client.aclose()
            self._client = None

    async def _post(self, payload: dict[str, object]) -> dict[str, object]:
        client = self._ensure_client()
        try:
            async with client.stream(
                "POST", self._endpoint, json=payload, headers=self._headers
            ) as response:
                raw = await self._read_capped(response)
        except httpx.HTTPError as exc:
            # Connect/timeout/protocol failures must surface as the
            # documented EmbeddingError, not a raw httpx exception that
            # callers catching EmbeddingError would miss.
            raise EmbeddingError(
                f"embedding endpoint {self._endpoint} request failed: {exc!r}"
            ) from exc
        snippet = raw[:200].decode("utf-8", "replace")
        if response.status_code >= _HTTP_ERROR_FLOOR:
            raise EmbeddingError(
                f"embedding endpoint {self._endpoint} returned {response.status_code}: {snippet}"
            )
        try:
            body = json.loads(raw)
        except ValueError as exc:
            # A 2xx with a non-JSON body (e.g. an HTML proxy page).
            raise EmbeddingError(
                f"embedding endpoint {self._endpoint} returned a non-JSON body: {snippet}"
            ) from exc
        if not isinstance(body, dict):
            raise EmbeddingError("embedding response was not a JSON object")
        return body

    async def _read_capped(self, response: httpx.Response) -> bytes:
        """Read the response body, aborting past ``_MAX_RESPONSE_BYTES``.

        Rejects an oversized ``Content-Length`` up front, then enforces
        the same ceiling while streaming so a chunked body without a
        declared length cannot slip past.
        """
        declared = response.headers.get("content-length")
        if declared is not None and declared.isdigit() and int(declared) > _MAX_RESPONSE_BYTES:
            raise EmbeddingError(
                f"embedding endpoint {self._endpoint} response too large: "
                f"{declared} bytes exceeds {_MAX_RESPONSE_BYTES}-byte cap"
            )
        chunks: list[bytes] = []
        total = 0
        async for chunk in response.aiter_bytes():
            total += len(chunk)
            if total > _MAX_RESPONSE_BYTES:
                raise EmbeddingError(
                    f"embedding endpoint {self._endpoint} response exceeded "
                    f"{_MAX_RESPONSE_BYTES}-byte cap"
                )
            chunks.append(chunk)
        return b"".join(chunks)

    async def embed_batch(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []
        text_list = list(texts)
        out: list[list[float]] = []
        # Chunk requests so we don't hand the server arbitrarily large
        # payloads when callers pass huge batches.
        for start in range(0, len(text_list), self._batch_size):
            chunk = text_list[start : start + self._batch_size]
            payload: dict[str, object] = {"input": chunk, "model": self._model}
            if self._configured_dim is not None:
                # OpenAI text-embedding-3-* and compatible providers
                # accept ``dimensions`` to truncate the output vector.
                payload["dimensions"] = self._configured_dim
            body = await self._post(payload)
            data = body.get("data")
            if not isinstance(data, list) or len(data) != len(chunk):
                raise EmbeddingError(f"expected {len(chunk)} embeddings, got {data!r:.80}")

            # OpenAI guarantees an ``index`` field; sort by it to
            # restore request order. A missing index must raise, not
            # default to 0 — defaulting silently collapses every
            # un-indexed item onto position 0 and corrupts the
            # text->vector alignment of the stored embeddings.
            def _index_of(r: dict[str, object]) -> int:
                idx = r.get("index")
                if isinstance(idx, bool) or not isinstance(idx, (int, float)):
                    raise EmbeddingError(
                        "embedding entry missing numeric 'index'; cannot guarantee vector order"
                    )
                return int(idx)

            for item in sorted(data, key=_index_of):
                vector = item.get("embedding")
                if not isinstance(vector, list):
                    raise EmbeddingError("embedding entry missing 'embedding' list")
                out.append([float(x) for x in vector])
        return out


class Embedder:
    """Async wrapper around a backend, with single and batch methods."""

    def __init__(self, backend: _Backend) -> None:
        self._backend = backend
        # Serialises ``warmup_async``: without it, two concurrent first
        # calls (e.g. ``store_episode`` and ``recall_episodes`` arriving
        # before the model is loaded) both see ``is_ready() == False``
        # and both run the backend warmup, which for the local backend
        # means downloading + loading a multi-GB model twice and
        # double-allocating VRAM. Lock construction is lazy under
        # asyncio so this is safe even when no loop is running yet.
        self._warmup_lock = asyncio.Lock()

    def warmup(self) -> None:
        """Force the backend to load now (for the local case, this downloads the model)."""
        self._backend.warmup()

    def is_ready(self) -> bool:
        """True once the backend can encode without further loading."""
        return self._backend.is_ready()

    async def warmup_async(self, ctx: Any | None = None) -> None:
        """Load the backend. The first call may pull multi-GB weights.

        Safe to call repeatedly and from concurrent coroutines: a single
        warmup runs to completion while other callers wait. If you pass
        a ``ctx``, we'll report progress through ``ctx.info`` and
        ``ctx.report_progress``.
        """
        if self._backend.is_ready():
            return
        async with self._warmup_lock:
            if self._backend.is_ready():
                return
            if ctx is not None:
                await ctx.info("Loading embedding model — first run may download model weights")
                await ctx.report_progress(progress=0, total=1)
            await asyncio.to_thread(self._backend.warmup)
            if ctx is not None:
                await ctx.report_progress(progress=1, total=1)
                await ctx.info("Embedding model ready")

    async def aclose(self) -> None:
        """Release whatever the backend is holding (httpx pool, etc.)."""
        await self._backend.aclose()

    async def embed(self, text: str) -> list[float]:
        out = await self._backend.embed_batch([text])
        if not out:
            raise EmbeddingError("backend returned no embeddings for non-empty input")
        return out[0]

    async def embed_batch(self, texts: Sequence[str]) -> list[list[float]]:
        return await self._backend.embed_batch(texts)

    async def dim(self) -> int:
        """Output vector width. We detect it on the first call and cache it.

        That way callers can size their own buffers or projectors
        without anyone having to declare a dimension up-front.
        """
        return await self._backend.dim()


def build_embedder(config: EmbeddingConfig) -> Embedder:
    """Pick a backend from the config. URL set means remote; otherwise local."""
    backend: _Backend
    if config.url:
        backend = RemoteBackend(
            config.url,
            model=config.model,
            api_key=config.api_key,
            timeout_seconds=config.timeout_seconds,
            dim=config.dim,
            batch_size=config.batch_size,
        )
    else:
        backend = LocalBackend(
            model=config.model,
            dim=config.dim,
            batch_size=config.batch_size,
            max_seq_length=config.max_seq_length,
        )
    return Embedder(backend)
