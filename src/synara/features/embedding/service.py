"""Embedding service.

Internal capability, not an MCP tool surface. Other features (memory
today, future memory/reasoning modules later) use an ``Embedder`` to
turn text into vectors without caring where the vectors came from.

``LocalBackend`` runs ONNX models through ``embed-anything`` (a Rust
backend). Encode is sync, so we run it on a worker thread to keep the
event loop free. Default model is ``nomic-embed-text-v1`` (768-d,
L2-normalised).

This replaced a PyTorch + sentence-transformers stack: ~5.1 GB of
dependencies became ~194 MB, and ``trust_remote_code`` disappeared
entirely — the previous default model executed arbitrary Python bundled
in its Hugging Face repo on every load.

Nomic conditions its embeddings on a task prefix, so storage and search
are separate calls (``Embedder.embed_documents`` vs ``embed_query``).
The prefix pair is resolved from the configured model id — see
``resolve_task_prefixes`` — and is empty for families that train without
one, which makes the two calls equivalent again.

``RemoteBackend`` POSTs to ``{base_url}/v1/embeddings``. That shape
covers ollama, the bundled ``simplevecdb-server``, OpenAI, and most
other providers without a per-provider adapter class.

The factory picks based on ``SYNARA_EMBEDDING_URL``: set it for
remote, leave it unset for local.
"""

from __future__ import annotations

import asyncio
import json
import math
import threading
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


def _index_of(r: dict[str, object]) -> int:
    """Extract the OpenAI ``index`` field, rejecting missing or non-numeric values.

    Hoisted to module scope so each ``RemoteBackend.embed_batch`` chunk
    reuses the same function object instead of rebuilding a closure
    per loop iteration. A missing index must raise, not default to 0 —
    defaulting silently collapses every un-indexed item onto position
    0 and corrupts the text->vector alignment of the stored embeddings.
    """
    idx = r.get("index")
    if isinstance(idx, bool) or not isinstance(idx, (int, float)):
        raise EmbeddingError(
            "embedding entry missing numeric 'index'; cannot guarantee vector order"
        )
    return int(idx)


class EmbeddingError(RuntimeError):
    """The backend gave us back something we can't use."""


# Retrieval models are trained with *their own* task prefixes, or with
# none at all. Feeding one family's prefix to another prepends literal
# noise: measured on ``AllMiniLML6V2Q``, ``"search_document: "`` moves a
# vector ~0.25 cosine, and the same sentence embedded as query vs
# document then sits ~0.10 apart -- a large slice of the relevance
# budget, spent on tokens the model was never trained to strip. So the
# prefixes come from the model id, and an unrecognised family gets none
# (symmetric encoding), which is the safe default for BGE / GTE / MiniLM
# / Mxbai / ModernBERT / Jina. Ordered: the first marker found in the
# normalised id wins.
_TASK_PREFIXES: tuple[tuple[str, tuple[str, str]], ...] = (
    ("nomic", ("search_document: ", "search_query: ")),
    ("e5", ("passage: ", "query: ")),
)


def _normalise_model_id(model_id: str) -> str:
    """Model id reduced to comparable letters+digits (``-``/``_``/``.``/``/`` dropped)."""
    for sep in ("-", "_", ".", "/"):
        model_id = model_id.replace(sep, "")
    return model_id.lower()


def resolve_task_prefixes(model_id: str) -> tuple[str, str]:
    """``(document, query)`` task prefixes for ``model_id``.

    ``("", "")`` when the model's family is unknown or trains without
    prefixes -- see ``_TASK_PREFIXES`` for why guessing is worse than
    abstaining.
    """
    normalised = _normalise_model_id(model_id)
    for marker, prefixes in _TASK_PREFIXES:
        if marker in normalised:
            return prefixes
    return ("", "")


def _truncate_normalise(vector: list[float], dim: int) -> list[float]:
    """Cut a Matryoshka embedding to ``dim`` and restore unit length.

    Truncation alone leaves the vector shorter than 1.0, which skews
    every cosine distance computed against it. Models whose native width
    is already <= ``dim`` are returned untouched.
    """
    if dim >= len(vector):
        return vector
    head = vector[:dim]
    norm = math.sqrt(sum(x * x for x in head))
    return [x / norm for x in head] if norm > 0 else head


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
    dim: pin the output width. Locally the vector is truncated and
        re-normalised, which is valid only for Matryoshka models
        (``nomic-embed-text-v1.5``, BGE-M3). The default model,
        ``nomic-embed-text-v1``, is *not* one of them -- truncating it
        degrades the vector, so leave this unset unless the configured
        model documents Matryoshka support. Remotely it goes in the
        request as ``dimensions`` (OpenAI text-embedding-3 and friends).
        Leave it as ``None`` to keep the model's native dimension.
    batch_size: encode chunk size locally, per-request size remotely.
    max_seq_length: token cap for the local model. Lower it to trade
        context for throughput. The remote backend ignores this; the
        server decides truncation. See
        ``LocalBackend._APPROX_CHARS_PER_TOKEN`` for how the cap is
        enforced now that there is no tokenizer to count with.
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
    """ONNX encode via ``embed-anything``, run on a worker thread.

    The Rust/ONNX runtime replaces the previous PyTorch +
    sentence-transformers stack: same vectors, ~5.1 GB of dependencies
    down to ~194 MB, and no ``trust_remote_code`` — the old default model
    executed arbitrary Python bundled in its repo at load time, which
    this backend has no mechanism to do.

    Default model is ``nomic-embed-text-v1`` (768-d, L2-normalised).
    Nomic is trained with asymmetric task prefixes, so callers must go
    through :meth:`embed_documents` / :meth:`embed_query` rather than
    embedding raw text — see ``resolve_task_prefixes``.
    """

    DEFAULT_MODEL = "nomic-embed-text-v1"
    DEFAULT_BATCH_SIZE = 64
    # Task prefixes for the *configured* model, filled in by ``__init__``
    # from ``resolve_task_prefixes``. Nomic's training objective
    # conditions the embedding on one of these; storing a document under
    # the query prefix (or vice versa) puts it in a subtly different
    # region of the space and quietly degrades recall, so the two paths
    # are kept distinct all the way down rather than sharing one "embed
    # some text" call. Empty for a model family that trains without
    # prefixes, which makes both paths symmetric again.
    DOCUMENT_PREFIX: str = ""
    QUERY_PREFIX: str = ""
    # ``max_seq_length`` is expressed in tokens, but dropping
    # sentence-transformers also dropped the tokenizer we would need to
    # count them. Rather than silently ignore the setting, the cap is
    # applied in characters using this ratio. Four characters per token
    # is the usual English average for BPE/WordPiece vocabularies, and
    # erring long is harmless: the ONNX runtime's own tokenizer still
    # truncates anything above the model's real limit, so this knob only
    # ever trades context for throughput -- it is not a safety bound.
    # Swap in a real tokenizer here if exact counts ever matter.
    _APPROX_CHARS_PER_TOKEN = 4

    # Process-level cache keyed by model id, so re-instantiating the
    # backend for an already-loaded model (hot reload, tests, a second
    # feature wiring its own embedder) reuses the loaded session instead
    # of re-reading the weights.
    _CACHE: ClassVar[dict[str, Any]] = {}
    # Guards the check-then-set on ``_CACHE`` and the load. ``warmup``
    # runs on a worker thread (``to_thread``), so two concurrent
    # ``embed_batch`` calls could otherwise both observe ``_model is
    # None`` and both pay the load. A ``threading.Lock`` (not asyncio) is
    # correct because the contended region is the sync ``warmup`` body.
    _CACHE_LOCK: ClassVar[threading.Lock] = threading.Lock()

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
        # ``Any``-typed because embed_anything ships no stubs; the
        # runtime contract is just ``embed_query(texts, embedder=...)``.
        self._model: Any = None
        self._configured_dim = dim
        self._dim: int | None = dim
        self._batch_size = batch_size or self.DEFAULT_BATCH_SIZE
        self._max_seq_length = max_seq_length
        self.DOCUMENT_PREFIX, self.QUERY_PREFIX = resolve_task_prefixes(self._model_id)

    def is_ready(self) -> bool:
        return self._model is not None

    async def dim(self) -> int:
        """Output dimension, cached once found.

        The configured value wins; otherwise a one-shot encode probe
        reports the model's native width. ONNX exposes no cheap
        dimension getter, so the probe is the only source of truth.
        """
        if self._dim is not None:
            return self._dim
        self._dim = await _probe_dim(self)
        return self._dim

    def warmup(self) -> None:
        """Load the ONNX session. The first call downloads weights."""
        if self._model is not None:
            return
        with self._CACHE_LOCK:
            if self._model is not None:
                return
            cached = self._CACHE.get(self._model_id)
            if cached is not None:
                self._model = cached
                return
            # Lazy import keeps the module importable without the ONNX
            # runtime installed (lint-only CI, or a remote-embedding
            # deployment that never loads a local model).
            from embed_anything import (  # noqa: PLC0415
                EmbeddingModel,
                ONNXModel,
                WhichModel,
            )

            # A bare name resolves against embed-anything's built-in ONNX
            # registry; anything containing "/" is treated as a Hugging
            # Face repo id so a custom or fine-tuned model still works.
            known = {name.lower(): name for name in dir(ONNXModel) if not name.startswith("_")}
            key = self._model_id.replace("-", "").replace("_", "").replace(".", "").lower()
            if "/" in self._model_id:
                self._model = EmbeddingModel.from_pretrained_onnx(
                    WhichModel.Bert,
                    hf_model_id=self._model_id,
                    path_in_repo="onnx/model.onnx",
                )
            elif key in known:
                self._model = EmbeddingModel.from_pretrained_onnx(
                    WhichModel.Bert, model_name=getattr(ONNXModel, known[key])
                )
            else:
                raise EmbeddingError(
                    f"unknown local embedding model {self._model_id!r}; pass a Hugging Face "
                    f"repo id (with '/') or one of: {', '.join(sorted(known.values()))}"
                )
            self._CACHE[self._model_id] = self._model

    async def aclose(self) -> None:
        """Nothing to release; the session stays loaded for the process."""

    async def embed_batch(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []
        if self._model is None:
            # warmup() loads (and on first run downloads) the model — a
            # multi-second blocking call. Off-load it so the event loop
            # is not frozen on the first embed request.
            await asyncio.to_thread(self.warmup)
        return await asyncio.to_thread(self._encode_to_lists, list(texts))

    def _cap(self, texts: list[str]) -> list[str]:
        """Apply the ``max_seq_length`` budget (see ``_APPROX_CHARS_PER_TOKEN``).

        Truncation happens after any task prefix has been prepended,
        because the prefix occupies real positions in the model's input.
        """
        if self._max_seq_length is None:
            return texts
        limit = self._max_seq_length * self._APPROX_CHARS_PER_TOKEN
        return [t[:limit] for t in texts]

    def _encode_to_lists(self, texts: list[str]) -> list[list[float]]:
        # Imported here rather than at module scope for the same reason
        # as in ``warmup``: the ONNX runtime is an optional install.
        import embed_anything  # noqa: PLC0415

        config = embed_anything.TextEmbedConfig(batch_size=self._batch_size)
        out = embed_anything.embed_query(self._cap(texts), embedder=self._model, config=config)
        vectors = [list(item.embedding) for item in out]
        if len(vectors) != len(texts):
            raise EmbeddingError(
                f"backend returned {len(vectors)} embeddings for {len(texts)} inputs; "
                "cannot guarantee text->vector alignment"
            )
        if self._configured_dim is not None:
            # Matryoshka truncation. Nomic *v1.5* and friends are trained
            # so a prefix of the vector is itself a valid embedding, but
            # the prefix must be re-normalised to stay unit-length or
            # cosine distances shift. Nothing here can verify the model
            # actually is Matryoshka; see ``EmbeddingConfig.dim``.
            vectors = [_truncate_normalise(v, self._configured_dim) for v in vectors]
        return vectors


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
                # Capture status while the response is still open, rather
                # than reading the attribute after the stream context has
                # exited.
                status_code = response.status_code
        except httpx.HTTPError as exc:
            # Connect/timeout/protocol failures must surface as the
            # documented EmbeddingError, not a raw httpx exception that
            # callers catching EmbeddingError would miss.
            raise EmbeddingError(
                f"embedding endpoint {self._endpoint} request failed: {exc!r}"
            ) from exc
        snippet = raw[:200].decode("utf-8", "replace")
        if status_code >= _HTTP_ERROR_FLOOR:
            raise EmbeddingError(
                f"embedding endpoint {self._endpoint} returned {status_code}: {snippet}"
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

    def _prefixes(self) -> tuple[str, str]:
        """``(document, query)`` prefixes for the active backend.

        Empty for the remote backend: the server owns whatever prompt
        conditioning its model needs, and prepending ours would corrupt
        it.
        """
        doc = getattr(self._backend, "DOCUMENT_PREFIX", "")
        query = getattr(self._backend, "QUERY_PREFIX", "")
        return doc, query

    async def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        """Embed text destined for storage.

        Distinct from :meth:`embed_query` because retrieval models are
        commonly trained with asymmetric task prefixes; embedding a
        stored document as though it were a query lands it in a
        different region of the space and silently degrades recall.
        """
        doc, _ = self._prefixes()
        if not doc:
            return await self._backend.embed_batch(texts)
        return await self._backend.embed_batch([doc + t for t in texts])

    @property
    def asymmetric(self) -> bool:
        """True when documents and queries land in different regions.

        A model trained with task prefixes encodes the same sentence
        differently depending on which side of the retrieval it is on --
        measured ~0.10 cosine apart. That is invisible to a plain search
        (which is exactly what the asymmetry is *for*), but any code that
        does arithmetic *between* a query vector and stored vectors has
        to know, because the two are then not directly combinable. See
        ``hippocampus/recall._document_space``.
        """
        doc, query = self._prefixes()
        return doc != query

    async def embed_query(self, text: str) -> list[float]:
        """Embed a search query (see :meth:`embed_documents`)."""
        _, query = self._prefixes()
        out = await self._backend.embed_batch([query + text if query else text])
        if not out:
            raise EmbeddingError("backend returned no embeddings for non-empty input")
        return out[0]

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
