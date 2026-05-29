"""Embedding service tests.

Local backend is exercised against the bundled simplevecdb embedder
(small bge-micro model — vector shape is asserted, not the values, so
this is a smoke test that the lazy ImportError path is not hit).

Remote backend is exercised against an httpx MockTransport so the test
runs offline and asserts the OpenAI-compatible request/response wire
format.
"""

from __future__ import annotations

import asyncio
import json
import time

import httpx
import pytest

from synara.features.embedding import (
    Embedder,
    EmbeddingConfig,
    EmbeddingError,
    LocalBackend,
    RemoteBackend,
    build_embedder,
)


# ---------------------------------------------------------------- local backend
@pytest.mark.slow
async def test_local_backend_returns_vectors() -> None:
    # Default model (Jina v5) requires trust_remote_code; opt in for the
    # real-model smoke test.
    vec = await Embedder(LocalBackend(trust_remote_code=True)).embed("hello world")
    assert isinstance(vec, list)
    assert len(vec) > 0
    assert all(isinstance(x, float) for x in vec)


@pytest.mark.slow
async def test_local_backend_batch_preserves_order_and_dim() -> None:
    vecs = await Embedder(LocalBackend(trust_remote_code=True)).embed_batch(
        ["alpha", "beta", "gamma"]
    )
    assert len(vecs) == 3
    dims = {len(v) for v in vecs}
    assert len(dims) == 1, "all vectors must share dimensionality"


async def test_local_backend_empty_batch_short_circuits() -> None:
    embedder = Embedder(LocalBackend())
    assert await embedder.embed_batch([]) == []


# --------------------------------------------------------------- remote backend
def _ok_handler(captured: list[httpx.Request]) -> httpx.MockTransport:
    """Build a MockTransport that records requests and returns a fake batch."""

    def handler(request: httpx.Request) -> httpx.Response:
        captured.append(request)
        body = json.loads(request.content)
        texts = body["input"] if isinstance(body["input"], list) else [body["input"]]
        # Return embeddings out-of-order so the client's sort-by-index is exercised.
        data = [
            {"object": "embedding", "embedding": [float(i), 0.5], "index": i}
            for i in range(len(texts))
        ]
        data.reverse()
        return httpx.Response(
            200,
            json={
                "object": "list",
                "data": data,
                "model": body.get("model", "default"),
                "usage": {"prompt_tokens": 1, "total_tokens": 1},
            },
        )

    return httpx.MockTransport(handler)


async def test_remote_backend_posts_openai_shape_and_orders_results() -> None:
    captured: list[httpx.Request] = []
    transport = _ok_handler(captured)
    async with httpx.AsyncClient(transport=transport) as client:
        backend = RemoteBackend(
            "https://embeddings.example",
            model="bge-small",
            api_key="secret-key",
            client=client,
        )
        vecs = await Embedder(backend).embed_batch(["a", "b", "c"])

    assert len(vecs) == 3
    # MockTransport returned reversed indices; client must restore order.
    assert vecs[0][0] == 0.0
    assert vecs[2][0] == 2.0
    assert len(captured) == 1
    sent = captured[0]
    assert sent.url.path == "/v1/embeddings"
    assert sent.headers["authorization"] == "Bearer secret-key"
    payload = json.loads(sent.content)
    assert payload == {"input": ["a", "b", "c"], "model": "bge-small"}


async def test_remote_backend_omits_auth_when_no_key() -> None:
    captured: list[httpx.Request] = []
    transport = _ok_handler(captured)
    async with httpx.AsyncClient(transport=transport) as client:
        backend = RemoteBackend("http://x", client=client)
        await Embedder(backend).embed("hi")
    assert "authorization" not in {h.lower() for h in captured[0].headers}


async def test_remote_backend_raises_on_http_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, text="boom")

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        backend = RemoteBackend("http://x", client=client)
        with pytest.raises(EmbeddingError, match="500"):
            await Embedder(backend).embed("hi")


async def test_remote_backend_raises_on_count_mismatch() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "object": "list",
                "data": [{"object": "embedding", "embedding": [1.0], "index": 0}],
                "model": "default",
                "usage": {},
            },
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        backend = RemoteBackend("http://x", client=client)
        with pytest.raises(EmbeddingError, match="expected 2"):
            await Embedder(backend).embed_batch(["a", "b"])


async def test_remote_backend_empty_batch_skips_http_call() -> None:
    captured: list[httpx.Request] = []
    transport = _ok_handler(captured)
    async with httpx.AsyncClient(transport=transport) as client:
        backend = RemoteBackend("http://x", client=client)
        assert await Embedder(backend).embed_batch([]) == []
    assert captured == []


# ---------------------------------------------------------------------- warmup
def test_local_backend_warmup_constructs_sentence_transformer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """warmup() must hand the configured model id + load kwargs to ST."""
    captured: dict[str, object] = {}

    class _FakeST:
        def __init__(self, model_id: str, **kwargs: object) -> None:
            captured["model_id"] = model_id
            captured["kwargs"] = kwargs

        def encode(self, *args: object, **kwargs: object) -> object:
            raise NotImplementedError

    monkeypatch.setattr("sentence_transformers.SentenceTransformer", _FakeST)
    monkeypatch.setattr(LocalBackend, "_CACHE", {})
    LocalBackend(model="some/repo", trust_remote_code=True).warmup()
    assert captured["model_id"] == "some/repo"
    kwargs = captured["kwargs"]
    assert isinstance(kwargs, dict)
    assert kwargs["trust_remote_code"] is True
    assert "device" in kwargs


def test_local_backend_warmup_uses_default_model_when_unset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    class _FakeST:
        def __init__(self, model_id: str, **kwargs: object) -> None:
            captured["model_id"] = model_id

        def encode(self, *args: object, **kwargs: object) -> object:
            raise NotImplementedError

    monkeypatch.setattr("sentence_transformers.SentenceTransformer", _FakeST)
    monkeypatch.setattr(LocalBackend, "_CACHE", {})
    LocalBackend().warmup()
    assert captured["model_id"] == LocalBackend.DEFAULT_MODEL


def test_local_backend_warmup_is_idempotent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []

    class _FakeST:
        def __init__(self, model_id: str, **kwargs: object) -> None:
            calls.append(model_id)

        def encode(self, *args: object, **kwargs: object) -> object:
            raise NotImplementedError

    monkeypatch.setattr("sentence_transformers.SentenceTransformer", _FakeST)
    monkeypatch.setattr(LocalBackend, "_CACHE", {})
    backend = LocalBackend()
    backend.warmup()
    backend.warmup()
    assert len(calls) == 1, "second warmup must not re-instantiate the model"


def test_local_backend_reuses_cached_model_across_instances(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Second LocalBackend(model=X) hits the class cache instead of
    re-loading. This is what unblocks the transformers↔Jina v5
    re-registration crash documented on ``LocalBackend._CACHE``."""
    calls: list[str] = []

    class _FakeST:
        def __init__(self, model_id: str, **kwargs: object) -> None:
            calls.append(model_id)

        def encode(self, *args: object, **kwargs: object) -> object:
            raise NotImplementedError

    monkeypatch.setattr("sentence_transformers.SentenceTransformer", _FakeST)
    monkeypatch.setattr(LocalBackend, "_CACHE", {})  # isolated per-test cache
    LocalBackend(model="repo/x").warmup()
    LocalBackend(model="repo/x").warmup()
    assert len(calls) == 1, "second instance must reuse cached model"


def test_local_backend_trust_remote_code_defaults_true(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The flag defaults on (opt-out); the default Jina v5 model requires it."""
    captured: dict[str, object] = {}

    class _FakeST:
        def __init__(self, model_id: str, **kwargs: object) -> None:
            captured["kwargs"] = kwargs

        def encode(self, *args: object, **kwargs: object) -> object:
            raise NotImplementedError

    monkeypatch.setattr("sentence_transformers.SentenceTransformer", _FakeST)
    monkeypatch.setattr(LocalBackend, "_CACHE", {})
    LocalBackend(model="repo/x").warmup()
    kwargs = captured["kwargs"]
    assert isinstance(kwargs, dict)
    assert kwargs["trust_remote_code"] is True


async def test_local_backend_concurrent_embed_loads_model_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Concurrent ``embed_batch`` calls on a cold backend must not
    double-load the model: the cache check-then-set is lock-guarded."""
    calls: list[str] = []
    dim = 4

    class _FakeST:
        def __init__(self, model_id: str, **kwargs: object) -> None:
            calls.append(model_id)
            # Simulate a slow multi-second load so the two concurrent
            # warmups overlap in the worker-thread pool.
            time.sleep(0.05)

        def get_sentence_embedding_dimension(self) -> int:
            return dim

        def encode(self, texts: object, **kwargs: object) -> object:
            seq = list(texts) if isinstance(texts, list) else [texts]
            return [[0.0] * dim for _ in seq]

    monkeypatch.setattr("sentence_transformers.SentenceTransformer", _FakeST)
    monkeypatch.setattr(LocalBackend, "_CACHE", {})
    backend = LocalBackend(model="repo/concurrent")
    out = await asyncio.gather(
        backend.embed_batch(["x"]),
        backend.embed_batch(["y"]),
        backend.dim(),
    )
    assert len(calls) == 1, "model must be constructed exactly once under concurrency"
    vecs_x, vecs_y, reported_dim = out
    assert reported_dim == dim
    assert len(vecs_x[0]) == dim
    assert len(vecs_y[0]) == dim


def test_remote_backend_warmup_is_noop() -> None:
    backend = RemoteBackend("http://x")
    # Must not raise and must not require network access — remote model
    # lifecycle is owned by the remote service, not the client.
    backend.warmup()


async def test_embedder_warmup_and_aclose_delegate_to_backend() -> None:
    calls: list[str] = []

    class _Spy:
        def warmup(self) -> None:
            calls.append("warm")

        def is_ready(self) -> bool:
            return "warm" in calls

        async def aclose(self) -> None:
            calls.append("close")

        async def embed_batch(self, texts: object) -> list[list[float]]:
            return []

        async def dim(self) -> int:
            return 0

    embedder = Embedder(_Spy())
    embedder.warmup()
    await embedder.aclose()
    assert calls == ["warm", "close"]


# ---------------------------------------------------------------------- aclose
async def test_remote_backend_reuses_owned_client_across_calls() -> None:
    captured: list[httpx.Request] = []
    transport = _ok_handler(captured)
    backend = RemoteBackend("http://x")
    # Stub the lazily-built client with one wired to MockTransport so the
    # reuse contract is observable without real network.
    backend._client = httpx.AsyncClient(transport=transport)
    backend._owned_client = True
    first = backend._ensure_client()
    second = backend._ensure_client()
    assert first is second, "owned client must be reused, not rebuilt"
    await backend.aclose()
    assert backend._client is None


async def test_remote_backend_does_not_close_injected_client() -> None:
    captured: list[httpx.Request] = []
    transport = _ok_handler(captured)
    async with httpx.AsyncClient(transport=transport) as client:
        backend = RemoteBackend("http://x", client=client)
        await backend.aclose()
        # Caller-owned client must remain usable after backend.aclose().
        await Embedder(backend).embed("hi")
    assert len(captured) == 1


# ---------------------------------------------------------------------- dim()
async def test_remote_backend_dim_uses_probe_and_caches() -> None:
    """First dim() probes the endpoint; further calls reuse the cached value."""
    captured: list[httpx.Request] = []
    transport = _ok_handler(captured)
    async with httpx.AsyncClient(transport=transport) as client:
        backend = RemoteBackend("http://x", client=client)
        d1 = await backend.dim()
        d2 = await backend.dim()
    # _ok_handler returns 2-dim vectors; both dim() calls match.
    assert d1 == 2
    assert d2 == 2
    # Exactly one probe POST — second call hit the cache.
    assert len(captured) == 1


async def test_embedder_dim_delegates_to_backend() -> None:
    captured: list[httpx.Request] = []
    transport = _ok_handler(captured)
    async with httpx.AsyncClient(transport=transport) as client:
        embedder = Embedder(RemoteBackend("http://x", client=client))
        assert await embedder.dim() == 2


def test_local_backend_dim_uses_st_getter_when_available(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """SentenceTransformer exposes get_sentence_embedding_dimension; the
    local backend should read it directly instead of running an encode."""
    encode_calls: list[object] = []

    class _FakeST:
        def __init__(self, model_id: str, **kwargs: object) -> None:
            self.model_id = model_id

        def get_sentence_embedding_dimension(self) -> int:
            return 384

        def encode(self, *args: object, **kwargs: object) -> object:
            encode_calls.append(args)
            raise AssertionError("dim() must not run encode when getter is available")

    monkeypatch.setattr("sentence_transformers.SentenceTransformer", _FakeST)
    monkeypatch.setattr("torch.cuda.is_available", lambda: False)
    monkeypatch.setattr(LocalBackend, "_CACHE", {})

    backend = LocalBackend(model="some/repo")
    assert asyncio.run(backend.dim()) == 384
    # Cached: second call must not re-warm or re-read.
    assert asyncio.run(backend.dim()) == 384
    assert encode_calls == []


def test_local_backend_dim_falls_back_to_probe(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When the SentenceTransformer doesn't expose the dimension getter
    (older / custom subclass), dim() falls back to a single-shot encode."""
    probe_inputs: list[list[str]] = []

    class _FakeST:
        def __init__(self, model_id: str, **kwargs: object) -> None:
            self.model_id = model_id

        def encode(self, texts: object, **kwargs: object) -> object:
            assert isinstance(texts, list)
            probe_inputs.append(list(texts))
            return [[0.0, 0.0, 0.0, 0.0, 0.0]]

    monkeypatch.setattr("sentence_transformers.SentenceTransformer", _FakeST)
    monkeypatch.setattr("torch.cuda.is_available", lambda: False)
    monkeypatch.setattr(LocalBackend, "_CACHE", {})

    backend = LocalBackend(model="probe/repo")
    assert asyncio.run(backend.dim()) == 5
    # Probe ran exactly once; cached result.
    assert asyncio.run(backend.dim()) == 5
    assert len(probe_inputs) == 1


# ------------------------------------------------------- variable-dim / batch
async def test_remote_backend_sends_dimensions_when_pinned() -> None:
    """When the user pins a dimension, the request body includes
    ``dimensions`` (OpenAI text-embedding-3 / compatible providers)."""
    captured: list[httpx.Request] = []
    transport = _ok_handler(captured)
    async with httpx.AsyncClient(transport=transport) as client:
        backend = RemoteBackend("http://x", client=client, dim=128)
        await Embedder(backend).embed("hi")
    body = json.loads(captured[0].content)
    assert body["dimensions"] == 128


async def test_remote_backend_omits_dimensions_when_unset() -> None:
    captured: list[httpx.Request] = []
    transport = _ok_handler(captured)
    async with httpx.AsyncClient(transport=transport) as client:
        backend = RemoteBackend("http://x", client=client)
        await Embedder(backend).embed("hi")
    body = json.loads(captured[0].content)
    assert "dimensions" not in body


async def test_remote_backend_chunks_by_batch_size() -> None:
    """A batch of N > batch_size texts is split into multiple POSTs."""
    captured: list[httpx.Request] = []
    transport = _ok_handler(captured)
    async with httpx.AsyncClient(transport=transport) as client:
        backend = RemoteBackend("http://x", client=client, batch_size=2)
        vecs = await Embedder(backend).embed_batch(["a", "b", "c", "d", "e"])
    assert len(vecs) == 5
    # 5 texts / batch_size 2 -> 3 POSTs (2 + 2 + 1).
    assert len(captured) == 3
    sizes = [len(json.loads(r.content)["input"]) for r in captured]
    assert sizes == [2, 2, 1]


def test_remote_backend_rejects_nonpositive_options() -> None:
    with pytest.raises(ValueError, match="dim"):
        RemoteBackend("http://x", dim=0)
    with pytest.raises(ValueError, match="batch_size"):
        RemoteBackend("http://x", batch_size=0)


def test_local_backend_passes_truncate_dim_to_st(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When config pins ``dim``, the local encode call must forward
    ``truncate_dim`` so Matryoshka models truncate before normalising."""
    captured_kwargs: dict[str, object] = {}

    class _FakeST:
        def __init__(self, model_id: str, **kwargs: object) -> None:
            self.max_seq_length = 8192

        def encode(self, texts: object, **kwargs: object) -> object:
            captured_kwargs.update(kwargs)
            assert isinstance(texts, list)
            return [[0.0] * 4 for _ in texts]

    monkeypatch.setattr("sentence_transformers.SentenceTransformer", _FakeST)
    monkeypatch.setattr("torch.cuda.is_available", lambda: False)
    monkeypatch.setattr(LocalBackend, "_CACHE", {})

    backend = LocalBackend(model="m", dim=4, batch_size=8)
    asyncio.run(backend.embed_batch(["x"]))
    assert captured_kwargs["truncate_dim"] == 4
    assert captured_kwargs["batch_size"] == 8


def test_local_backend_uses_tensor_path_on_cuda(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When the model lives on CUDA, encode() must request a tensor
    (preserves bf16 in GPU memory) instead of a numpy array."""
    captured_kwargs: dict[str, object] = {}

    class _FakeDevice:
        type = "cuda"

    class _FakeTensor:
        def __init__(self, rows: list[list[float]]) -> None:
            self._rows = rows

        def float(self):  # type: ignore[no-untyped-def]
            return self

        def cpu(self):  # type: ignore[no-untyped-def]
            return self

        def tolist(self):  # type: ignore[no-untyped-def]
            return self._rows

    class _FakeST:
        device = _FakeDevice()
        max_seq_length = 8192

        def __init__(self, model_id: str, **kwargs: object) -> None:
            pass

        def encode(self, texts: object, **kwargs: object) -> object:
            captured_kwargs.update(kwargs)
            assert isinstance(texts, list)
            return _FakeTensor([[0.5, 0.25] for _ in texts])

    monkeypatch.setattr("sentence_transformers.SentenceTransformer", _FakeST)
    monkeypatch.setattr("torch.cuda.is_available", lambda: True)
    monkeypatch.setattr(LocalBackend, "_CACHE", {})

    vecs = asyncio.run(LocalBackend(model="m").embed_batch(["x"]))
    assert vecs == [[0.5, 0.25]]
    assert captured_kwargs.get("convert_to_tensor") is True
    assert "convert_to_numpy" not in captured_kwargs


def test_local_backend_omits_truncate_dim_when_unpinned(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured_kwargs: dict[str, object] = {}

    class _FakeST:
        def __init__(self, model_id: str, **kwargs: object) -> None:
            self.max_seq_length = 8192

        def encode(self, texts: object, **kwargs: object) -> object:
            captured_kwargs.update(kwargs)
            return [[0.0]]

    monkeypatch.setattr("sentence_transformers.SentenceTransformer", _FakeST)
    monkeypatch.setattr("torch.cuda.is_available", lambda: False)
    monkeypatch.setattr(LocalBackend, "_CACHE", {})

    asyncio.run(LocalBackend(model="m").embed_batch(["x"]))
    assert "truncate_dim" not in captured_kwargs


def test_local_backend_sets_max_seq_length_on_warmup(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _FakeST:
        def __init__(self, model_id: str, **kwargs: object) -> None:
            self.max_seq_length = 8192

        def encode(self, *a: object, **kw: object) -> object:
            return [[0.0]]

    monkeypatch.setattr("sentence_transformers.SentenceTransformer", _FakeST)
    monkeypatch.setattr("torch.cuda.is_available", lambda: False)
    monkeypatch.setattr(LocalBackend, "_CACHE", {})

    backend = LocalBackend(model="m", max_seq_length=512)
    backend.warmup()
    assert backend._model.max_seq_length == 512


def test_local_backend_rejects_nonpositive_options() -> None:
    with pytest.raises(ValueError, match="dim"):
        LocalBackend(dim=0)
    with pytest.raises(ValueError, match="batch_size"):
        LocalBackend(batch_size=0)
    with pytest.raises(ValueError, match="max_seq_length"):
        LocalBackend(max_seq_length=0)


async def test_local_backend_dim_returns_configured_value_without_warmup(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Pinning ``dim`` in config skips both the model getter and the
    encode probe — dim() returns the configured value immediately."""
    monkeypatch.setattr(LocalBackend, "_CACHE", {})
    backend = LocalBackend(model="m", dim=512)
    assert await backend.dim() == 512
    assert not backend.is_ready(), "configured dim must not force a warmup"


# --------------------------------------------------------------- factory wiring
def test_build_embedder_picks_local_when_no_url() -> None:
    embedder = build_embedder(EmbeddingConfig())
    # Reach into the private slot intentionally — backend selection is the
    # contract under test.
    assert isinstance(embedder._backend, LocalBackend)


def test_build_embedder_picks_remote_when_url_set() -> None:
    embedder = build_embedder(EmbeddingConfig(url="http://x", api_key="k"))
    assert isinstance(embedder._backend, RemoteBackend)


# ------------------------------------------------------ error-contract (C7/Imp)
async def test_remote_backend_wraps_network_error_as_embedding_error() -> None:
    """Connect/timeout failures must surface as EmbeddingError, not raw httpx."""

    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused", request=request)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        backend = RemoteBackend("http://x", client=client)
        with pytest.raises(EmbeddingError, match="request failed"):
            await Embedder(backend).embed("hi")


async def test_remote_backend_wraps_non_json_body_as_embedding_error() -> None:
    """A 2xx with a non-JSON body (e.g. an HTML proxy page) -> EmbeddingError."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text="<html>not json</html>")

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        backend = RemoteBackend("http://x", client=client)
        with pytest.raises(EmbeddingError, match="non-JSON"):
            await Embedder(backend).embed("hi")


async def test_remote_backend_raises_on_missing_index() -> None:
    """Missing 'index' must raise, not silently collapse vector order."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "object": "list",
                "data": [
                    {"object": "embedding", "embedding": [1.0]},
                    {"object": "embedding", "embedding": [2.0]},
                ],
                "model": "default",
                "usage": {},
            },
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        backend = RemoteBackend("http://x", client=client)
        with pytest.raises(EmbeddingError, match="index"):
            await Embedder(backend).embed_batch(["a", "b"])


def test_embedding_config_api_key_absent_from_repr() -> None:
    """The Bearer token must not leak through the dataclass repr."""
    cfg = EmbeddingConfig(url="http://x", api_key="super-secret-token")
    assert "super-secret-token" not in repr(cfg)
