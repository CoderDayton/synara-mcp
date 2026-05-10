"""Embedding service tests.

Local backend is exercised against the bundled simplevecdb embedder
(small bge-micro model — vector shape is asserted, not the values, so
this is a smoke test that the lazy ImportError path is not hit).

Remote backend is exercised against an httpx MockTransport so the test
runs offline and asserts the OpenAI-compatible request/response wire
format.
"""

from __future__ import annotations

import json

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
    vec = await Embedder(LocalBackend()).embed("hello world")
    assert isinstance(vec, list)
    assert len(vec) > 0
    assert all(isinstance(x, float) for x in vec)


@pytest.mark.slow
async def test_local_backend_batch_preserves_order_and_dim() -> None:
    vecs = await Embedder(LocalBackend()).embed_batch(["alpha", "beta", "gamma"])
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
    LocalBackend(model="some/repo").warmup()
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


# --------------------------------------------------------------- factory wiring
def test_build_embedder_picks_local_when_no_url() -> None:
    embedder = build_embedder(EmbeddingConfig())
    # Reach into the private slot intentionally — backend selection is the
    # contract under test.
    assert isinstance(embedder._backend, LocalBackend)


def test_build_embedder_picks_remote_when_url_set() -> None:
    embedder = build_embedder(EmbeddingConfig(url="http://x", api_key="k"))
    assert isinstance(embedder._backend, RemoteBackend)
