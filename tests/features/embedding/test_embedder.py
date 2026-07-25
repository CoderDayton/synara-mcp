"""Embedding service tests.

The local backend runs ONNX models through embed-anything. Fast tests
patch that surface via ``_install_fake_onnx`` so they stay offline; the
few that load real weights are marked ``slow`` and assert vector shape
rather than values.

Remote backend is exercised against an httpx MockTransport so the test
runs offline and asserts the OpenAI-compatible request/response wire
format.
"""

from __future__ import annotations

import asyncio
import json
import math
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
from synara.features.embedding.service import resolve_task_prefixes


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


@pytest.mark.slow
async def test_local_backend_applies_asymmetric_task_prefixes() -> None:
    """Nomic conditions on a task prefix, so storage and search must not
    embed identical text identically -- if they do, the prefixes are not
    being applied and recall quality silently degrades."""
    embedder = Embedder(LocalBackend())
    doc = (await embedder.embed_documents(["the deploy script needs sudo"]))[0]
    query = await embedder.embed_query("the deploy script needs sudo")
    assert doc != query


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
class _FakeEmbedding:
    """Stands in for embed_anything's EmbedData (only ``.embedding`` is read)."""

    def __init__(self, values: list[float]) -> None:
        self.embedding = values


def _install_fake_onnx(
    monkeypatch: pytest.MonkeyPatch,
    *,
    dim: int = 4,
    load_delay: float = 0.0,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    """Patch the embed_anything surface LocalBackend actually calls.

    ``LocalBackend`` imports embed_anything lazily inside ``warmup`` and
    ``_encode_to_lists``, so patching module attributes is enough — the
    names resolve at call time. Returns ``(loads, captured)``: every
    model construction, and the last encode's texts/config.
    """
    loads: list[dict[str, object]] = []
    captured: dict[str, object] = {}

    class _FakeModel:
        def __init__(self, label: object) -> None:
            self.label = label

    class _FakeEmbeddingModel:
        @staticmethod
        def from_pretrained_onnx(
            which: object,
            *,
            model_name: object = None,
            hf_model_id: object = None,
            path_in_repo: object = None,
        ) -> _FakeModel:
            loads.append(
                {
                    "which": which,
                    "model_name": model_name,
                    "hf_model_id": hf_model_id,
                    "path_in_repo": path_in_repo,
                }
            )
            if load_delay:
                # Simulate a slow load so concurrent warmups overlap.
                time.sleep(load_delay)
            return _FakeModel(hf_model_id or model_name)

    def _fake_embed_query(
        texts: list[str], *, embedder: object, config: object = None
    ) -> list[_FakeEmbedding]:
        captured["texts"] = list(texts)
        captured["config"] = config
        # Unit-norm so truncation genuinely needs re-normalising.
        return [_FakeEmbedding([0.5] * dim) for _ in texts]

    class _FakeConfig:
        def __init__(self, **kwargs: object) -> None:
            self.kwargs = kwargs

    monkeypatch.setattr("embed_anything.EmbeddingModel", _FakeEmbeddingModel)
    monkeypatch.setattr("embed_anything.embed_query", _fake_embed_query)
    monkeypatch.setattr("embed_anything.TextEmbedConfig", _FakeConfig)
    monkeypatch.setattr(LocalBackend, "_CACHE", {})
    return loads, captured


def test_local_backend_warmup_resolves_hf_repo_id(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A model id containing '/' is treated as a Hugging Face repo."""
    loads, _ = _install_fake_onnx(monkeypatch)
    LocalBackend(model="some/repo").warmup()
    assert loads[0]["hf_model_id"] == "some/repo"
    assert loads[0]["path_in_repo"] == "onnx/model.onnx"
    assert loads[0]["model_name"] is None


def test_local_backend_warmup_resolves_builtin_enum_name(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A bare name resolves against embed-anything's ONNX registry.

    Matching is punctuation-insensitive so the documented
    ``nomic-embed-text-v1`` reaches the ``NomicEmbedTextV1`` enum member.
    """
    loads, _ = _install_fake_onnx(monkeypatch)
    LocalBackend(model="nomic-embed-text-v1").warmup()
    assert loads[0]["hf_model_id"] is None
    assert loads[0]["model_name"] is not None


def test_local_backend_rejects_unknown_model_name(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An unrecognised bare name fails loudly, listing the valid options.

    Without this it would fall through to a confusing loader error far
    from the actual mistake (a typo in SYNARA_EMBEDDING_MODEL).
    """
    _install_fake_onnx(monkeypatch)
    with pytest.raises(EmbeddingError, match="unknown local embedding model"):
        LocalBackend(model="not-a-real-model").warmup()


def test_local_backend_warmup_uses_default_model_when_unset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    loads, _ = _install_fake_onnx(monkeypatch)
    LocalBackend().warmup()
    assert loads, "warmup must construct a model"
    assert LocalBackend.DEFAULT_MODEL == "nomic-embed-text-v1"


def test_local_backend_warmup_is_idempotent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    loads, _ = _install_fake_onnx(monkeypatch)
    backend = LocalBackend()
    backend.warmup()
    backend.warmup()
    assert len(loads) == 1, "second warmup must not re-instantiate the model"


def test_local_backend_reuses_cached_model_across_instances(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A second LocalBackend for the same id reuses the loaded session."""
    loads, _ = _install_fake_onnx(monkeypatch)
    LocalBackend(model="repo/x").warmup()
    LocalBackend(model="repo/x").warmup()
    assert len(loads) == 1, "second instance must reuse cached model"


async def test_local_backend_concurrent_embed_loads_model_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Concurrent ``embed_batch`` calls on a cold backend must not
    double-load: the cache check-then-set is lock-guarded."""
    dim = 4
    loads, _ = _install_fake_onnx(monkeypatch, dim=dim, load_delay=0.05)
    backend = LocalBackend(model="repo/concurrent")
    vecs_x, vecs_y, reported_dim = await asyncio.gather(
        backend.embed_batch(["x"]),
        backend.embed_batch(["y"]),
        backend.dim(),
    )
    assert len(loads) == 1, "model must be constructed exactly once under concurrency"
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


def test_local_backend_dim_uses_probe_and_caches(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """ONNX exposes no cheap dimension getter, so dim() probes once.

    The old backend could read ``get_sentence_embedding_dimension``; the
    encode probe is now the only source of truth, and it must be cached
    so repeated calls don't re-encode.
    """
    _, captured = _install_fake_onnx(monkeypatch, dim=5)
    backend = LocalBackend(model="probe/repo")
    assert asyncio.run(backend.dim()) == 5
    captured.clear()
    assert asyncio.run(backend.dim()) == 5
    assert captured == {}, "second dim() must use the cached value"


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


def test_local_backend_truncates_and_renormalises_when_dim_pinned(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A pinned ``dim`` truncates the vector AND restores unit length.

    Truncation alone leaves the vector short of 1.0, which skews every
    cosine distance computed against it — so the re-normalisation is the
    part that actually matters here.
    """
    _install_fake_onnx(monkeypatch, dim=4)
    vecs = asyncio.run(LocalBackend(model="repo/m", dim=2, batch_size=8).embed_batch(["x"]))
    assert len(vecs[0]) == 2
    norm = math.sqrt(sum(v * v for v in vecs[0]))
    assert norm == pytest.approx(1.0)


def test_local_backend_keeps_native_dim_when_unpinned(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_onnx(monkeypatch, dim=4)
    vecs = asyncio.run(LocalBackend(model="repo/m").embed_batch(["x"]))
    assert len(vecs[0]) == 4


def test_local_backend_forwards_batch_size(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, captured = _install_fake_onnx(monkeypatch)
    asyncio.run(LocalBackend(model="repo/m", batch_size=8).embed_batch(["x"]))
    config = captured["config"]
    assert getattr(config, "kwargs", {})["batch_size"] == 8


def test_local_backend_caps_input_by_max_seq_length(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``max_seq_length`` is honoured by truncating the input text.

    Dropping sentence-transformers dropped the tokenizer too, so the cap
    is applied in characters via ``_APPROX_CHARS_PER_TOKEN``. The setting
    must still have a visible effect rather than being silently ignored.
    """
    _, captured = _install_fake_onnx(monkeypatch)
    backend = LocalBackend(model="repo/m", max_seq_length=10)
    asyncio.run(backend.embed_batch(["y" * 500]))
    texts = captured["texts"]
    assert isinstance(texts, list)
    assert len(texts[0]) == 10 * LocalBackend._APPROX_CHARS_PER_TOKEN


def test_local_backend_unbounded_input_when_max_seq_length_unset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, captured = _install_fake_onnx(monkeypatch)
    asyncio.run(LocalBackend(model="repo/m").embed_batch(["y" * 500]))
    texts = captured["texts"]
    assert isinstance(texts, list)
    assert len(texts[0]) == 500


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
    backend = LocalBackend(model="repo/m", dim=512)
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


@pytest.mark.parametrize(
    ("model", "expected"),
    [
        ("nomic-embed-text-v1", ("search_document: ", "search_query: ")),
        ("NomicEmbedTextV15", ("search_document: ", "search_query: ")),
        ("nomic-ai/nomic-embed-text-v1.5", ("search_document: ", "search_query: ")),
        ("MultilingualE5Large", ("passage: ", "query: ")),
        ("intfloat/multilingual-e5-large", ("passage: ", "query: ")),
        # Families trained without task prefixes must get none: prepending
        # Nomic's tokens moves a MiniLM/BGE vector ~0.25 cosine and pushes a
        # query ~0.1 away from its own document, all of it noise.
        ("BGESmallENV15", ("", "")),
        ("BAAI/bge-base-en-v1.5", ("", "")),
        ("AllMiniLML6V2Q", ("", "")),
        ("GTEBaseENV15", ("", "")),
        ("MxbaiEmbedLargeV1", ("", "")),
    ],
)
def test_task_prefixes_follow_the_model_family(model: str, expected: tuple[str, str]) -> None:
    assert resolve_task_prefixes(model) == expected
    backend = LocalBackend(model=model)
    assert expected == (backend.DOCUMENT_PREFIX, backend.QUERY_PREFIX)
    embedder = Embedder(backend)
    assert embedder._prefixes() == expected
    # The flag consumers key off: it must say "asymmetric" exactly when
    # the two sides encode differently, so a symmetric model never pays
    # for the second encode the correction needs.
    assert embedder.asymmetric is (expected[0] != expected[1])


def test_remote_backend_is_symmetric() -> None:
    """The server owns its own prompt conditioning; we add none, so a
    remote embedder's query and document vectors are the same vector."""
    assert Embedder(RemoteBackend("http://x", model="nomic-embed-text-v1")).asymmetric is False
