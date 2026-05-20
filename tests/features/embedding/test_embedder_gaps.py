"""Coverage of error paths and lazy-init branches in features.embedding.service."""

from __future__ import annotations

from collections.abc import Sequence

import httpx
import pytest

from synara.features.embedding import (
    Embedder,
    EmbeddingError,
    RemoteBackend,
)
from synara.features.embedding import service as svc_mod


# ---------------------------------------------------------- _probe_dim guard
class _EmptyBackend:
    async def embed_batch(self, texts: Sequence[str]) -> list[list[float]]:
        return [[]]  # well-formed list, but the vector itself is empty

    def warmup(self) -> None: ...
    def is_ready(self) -> bool:
        return True

    async def aclose(self) -> None: ...
    async def dim(self) -> int:
        return await svc_mod._probe_dim(self)


async def test_probe_dim_raises_on_empty_vector() -> None:
    with pytest.raises(EmbeddingError, match="empty embedding"):
        await svc_mod._probe_dim(_EmptyBackend())


# ---------------------------------------------------------- RemoteBackend.is_ready
def test_remote_backend_is_ready_is_true_before_warmup() -> None:
    """Remote backend has no model to load locally, so it's always ready."""
    assert RemoteBackend("http://x").is_ready() is True


# ---------------------------------------------------------- _ensure_client lazy build
async def test_remote_backend_lazy_builds_client_when_none_injected() -> None:
    backend = RemoteBackend("http://x", timeout_seconds=5.0)
    assert backend._client is None
    client = backend._ensure_client()
    assert isinstance(client, httpx.AsyncClient)
    # Second call must reuse it.
    assert backend._ensure_client() is client
    await backend.aclose()


# ---------------------------------------------------------- _post non-dict body
async def test_remote_backend_raises_when_json_body_is_not_dict() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=[1, 2, 3])  # array, not object

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        backend = RemoteBackend("http://x", client=client)
        with pytest.raises(EmbeddingError, match="not a JSON object"):
            await Embedder(backend).embed("hi")


# ---------------------------------------------------------- _read_capped
async def test_remote_backend_rejects_oversized_content_length() -> None:
    huge = svc_mod._MAX_RESPONSE_BYTES + 1

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            headers={"content-length": str(huge), "content-type": "application/json"},
            content=b"{}",  # body itself is short; the header is what we cap on
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        backend = RemoteBackend("http://x", client=client)
        with pytest.raises(EmbeddingError, match="response too large"):
            await Embedder(backend).embed("hi")


class _NoLengthStream(httpx.AsyncByteStream):
    """Streams chunks without declaring Content-Length, exercising the
    in-flight cap rather than the up-front declared-length guard."""

    def __init__(self, data: bytes, chunk: int) -> None:
        self._data = data
        self._chunk = chunk

    async def __aiter__(self):  # type: ignore[no-untyped-def]
        for i in range(0, len(self._data), self._chunk):
            yield self._data[i : i + self._chunk]

    async def aclose(self) -> None:
        return None


async def test_remote_backend_rejects_oversized_streamed_body(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Body without Content-Length still hits the cap mid-stream."""
    # Drop the cap drastically so the test stays cheap.
    monkeypatch.setattr(svc_mod, "_MAX_RESPONSE_BYTES", 8)
    payload = b"a" * 32

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, stream=_NoLengthStream(payload, chunk=4))

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        backend = RemoteBackend("http://x", client=client)
        with pytest.raises(EmbeddingError, match="exceeded"):
            await Embedder(backend).embed("hi")


# ---------------------------------------------------------- malformed entry
async def test_remote_backend_raises_when_embedding_field_missing() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "object": "list",
                "data": [{"object": "embedding", "index": 0}],  # no "embedding"
                "model": "default",
                "usage": {},
            },
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        backend = RemoteBackend("http://x", client=client)
        with pytest.raises(EmbeddingError, match="missing 'embedding' list"):
            await Embedder(backend).embed("hi")


# ---------------------------------------------------------- Embedder facade
async def test_embedder_is_ready_delegates_to_backend() -> None:
    flag = {"ready": False}

    class _B:
        async def embed_batch(self, texts: Sequence[str]) -> list[list[float]]:
            return []

        def warmup(self) -> None: ...
        def is_ready(self) -> bool:
            return flag["ready"]

        async def aclose(self) -> None: ...
        async def dim(self) -> int:
            return 0

    embedder = Embedder(_B())
    assert embedder.is_ready() is False
    flag["ready"] = True
    assert embedder.is_ready() is True


async def test_embedder_embed_raises_when_backend_returns_empty() -> None:
    class _Empty:
        async def embed_batch(self, texts: Sequence[str]) -> list[list[float]]:
            return []  # backend dropped the single input — protocol violation

        def warmup(self) -> None: ...
        def is_ready(self) -> bool:
            return True

        async def aclose(self) -> None: ...
        async def dim(self) -> int:
            return 0

    with pytest.raises(EmbeddingError, match="no embeddings"):
        await Embedder(_Empty()).embed("hi")


# ---------------------------------------------------------- warmup_async
class _SpyBackend:
    def __init__(self, ready_after_warmup: bool = True) -> None:
        self.warm_calls = 0
        self._ready = False
        self._ready_after = ready_after_warmup

    async def embed_batch(self, texts: Sequence[str]) -> list[list[float]]:
        return []

    def is_ready(self) -> bool:
        return self._ready

    def warmup(self) -> None:
        self.warm_calls += 1
        self._ready = self._ready_after

    async def aclose(self) -> None: ...
    async def dim(self) -> int:
        return 0


async def test_warmup_async_skips_when_already_ready() -> None:
    backend = _SpyBackend()
    backend._ready = True
    await Embedder(backend).warmup_async()
    assert backend.warm_calls == 0


async def test_warmup_async_runs_backend_warmup_without_ctx() -> None:
    backend = _SpyBackend()
    await Embedder(backend).warmup_async()
    assert backend.warm_calls == 1


async def test_warmup_async_reports_progress_when_ctx_supplied() -> None:
    backend = _SpyBackend()
    info_calls: list[str] = []
    progress_log: list[tuple[float, float]] = []

    class _Ctx:
        async def info(self, msg: str) -> None:
            info_calls.append(msg)

        async def report_progress(self, *, progress: float, total: float) -> None:
            progress_log.append((progress, total))

    await Embedder(backend).warmup_async(ctx=_Ctx())
    assert backend.warm_calls == 1
    assert progress_log == [(0, 1), (1, 1)]
    assert len(info_calls) == 2
    assert "Loading" in info_calls[0]
    assert "ready" in info_calls[1]


async def test_warmup_async_double_check_under_lock(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """If is_ready flips to True between the outer check and the lock,
    the inner check must short-circuit and skip the thread call."""
    backend = _SpyBackend()
    flipped = {"value": False}

    real_is_ready = backend.is_ready

    def flipping_is_ready() -> bool:
        # First call (outer): False.
        # Once we've passed the outer guard, simulate another coroutine
        # finishing the warmup by flipping the flag.
        if not flipped["value"]:
            flipped["value"] = True
            return False
        return True

    monkeypatch.setattr(backend, "is_ready", flipping_is_ready)
    await Embedder(backend).warmup_async()
    # The inner check saw True and skipped backend.warmup().
    assert backend.warm_calls == 0
    # Restore so cleanup doesn't blow up.
    monkeypatch.setattr(backend, "is_ready", real_is_ready)
