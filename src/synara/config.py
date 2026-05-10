"""Runtime settings sourced from environment + argv."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from synara.features.embedding import EmbeddingConfig

Transport = Literal["stdio", "http", "sse", "streamable-http"]
_VALID_TRANSPORTS: tuple[Transport, ...] = ("stdio", "http", "sse", "streamable-http")


def default_db_path() -> str:
    """Per-user vector DB path: XDG_CACHE_HOME or ~/.cache.

    Directory created lazily by the consumer.
    """
    base = os.environ.get("XDG_CACHE_HOME")
    cache_root = Path(base) if base else Path.home() / ".cache"
    return str(cache_root / "synara-mcp" / "synara.db")


@dataclass(frozen=True, slots=True)
class Settings:
    log_level: str
    transport: Transport
    db_path: str = field(default_factory=default_db_path)
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)

    @classmethod
    def from_env(cls, argv: list[str] | None = None) -> Settings:
        del argv  # reserved for future CLI flags
        raw_transport = os.environ.get("SYNARA_TRANSPORT", "stdio")
        if raw_transport not in _VALID_TRANSPORTS:
            raise ValueError(f"SYNARA_TRANSPORT={raw_transport!r} not in {_VALID_TRANSPORTS}")

        raw_timeout = os.environ.get("SYNARA_EMBEDDING_TIMEOUT", "30")
        try:
            timeout_seconds = float(raw_timeout)
        except ValueError as exc:
            raise ValueError(f"SYNARA_EMBEDDING_TIMEOUT={raw_timeout!r} must be a number") from exc
        if timeout_seconds <= 0:
            raise ValueError(f"SYNARA_EMBEDDING_TIMEOUT={raw_timeout!r} must be positive")

        dim = _positive_int_env("SYNARA_EMBEDDING_DIM")
        batch_size = _positive_int_env("SYNARA_EMBEDDING_BATCH_SIZE")
        max_seq_length = _positive_int_env("SYNARA_EMBEDDING_MAX_SEQ_LENGTH")

        return cls(
            log_level=os.environ.get("SYNARA_LOG_LEVEL", "INFO").upper(),
            transport=raw_transport,
            db_path=os.environ.get("SYNARA_DB_PATH") or default_db_path(),
            embedding=EmbeddingConfig(
                model=os.environ.get("SYNARA_EMBEDDING_MODEL") or None,
                url=os.environ.get("SYNARA_EMBEDDING_URL") or None,
                api_key=os.environ.get("SYNARA_EMBEDDING_API_KEY") or None,
                timeout_seconds=timeout_seconds,
                dim=dim,
                batch_size=batch_size if batch_size is not None else 64,
                max_seq_length=max_seq_length,
            ),
        )


def _positive_int_env(name: str) -> int | None:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return None
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(f"{name}={raw!r} must be an integer") from exc
    if value <= 0:
        raise ValueError(f"{name}={raw!r} must be positive")
    return value
