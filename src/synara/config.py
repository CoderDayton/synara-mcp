"""Runtime settings sourced from environment + argv."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Literal

Transport = Literal["stdio", "http", "sse", "streamable-http"]
_VALID_TRANSPORTS: tuple[Transport, ...] = ("stdio", "http", "sse", "streamable-http")


@dataclass(frozen=True, slots=True)
class Settings:
    log_level: str
    transport: Transport
    db_path: str = ":memory:"

    @classmethod
    def from_env(cls, argv: list[str] | None = None) -> Settings:
        del argv  # reserved for future CLI flags
        raw_transport = os.environ.get("SYNARA_TRANSPORT", "stdio")
        if raw_transport not in _VALID_TRANSPORTS:
            raise ValueError(f"SYNARA_TRANSPORT={raw_transport!r} not in {_VALID_TRANSPORTS}")
        return cls(
            log_level=os.environ.get("SYNARA_LOG_LEVEL", "INFO").upper(),
            transport=raw_transport,
            db_path=os.environ.get("SYNARA_DB_PATH", ":memory:"),
        )
