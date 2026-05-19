"""Dashboard runtime settings sourced from environment.

Stdlib-only by contract (see package docstring): imported transitively
by :mod:`synara.config`.

Security posture: bind ``127.0.0.1`` by default; a token is optional for
loopback but **required** before any non-loopback bind is permitted —
enforced as a hard ``ValueError`` at startup so a write-capable console
can never be silently exposed to the network.
"""

from __future__ import annotations

import ipaddress
import os
from dataclasses import dataclass, field

_TRUE = {"1", "true", "yes", "on"}
_FALSE = {"0", "false", "no", "off"}
_LOOPBACK_NAMES = {"localhost"}

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8765
_MAX_PORT = 65535


def _bool_env(name: str, *, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    value = raw.strip().lower()
    if value in _TRUE:
        return True
    if value in _FALSE:
        return False
    raise ValueError(f"{name}={raw!r} must be a boolean (true/false)")


def _port_env(name: str, *, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(f"{name}={raw!r} must be an integer") from exc
    if not 1 <= value <= _MAX_PORT:
        raise ValueError(f"{name}={raw!r} must be in 1..{_MAX_PORT}")
    return value


def _host_is_loopback(host: str) -> bool:
    candidate = host.strip().lower()
    if candidate in _LOOPBACK_NAMES:
        return True
    try:
        return ipaddress.ip_address(candidate).is_loopback
    except ValueError:
        # A hostname we cannot resolve here: treat as non-loopback so the
        # token requirement errs on the safe side.
        return False


@dataclass(frozen=True, slots=True)
class DashboardConfig:
    """Parallel FastAPI console settings.

    enabled: master gate (``SYNARA_DASHBOARD``); nothing starts unless set.
    host/port: bind address (defaults to loopback:8765).
    token: optional bearer secret; required for any non-loopback bind.
    """

    enabled: bool = False
    host: str = DEFAULT_HOST
    port: int = DEFAULT_PORT
    token: str | None = field(default=None, repr=False)

    @property
    def is_loopback(self) -> bool:
        return _host_is_loopback(self.host)

    @classmethod
    def from_env(cls) -> DashboardConfig:
        cfg = cls(
            enabled=_bool_env("SYNARA_DASHBOARD", default=False),
            host=os.environ.get("SYNARA_DASHBOARD_HOST") or DEFAULT_HOST,
            port=_port_env("SYNARA_DASHBOARD_PORT", default=DEFAULT_PORT),
            token=os.environ.get("SYNARA_DASHBOARD_TOKEN") or None,
        )
        if cfg.enabled and not cfg.is_loopback and cfg.token is None:
            raise ValueError(
                "SYNARA_DASHBOARD is enabled with a non-loopback "
                f"SYNARA_DASHBOARD_HOST={cfg.host!r} but no "
                "SYNARA_DASHBOARD_TOKEN set; refusing to expose a "
                "write-capable admin console without a token"
            )
        return cfg
