"""Bearer auth for the dashboard.

Posture (see :class:`DashboardConfig`): a token is optional on loopback
and mandatory before any non-loopback bind (enforced at config load).
This dependency is the request-time half: when a token is configured it
is required on *every* request and compared in constant time; when no
token is configured (loopback dev default) requests pass through.
"""

from __future__ import annotations

import hmac
from collections.abc import Awaitable, Callable

from fastapi import HTTPException, Request, status

from .config import DashboardConfig

AuthDependency = Callable[[Request], Awaitable[None]]


def make_auth_dependency(config: DashboardConfig) -> AuthDependency:
    token = config.token

    async def _require_bearer(request: Request) -> None:
        if token is None:
            return
        header = request.headers.get("Authorization", "")
        scheme, _, presented = header.partition(" ")
        if scheme.lower() != "bearer" or not hmac.compare_digest(presented, token):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="invalid or missing bearer token",
                headers={"WWW-Authenticate": "Bearer"},
            )

    return _require_bearer
