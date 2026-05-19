"""Dashboard API routers.

Each module exposes an ``APIRouter`` mounted under ``/api`` by
:func:`synara.features.dashboard.app.build_dashboard_app`. Routers
delegate to the shared :class:`MemoryService` — no memory/SR/forget
logic lives here.
"""

from __future__ import annotations
