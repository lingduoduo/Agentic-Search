"""Admin observability surface API."""

from __future__ import annotations

from fastapi import APIRouter
from fastapi import Depends

from src.backend.auth import AuthenticatedUser
from src.backend.configs import AppSettings
from src.backend.db import AgenticSearchStore
from src.backend.observability import AdminSurfaceSummary
from src.backend.observability import build_admin_surface_summary
from src.backend.servers._auth import make_require_admin


def create_admin_surface_router(
    store: AgenticSearchStore,
    app_settings: AppSettings,
) -> APIRouter:
    router = APIRouter(prefix="/admin/observability", tags=["admin-observability"])
    _require_admin = make_require_admin(app_settings)

    @router.get("/summary")
    def get_admin_surface_summary(
        _: AuthenticatedUser = Depends(_require_admin),
    ) -> AdminSurfaceSummary:
        return build_admin_surface_summary(store, app_settings)

    return router


__all__ = ["create_admin_surface_router"]
