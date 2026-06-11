"""Usage report API.

py.
Celery-based async generation is replaced with synchronous report creation.
External imports are replaced with project-local equivalents.
"""

from __future__ import annotations

import io

from fastapi import APIRouter
from fastapi import Depends
from fastapi import HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from src.internal.auth import AuthenticatedUser
from src.internal.configs import AppSettings
from src.internal.db import AgenticSearchStore
from src.internal.servers.reporting.generation import create_new_usage_report
from src.internal.servers.reporting.models import UsageReportMetadata
from src.internal.servers._auth import make_require_admin


class GenerateUsageReportParams(BaseModel):
    period_from: str | None = None
    period_to: str | None = None


def create_reporting_router(
    store: AgenticSearchStore,
    app_settings: AppSettings,
) -> APIRouter:
    """Return an APIRouter with usage-report admin endpoints."""

    router = APIRouter(prefix="/admin", tags=["reporting"])

    _require_admin = make_require_admin(app_settings)

    @router.post("/usage-report", status_code=201)
    def generate_report(
        params: GenerateUsageReportParams,
        user: AuthenticatedUser = Depends(_require_admin),
    ) -> UsageReportMetadata:
        """Generate a usage report synchronously and return its metadata.

        The original version dispatched this to a Celery task.
        Here the ZIP is assembled in-memory and stored in the DB immediately.
        """
        return create_new_usage_report(
            store=store,
            requestor_id=user.id,
            period_from=params.period_from,
            period_to=params.period_to,
        )

    @router.get("/usage-report")
    def list_reports(
        _: AuthenticatedUser = Depends(_require_admin),
    ) -> list[UsageReportMetadata]:
        """Return metadata for all stored usage reports."""
        rows = store.get_all_usage_reports()
        return [
            UsageReportMetadata(
                report_name=row["report_name"],
                requestor_id=row["requestor_id"],
                time_created=row["time_created"] or "",
                period_from=row["period_from"],
                period_to=row["period_to"],
            )
            for row in rows
        ]

    @router.get("/usage-report/{report_name}")
    def download_report(
        report_name: str,
        _: AuthenticatedUser = Depends(_require_admin),
    ) -> StreamingResponse:
        """Stream the ZIP for an existing usage report."""
        data = store.get_usage_report_data(report_name)
        if data is None:
            raise HTTPException(
                status_code=404,
                detail=f"Report '{report_name}' not found.",
            )

        def _iter_bytes() -> object:
            yield from io.BytesIO(data)

        return StreamingResponse(
            _iter_bytes(),
            media_type="application/zip",
            headers={"Content-Disposition": f"attachment; filename={report_name}"},
        )

    return router


__all__ = ["GenerateUsageReportParams", "create_reporting_router"]
