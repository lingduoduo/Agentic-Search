"""Query history admin API.

py.
Celery-based async export is replaced with synchronous streaming CSV.
External imports are replaced with project-local equivalents.
"""

from __future__ import annotations

import csv
import io
import logging

from fastapi import APIRouter
from fastapi import Depends
from fastapi import HTTPException
from fastapi import Query
from fastapi.responses import StreamingResponse

from src.backend.auth import AuthenticatedUser
from src.backend.configs import AppSettings
from src.backend.db import AgenticSearchStore
from src.backend.servers.query_history.models import ChatSessionMinimal
from src.backend.servers.query_history.models import ChatSessionSnapshot
from src.backend.servers.query_history.models import PaginatedReturn
from src.backend.servers.query_history.models import QuestionAnswerPairSnapshot
from src.backend.servers._auth import make_require_admin

logger = logging.getLogger(__name__)

_CSV_HEADERS = [
    "session_id",
    "message_pair_num",
    "user_message",
    "ai_response",
    "retrieved_documents",
    "feedback_type",
    "feedback_text",
    "user_id",
    "time_created",
    "flow_type",
]


def create_query_history_router(
    store: AgenticSearchStore,
    app_settings: AppSettings,
) -> APIRouter:
    """Return an APIRouter with query-history admin endpoints."""

    router = APIRouter(prefix="/admin", tags=["query-history"])

    _require_admin = make_require_admin(app_settings)

    @router.get("/chat-sessions")
    def get_chat_sessions_for_user(
        user_id: str,
        limit: int = Query(100, ge=1, le=500),
        _: AuthenticatedUser = Depends(_require_admin),
    ) -> list[dict[str, str | None]]:
        """Return sessions belonging to a specific user."""
        sessions = store.list_sessions_for_user(user_id, limit=limit)
        return [
            {
                "id": s.id,
                "title": s.title,
                "user_id": s.user_id,
                "created_at": s.created_at,
                "updated_at": s.updated_at,
            }
            for s in sessions
        ]

    @router.get("/chat-session-history")
    def get_chat_session_history(
        page_num: int = Query(0, ge=0),
        page_size: int = Query(10, ge=1, le=200),
        start_time: str | None = None,
        end_time: str | None = None,
        _: AuthenticatedUser = Depends(_require_admin),
    ) -> PaginatedReturn:
        """Return a paginated list of chat sessions with minimal detail."""
        sessions = store.get_paginated_chat_sessions(
            page_num=page_num,
            page_size=page_size,
            start_time=start_time,
            end_time=end_time,
        )
        total = store.get_chat_sessions_count(
            start_time=start_time,
            end_time=end_time,
        )
        items = [
            ChatSessionMinimal.from_records(
                session=s,
                messages=store.list_chat_messages(s.id),
            )
            for s in sessions
        ]
        return PaginatedReturn(items=items, total_items=total)

    @router.get("/chat-session-history/{session_id}")
    def get_chat_session_detail(
        session_id: str,
        _: AuthenticatedUser = Depends(_require_admin),
    ) -> ChatSessionSnapshot:
        """Return the full message history for a single chat session."""
        session = store.get_chat_session(session_id)
        if session is None:
            raise HTTPException(
                status_code=404,
                detail=f"Chat session '{session_id}' not found.",
            )
        messages = store.list_chat_messages(session_id)
        return ChatSessionSnapshot.from_records(session=session, messages=messages)

    @router.get("/query-history/export")
    def export_query_history(
        start_time: str | None = None,
        end_time: str | None = None,
        _: AuthenticatedUser = Depends(_require_admin),
    ) -> StreamingResponse:
        """Stream a CSV export of all question/answer pairs in the given window.

        The original version dispatched a Celery task and required a
        separate download step. Here the export is generated synchronously and
        streamed directly to the caller.
        """
        PAGE_SIZE = 100

        def _generate_csv() -> object:
            buf = io.StringIO()
            writer = csv.DictWriter(buf, fieldnames=_CSV_HEADERS)
            writer.writeheader()
            yield buf.getvalue()

            page = 0
            while True:
                sessions = store.get_paginated_chat_sessions(
                    page_num=page,
                    page_size=PAGE_SIZE,
                    start_time=start_time,
                    end_time=end_time,
                )
                if not sessions:
                    break
                for session in sessions:
                    messages = store.list_chat_messages(session.id)
                    snapshot = ChatSessionSnapshot.from_records(
                        session=session, messages=messages
                    )
                    for pair in QuestionAnswerPairSnapshot.from_snapshot(snapshot):
                        buf = io.StringIO()
                        writer = csv.DictWriter(buf, fieldnames=_CSV_HEADERS)
                        writer.writerow(pair.to_csv_row())
                        yield buf.getvalue()
                if len(sessions) < PAGE_SIZE:
                    break
                page += 1

        filename = "query_history_export.csv"
        return StreamingResponse(
            _generate_csv(),
            media_type="text/csv",
            headers={"Content-Disposition": f"attachment; filename={filename}"},
        )

    return router


__all__ = ["create_query_history_router"]
