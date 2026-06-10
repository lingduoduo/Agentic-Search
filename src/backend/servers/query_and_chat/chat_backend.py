"""Chat session management router for Agentic Search.

Provides CRUD endpoints for chat sessions backed by AgenticSearchStore.
The send-message flow is handled by POST /api/agent in the web app.
"""

from __future__ import annotations

import logging

from fastapi import APIRouter
from fastapi import HTTPException
from fastapi import Request

from src.backend.auth import AuthenticatedUser
from src.backend.auth import user_from_headers
from src.backend.db import AgenticSearchStore
from src.backend.servers.query_and_chat.models import ChatFeedbackRequest
from src.backend.servers.query_and_chat.models import ChatMessageDetail
from src.backend.servers.query_and_chat.models import ChatRenameRequest
from src.backend.servers.query_and_chat.models import ChatSessionCreationRequest
from src.backend.servers.query_and_chat.models import ChatSessionDetailResponse
from src.backend.servers.query_and_chat.models import ChatSessionDetails
from src.backend.servers.query_and_chat.models import ChatSessionsResponse
from src.backend.servers.query_and_chat.models import RenameChatSessionResponse

logger = logging.getLogger(__name__)


def create_chat_router(store: AgenticSearchStore) -> APIRouter:
    """Return an APIRouter for chat session endpoints bound to *store*."""

    router = APIRouter(prefix="/chat", tags=["chat"])

    def _get_user(request: Request) -> AuthenticatedUser | None:
        return user_from_headers(request.headers)

    @router.get("/get-user-chat-sessions")
    def get_user_chat_sessions(
        request: Request,
        page_size: int = 50,
    ) -> ChatSessionsResponse:
        user = _get_user(request)
        if user is None or user.is_anonymous:
            return ChatSessionsResponse(sessions=[], has_more=False)
        sessions = store.list_sessions_for_user(user.id, limit=page_size + 1)
        has_more = len(sessions) > page_size
        return ChatSessionsResponse(
            sessions=[
                ChatSessionDetails(
                    id=s.id,
                    title=s.title,
                    created_at=s.created_at or "",
                    updated_at=s.updated_at or "",
                )
                for s in sessions[:page_size]
            ],
            has_more=has_more,
        )

    @router.get("/get-chat-session/{session_id}")
    def get_chat_session(
        session_id: str,
        request: Request,
    ) -> ChatSessionDetailResponse:
        session = store.get_chat_session(session_id)
        if session is None:
            raise HTTPException(status_code=404, detail="Chat session not found")
        messages = store.list_chat_messages(session_id)
        return ChatSessionDetailResponse(
            session_id=session_id,
            title=session.title,
            messages=[
                ChatMessageDetail(
                    id=m.id,
                    session_id=m.session_id,
                    role=m.role,
                    content=m.content,
                    created_at=m.created_at or "",
                )
                for m in messages
            ],
        )

    @router.post("/create-chat-session")
    def create_new_chat_session(
        req: ChatSessionCreationRequest,
        request: Request,
    ) -> dict[str, str]:
        user = _get_user(request)
        session = store.create_chat_session(
            user_id=user.id if user and not user.is_anonymous else None,
            title=req.title,
        )
        return {"chat_session_id": session.id}

    @router.put("/rename-chat-session")
    def rename_chat_session(
        req: ChatRenameRequest,
        request: Request,
    ) -> RenameChatSessionResponse:
        found = store.update_chat_session_title(req.chat_session_id, req.name)
        if not found:
            raise HTTPException(status_code=404, detail="Chat session not found")
        return RenameChatSessionResponse(new_name=req.name)

    @router.delete("/delete-chat-session/{session_id}")
    def delete_chat_session_by_id(
        session_id: str,
        request: Request,
    ) -> None:
        found = store.delete_chat_session(session_id)
        if not found:
            raise HTTPException(status_code=404, detail="Chat session not found")

    @router.post("/create-chat-message-feedback")
    def create_chat_feedback(feedback: ChatFeedbackRequest) -> None:
        logger.debug(
            "Feedback received for message %s: positive=%s",
            feedback.chat_message_id,
            feedback.is_positive,
        )

    return router


__all__ = ["create_chat_router"]
