"""FastAPI app for a browser-based search and agent experience."""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from starlette.middleware.base import BaseHTTPMiddleware

from src.auth import AuthenticatedUser
from src.auth import user_from_headers
from src.configs import AppSettings
from src.configs import Tier
from src.configs import is_path_allowed_for_tier
from src.configs import load_app_settings
from src.context import ChatMessage
from src.context import LLMClient
from src.context import answer_with_retrieval
from src.context.models import AnswerGenerationResult
from src.context.models import ContextDocument
from src.context.preprocessing.access_filters import build_user_only_filters
from src.db import AgenticSearchStore

from .static import APP_CSS
from .static import APP_HTML
from .static import APP_JS

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SearchExperienceSettings:
    """Runtime settings for the browser search app."""

    search_url: str = "http://localhost:8000/retrieve"
    top_k: int = 5
    db_path: str | Path = ":memory:"

    @classmethod
    def from_app_settings(
        cls,
        settings: AppSettings | None = None,
    ) -> "SearchExperienceSettings":
        app_settings = settings or load_app_settings()
        return cls(
            search_url=app_settings.services.retrieval_url,
            top_k=app_settings.services.web_top_k,
            db_path=app_settings.services.web_db_path,
        )


class SessionCreateRequest(BaseModel):
    title: str | None = None
    user_id: str | None = None


class ChatMessageView(BaseModel):
    role: str
    content: str


class ChatSessionView(BaseModel):
    id: str
    title: str | None = None
    user_id: str | None = None
    messages: list[ChatMessageView] = Field(default_factory=list)


class SourceDocumentView(BaseModel):
    id: str
    citation: str
    title: str
    content: str
    url: str | None = None
    score: float = 0.0
    metadata: dict[str, object] = Field(default_factory=dict)


class AgentExperienceRequest(BaseModel):
    query: str = Field(..., min_length=1)
    session_id: str | None = None
    user_id: str | None = None
    search_url: str | None = None
    top_k: int = Field(default=5, ge=1, le=20)


class AgentExperienceResponse(BaseModel):
    session_id: str
    answer: str
    citations: list[str]
    documents: list[SourceDocumentView]
    messages: list[ChatMessageView]


def create_web_app(
    settings: SearchExperienceSettings | None = None,
    *,
    app_settings: AppSettings | None = None,
    store: AgenticSearchStore | None = None,
    llm: LLMClient | None = None,
) -> FastAPI:
    """Create the user-facing web app.

    The app serves a dependency-free HTML/JS interface and a small JSON API that
    runs the repo's retrieval-grounded answer pipeline. Chat state is persisted
    through `AgenticSearchStore`, which defaults to an in-memory SQLite DB.
    """

    resolved = app_settings or load_app_settings()
    settings = settings or SearchExperienceSettings.from_app_settings(resolved)
    owns_store = store is None
    db = store or AgenticSearchStore(settings.db_path)

    @asynccontextmanager
    async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
        try:
            yield
        finally:
            if owns_store:
                db.close()

    app = FastAPI(title="Agentic Search Web", lifespan=lifespan)
    if resolved.license_enforcement_enabled:
        app.add_middleware(_LicenseMiddleware, tier=Tier.FREE)
    frontend_dist = _frontend_dist_path()

    @app.get("/health")
    def healthcheck() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/", response_class=HTMLResponse)
    def index() -> str:
        if frontend_dist:
            return (frontend_dist / "index.html").read_text(encoding="utf-8")
        return APP_HTML

    @app.get("/assets/app.css")
    def stylesheet() -> Response:
        return Response(APP_CSS, media_type="text/css")

    @app.get("/assets/app.js")
    def javascript() -> Response:
        return Response(APP_JS, media_type="application/javascript")

    if frontend_dist:
        app.mount(
            "/assets",
            StaticFiles(directory=frontend_dist / "assets"),
            name="frontend-assets",
        )

    @app.post("/api/sessions")
    def create_session(request: SessionCreateRequest) -> ChatSessionView:
        session = db.create_chat_session(
            user_id=request.user_id,
            title=request.title or "Search session",
            metadata={"source": "web"},
        )
        return ChatSessionView(
            id=session.id,
            title=session.title,
            user_id=session.user_id,
            messages=[],
        )

    @app.get("/api/sessions/{session_id}")
    def get_session(session_id: str) -> ChatSessionView:
        session = db.get_chat_session(session_id)
        if session is None:
            raise HTTPException(status_code=404, detail="Session not found")
        return ChatSessionView(
            id=session.id,
            title=session.title,
            user_id=session.user_id,
            messages=[
                ChatMessageView(role=message.role, content=message.content)
                for message in db.list_chat_messages(session.id)
            ],
        )

    @app.post("/api/agent")
    async def run_agent(
        request: AgentExperienceRequest,
        http_request: Request,
    ) -> AgentExperienceResponse:
        query = request.query.strip()
        if not query:
            raise HTTPException(status_code=422, detail="query is required")

        auth_user = _optional_user_from_request(http_request)
        user_id = request.user_id or (auth_user.id if auth_user else None)
        session_request = _copy_agent_request(request, user_id=user_id)
        session_id = _ensure_session(db, session_request)
        history = [
            ChatMessage(role=message.role, content=message.content)
            for message in db.list_chat_messages(session_id)
        ]
        db.add_chat_message(session_id, role="user", content=query)

        try:
            result = await answer_with_retrieval(
                query,
                llm=llm,
                chat_history=history,
                search_url=request.search_url or settings.search_url,
                top_k=request.top_k or settings.top_k,
                filters=(
                    build_user_only_filters(
                        auth_user.id,
                        email=auth_user.email,
                        group_ids=auth_user.group_ids,
                    )
                    if auth_user
                    else (build_user_only_filters(user_id) if user_id else None)
                ),
            )
        except Exception as exc:
            logger.exception("Agent search failed: %s", exc)
            search_url = request.search_url or settings.search_url
            detail = (
                (
                    f"Cannot reach retrieval server at {search_url}. "
                    "Start it with: python3 -m src.servers.retrieval.retrieval "
                    "--retrieval_method bm25"
                )
                if "connect" in str(exc).lower() or "retriev" in str(exc).lower()
                else "Agent search failed"
            )
            raise HTTPException(status_code=502, detail=detail) from exc

        db.add_chat_message(
            session_id,
            role="assistant",
            content=result.answer,
            metadata={
                "citations": result.citations,
                "document_ids": [document.id for document in result.context.documents],
            },
        )
        messages = [
            ChatMessageView(role=message.role, content=message.content)
            for message in db.list_chat_messages(session_id)
        ]
        return _response_from_result(session_id, result, messages)

    return app


def _ensure_session(
    store: AgenticSearchStore,
    request: AgentExperienceRequest,
) -> str:
    if request.session_id and store.get_chat_session(request.session_id):
        return request.session_id
    session = store.create_chat_session(
        user_id=request.user_id,
        title=request.query[:80],
        metadata={"source": "web"},
        session_id=request.session_id,
    )
    return session.id


def _copy_agent_request(
    request: AgentExperienceRequest,
    *,
    user_id: str | None,
) -> AgentExperienceRequest:
    model_copy = getattr(request, "model_copy", None)
    if callable(model_copy):
        return model_copy(update={"user_id": user_id})
    return request.copy(update={"user_id": user_id})


def _response_from_result(
    session_id: str,
    result: AnswerGenerationResult,
    messages: list[ChatMessageView],
) -> AgentExperienceResponse:
    return AgentExperienceResponse(
        session_id=session_id,
        answer=result.answer,
        citations=result.citations,
        documents=[_document_view(document) for document in result.context.documents],
        messages=messages,
    )


def _document_view(document: ContextDocument) -> SourceDocumentView:
    return SourceDocumentView(
        id=document.id,
        citation=document.citation,
        title=document.title,
        content=document.content,
        url=document.url,
        score=document.score,
        metadata=document.metadata,
    )


def _optional_user_from_request(request: Request) -> AuthenticatedUser | None:
    return user_from_headers(request.headers)


class _LicenseMiddleware(BaseHTTPMiddleware):
    def __init__(self, app: FastAPI, *, tier: Tier) -> None:
        super().__init__(app)
        self.tier = tier

    async def dispatch(self, request: Request, call_next):
        if not is_path_allowed_for_tier(request.url.path, self.tier):
            return Response(
                content='{"detail":"Feature not available on current tier."}',
                status_code=403,
                media_type="application/json",
            )
        return await call_next(request)


def _frontend_dist_path() -> Path | None:
    dist = Path(__file__).resolve().parents[3] / "web" / "dist"
    if (dist / "index.html").exists() and (dist / "assets").is_dir():
        return dist
    return None


app = create_web_app()
