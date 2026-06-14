"""FastAPI app for a browser-based search and agent experience."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from itertools import islice
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, Response
from fastapi.staticfiles import StaticFiles
from dotenv import load_dotenv
from pydantic import BaseModel, Field

from src.internal.auth import AuthenticatedUser
from src.internal.auth import user_from_headers
from src.internal.db.models import UserRecord
from src.internal.configs import AppSettings
from src.internal.configs import load_app_settings
from src.internal.llm.interfaces import LLMConfig
from src.internal.llm.providers import OpenAICompatibleLLM
from src.internal.search.process_search_query import run_expanded_search
from src.internal.servers.secondary_llm_flows import expand_keywords
from src.internal.servers.secondary_llm_flows.query_expansion import (
    with_temporal_context,
)
from src.agents.agentic_rag import AgenticRAGConfig, AgenticRAGLoop
from src.context import ChatMessage
from src.context import LLMClient
from src.context import answer_with_retrieval
from src.context import build_context_bundle
from src.context.utils import mmr_rerank
from src.context.models import AnswerGenerationResult
from src.context.models import ContextDocument
from src.context.models import SearchFilters
from src.context.preprocessing.access_filters import build_user_only_filters
from src.internal.db import AgenticSearchStore
from src.internal.hooks import HookPoint
from src.internal.hooks import HookRegistry
from src.internal.hooks import HookSoftFailed
from src.internal.hooks import execute_hook
from src.internal.servers.admin_surface.api import create_admin_surface_router
from src.internal.servers.analytics.api import create_analytics_router
from src.internal.servers.web.auth_check import PUBLIC_ENDPOINT_SPECS
from src.internal.servers.web.auth_check import check_router_auth
from src.internal.servers.billing.api import create_billing_router
from src.internal.servers.connectors.api import create_connectors_router
from src.internal.servers.tools.api import create_tools_router
from src.internal.servers.documents.cc_pair import create_documents_router
from src.internal.servers.enterprise_settings.api import (
    create_enterprise_settings_routers,
)
from src.internal.servers.evals.api import create_evals_router
from src.internal.servers.features.hooks.api import create_hooks_router
from src.internal.servers.license.api import create_license_router
from src.internal.servers.manage.standard_answer import create_manage_router
from src.internal.servers.middleware.license_enforcement import (
    add_license_enforcement_middleware,
)
from src.internal.servers.middleware.tenant_tracking import (
    add_api_server_tenant_id_middleware,
)
from src.internal.servers.middleware.tier_gate import add_tier_gate_middleware
from src.internal.servers.oauth.api import create_oauth_router
from src.internal.servers.query_and_chat.chat_backend import create_chat_router
from src.internal.servers.query_and_chat.query_backend import (
    basic_router as query_basic_router,
)
from src.internal.servers.query_and_chat.search_backend import create_search_router
from src.internal.servers.query_history.api import create_query_history_router
from src.internal.servers.reporting.api import create_reporting_router
from src.internal.servers.scim.api import create_scim_router
from src.internal.servers.scim.api import register_scim_exception_handlers
from src.internal.servers.web.seeding import seed_db
from src.internal.servers.settings.api import create_settings_router
from src.internal.servers.tenants.api import router as tenants_router
from src.internal.servers.token_rate_limits.api import create_token_rate_limits_router
from src.internal.servers.user_group.api import create_user_group_router
from src.internal.servers.users.api import create_users_router
from src.tools import SearchPage
from src.tools import fetch_pages_concurrently
from src.tools import search_tool

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
    metadata: dict[str, object] = Field(default_factory=dict)


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
    source_provider: str = Field(
        default="retrieval",
        description=(
            "'retrieval', 'serpapi', 'browser', or 'all'. "
            "Browser uses the retrieval-compatible URL in search_url."
        ),
    )
    mode: str = Field(
        default="chat_once",
        description=(
            "'search_tool', 'hybrid_search', 'chat_once', or 'chat_loop'. "
            "'standard' and 'agentic_rag' are accepted as legacy aliases."
        ),
    )


class AgentExperienceResponse(BaseModel):
    session_id: str
    answer: str
    citations: list[str]
    documents: list[SourceDocumentView]
    messages: list[ChatMessageView]
    hook_metadata: dict[str, object] = Field(default_factory=dict)


class QueryProcessingHookResponse(BaseModel):
    query: str | None = None
    metadata: dict[str, object] = Field(default_factory=dict)


@dataclass(frozen=True)
class _HybridSearchResult:
    executed_queries: list[str]
    documents: list[ContextDocument]


def _register_routers(
    app: FastAPI,
    db: AgenticSearchStore,
    settings: AppSettings,
    search_url: str,
) -> None:
    """Attach all API routers and exception handlers to *app*."""

    # --- User auth ---
    app.include_router(create_users_router(db, settings))

    # --- Core search & chat ---
    app.include_router(create_chat_router(db))
    app.include_router(create_search_router(db, search_url=search_url))
    app.include_router(query_basic_router)
    app.include_router(create_query_history_router(db, settings))

    # --- Documents & connectors ---
    app.include_router(create_documents_router(db, settings))
    app.include_router(create_connectors_router(db, settings))
    app.include_router(create_tools_router(settings))

    # --- Users & groups ---
    app.include_router(create_user_group_router(db, settings))
    app.include_router(tenants_router)

    # --- Admin: enterprise settings, evals, hooks ---
    ee_admin_router, ee_basic_router = create_enterprise_settings_routers(settings)
    app.include_router(ee_admin_router)
    app.include_router(ee_basic_router)
    app.include_router(create_evals_router(settings, search_url=search_url))
    app.include_router(create_hooks_router(db, settings))

    # --- Admin: license, billing ---
    app.include_router(create_license_router(settings))
    app.include_router(create_billing_router(settings))

    # --- Admin: rate limits, standard answers, reporting ---
    app.include_router(create_admin_surface_router(db, settings))
    app.include_router(create_token_rate_limits_router(db, settings))
    app.include_router(create_manage_router(db, settings))
    app.include_router(create_reporting_router(db, settings))

    # --- Admin: OAuth, settings, SCIM ---
    app.include_router(create_oauth_router(settings))
    app.include_router(create_settings_router(settings))
    app.include_router(create_scim_router(db))
    register_scim_exception_handlers(app)

    # --- Analytics ---
    app.include_router(create_analytics_router(db, settings))


def create_web_app(
    settings: SearchExperienceSettings | None = None,
    *,
    app_settings: AppSettings | None = None,
    store: AgenticSearchStore | None = None,
    llm: LLMClient | None = None,
    hook_registry: HookRegistry | None = None,
) -> FastAPI:
    """Create the user-facing web app.

    The app serves a dependency-free HTML/JS interface and a small JSON API that
    runs the repo's retrieval-grounded answer pipeline. Chat state is persisted
    through `AgenticSearchStore`, which defaults to an in-memory SQLite DB.
    """
    load_dotenv()
    resolved = app_settings or load_app_settings()
    settings = settings or SearchExperienceSettings.from_app_settings(resolved)
    owns_store = store is None
    db = store or AgenticSearchStore(settings.db_path)
    if llm is None:
        import os

        api_key = resolved.llm.api_key or os.environ.get("OPENAI_API_KEY")
        if api_key:
            llm = OpenAICompatibleLLM(
                LLMConfig(
                    model_provider=resolved.llm.model_provider,
                    model_name=resolved.llm.model_name,
                    api_key=api_key,
                    api_base=resolved.llm.api_base,
                    max_input_tokens=resolved.llm.max_input_tokens,
                )
            )

    @asynccontextmanager
    async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
        seed_db(db)
        check_router_auth(_app, PUBLIC_ENDPOINT_SPECS)
        _app.state.search_agent_manager = None
        _app.state.search_agent_tokenizer = None
        if resolved.search_agent_server_url:
            try:
                from transformers import AutoTokenizer
                from examples.run_agentic_search import OpenAIServerManager

                model = resolved.search_agent_model or "Qwen/Qwen2.5-1.5B-Instruct"
                tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
                if tokenizer.pad_token_id is None:
                    tokenizer.pad_token_id = tokenizer.eos_token_id
                manager = OpenAIServerManager(
                    tokenizer=tokenizer,
                    base_url=resolved.search_agent_server_url,
                    model=model,
                )
                _app.state.search_agent_tokenizer = tokenizer
                _app.state.search_agent_manager = manager
                logger.info(
                    "search_agent: connected to OpenAI-compatible server at %s (model %s)",
                    resolved.search_agent_server_url,
                    model,
                )
            except Exception:
                logger.exception(
                    "search_agent: failed to init OpenAI server manager — mode will return 400"
                )
        elif resolved.search_agent_model:
            try:
                from transformers import AutoTokenizer
                from examples.run_agentic_search import LocalServerManager

                tokenizer = AutoTokenizer.from_pretrained(
                    resolved.search_agent_model,
                    trust_remote_code=True,
                    local_files_only=True,
                )
                if tokenizer.pad_token_id is None:
                    tokenizer.pad_token_id = tokenizer.eos_token_id
                manager = LocalServerManager(
                    model_path=resolved.search_agent_model,
                    device=resolved.search_agent_device,
                    allow_unsafe_mps=True,
                    local_files_only=True,
                )
                _app.state.search_agent_tokenizer = tokenizer
                _app.state.search_agent_manager = manager
                logger.info(
                    "search_agent: loaded %s on %s",
                    resolved.search_agent_model,
                    resolved.search_agent_device,
                )
            except Exception:
                logger.exception(
                    "search_agent: failed to load model — mode will return 400"
                )
        try:
            yield
        finally:
            if owns_store:
                db.close()

    app = FastAPI(title="Agentic Search Web", lifespan=lifespan)
    add_api_server_tenant_id_middleware(app, resolved)
    add_license_enforcement_middleware(app, resolved)
    add_tier_gate_middleware(app, resolved)

    _register_routers(app, db, resolved, search_url=settings.search_url)

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
                ChatMessageView(
                    role=message.role,
                    content=message.content,
                    metadata=message.metadata,
                )
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
        hook_metadata: dict[str, object] = {}

        auth_user = _optional_user_from_request(http_request)
        user_id = request.user_id or (auth_user.id if auth_user else None)
        hook_result = execute_hook(
            hook_point=HookPoint.QUERY_PROCESSING,
            payload={"query": query, "user_id": user_id},
            response_type=QueryProcessingHookResponse,
            registry=hook_registry,
        )
        if isinstance(hook_result, QueryProcessingHookResponse):
            if hook_result.query and hook_result.query.strip():
                query = hook_result.query.strip()
            hook_metadata = hook_result.metadata
        elif isinstance(hook_result, HookSoftFailed):
            hook_metadata = {"query_processing_hook_error": hook_result.error_message}

        mode = _normalize_agent_mode(request.mode)
        session_request = _copy_agent_request(request, user_id=user_id)
        session_id = _ensure_session(db, session_request, auth_user=auth_user)
        history = [
            ChatMessage(role=message.role, content=message.content)
            for message in db.list_chat_messages(session_id)
        ]
        db.add_chat_message(session_id, role="user", content=query)

        search_url = request.search_url or settings.search_url
        top_k = request.top_k or settings.top_k
        filters = (
            build_user_only_filters(
                auth_user.id,
                email=auth_user.email,
                group_ids=auth_user.group_ids,
            )
            if auth_user
            else (build_user_only_filters(user_id) if user_id else None)
        )

        try:
            if mode == "search_tool":
                source_provider = _normalize_source_provider(request.source_provider)
                documents = await _run_direct_search(
                    query,
                    source_provider=source_provider,
                    search_url=search_url,
                    top_k=top_k,
                )
                answer = _search_only_answer(
                    "Direct search tool",
                    queries=[query],
                    documents=documents,
                    source_provider=source_provider,
                )
                db.add_chat_message(
                    session_id,
                    role="assistant",
                    content=answer,
                    metadata={
                        "citations": [doc.citation for doc in documents],
                        "document_ids": [doc.id for doc in documents],
                        "hooks": hook_metadata,
                        "mode": mode,
                        "source_provider": source_provider,
                    },
                )
                messages = [
                    ChatMessageView(role=m.role, content=m.content, metadata=m.metadata)
                    for m in db.list_chat_messages(session_id)
                ]
                return AgentExperienceResponse(
                    session_id=session_id,
                    answer=answer,
                    citations=[doc.citation for doc in documents],
                    documents=[_document_view(doc) for doc in documents],
                    messages=messages,
                    hook_metadata=hook_metadata,
                )

            if mode == "hybrid_search":
                source_provider = _normalize_source_provider(request.source_provider)
                search_result = await _run_hybrid_search(
                    query,
                    llm=llm,
                    search_url=search_url,
                    top_k=top_k,
                    filters=filters,
                    source_provider=source_provider,
                )
                answer = _search_only_answer(
                    "Hybrid search",
                    queries=search_result.executed_queries,
                    documents=search_result.documents,
                    source_provider=source_provider,
                )
                db.add_chat_message(
                    session_id,
                    role="assistant",
                    content=answer,
                    metadata={
                        "citations": [doc.citation for doc in search_result.documents],
                        "document_ids": [doc.id for doc in search_result.documents],
                        "hooks": hook_metadata,
                        "mode": mode,
                        "source_provider": source_provider,
                        "executed_queries": search_result.executed_queries,
                    },
                )
                messages = [
                    ChatMessageView(role=m.role, content=m.content, metadata=m.metadata)
                    for m in db.list_chat_messages(session_id)
                ]
                return AgentExperienceResponse(
                    session_id=session_id,
                    answer=answer,
                    citations=[doc.citation for doc in search_result.documents],
                    documents=[_document_view(doc) for doc in search_result.documents],
                    messages=messages,
                    hook_metadata=hook_metadata,
                )

            if mode == "chat_loop":
                rag_loop = AgenticRAGLoop(
                    AgenticRAGConfig(
                        max_rounds=3, topk=top_k, retrieval_url=search_url
                    ),
                    llm=llm,
                )
                rag = await rag_loop.run(query, chat_history=history)
                db.add_chat_message(
                    session_id,
                    role="assistant",
                    content=rag.answer,
                    metadata={
                        "citations": rag.citations,
                        "document_ids": [doc.id for doc in rag.context.documents],
                        "hooks": hook_metadata,
                        "rounds_used": rag.rounds_used,
                        "mode": mode,
                    },
                )
                messages = [
                    ChatMessageView(role=m.role, content=m.content, metadata=m.metadata)
                    for m in db.list_chat_messages(session_id)
                ]
                return AgentExperienceResponse(
                    session_id=session_id,
                    answer=rag.answer,
                    citations=rag.citations,
                    documents=[_document_view(doc) for doc in rag.context.documents],
                    messages=messages,
                    hook_metadata=hook_metadata,
                )
            if mode == "search_agent":
                from src.agents.search import SearchAgentLoop, SearchAgentLoopConfig

                manager = getattr(http_request.app.state, "search_agent_manager", None)
                tokenizer = getattr(
                    http_request.app.state, "search_agent_tokenizer", None
                )
                if manager is None or tokenizer is None:
                    raise HTTPException(
                        status_code=400,
                        detail=(
                            "search_agent mode is not configured. "
                            "Set SEARCH_AGENT_MODEL in .env and restart the server."
                        ),
                    )
                loop = SearchAgentLoop(
                    tokenizer=tokenizer,
                    server_manager=manager,
                    search_config=SearchAgentLoopConfig(
                        search_url=search_url,
                        topk=top_k,
                        max_turns=3,
                    ),
                )
                output = await loop.run(
                    [{"role": "user", "content": query}],
                    sampling_params={"temperature": 0.0, "max_tokens": 256},
                )
                answer = output.final_answer or ""
                sa_documents: list[ContextDocument] = []
                if output.context is not None:
                    for sc in output.context.turns:
                        for result in sc.results:
                            sa_documents.append(
                                ContextDocument.from_search_result(
                                    result, index=len(sa_documents) + 1
                                )
                            )
                sa_documents = _dedupe_documents(sa_documents)
                db.add_chat_message(
                    session_id,
                    role="assistant",
                    content=answer,
                    metadata={
                        "citations": [doc.citation for doc in sa_documents],
                        "document_ids": [doc.id for doc in sa_documents],
                        "hooks": hook_metadata,
                        "mode": mode,
                        "num_turns": output.num_turns,
                    },
                )
                messages = [
                    ChatMessageView(role=m.role, content=m.content, metadata=m.metadata)
                    for m in db.list_chat_messages(session_id)
                ]
                return AgentExperienceResponse(
                    session_id=session_id,
                    answer=answer,
                    citations=[doc.citation for doc in sa_documents],
                    documents=[_document_view(doc) for doc in sa_documents],
                    messages=messages,
                    hook_metadata=hook_metadata,
                )

            if mode == "tool_agent":
                from src.agents.tool_calling import ToolAgentLoop, ToolAgentLoopConfig
                from src.tools import build_search_tool, tool_registry

                manager = getattr(http_request.app.state, "search_agent_manager", None)
                tokenizer = getattr(
                    http_request.app.state, "search_agent_tokenizer", None
                )
                if manager is None or tokenizer is None:
                    raise HTTPException(
                        status_code=400,
                        detail=(
                            "tool_agent mode requires SEARCH_AGENT_MODEL or "
                            "SEARCH_AGENT_SERVER_URL to be set."
                        ),
                    )
                tools = [
                    build_search_tool(search_url=search_url)
                ] + tool_registry.list_tools()
                loop = ToolAgentLoop(
                    tokenizer=tokenizer,
                    server_manager=manager,
                    tools=tools,
                    config=ToolAgentLoopConfig(
                        tool_parser_format=resolved.tool_agent_parser,
                    ),
                )
                output = await loop.run(
                    [{"role": "user", "content": query}],
                    sampling_params={"temperature": 0.0, "max_tokens": 512},
                )
                answer = output.final_answer or next(
                    (
                        m["content"]
                        for m in reversed(output.trajectory_messages)
                        if m.get("role") == "assistant"
                    ),
                    "",
                )
                db.add_chat_message(
                    session_id,
                    role="assistant",
                    content=answer,
                    metadata={
                        "mode": mode,
                        "hooks": hook_metadata,
                        "num_turns": output.num_turns,
                    },
                )
                messages = [
                    ChatMessageView(role=m.role, content=m.content, metadata=m.metadata)
                    for m in db.list_chat_messages(session_id)
                ]
                return AgentExperienceResponse(
                    session_id=session_id,
                    answer=answer,
                    citations=[],
                    documents=[],
                    messages=messages,
                    hook_metadata=hook_metadata,
                )

            result = await answer_with_retrieval(
                query,
                llm=llm,
                chat_history=history,
                search_url=search_url,
                top_k=top_k,
                filters=filters,
            )
        except HTTPException:
            raise
        except Exception as exc:
            logger.exception("Agent search failed: %s", exc)
            search_url = request.search_url or settings.search_url
            detail = (
                (
                    f"Cannot reach retrieval server at {search_url}. "
                    "Start it with: python3 -m src.internal.servers.retrieval.retrieval_server "
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
                "hooks": hook_metadata,
                "mode": mode,
            },
        )
        messages = [
            ChatMessageView(
                role=message.role, content=message.content, metadata=message.metadata
            )
            for message in db.list_chat_messages(session_id)
        ]
        return _response_from_result(
            session_id, result, messages, hook_metadata=hook_metadata
        )

    return app


def _ensure_session(
    store: AgenticSearchStore,
    request: AgentExperienceRequest,
    auth_user: AuthenticatedUser | None = None,
) -> str:
    if request.session_id and store.get_chat_session(request.session_id):
        return request.session_id
    if auth_user is not None and request.user_id:
        store.upsert_user(UserRecord(id=auth_user.id, email=auth_user.email))
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


_MODE_ALIASES = {
    "standard": "chat_once",
    "agentic_rag": "chat_loop",
}
_VALID_AGENT_MODES = {
    "search_tool",
    "hybrid_search",
    "chat_once",
    "chat_loop",
    "search_agent",
    "tool_agent",
}


def _normalize_agent_mode(mode: str) -> str:
    requested = mode.strip().lower()
    normalized = _MODE_ALIASES.get(requested, requested)
    if normalized not in _VALID_AGENT_MODES:
        valid = ", ".join(sorted(_VALID_AGENT_MODES))
        raise HTTPException(status_code=422, detail=f"mode must be one of: {valid}")
    return normalized


_SOURCE_PROVIDER_ALIASES = {
    "local": "retrieval",
    "direct": "retrieval",
    "serp": "serpapi",
    "web": "all",
}
_VALID_SOURCE_PROVIDERS = {
    "retrieval",
    "serpapi",
    "browser",
    "all",
}
_SOURCE_PROVIDER_LABELS = {
    "retrieval": "Local Retrieval",
    "serpapi": "SerpAPI",
    "browser": "Browser Retrieval",
    "all": "All Active Sources",
}


def _normalize_source_provider(source_provider: str) -> str:
    requested = source_provider.strip().lower()
    normalized = _SOURCE_PROVIDER_ALIASES.get(requested, requested)
    if normalized not in _VALID_SOURCE_PROVIDERS:
        valid = ", ".join(sorted(_VALID_SOURCE_PROVIDERS))
        raise HTTPException(
            status_code=422,
            detail=f"source_provider must be one of: {valid}",
        )
    return normalized


def _source_providers_for(source_provider: str) -> list[str]:
    if source_provider == "all":
        return ["retrieval", "serpapi"]
    return [source_provider]


_WEB_PROVIDERS = {"serpapi"}


def _is_web_provider(source_provider: str) -> bool:
    """Returns True for providers that return URL snippets needing full-page fetch."""
    return source_provider in _WEB_PROVIDERS


def _tool_provider_for(source_provider: str):
    return "retrieval" if source_provider == "browser" else source_provider


def _source_label(source_provider: str) -> str:
    return _SOURCE_PROVIDER_LABELS.get(source_provider, source_provider)


async def _run_direct_search(
    query: str,
    *,
    source_provider: str,
    search_url: str,
    top_k: int,
) -> list[ContextDocument]:
    # Over-fetch so MMR has candidates beyond top_k to diversify from.
    fetch_k = top_k * 2
    documents: list[ContextDocument] = []
    for provider in _source_providers_for(source_provider):
        pages = await search_tool(
            query,
            provider=_tool_provider_for(provider),
            search_url=search_url,
            page_size=fetch_k,
        )
        if _is_web_provider(provider):
            pages = await fetch_pages_concurrently(pages, max_chars=2000)
        documents.extend(
            _documents_from_search_pages(
                pages,
                source_provider=provider,
                query=query,
                start_index=len(documents) + 1,
            )
        )
    deduped = _dedupe_documents(documents)
    diversified = mmr_rerank(deduped, topk=top_k)
    return _reindex_documents(diversified)


async def _run_hybrid_search(
    query: str,
    *,
    llm: LLMClient | None,
    search_url: str,
    top_k: int,
    filters: SearchFilters | None,
    source_provider: str,
) -> _HybridSearchResult:
    if source_provider in {"retrieval", "browser"}:
        # Over-fetch so MMR has candidates beyond top_k to diversify from.
        search_result = await run_expanded_search(
            query,
            llm=llm,
            search_url=search_url,
            top_k=top_k * 2,
            filters=filters,
            expand=True,
        )
        context = build_context_bundle(
            query,
            search_result.results,
            max_documents=top_k * 2,
        )
        diversified = mmr_rerank(context.documents, topk=top_k)
        return _HybridSearchResult(
            executed_queries=search_result.executed_queries,
            documents=[
                _document_with_metadata(
                    doc,
                    source_provider=source_provider,
                    query=query,
                    entry_point="hybrid_search",
                )
                for doc in diversified
            ],
        )

    executed_queries = _expanded_queries(query, llm)
    documents: list[ContextDocument] = []
    for provider in _source_providers_for(source_provider):
        tool_provider = _tool_provider_for(provider)
        # Run all expanded queries concurrently for this provider
        page_lists: list[list[SearchPage]] = list(
            await asyncio.gather(
                *[
                    search_tool(
                        expanded_query,
                        provider=tool_provider,
                        search_url=search_url,
                        page_size=top_k,
                    )
                    for expanded_query in executed_queries
                ]
            )
        )
        if _is_web_provider(provider):
            all_pages = [p for pages in page_lists for p in pages]
            enriched = await fetch_pages_concurrently(all_pages, max_chars=2000)
            # Re-partition enriched pages back into per-query slices
            it = iter(enriched)
            page_lists = [list(islice(it, len(pages))) for pages in page_lists]
        for expanded_query, pages in zip(executed_queries, page_lists):
            documents.extend(
                _documents_from_search_pages(
                    pages,
                    source_provider=provider,
                    query=expanded_query,
                    start_index=len(documents) + 1,
                    entry_point="hybrid_search",
                )
            )
    deduped = _dedupe_documents(documents)
    diversified = mmr_rerank(deduped, topk=top_k)
    return _HybridSearchResult(
        executed_queries=executed_queries,
        documents=_reindex_documents(diversified),
    )


def _expanded_queries(query: str, llm: LLMClient | None) -> list[str]:
    if llm is None:
        expansions = []
    else:
        try:
            expansions = expand_keywords(query, llm)
        except Exception:
            logger.exception("Query expansion failed for hybrid web search")
            expansions = []
    queries = [query] + [e for e in expansions if e != query]
    temporal = with_temporal_context(query)
    if temporal != query and temporal not in queries:
        queries.append(temporal)
    return queries


def _documents_from_search_pages(
    pages: list[SearchPage],
    *,
    source_provider: str,
    query: str,
    start_index: int = 1,
    entry_point: str = "search_tool",
) -> list[ContextDocument]:
    documents: list[ContextDocument] = []
    for offset, page in enumerate(pages):
        index = start_index + offset
        documents.append(
            ContextDocument(
                id=f"D{index}",
                title=page.title
                or ("Search error" if page.error else f"Result {index}"),
                content=page.error or page.summary or "No summary available.",
                url=page.url or None,
                score=0.0,
                metadata={
                    "entry_point": entry_point,
                    "source": _source_label(source_provider),
                    "source_provider": source_provider,
                    "query": query,
                },
            )
        )
    return documents


def _document_with_metadata(
    document: ContextDocument,
    *,
    source_provider: str,
    query: str,
    entry_point: str,
) -> ContextDocument:
    return ContextDocument(
        id=document.id,
        title=document.title,
        content=document.content,
        url=document.url,
        score=document.score,
        metadata={
            **document.metadata,
            "entry_point": entry_point,
            "source": _source_label(source_provider),
            "source_provider": source_provider,
            "query": query,
        },
    )


def _dedupe_documents(documents: list[ContextDocument]) -> list[ContextDocument]:
    deduped: list[ContextDocument] = []
    seen: set[tuple[str | None, str]] = set()
    for document in documents:
        key = (document.url, document.content[:160])
        if key in seen:
            continue
        seen.add(key)
        deduped.append(document)
    return deduped


def _reindex_documents(documents: list[ContextDocument]) -> list[ContextDocument]:
    return [
        ContextDocument(
            id=f"D{index}",
            title=document.title,
            content=document.content,
            url=document.url,
            score=document.score,
            metadata=document.metadata,
        )
        for index, document in enumerate(documents, 1)
    ]


def _search_only_answer(
    label: str,
    *,
    queries: list[str],
    documents: list[ContextDocument],
    source_provider: str,
) -> str:
    query_lines = "\n".join(f"- {query}" for query in queries)
    if not documents:
        return (
            f"{label} returned no results from {_source_label(source_provider)}.\n\n"
            f"Executed queries:\n{query_lines}"
        )
    citation_list = ", ".join(doc.citation for doc in documents)
    return (
        f"{label} returned {len(documents)} result(s) from "
        f"{_source_label(source_provider)}.\n\n"
        f"Executed queries:\n{query_lines}\n\n"
        f"Open the Sources panel to inspect {citation_list}."
    )


def _response_from_result(
    session_id: str,
    result: AnswerGenerationResult,
    messages: list[ChatMessageView],
    *,
    hook_metadata: dict[str, object] | None = None,
) -> AgentExperienceResponse:
    return AgentExperienceResponse(
        session_id=session_id,
        answer=result.answer,
        citations=result.citations,
        documents=[_document_view(document) for document in result.context.documents],
        messages=messages,
        hook_metadata=hook_metadata or {},
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


def _frontend_dist_path() -> Path | None:
    dist = Path(__file__).resolve().parents[3] / "web" / "dist"
    if (dist / "index.html").exists() and (dist / "assets").is_dir():
        return dist
    return None


app = create_web_app()
