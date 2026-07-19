"""FastAPI app for a browser-based search and agent experience."""

from __future__ import annotations

import asyncio
import json as _json
import logging
import uuid as _uuid
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from dataclasses import asdict, dataclass
from itertools import islice
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, Response, StreamingResponse
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
from src.internal.utils.embedding_gate import (
    gate_embedder,
    make_cosine_fn,
    search_direct_cos_min,
)
from src.internal.utils.text_processing import levenshtein_lt2, normalize_for_match
from src.internal.search.process_search_query import run_expanded_search
from src.internal.servers.secondary_llm_flows import expand_keywords
from src.internal.servers.secondary_llm_flows.query_expansion import (
    with_temporal_context,
)
from src.agents.search import AgenticRAGConfig, AgenticRAGLoop
from src.agents.core.base import OnTurnCallback
from src.agents.core.control_flow_trace import (
    ControlFlowEvent,
    ControlFlowRecorder,
    EventSink,
)
from src.context import ChatMessage
from src.context import LLMClient
from src.context import answer_with_retrieval
from src.context import build_context_bundle
from src.context.models import AnswerGenerationResult
from src.context.models import ContextDocument
from src.context.models import SearchFilters
from src.context.search import SearchResult, citation_key
from src.context.preprocessing.access_filters import build_user_only_filters
from src.internal.db import AgenticSearchStore
from src.internal.hooks import HookPoint
from src.internal.hooks import HookRegistry
from src.internal.hooks import HookSoftFailed
from src.internal.hooks import execute_hook
from src.internal.search_pipeline.models import CandidateSet
from src.internal.search_pipeline.models import GeneratedAnswer
from src.internal.search_pipeline.models import RankedEvidence
from src.internal.search_pipeline.pipeline import SearchPipeline
from src.internal.search_pipeline.ranking import DefaultRankingStage
from src.internal.search_pipeline.stages import RerankHTTPRankingStage
from src.internal.servers.admin_surface.api import create_admin_surface_router
from src.internal.servers.analytics.api import create_analytics_router
from src.internal.servers.web import request_capture as _capture
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
from src.tools.knowledge_base import seed_tools, tool_knowledge_base
from src.tools.registry import tool_registry
from src.internal.servers.web.tool_approval import (
    ApprovalConflict,
    ApprovalExpired,
    ApprovalForbidden,
    ApprovalNotFound,
    ToolApprovalBroker,
)
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
from .intent_routing import (
    RouteStrategy,
    _infer_intent_from_output,
    route_query,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SearchExperienceSettings:
    """Runtime settings for the browser search app."""

    search_url: str = "http://localhost:8001/retrieve"
    top_k: int = 5
    db_path: str | Path = ":memory:"
    browser_search_url: str | None = None
    rerank_url: str | None = None
    # When False (default), the retrieval URL is resolved server-side and any
    # client-supplied search_url is ignored — this closes the SSRF hole where a
    # client could point the backend at an arbitrary URL. Enable only for local
    # dev/debugging via AGENTIC_SEARCH_ALLOW_CLIENT_RETRIEVAL_URL=true.
    allow_client_search_url: bool = False
    # Dev-only observability console. Off by default; never enable in prod.
    debug_panels: bool = False

    @classmethod
    def from_app_settings(
        cls,
        settings: AppSettings | None = None,
    ) -> "SearchExperienceSettings":
        import os

        def _flag(name: str) -> bool:
            return os.environ.get(name, "").strip().lower() in {"1", "true", "yes"}

        app_settings = settings or load_app_settings()
        return cls(
            search_url=app_settings.services.retrieval_url,
            top_k=app_settings.services.web_top_k,
            db_path=app_settings.services.web_db_path,
            allow_client_search_url=_flag("AGENTIC_SEARCH_ALLOW_CLIENT_RETRIEVAL_URL"),
            debug_panels=_flag("AGENTIC_SEARCH_DEBUG_PANELS"),
        )


class SessionCreateRequest(BaseModel):
    title: str | None = None
    user_id: str | None = None


class ToolApprovalDecisionRequest(BaseModel):
    decision: Literal["approve", "deny"]


class ToolApprovalDecisionResponse(BaseModel):
    id: str
    decision: str


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
        default="auto",
        description=(
            "'auto' (default — internal retrieval first, then SerpAPI and the "
            "configured browser fallback when evidence is weak or empty), "
            "'retrieval', 'serpapi', 'google', 'browser', or 'all'. An explicit "
            "value other "
            "than 'auto' forces a search against that single provider (dev only)."
        ),
    )
    mode: str | None = Field(
        default=None,
        description=(
            "Optional explicit mode override: 'search_tool', 'hybrid_search', 'chat_once', "
            "'chat_loop', 'search_agent', 'tool_agent'. When None, intent is auto-detected."
        ),
    )


class ToolCallView(BaseModel):
    tool_name: str
    status: str
    arguments: dict[str, object]
    result_summary: str
    latency_ms: int
    error: str | None = None


class ControlFlowEventView(BaseModel):
    sequence: int
    timestamp: str
    turn: int
    component: str
    action: str
    status: str
    duration_ms: int | None = None
    details: dict[str, str | int | float | bool | None] = Field(default_factory=dict)


class AgentExperienceResponse(BaseModel):
    session_id: str
    answer: str
    citations: list[str]
    documents: list[SourceDocumentView]
    messages: list[ChatMessageView]
    hook_metadata: dict[str, object] = Field(default_factory=dict)
    intent: str = "chat"  # "search" | "chat" | "tool"
    tool_calls: list[ToolCallView] = Field(default_factory=list)
    control_flow_trace: list[ControlFlowEventView] = Field(default_factory=list)


class QueryProcessingHookResponse(BaseModel):
    query: str | None = None
    metadata: dict[str, object] = Field(default_factory=dict)


def _control_flow_event_view(event: ControlFlowEvent) -> ControlFlowEventView:
    return ControlFlowEventView(**event.to_dict())


async def _request_tool_approval(
    broker: ToolApprovalBroker,
    owner_user_id: str,
    approval_request,
    queue: asyncio.Queue[dict],
):
    registered = asyncio.Event()
    publish_task: asyncio.Task[None] | None = None

    def publish(view) -> None:
        nonlocal publish_task
        publish_task = asyncio.create_task(
            queue.put({"type": "approval_required", "approval": asdict(view)})
        )
        registered.set()

    request_task = asyncio.create_task(
        broker.request(owner_user_id, approval_request, on_registered=publish)
    )
    registration_task = asyncio.create_task(registered.wait())
    try:
        done, _ = await asyncio.wait(
            (request_task, registration_task),
            return_when=asyncio.FIRST_COMPLETED,
        )
        if request_task in done and not registered.is_set():
            return await request_task

        await registration_task
        assert publish_task is not None
        done, _ = await asyncio.wait(
            (request_task, publish_task),
            return_when=asyncio.FIRST_COMPLETED,
        )
        if request_task in done and not publish_task.done():
            return await request_task
        await publish_task
        return await request_task
    finally:
        tracked_tasks = [request_task, registration_task]
        if publish_task is not None:
            tracked_tasks.append(publish_task)
        for task in tracked_tasks:
            if not task.done():
                task.cancel()
        await asyncio.gather(*tracked_tasks, return_exceptions=True)


@dataclass(frozen=True)
class _HybridSearchResult:
    executed_queries: list[str]
    documents: list[ContextDocument]
    status: str = "ok"  # "ok" | "empty" | "unreachable"
    ranking: dict | None = None


class _RankedDocumentList(list[ContextDocument]):
    """List-compatible ranked documents carrying the ranking stage facts."""

    def __init__(self, documents: list[ContextDocument], ranking: dict) -> None:
        super().__init__(documents)
        self.ranking = ranking


def _ranking_metadata(
    documents: list[ContextDocument], *, fallback_operations: list[str]
) -> dict:
    ranking = getattr(documents, "ranking", None)
    if ranking is not None:
        return dict(ranking)
    return {
        "operations": fallback_operations,
        "candidate_count": len(documents),
    }


def _register_routers(
    app: FastAPI,
    db: AgenticSearchStore,
    settings: AppSettings,
    search_url: str,
    debug_panels: bool = False,
    llm: LLMClient | None = None,
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
    app.include_router(create_evals_router(settings, search_url=search_url, db=db))
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

    # --- Retrieval feedback ---
    from src.internal.servers.retrieval.feedback_router import create_feedback_router

    app.include_router(create_feedback_router(db))

    # --- Dev console (gated; dev-only observability) ---
    if debug_panels:
        from src.internal.servers.web.debug_router import create_debug_router

        app.include_router(create_debug_router(search_url=search_url, llm=llm, db=db))


def _extract_tool_calls_and_docs(output) -> tuple[list, list]:
    """Parse a ToolAgentLoop action_trace into ToolCallView + ContextDocument lists."""
    tool_calls: list[ToolCallView] = []
    documents: list = []
    if not output.action_trace:
        return tool_calls, documents
    for line in output.action_trace.split("\n"):
        if not line.strip():
            continue
        try:
            rec = _json.loads(line)
            tool_name = rec.get("tool_name", "")
            perf = rec.get("performance", {})
            latency_ms = round(perf.get("execution_time", 0.0) * 1000)
            status_raw = str(rec.get("status", "failed")).lower()
            is_completed = "completed" in status_raw
            result = rec.get("result")
            decoded_result = result
            if isinstance(result, str):
                try:
                    decoded_result = _json.loads(result)
                except Exception:
                    pass
            if isinstance(decoded_result, list):
                result_summary = f"{len(decoded_result)} items"
            elif result is not None:
                result_summary = str(result)[:200]
            else:
                result_summary = ""
            args = rec.get("arguments") or {}
            args = args if isinstance(args, dict) else {}
            tool_calls.append(
                ToolCallView(
                    tool_name=tool_name,
                    status="completed" if is_completed else "failed",
                    arguments=args,
                    result_summary=result_summary,
                    latency_ms=latency_ms,
                    error=rec.get("error_message"),
                )
            )
            if tool_name == "search_routing_tool" and result:
                raw = _json.loads(result) if isinstance(result, str) else result
                if isinstance(raw, list):
                    for i, item in enumerate(raw, 1):
                        documents.append(
                            ContextDocument(
                                id=f"D{i}",
                                title=item.get("title", ""),
                                content=item.get("content", ""),
                                url=item.get("url"),
                                score=0.0,
                                metadata={"source": "search_routing_tool"},
                            )
                        )
        except Exception:
            pass
    return tool_calls, documents


class _WebHybridRetrievalStage:
    """Adapt the established hybrid/provider policy to normalized candidates."""

    def __init__(
        self,
        *,
        llm,
        search_url,
        browser_search_url,
        rerank_url,
        source_provider,
    ) -> None:
        self._llm = llm
        self._search_url = search_url
        self._browser_search_url = browser_search_url
        self._rerank_url = rerank_url
        self._source_provider = source_provider

    async def retrieve(self, query, history, filters, top_k) -> CandidateSet:
        result = await _run_hybrid_search(
            query,
            llm=self._llm,
            search_url=self._search_url,
            browser_search_url=self._browser_search_url,
            rerank_url=self._rerank_url,
            top_k=top_k,
            filters=filters,
            source_provider=self._source_provider,
        )
        candidates = [
            SearchResult(
                contents=document.content,
                score=document.score,
                title=document.title,
                url=document.url,
                metadata=document.metadata,
            )
            for document in result.documents
        ]
        return CandidateSet(
            query=query,
            candidates=candidates,
            provider=self._source_provider,
            filters=filters,
            metadata={
                "status": result.status,
                "executed_queries": result.executed_queries,
            },
        )


class _WebEvidenceStage:
    """Preserve the already-ranked hybrid result while normalizing its IDs."""

    async def rank(self, query, candidates, top_k) -> RankedEvidence:
        documents = DefaultRankingStage.documents(candidates)[:top_k]
        return RankedEvidence(
            query=query,
            evidence=documents,
            metadata={
                "operations": ["hybrid_ranking"],
                "executed_queries": candidates.metadata.get("executed_queries", []),
            },
        )


class _WebSearchAnswerStage:
    """Keep the existing deterministic search answer behind inference protocol."""

    def __init__(self, source_provider: str) -> None:
        self._source_provider = source_provider

    async def generate(self, query, history, evidence) -> GeneratedAnswer:
        answer = _search_only_answer(
            "Search",
            queries=evidence.metadata.get("executed_queries", [query]),
            documents=evidence.evidence,
            source_provider=self._source_provider,
        )
        return GeneratedAnswer(
            answer=answer,
            citations=[document.citation for document in evidence.evidence],
        )


async def _auto_search_pipeline(
    query: str,
    *,
    llm,
    search_url: str,
    browser_search_url,
    rerank_url,
    top_k: int,
    filters,
    history: list,
    source_provider: str,
    extra: dict,
) -> tuple:
    """Run the shared stage composer as the grounded degraded fallback."""
    pipeline = SearchPipeline(
        _WebHybridRetrievalStage(
            llm=llm,
            search_url=search_url,
            browser_search_url=browser_search_url,
            rerank_url=rerank_url,
            source_provider=source_provider,
        ),
        _WebEvidenceStage(),
        _WebSearchAnswerStage(source_provider),
    )
    result = await pipeline.run(query, history, filters, top_k, source_provider)
    result[4].update(extra)
    return result


# ---------------------------------------------------------------------------
# Loop runners — single source of truth for building + running each agent loop.
# Each returns the canonical (answer, citations, documents, intent, extra) tuple
# consumed by both _run_auto_routed and the explicit-mode chain. Runners hold no
# capability policy: degradation lives in _run_auto_routed and the explicit-mode
# 400 guards live at those call sites.
# ---------------------------------------------------------------------------


def _search_agent_documents(output) -> list[ContextDocument]:
    """Extract documents from a SearchAgentLoop output, preserving each result's
    real ``[RxQyDz]`` citation label so answer markers resolve to source cards.

    Rounds/queries/docs are enumerated 1-based to match
    ``SearchAgentLoop._format_round_information`` (the labels the model cited).
    Dedup is intentionally skipped here: every cited label needs its own card so
    no citation dangles, even when the same doc is retrieved under two queries.
    """
    documents: list[ContextDocument] = []
    rounds = getattr(output.context, "rounds", None) or []
    for round_idx, round_ctxs in enumerate(rounds, 1):
        for query_idx, ctx in enumerate(round_ctxs, 1):
            for doc_idx, result in enumerate(ctx.results, 1):
                key = citation_key(round_idx, query_idx, doc_idx)
                base = ContextDocument.from_search_result(result, index=doc_idx)
                documents.append(
                    ContextDocument(
                        id=key,
                        title=base.title,
                        content=base.content,
                        url=base.url,
                        score=base.score,
                        metadata=base.metadata,
                    )
                )
    return documents


async def _run_search_agent(
    query: str,
    *,
    manager,
    tokenizer,
    search_url: str,
    top_k: int,
    history: list | None = None,
    filters: dict | None = None,
    allow_internal_knowledge_answer: bool = True,
    on_turn=None,
    on_trace=None,
) -> tuple:
    """Run the multi-turn SearchAgentLoop. Assumes a local model is configured."""
    from src import get_registered_agent_loop, resolve_agent_name
    from src.agents.search import SearchAgentLoopConfig

    loop_cls = get_registered_agent_loop(resolve_agent_name("search_agent"))
    loop = loop_cls(
        tokenizer=tokenizer,
        server_manager=manager,
        search_config=SearchAgentLoopConfig(
            search_url=search_url,
            topk=top_k,
            max_turns=3,
            filters=filters,
            allow_internal_knowledge_answer=allow_internal_knowledge_answer,
        ),
    )
    output = await loop.run(
        _build_search_agent_messages(query, history or []),
        sampling_params={"temperature": 0.0, "max_tokens": 256},
        on_turn=on_turn,
        on_trace=on_trace,
    )
    documents = _search_agent_documents(output)
    extra = {
        "control_flow_trace": output.control_flow_trace,
        "num_turns": output.num_turns,
    }
    return (
        output.final_answer or "",
        [d.citation for d in documents],
        documents,
        "search",
        extra,
    )


async def _run_agentic_rag(
    query: str,
    *,
    llm,
    search_url: str,
    top_k: int,
    filters=None,
    history: list,
) -> tuple:
    """Run the AgenticRAGLoop (decompose + HyDE). Assumes an LLM is configured."""
    rag_loop = AgenticRAGLoop(
        AgenticRAGConfig(
            max_rounds=3,
            topk=top_k,
            retrieval_url=search_url,
            filters=filters,
        ),
        llm=llm,
    )
    from uuid import uuid4

    recorder = ControlFlowRecorder(uuid4().hex)
    rag = await rag_loop.run(query, chat_history=history, recorder=recorder)
    return (
        rag.answer,
        rag.citations,
        rag.context.documents,
        "chat",
        {
            "rounds_used": rag.rounds_used,
            "control_flow_trace": recorder.snapshot(),
        },
    )


async def _run_tool_agent(
    query: str,
    *,
    manager,
    tokenizer,
    search_url: str,
    history: list,
    resolved,
    on_turn=None,
    on_approval=None,
    with_search_tool: bool,
) -> tuple:
    """Run the ToolAgentLoop. Assumes a local model is configured.

    ``answer`` is ``output.final_answer or ""`` with no fallback applied; the
    last assistant message is exposed in ``extra["_assistant_fallback"]`` so the
    auto-route (degrade on empty) and explicit mode (fall back to it) can each
    apply their own policy. Callers must pop ``_assistant_fallback`` before it
    reaches the response/metadata.
    """
    from src.agents.tool import ToolAgentLoop, ToolAgentLoopConfig
    from src.tools import build_search_tool, tool_registry

    tools = list(tool_registry.list_tools())
    if with_search_tool:
        tools = [build_search_tool(search_url=search_url)] + [
            t for t in tools if t.name != "search"
        ]
    loop = ToolAgentLoop(
        tokenizer=tokenizer,
        server_manager=manager,
        tools=tools,
        config=ToolAgentLoopConfig(
            tool_parser_format=resolved.tool_agent_parser,
            approval_timeout_seconds=getattr(
                resolved, "tool_approval_timeout_seconds", 60.0
            ),
        ),
    )
    messages = [{"role": m.role, "content": m.content} for m in history] + [
        {"role": "user", "content": query}
    ]
    output = await loop.run(
        messages,
        sampling_params={"temperature": 0.0, "max_tokens": 512},
        on_turn=on_turn,
        on_approval=on_approval,
    )
    tool_calls, documents = _extract_tool_calls_and_docs(output)
    fallback = next(
        (
            m["content"]
            for m in reversed(output.trajectory_messages)
            if m.get("role") == "assistant"
        ),
        "",
    )
    extra = {
        "tool_calls": tool_calls,
        "num_turns": output.num_turns,
        "_assistant_fallback": fallback,
    }
    return (
        output.final_answer or "",
        [d.citation for d in documents],
        documents,
        _infer_intent_from_output(output),
        extra,
    )


def _finalize_response(
    db,
    session_id: str,
    *,
    query: str,
    answer: str,
    citations: list,
    documents: list,
    intent: str,
    hook_metadata: dict,
    extra: dict,
    mode: str,
) -> "AgentExperienceResponse":
    """Persist the assistant turn and build the response. Single response tail.

    ``extra`` carries loop-specific data on the canonical tuple. ``tool_calls``
    and ``control_flow_trace`` are response fields (not metadata); everything else
    in ``extra`` is merged into the persisted metadata and the response's
    ``hook_metadata``. The private ``_assistant_fallback`` key is dropped.
    """
    extra = dict(extra)
    extra.pop("_assistant_fallback", None)
    tool_calls = extra.pop("tool_calls", [])
    raw_trace = extra.pop("control_flow_trace", None)
    trace_views = (
        [_control_flow_event_view(event) for event in raw_trace] if raw_trace else []
    )
    pipeline_stages = _capture.pipeline_stage_summary(
        query,
        citations=citations,
        documents=documents,
        metadata=extra,
    )
    metadata = {
        "citations": citations,
        "document_ids": [d.id for d in documents],
        "hooks": hook_metadata,
        "mode": mode,
        "intent": intent,
        "pipeline_stages": pipeline_stages,
        **extra,
    }
    if trace_views:
        metadata["control_flow_trace"] = [view.model_dump() for view in trace_views]
    _capture.record_pipeline_stages(pipeline_stages)
    _capture.record_stage(
        "final",
        "answer",
        {
            "answer": answer,
            "citations": citations,
            "documents": [d.id for d in documents],
            "intent": intent,
            "route": extra.get("route"),
            "route_degraded": extra.get("route_degraded"),
        },
    )
    db.add_chat_message(session_id, role="assistant", content=answer, metadata=metadata)
    messages = [
        ChatMessageView(role=m.role, content=m.content, metadata=m.metadata)
        for m in db.list_chat_messages(session_id)
    ]
    return AgentExperienceResponse(
        session_id=session_id,
        answer=answer,
        citations=citations,
        documents=[_document_view(d) for d in documents],
        messages=messages,
        hook_metadata={**hook_metadata, **extra},
        intent=intent,
        tool_calls=tool_calls,
        control_flow_trace=trace_views,
    )


def _direct_gate_decision(
    query: str,
    docs: list[ContextDocument],
    *,
    cos_min: float,
    cosine_fn: Callable[[str, str], float | None],
) -> tuple[bool, str, float, float | None]:
    """Tiered strong/weak gate over the rank-1 retrieval result.

    exact title match → direct; typo (Levenshtein<2) confirmed by cosine → direct;
    semantic cosine > cos_min → direct; otherwise escalate. Backend-independent.
    """
    top_score = max((d.score or 0.0 for d in docs), default=0.0)
    if not docs:
        return False, "weak", top_score, None

    top = docs[0]
    q = normalize_for_match(query)
    t = normalize_for_match(top.title)

    if q and q == t:
        return True, "exact", top_score, None

    passage = f"{top.title} {top.content}".strip()

    if q and t and levenshtein_lt2(q, t):
        cosine = cosine_fn(query, passage)
        if cosine is not None and cosine > cos_min:
            return True, "fuzzy", top_score, cosine
        return False, "weak", top_score, cosine

    cosine = cosine_fn(query, passage)
    if cosine is not None and cosine > cos_min:
        return True, "semantic", top_score, cosine
    return False, "weak", top_score, cosine


async def _run_search_direct_or_escalate(
    query: str,
    *,
    manager,
    tokenizer,
    llm,
    search_url: str,
    browser_search_url,
    rerank_url,
    top_k: int,
    filters,
    history: list,
    source_provider: str,
    on_turn=None,
) -> tuple:
    """Direct retrieval first; return docs when the query matches, else escalate.

    A tiered match gate (`_direct_gate_decision`) compares the query to the
    rank-1 result: exact title match, or a typo (Levenshtein<2) confirmed by
    cosine, or semantic cosine > threshold → return the ranked docs with a
    non-LLM summary (no agent loop). Otherwise escalate to the SearchAgentLoop
    (local model) or the degraded pipeline, preserving today's behavior.
    """

    # Access-control safety: the direct-retrieval short-circuit
    # (`_run_direct_search`) and the SearchAgentLoop escalation
    # (`_run_search_agent`) do not thread per-user access filters into
    # retrieval — `search_tool` and the loop take no `filters`. When filters are
    # present (authenticated / multi-tenant), those paths would retrieve across
    # every user's documents. Route such queries through the filter-aware
    # retrieval pipeline instead, which passes `filters` down to retrieval.
    if filters:
        return await _auto_search_pipeline(
            query,
            llm=llm,
            search_url=search_url,
            browser_search_url=browser_search_url,
            rerank_url=rerank_url,
            top_k=top_k,
            filters=filters,
            history=history,
            source_provider=source_provider,
            extra={
                "search_mode": "filtered_pipeline",
                "route_reason": "access_filters_present",
            },
        )

    async def _escalate(top_score: float, reason: str) -> tuple:
        escalate_extra = {
            "search_mode": "escalated",
            "top_score": top_score,
            "escalate_reason": reason,
        }
        has_local_model = manager is not None and tokenizer is not None
        if has_local_model:
            answer, citations, docs, intent, run_extra = await _run_search_agent(
                query,
                manager=manager,
                tokenizer=tokenizer,
                search_url=search_url,
                top_k=top_k,
                history=history,
                filters=filters,
                allow_internal_knowledge_answer=False,
                on_turn=on_turn,
                on_trace=None,
            )
            run_extra.update(escalate_extra)
            return answer, citations, docs, intent, run_extra

        escalate_extra["route_degraded"] = "no_local_model"
        return await _auto_search_pipeline(
            query,
            llm=llm,
            search_url=search_url,
            browser_search_url=browser_search_url,
            rerank_url=rerank_url,
            top_k=top_k,
            filters=filters,
            history=history,
            source_provider=source_provider,
            extra=escalate_extra,
        )

    # Explicit non-default source: honor it via the existing escalation path
    # (no direct-first local retrieval short-circuit).
    if source_provider not in ("auto", "retrieval"):
        return await _escalate(0.0, "explicit_source")

    internal_unreachable = False
    try:
        documents = await _run_direct_search(
            query,
            source_provider="retrieval",
            search_url=search_url,
            rerank_url=rerank_url,
            top_k=top_k,
            filters=filters,
        )
    except Exception:
        internal_unreachable = True
        documents = []
    real = [d for d in documents if not d.metadata.get("error")]
    if documents and not real:
        internal_unreachable = True
    is_strong, tier, top_score, cosine = await asyncio.to_thread(
        _direct_gate_decision,
        query,
        real,
        cos_min=search_direct_cos_min(),
        cosine_fn=make_cosine_fn(gate_embedder()),
    )
    _capture.record_stage(
        "search",
        "direct_retrieval",
        {
            "query": query,
            "top_k": top_k,
            "top_score": top_score,
            "documents": [
                {"id": d.id, "title": d.title, "score": d.score} for d in real
            ],
        },
    )

    if is_strong:
        ranking = _ranking_metadata(documents, fallback_operations=["direct_ranking"])
        ranking["operations"] = [*ranking["operations"], "sufficiency_gate"]
        _capture.record_stage(
            "search",
            "sufficiency",
            {"mode": "direct", "tier": tier, "cosine": cosine, "top_score": top_score},
        )
        answer = _search_only_answer(
            "Direct retrieval",
            queries=[query],
            documents=real,
            source_provider="retrieval",
        )
        return (
            answer,
            [d.citation for d in real],
            real,
            "search",
            {
                "search_mode": "direct",
                "tier": tier,
                "top_score": top_score,
                "source_provider": "retrieval",
                "retrieval_query": query,
                "ranking": ranking,
                "inference": {"mode": "deterministic", "model": None},
            },
        )

    _capture.record_stage(
        "search",
        "sufficiency",
        {"mode": "escalated", "tier": tier, "cosine": cosine, "top_score": top_score},
    )

    if source_provider == "auto":
        external_providers = ["serpapi"]
        if browser_search_url:
            external_providers.append("browser")
        providers_attempted = 1
        unreachable_providers = 1 if internal_unreachable else 0

        for provider in external_providers:
            providers_attempted += 1
            try:
                external_documents = await _run_direct_search(
                    query,
                    source_provider=provider,
                    search_url=search_url,
                    browser_search_url=(
                        browser_search_url if provider == "browser" else None
                    ),
                    rerank_url=rerank_url,
                    top_k=top_k,
                    filters=None,
                )
            except Exception as exc:
                logger.warning("%s fallback failed for %r: %s", provider, query, exc)
                unreachable_providers += 1
                external_documents = []

            external_real = [
                doc for doc in external_documents if not doc.metadata.get("error")
            ]
            if external_documents and not external_real:
                unreachable_providers += 1
            if external_real:
                _capture.record_stage(
                    "search",
                    "external_fallback",
                    {"provider": provider, "documents": len(external_real)},
                )
                answer = _search_only_answer(
                    "External search",
                    queries=[query],
                    documents=external_real,
                    source_provider=provider,
                )
                return (
                    answer,
                    [doc.citation for doc in external_real],
                    external_real,
                    "search",
                    {
                        "search_mode": "external_fallback",
                        "external_provider": provider,
                        "source_provider": provider,
                        "retrieval_query": query,
                        "top_score": top_score,
                        "ranking": _ranking_metadata(
                            external_documents,
                            fallback_operations=["direct_ranking"],
                        ),
                        "inference": {"mode": "deterministic", "model": None},
                    },
                )

        _capture.record_stage(
            "search",
            "external_fallback",
            {"provider": "none", "documents": 0},
        )
        all_unreachable = unreachable_providers == providers_attempted
        return (
            (
                "No sources are reachable right now. Please try again shortly."
                if all_unreachable
                else f"No results found for: {query}"
            ),
            [],
            [],
            "search",
            {
                "search_mode": "external_empty",
                "top_score": top_score,
            },
        )

    return await _escalate(top_score, "weak_retrieval")


async def _run_auto_routed(
    query: str,
    *,
    llm,
    manager,
    tokenizer,
    search_url: str,
    browser_search_url,
    rerank_url,
    top_k: int,
    filters,
    history: list,
    resolved,
    source_provider: str = "retrieval",
    on_turn=None,
    on_approval=None,
) -> tuple:
    """3-way agentic routing. Returns (answer, citations, documents, intent, extra).

    `route_query` picks one of {chat, search, tool}; dispatch is
    capability-aware and degrades gracefully when a required backend
    (local model vs LLM client) is unavailable. `extra["route"]` records the
    chosen strategy and `extra["route_degraded"]` records any degradation.
    """
    extra: dict = {}
    explicit_source = source_provider != "auto"
    has_local_model = manager is not None and tokenizer is not None

    strategy = route_query(
        query,
        llm=llm,
        explicit_source=explicit_source,
    )
    extra["route"] = strategy.value

    # ---- TOOL: run ToolAgentLoop with the real registered tools ----
    if strategy is RouteStrategy.TOOL:
        if has_local_model:
            result = None
            try:
                result = await _run_tool_agent(
                    query,
                    manager=manager,
                    tokenizer=tokenizer,
                    search_url=search_url,
                    history=history,
                    resolved=resolved,
                    on_turn=on_turn,
                    on_approval=on_approval,
                    with_search_tool=False,
                )
            except Exception as exc:
                logger.warning("ToolAgentLoop failed, degrading to RAG: %s", exc)
            if result is not None:
                answer, citations, documents, intent, run_extra = result
                run_extra.pop("_assistant_fallback", None)
                if answer.strip():
                    extra.update(run_extra)
                    return answer, citations, documents, intent, extra
            extra["route_degraded"] = "tool_unavailable"
        else:
            extra["route_degraded"] = "no_local_model"
        strategy = RouteStrategy.CHAT

    # ---- SEARCH: direct retrieval first, escalate to the agent loop if weak ----
    if strategy is RouteStrategy.SEARCH:
        (
            answer,
            citations,
            documents,
            intent,
            run_extra,
        ) = await _run_search_direct_or_escalate(
            query,
            manager=manager,
            tokenizer=tokenizer,
            llm=llm,
            search_url=search_url,
            browser_search_url=browser_search_url,
            rerank_url=rerank_url,
            top_k=top_k,
            filters=filters,
            history=history,
            source_provider=source_provider,
            on_turn=on_turn,
        )
        extra.update(run_extra)
        return answer, citations, documents, intent, extra

    # ---- CHAT: grounded synthesis via AgenticRAGLoop (degrade to pipeline) ----
    if strategy is RouteStrategy.CHAT:
        if llm is not None:
            answer, citations, documents, intent, run_extra = await _run_agentic_rag(
                query,
                llm=llm,
                search_url=search_url,
                top_k=top_k,
                filters=filters,
                history=history,
            )
            extra.update(run_extra)
            return answer, citations, documents, intent, extra
        extra.setdefault("route_degraded", "no_llm")
        return await _auto_search_pipeline(
            query,
            llm=llm,
            search_url=search_url,
            browser_search_url=browser_search_url,
            rerank_url=rerank_url,
            top_k=top_k,
            filters=filters,
            history=history,
            source_provider=source_provider,
            extra=extra,
        )


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
    if resolved.auth.dev_admin_bypass:
        logger.warning(
            "ADMIN AUTH BYPASSED via AGENTIC_SEARCH_DEV_ADMIN — every admin "
            "endpoint accepts unauthenticated requests. Dev only; never set "
            "this in production."
        )
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
        seed_tools(
            tool_registry,
            tools=tool_knowledge_base(
                search_url=resolved.services.retrieval_url, llm=llm
            ),
        )
        check_router_auth(_app, PUBLIC_ENDPOINT_SPECS)
        _app.state.search_agent_manager = None
        _app.state.search_agent_tokenizer = None
        if resolved.search_agent_server_url:
            try:
                from transformers import AutoTokenizer
                from src.model.serving import build_server_manager

                model = resolved.search_agent_model or "Qwen/Qwen2.5-1.5B-Instruct"
                tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
                if tokenizer.pad_token_id is None:
                    tokenizer.pad_token_id = tokenizer.eos_token_id
                manager = build_server_manager(
                    tokenizer,
                    server_url=resolved.search_agent_server_url,
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
                from src.model.serving import build_server_manager

                tokenizer = AutoTokenizer.from_pretrained(
                    resolved.search_agent_model,
                    trust_remote_code=True,
                    local_files_only=True,
                )
                if tokenizer.pad_token_id is None:
                    tokenizer.pad_token_id = tokenizer.eos_token_id
                manager = build_server_manager(
                    tokenizer,
                    model=resolved.search_agent_model,
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
            gate_embedder()  # warm the direct-gate e5 model (no-op when SEMANTIC=0)
        except Exception:
            logger.exception("direct-gate: embedder warmup failed")
        try:
            yield
        finally:
            if owns_store:
                db.close()

    app = FastAPI(title="Agentic Search Web", lifespan=lifespan)
    app.state.tool_approval_broker = ToolApprovalBroker(
        resolved.tool_approval_timeout_seconds
    )
    import os as _os
    from src.internal.servers.web.request_capture_store import RequestCaptureStore

    app.state.request_captures = RequestCaptureStore(
        max_size=int(_os.environ.get("AGENTIC_SEARCH_REQUEST_CAPTURE_MAX", "20"))
    )
    add_api_server_tenant_id_middleware(app, resolved)
    add_license_enforcement_middleware(app, resolved)
    add_tier_gate_middleware(app, resolved)

    _register_routers(
        app,
        db,
        resolved,
        search_url=settings.search_url,
        debug_panels=settings.debug_panels,
        llm=llm,
    )

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

    async def _run_agent_impl(
        request: AgentExperienceRequest,
        http_request: Request,
        *,
        request_id: str,
        on_turn: "OnTurnCallback | None" = None,
        on_trace: EventSink | None = None,
        on_approval=None,
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

        mode_str = request.mode.strip().lower() if request.mode else None
        normalized_mode = _normalize_agent_mode(mode_str) if mode_str else None

        session_request = _copy_agent_request(request, user_id=user_id)
        session_id = _ensure_session(db, session_request, auth_user=auth_user)
        history = _trim_history(
            [
                ChatMessage(role=message.role, content=message.content)
                for message in db.list_chat_messages(session_id)
            ]
        )
        db.add_chat_message(session_id, role="user", content=query)

        # Resolve the retrieval URL server-side. A client-supplied search_url is
        # honored only when explicitly allowed (dev/debug); otherwise it is
        # ignored so the backend, not the browser, decides what it will fetch.
        search_url = (
            request.search_url
            if settings.allow_client_search_url and request.search_url
            else settings.search_url
        )
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

        manager = getattr(http_request.app.state, "search_agent_manager", None)
        tokenizer = getattr(http_request.app.state, "search_agent_tokenizer", None)

        capture_on = (
            getattr(http_request.app.state, "request_captures", None) is not None
        )
        capture_token = (
            _capture.start_capture(request_id, query)
            if capture_on and settings.debug_panels
            else None
        )

        try:
            try:
                if normalized_mode is None:
                    # Auto-routing path
                    (
                        answer,
                        citations,
                        documents,
                        intent,
                        extra,
                    ) = await _run_auto_routed(
                        query,
                        llm=llm,
                        manager=manager,
                        tokenizer=tokenizer,
                        search_url=search_url,
                        browser_search_url=settings.browser_search_url,
                        rerank_url=settings.rerank_url,
                        top_k=top_k,
                        filters=filters,
                        history=history,
                        resolved=resolved,
                        source_provider=_normalize_source_provider(
                            request.source_provider
                        ),
                        on_turn=on_turn,
                        on_approval=on_approval,
                    )
                    _cap = _capture.active()
                    if _cap is not None:
                        _cap.route = extra.get("route")
                        _cap.route_degraded = extra.get("route_degraded")
                    return _finalize_response(
                        db,
                        session_id,
                        query=query,
                        answer=answer,
                        citations=citations,
                        documents=documents,
                        intent=intent,
                        hook_metadata=hook_metadata,
                        extra=extra,
                        mode="auto",
                    )

                # Explicit mode — pin the dispatch, then share the runners + tail.
                mode = normalized_mode

                if mode == "search_tool":
                    source_provider = _normalize_source_provider(
                        request.source_provider
                    )
                    documents = await _run_direct_search(
                        query,
                        source_provider=source_provider,
                        search_url=search_url,
                        browser_search_url=settings.browser_search_url,
                        rerank_url=settings.rerank_url,
                        top_k=top_k,
                        filters=filters,
                    )
                    answer = _search_only_answer(
                        "Direct search tool",
                        queries=[query],
                        documents=documents,
                        source_provider=source_provider,
                    )
                    return _finalize_response(
                        db,
                        session_id,
                        query=query,
                        answer=answer,
                        citations=[doc.citation for doc in documents],
                        documents=documents,
                        intent="search",
                        hook_metadata=hook_metadata,
                        extra={
                            "source_provider": source_provider,
                            "retrieval_query": query,
                            "ranking": _ranking_metadata(
                                documents, fallback_operations=["direct_ranking"]
                            ),
                            "inference": {"mode": "deterministic", "model": None},
                        },
                        mode=mode,
                    )

                if mode == "hybrid_search":
                    source_provider = _normalize_source_provider(
                        request.source_provider
                    )
                    search_result = await _run_hybrid_search(
                        query,
                        llm=llm,
                        search_url=search_url,
                        browser_search_url=settings.browser_search_url,
                        rerank_url=settings.rerank_url,
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
                    return _finalize_response(
                        db,
                        session_id,
                        query=query,
                        answer=answer,
                        citations=[doc.citation for doc in search_result.documents],
                        documents=search_result.documents,
                        intent="search",
                        hook_metadata=hook_metadata,
                        extra={
                            "source_provider": source_provider,
                            "retrieval_query": query,
                            "executed_queries": search_result.executed_queries,
                            "ranking": search_result.ranking
                            or {
                                "operations": ["hybrid_ranking"],
                                "candidate_count": len(search_result.documents),
                            },
                            "inference": {"mode": "deterministic", "model": None},
                        },
                        mode=mode,
                    )

                if mode == "chat_loop":
                    (
                        answer,
                        citations,
                        documents,
                        intent,
                        extra,
                    ) = await _run_agentic_rag(
                        query,
                        llm=llm,
                        search_url=search_url,
                        top_k=top_k,
                        filters=filters,
                        history=history,
                    )
                    return _finalize_response(
                        db,
                        session_id,
                        query=query,
                        answer=answer,
                        citations=citations,
                        documents=documents,
                        intent=intent,
                        hook_metadata=hook_metadata,
                        extra=extra,
                        mode=mode,
                    )

                if mode == "search_agent":
                    if manager is None or tokenizer is None:
                        raise HTTPException(
                            status_code=400,
                            detail=(
                                "search_agent mode is not configured. "
                                "Set SEARCH_AGENT_MODEL in .env and restart the server."
                            ),
                        )
                    (
                        answer,
                        citations,
                        documents,
                        intent,
                        extra,
                    ) = await _run_search_agent(
                        query,
                        manager=manager,
                        tokenizer=tokenizer,
                        search_url=search_url,
                        top_k=top_k,
                        history=history,
                        filters=filters,
                        on_turn=on_turn,
                        on_trace=on_trace,
                    )
                    return _finalize_response(
                        db,
                        session_id,
                        query=query,
                        answer=answer,
                        citations=citations,
                        documents=documents,
                        intent=intent,
                        hook_metadata=hook_metadata,
                        extra=extra,
                        mode=mode,
                    )

                if mode == "tool_agent":
                    if manager is None or tokenizer is None:
                        raise HTTPException(
                            status_code=400,
                            detail=(
                                "tool_agent mode requires a local model. "
                                "Set SEARCH_AGENT_MODEL or SEARCH_AGENT_SERVER_URL in .env and restart."
                            ),
                        )
                    answer, citations, documents, intent, extra = await _run_tool_agent(
                        query,
                        manager=manager,
                        tokenizer=tokenizer,
                        search_url=search_url,
                        history=history,
                        resolved=resolved,
                        on_turn=on_turn,
                        on_approval=on_approval,
                        with_search_tool=True,
                    )
                    # Explicit mode falls back to the last assistant message on an
                    # empty final answer (the auto-route instead degrades to RAG).
                    answer = answer or extra.pop("_assistant_fallback", "")
                    return _finalize_response(
                        db,
                        session_id,
                        query=query,
                        answer=answer,
                        citations=citations,
                        documents=documents,
                        intent=intent,
                        hook_metadata=hook_metadata,
                        extra=extra,
                        mode=mode,
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
                logger.exception("Agent dispatch error: %s", exc)
                detail = (
                    str(exc)
                    if str(exc).strip()
                    else "Unexpected error during agent dispatch"
                )
                raise HTTPException(status_code=502, detail=detail) from exc

            return _finalize_response(
                db,
                session_id,
                query=query,
                answer=result.answer,
                citations=result.citations,
                documents=result.context.documents,
                intent="chat",
                hook_metadata=hook_metadata,
                extra={},
                mode=mode,
            )
        finally:
            cap = _capture.active()
            if cap is not None:
                cap.finish()
                http_request.app.state.request_captures.put(cap.snapshot())
            if capture_token is not None:
                _capture.reset_capture(capture_token)

    @app.post("/api/agent")
    async def run_agent(
        request: AgentExperienceRequest,
        http_request: Request,
    ) -> AgentExperienceResponse:
        request_id = _uuid.uuid4().hex
        return await _run_agent_impl(
            request, http_request, request_id=request_id, on_turn=None
        )

    @app.post(
        "/api/agent/approvals/{approval_id}",
        response_model=ToolApprovalDecisionResponse,
    )
    async def decide_tool_approval(
        approval_id: str,
        request: ToolApprovalDecisionRequest,
        http_request: Request,
    ) -> ToolApprovalDecisionResponse:
        from src.agents.tool import ApprovalDecision

        auth_user = _optional_user_from_request(http_request)
        if auth_user is None:
            raise HTTPException(status_code=401, detail="Authentication required")
        try:
            await http_request.app.state.tool_approval_broker.decide(
                approval_id,
                auth_user.id,
                ApprovalDecision(request.decision),
            )
        except ApprovalForbidden as exc:
            raise HTTPException(status_code=403, detail="Approval forbidden") from exc
        except ApprovalNotFound as exc:
            raise HTTPException(status_code=404, detail="Approval not found") from exc
        except ApprovalConflict as exc:
            raise HTTPException(
                status_code=409, detail="Approval already decided"
            ) from exc
        except ApprovalExpired as exc:
            raise HTTPException(status_code=410, detail="Approval expired") from exc
        return ToolApprovalDecisionResponse(id=approval_id, decision=request.decision)

    @app.post("/api/agent/stream")
    async def stream_agent(
        request: AgentExperienceRequest,
        http_request: Request,
    ) -> StreamingResponse:
        """Stream agent progress as Server-Sent Events.

        Emits:
          {"type": "progress", "turn": N, "text": "..."}  — per-turn progress
          {"type": "answer",   "text": "..."}             — final answer text
          {"type": "done",     "session_id": "...", "citations": [...], "documents": [...], "intent": "..."}
          {"type": "error",    "detail": "..."}           — on failure
        """

        def _sse(data: dict) -> str:
            return f"data: {_json.dumps(data)}\n\n"

        queue: asyncio.Queue[dict] = asyncio.Queue(maxsize=100)
        dropped_trace_events = 0
        auth_user = _optional_user_from_request(http_request)
        request_id = _uuid.uuid4().hex

        async def on_turn(turn: int, tool_name: "str | None", doc_count: int) -> None:
            text = (
                f"{tool_name} · {doc_count} docs" if tool_name else "writing answer..."
            )
            await queue.put({"type": "progress", "turn": turn, "text": text})

        async def on_trace(event: ControlFlowEvent) -> None:
            nonlocal dropped_trace_events
            try:
                queue.put_nowait(
                    {
                        "type": "trace",
                        "event": _control_flow_event_view(event).model_dump(),
                    }
                )
            except asyncio.QueueFull:
                dropped_trace_events += 1

        async def on_approval(approval_request):
            return await _request_tool_approval(
                http_request.app.state.tool_approval_broker,
                auth_user.id,
                approval_request,
                queue,
            )

        async def _generate():
            task = asyncio.create_task(
                _run_agent_impl(
                    request,
                    http_request,
                    request_id=request_id,
                    on_turn=on_turn,
                    on_trace=on_trace,
                    on_approval=on_approval if auth_user is not None else None,
                )
            )
            try:
                while not task.done():
                    try:
                        item = await asyncio.wait_for(queue.get(), timeout=0.05)
                        yield _sse(item)
                    except asyncio.TimeoutError:
                        continue
                while not queue.empty():
                    yield _sse(queue.get_nowait())
                result: AgentExperienceResponse = task.result()
                yield _sse({"type": "answer", "text": result.answer})
                yield _sse(
                    {
                        "type": "done",
                        "request_id": request_id,
                        "session_id": result.session_id,
                        "citations": result.citations,
                        "documents": [d.model_dump() for d in result.documents],
                        "intent": result.intent,
                        "route": result.hook_metadata.get("route"),
                        "route_degraded": result.hook_metadata.get("route_degraded"),
                        "tool_calls": [tc.model_dump() for tc in result.tool_calls],
                        "control_flow_trace": [
                            event.model_dump() for event in result.control_flow_trace
                        ],
                    }
                )
                if dropped_trace_events:
                    logger.warning(
                        "dropped %d live control-flow trace events",
                        dropped_trace_events,
                    )
            except BaseException as exc:
                if not task.done():
                    task.cancel()
                    try:
                        await task
                    except (asyncio.CancelledError, Exception):
                        pass
                if isinstance(exc, asyncio.CancelledError):
                    return  # client disconnected
                if isinstance(exc, HTTPException):
                    yield _sse({"type": "error", "detail": exc.detail})
                else:
                    yield _sse({"type": "error", "detail": str(exc)})

        return StreamingResponse(
            _generate(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",
            },
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

MAX_HISTORY_MESSAGES = 40


def _trim_history(history: list, max_messages: int = MAX_HISTORY_MESSAGES) -> list:
    """Keep only the tail of chat history to avoid overflowing the LLM context."""
    if len(history) <= max_messages:
        return history
    return history[-max_messages:]


# Search mode stacks long <information> observations on top of history each turn,
# so cap threaded history tighter than MAX_HISTORY_MESSAGES.
SEARCH_AGENT_HISTORY_MESSAGES = 6


def _build_search_agent_messages(query: str, history: list) -> list[dict[str, str]]:
    """Build the SearchAgentLoop message buffer: capped prior turns + the query.

    History is capped to the last ``SEARCH_AGENT_HISTORY_MESSAGES`` messages and
    mapped to ``{"role", "content"}`` dicts; the current user query is appended
    last. ``SearchAgentLoop._with_system_prompt`` prepends the system prompt.
    """
    capped = _trim_history(history, max_messages=SEARCH_AGENT_HISTORY_MESSAGES)
    messages = [{"role": m.role, "content": m.content} for m in capped]
    messages.append({"role": "user", "content": query})
    return messages


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
    "auto",
    "retrieval",
    "serpapi",
    "browser",
    "all",
}
_SOURCE_PROVIDER_LABELS = {
    "auto": "Auto (internal + web)",
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


# Default provider set when the user does not pick a source: fan out to internal
# RAG + fast web search, merged. Browser is excluded (too slow for the default).
_DEFAULT_FANOUT_PROVIDERS = ["retrieval", "serpapi"]


def _source_providers_for(source_provider: str) -> list[str]:
    if source_provider == "all":
        return ["retrieval", "serpapi", "browser"]
    if source_provider == "auto":
        return list(_DEFAULT_FANOUT_PROVIDERS)
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
    browser_search_url: str | None = None,
    rerank_url: str | None = None,
    top_k: int,
    filters: dict | None = None,
) -> list[ContextDocument]:
    # Over-fetch so MMR has candidates beyond top_k to diversify from.
    fetch_k = top_k * 2
    documents: list[ContextDocument] = []
    for provider in _source_providers_for(source_provider):
        if provider == "browser":
            if browser_search_url:
                browser_docs = await _run_browser_search(
                    query,
                    browser_search_url=browser_search_url,
                    top_k=fetch_k,
                    existing_count=len(documents),
                )
                documents.extend(browser_docs)
            continue
        pages = await search_tool(
            query,
            provider=provider,
            search_url=search_url,
            page_size=fetch_k,
            # Access filters apply to the internal corpus only (search_tool
            # forwards them for the 'retrieval' provider and ignores them for
            # web providers).
            filters=filters,
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
    # Sidecar: add browser when it isn't already the primary or part of "all".
    # "auto" (internal + serpapi) deliberately excludes the slow browser.
    if browser_search_url and source_provider not in {"browser", "all", "auto"}:
        browser_docs = await _run_browser_search(
            query,
            browser_search_url=browser_search_url,
            top_k=fetch_k,
            existing_count=len(documents),
        )
        documents.extend(browser_docs)
    return await _rank_documents(documents, query, rerank_url, top_k)


async def _run_browser_search(
    query: str,
    *,
    browser_search_url: str,
    top_k: int,
    existing_count: int,
) -> list[ContextDocument]:
    """Call a running browser.py /retrieve server and convert results to ContextDocuments."""
    try:
        pages = await search_tool(
            query,
            provider="retrieval",
            search_url=browser_search_url,
            page_size=top_k,
        )
    except Exception as exc:
        logger.warning("Browser search failed for %r: %s", query, exc)
        return []
    return _documents_from_search_pages(
        pages,
        source_provider="browser",
        query=query,
        start_index=existing_count + 1,
    )


async def _rerank_documents(
    docs: list[ContextDocument],
    query: str,
    rerank_url: str,
) -> list[ContextDocument]:
    """Compatibility wrapper around the shared ranking policy."""
    return await _rank_documents(docs, query, rerank_url, len(docs))


async def _rank_documents(
    documents: list[ContextDocument],
    query: str,
    rerank_url: str | None,
    top_k: int,
) -> list[ContextDocument]:
    """Adapt web documents to the shared default ranking policy."""
    candidates = CandidateSet(
        query=query,
        provider="web",
        candidates=[
            SearchResult(
                contents=document.content,
                title=document.title,
                url=document.url,
                score=document.score,
                metadata=document.metadata,
            )
            for document in documents
        ],
    )
    reranker = (
        RerankHTTPRankingStage(
            rerank_url,
            document_contents=lambda candidate: (
                f"{candidate.title}\n{candidate.contents}"
            ),
            send_top_k=False,
        )
        if rerank_url
        else None
    )
    ranked = await DefaultRankingStage(reranker=reranker).rank(query, candidates, top_k)
    return _RankedDocumentList(ranked.evidence, ranked.metadata)


def _provider_error_doc(provider: str, message: str) -> ContextDocument:
    """A placeholder doc marking a provider that failed/timed out. Filtered from
    results; its presence signals 'unreachable' when no real docs were found."""
    return ContextDocument(
        id="D0",
        title="Search error",
        content=message,
        url=None,
        score=0.0,
        metadata={
            "source_provider": provider,
            "source": _source_label(provider),
            "error": True,
        },
    )


async def _finalize_hybrid(
    documents: list[ContextDocument],
    *,
    executed_queries: list[str],
    query: str,
    rerank_url: str | None,
    top_k: int,
) -> _HybridSearchResult:
    real = [d for d in documents if not d.metadata.get("error")]
    errored = [d for d in documents if d.metadata.get("error")]
    if not real:
        status = "unreachable" if errored else "empty"
        return _HybridSearchResult(
            executed_queries=executed_queries,
            documents=[],
            status=status,
            ranking={"operations": [], "candidate_count": 0},
        )
    diversified = await _rank_documents(real, query, rerank_url, top_k)
    ranking = _ranking_metadata(diversified, fallback_operations=["hybrid_ranking"])
    return _HybridSearchResult(
        executed_queries=executed_queries,
        documents=_reindex_documents(diversified),
        status="ok" if diversified else "empty",
        ranking=ranking,
    )


async def _run_hybrid_search(
    query: str,
    *,
    llm: LLMClient | None,
    search_url: str,
    browser_search_url: str | None = None,
    rerank_url: str | None = None,
    top_k: int,
    filters: SearchFilters | None,
    source_provider: str,
) -> _HybridSearchResult:
    if source_provider == "retrieval":
        # Path A — corpus retrieval with its own query expansion pipeline.
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
        diversified = await _rank_documents(context.documents, query, None, top_k)
        ranking = _ranking_metadata(diversified, fallback_operations=["hybrid_ranking"])
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
            status="ok" if diversified else "empty",
            ranking=ranking,
        )

    if source_provider == "browser":
        # Path B-browser — playwright browser server; no query expansion (too slow).
        browser_docs: list[ContextDocument] = []
        if browser_search_url:
            browser_docs = await _run_browser_search(
                query,
                browser_search_url=browser_search_url,
                top_k=top_k * 2,
                existing_count=0,
            )
        diversified_b = await _rank_documents(browser_docs, query, rerank_url, top_k)
        ranking = _ranking_metadata(
            diversified_b, fallback_operations=["hybrid_ranking"]
        )
        return _HybridSearchResult(
            executed_queries=[query],
            documents=_reindex_documents(diversified_b),
            status="ok" if diversified_b else "empty",
            ranking=ranking,
        )

    executed_queries = _expanded_queries(query, llm)

    async def _fetch_provider(provider: str) -> list[ContextDocument]:
        if provider == "browser":
            if not browser_search_url:
                return []
            # IDs are globally reassigned by _finalize_hybrid -> _reindex_documents, so starting at 0 here is safe.
            return await _run_browser_search(
                query,
                browser_search_url=browser_search_url,
                top_k=top_k * 2,
                existing_count=0,
            )
        page_lists: list[list[SearchPage]] = list(
            await asyncio.gather(
                *[
                    search_tool(
                        expanded_query,
                        provider=provider,
                        search_url=search_url,
                        page_size=top_k,
                        timeout_seconds=5,
                        max_retries=1,
                        **({"filters": filters} if provider == "retrieval" else {}),
                    )
                    for expanded_query in executed_queries
                ]
            )
        )
        if _is_web_provider(provider):
            all_pages = [p for pages in page_lists for p in pages]
            enriched = await fetch_pages_concurrently(all_pages, max_chars=2000)
            it = iter(enriched)
            page_lists = [list(islice(it, len(pages))) for pages in page_lists]
        docs: list[ContextDocument] = []
        for expanded_query, pages in zip(executed_queries, page_lists):
            docs.extend(
                _documents_from_search_pages(
                    pages,
                    source_provider=provider,
                    query=expanded_query,
                    start_index=len(docs) + 1,
                    entry_point="hybrid_search",
                )
            )
        return docs

    async def _fetch_provider_guarded(provider: str) -> list[ContextDocument]:
        try:
            return await asyncio.wait_for(_fetch_provider(provider), timeout=8.0)
        except Exception as exc:  # timeout or provider error → mark unreachable
            logger.warning("Provider %s failed/timed out: %s", provider, exc)
            return [_provider_error_doc(provider, str(exc))]

    def _has_usable(docs: list[ContextDocument]) -> bool:
        return any(not doc.metadata.get("error") for doc in docs)

    async def _fetch_web_with_fallback() -> list[ContextDocument]:
        """Cascade the web leg for 'auto': try fast web search (SerpAPI) first;
        if it yields no usable results (missing key, error, or empty), fall back
        to the browser provider. Keeps internal+web hybrid useful without a key."""
        web_docs = await _fetch_provider_guarded("serpapi")
        if _has_usable(web_docs):
            return web_docs
        if browser_search_url:
            browser_docs = await _fetch_provider_guarded("browser")
            if _has_usable(browser_docs):
                return browser_docs
        return web_docs  # propagate error doc so status reflects 'unreachable'

    if source_provider == "auto":
        # Local retrieval ∥ (SerpAPI → browser fallback); MMR picks top_k below.
        legs = await asyncio.gather(
            _fetch_provider_guarded("retrieval"),
            _fetch_web_with_fallback(),
        )
    else:
        legs = await asyncio.gather(
            *[
                _fetch_provider_guarded(provider)
                for provider in _source_providers_for(source_provider)
            ]
        )
    documents = [doc for docs in legs for doc in docs]
    return await _finalize_hybrid(
        documents,
        executed_queries=executed_queries,
        query=query,
        rerank_url=rerank_url,
        top_k=top_k,
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
                score=page.score,
                metadata={
                    "entry_point": entry_point,
                    "source": _source_label(source_provider),
                    "source_provider": source_provider,
                    "query": query,
                    "error": bool(page.error),
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


def _reindex_documents(documents: list[ContextDocument]) -> list[ContextDocument]:
    return [
        ContextDocument(
            id=f"D{index}",
            title=document.title,
            content=document.content,
            url=document.url,
            score=document.score,
            metadata={**document.metadata, "mmr_rank": index},
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
