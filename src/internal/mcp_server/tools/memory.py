"""MCP tools for user-memory management (thin wrappers over memory.service)."""

from __future__ import annotations

import logging
import os
from dataclasses import asdict
from typing import Any

from src.internal.configs.app_configs import load_app_settings
from src.internal.db.store import AgenticSearchStore
from src.internal.llm.interfaces import LLMConfig
from src.internal.llm.providers import OpenAICompatibleLLM
from src.internal.memory import service
from src.internal.memory.service import DEFAULT_MEMORY_USER_ID, maybe_build_encoder

from ..api import mcp_server
from ..utils import require_access_token

logger = logging.getLogger(__name__)

_STORE: AgenticSearchStore | None = None
_WARNED_MEMORY_DB = False


def _get_store() -> AgenticSearchStore:
    global _STORE, _WARNED_MEMORY_DB
    if _STORE is None:
        db_path = load_app_settings().services.web_db_path
        if str(db_path) == ":memory:" and not _WARNED_MEMORY_DB:
            _WARNED_MEMORY_DB = True
            logger.warning(
                "Memory store is using an in-memory (:memory:) SQLite DB. This MCP "
                "server runs as a separate process from the web backend, so it "
                "will not share chat sessions or memories with it, and memories "
                "will not persist across restarts. Set AGENTIC_SEARCH_WEB_DB_PATH "
                "to the same real file path used by the web backend to share data."
            )
        _STORE = AgenticSearchStore(db_path)
    return _STORE


def _resolve_user_id() -> str:
    try:
        token = require_access_token()
        sub = (getattr(token, "claims", None) or {}).get("sub")
        if isinstance(sub, str) and sub.strip():
            return sub.strip()
    except Exception:  # noqa: BLE001 — unauthenticated/local falls back
        pass
    # Read at call time, not import time, so the same env var governs this
    # separate process as governs the web router. Enforcing only there would
    # leave MCP as an unguarded second door to the same shared bucket.
    if os.getenv("AGENTIC_SEARCH_MEMORY_REQUIRE_AUTH", "").lower() in {
        "1",
        "true",
        "yes",
    }:
        raise PermissionError(
            "AGENTIC_SEARCH_MEMORY_REQUIRE_AUTH is set: memory tools need an "
            "authenticated caller."
        )
    return DEFAULT_MEMORY_USER_ID


def _build_llm() -> OpenAICompatibleLLM | None:
    api_key = os.getenv("OPENAI_API_KEY") or os.getenv("AGENTIC_SEARCH_LLM_API_KEY")
    if not api_key:
        return None
    return OpenAICompatibleLLM(
        LLMConfig(
            model_provider=os.getenv("AGENTIC_SEARCH_LLM_PROVIDER", "openai"),
            model_name=os.getenv("AGENTIC_SEARCH_LLM_MODEL", "gpt-4o-mini"),
            api_key=api_key,
            api_base=os.getenv("AGENTIC_SEARCH_LLM_API_BASE"),
        )
    )


@mcp_server.tool()
async def save_memory(text: str) -> dict[str, Any]:
    """Save one explicit long-term memory (a contextual sentence) for the user."""
    try:
        memory_id = service.save_memory(_get_store(), _resolve_user_id(), text)
        if memory_id is None:
            return {"status": "empty", "message": "content was empty; nothing saved"}
        return {"status": "ok", "memory_id": memory_id}
    except Exception as exc:  # noqa: BLE001
        return {"status": "error", "message": str(exc)}


@mcp_server.tool()
async def update_memory_from_conversation(
    session_id: str | None = None,
) -> dict[str, Any]:
    """Read the user's conversation(s) + memories and reconcile memories via the LLM."""
    llm = _build_llm()
    if llm is None:
        return {"status": "error", "message": "no LLM configured (set OPENAI_API_KEY)"}
    try:
        return await service.curate_from_conversation(
            _get_store(), _resolve_user_id(), llm, session_id=session_id
        )
    except Exception as exc:  # noqa: BLE001
        return {"status": "error", "message": str(exc)}


@mcp_server.tool()
async def generate_user_profile() -> dict[str, Any]:
    """Consolidate the user's memories into a structured {topic, subtopic, content} profile."""
    llm = _build_llm()
    if llm is None:
        return {"status": "error", "message": "no LLM configured (set OPENAI_API_KEY)"}
    try:
        entries = service.generate_user_profile(_get_store(), _resolve_user_id(), llm)
        return {"status": "ok", "profile": [asdict(e) for e in entries]}
    except Exception as exc:  # noqa: BLE001
        return {"status": "error", "message": str(exc)}


@mcp_server.tool()
async def get_user_profile() -> dict[str, Any]:
    """Return the persisted structured user profile."""
    try:
        entries = service.get_user_profile(_get_store(), _resolve_user_id())
        return {"status": "ok", "profile": [asdict(e) for e in entries]}
    except Exception as exc:  # noqa: BLE001
        return {"status": "error", "message": str(exc)}


@mcp_server.tool()
async def search_memories(query: str, max_results: int = 5) -> dict[str, Any]:
    """Semantically (or lexically) search the user's memories."""
    try:
        hits = service.search_memories(
            _get_store(),
            _resolve_user_id(),
            query,
            max_results=max_results,
            encoder=maybe_build_encoder(),
        )
        return {
            "status": "ok",
            "results": [
                {"id": r.id, "text": r.memory_text, "score": s} for r, s in hits
            ],
        }
    except Exception as exc:  # noqa: BLE001
        return {"status": "error", "message": str(exc)}


@mcp_server.tool()
async def consolidate_memories(resolve_conflicts: bool = True) -> dict[str, Any]:
    """Deterministically dedup + resolve tagged conflicts in the user's memories."""
    try:
        report = service.consolidate_memories(
            _get_store(), _resolve_user_id(), resolve_conflicts=resolve_conflicts
        )
        return {"status": "ok", "report": report}
    except Exception as exc:  # noqa: BLE001
        return {"status": "error", "message": str(exc)}
