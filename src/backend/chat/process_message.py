"""
IMPORTANT: familiarize yourself with the design concepts prior to contributing to this file.
An overview can be found in the README.md file in this directory.
"""

from __future__ import annotations

import logging as _logging

import contextvars
import io
import os
import queue
import re
import threading
import traceback
from collections.abc import Callable
from collections.abc import Generator
from concurrent.futures import ThreadPoolExecutor
from contextvars import Token
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Final
from uuid import UUID

from pydantic import BaseModel

if TYPE_CHECKING:
    from sqlalchemy.orm import Session
from src.backend.chat.chat_processing_checker import set_processing_status
from src.backend.chat.chat_state import AvailableFiles
from src.backend.chat.chat_state import ChatStateContainer
from src.backend.chat.chat_state import ChatTurnSetup
from src.backend.chat.chat_utils import build_file_context
from src.backend.chat.chat_utils import convert_chat_history
from src.backend.chat.chat_utils import create_chat_history_chain
from src.backend.chat.chat_utils import create_chat_session_from_request
from src.backend.chat.chat_utils import get_persona_prompt
from src.backend.chat.chat_utils import is_last_assistant_message_clarification
from src.backend.chat.chat_utils import load_all_chat_files
from src.backend.chat.compression import calculate_total_history_tokens
from src.backend.chat.compression import compress_chat_history
from src.backend.chat.compression import find_summary_for_branch
from src.backend.chat.compression import get_compression_params
from src.backend.chat.emitter import Emitter
from src.backend.chat.llm_loop import EmptyLLMResponseError
from src.backend.chat.llm_loop import run_llm_loop
from src.backend.chat.models import AnswerStream
from src.backend.chat.models import AnswerStreamPart
from src.backend.chat.models import ChatBasicResponse
from src.backend.chat.models import ChatFullResponse
from src.backend.chat.models import ChatLoadedFile
from src.backend.chat.models import ChatMessageSimple
from src.backend.chat.models import ContextFileMetadata
from src.backend.chat.models import CreateChatSessionID
from src.backend.chat.models import ExtractedContextFiles
from src.backend.chat.models import FileToolMetadata
from src.backend.chat.models import SearchParams
from src.backend.chat.models import StreamingError
from src.backend.chat.models import ToolCallResponse
from src.backend.chat.prompt_utils import calculate_reserved_tokens
from src.backend.chat.save_chat import save_chat_turn
from src.backend.chat.stop_signal_checker import is_connected as check_stop_signal
from src.backend.chat.stop_signal_checker import reset_cancel_status
from src.backend.configs.app_configs import INTEGRATION_TESTS_MODE
from src.backend.configs.constants import DEFAULT_PERSONA_ID
from src.backend.configs.constants import DocumentSource
from src.backend.servers.query_history.models import MessageType
from src.backend.configs.constants import MilestoneRecordType
from src.retrieval.models import SearchDoc
from src.backend.db.memory import get_memories
from src.backend.db.store import get_session_with_current_tenant
from src.backend.db.models import ChatMessage
from src.backend.db.models import Persona
from src.backend.db.models import UserFile
from src.backend.file_store.models import ChatFileType
from src.backend.file_store.models import InMemoryChatFile
from src.backend.cache.interface import InMemoryCache as _InMemoryCache
from src.backend.llm.interfaces import LLM
from src.backend.llm.interfaces import LLMConfig
from src.backend.llm.interfaces import LLMUserIdentity
from src.backend.llm.providers import OpenAICompatibleLLM
from src.backend.db.models import SlackContext
from src.backend.servers.query_and_chat.streaming_models import MessageResponseIDInfo
from src.backend.servers.query_and_chat.streaming_models import (
    MultiModelMessageResponseIDInfo,
)
from src.backend.servers.query_and_chat.models import SendMessageRequest
from src.backend.servers.query_and_chat.models import Placement
from src.backend.servers.query_and_chat.streaming_models import CitationInfo
from src.backend.servers.query_and_chat.streaming_models import OverallStop
from src.backend.servers.query_and_chat.streaming_models import Packet
from src.backend.chat.llm_step import AgentResponseDelta
from src.backend.chat.llm_step import AgentResponseStart
from src.backend.db.models import User
from src.backend.hooks.executor import HookPoint
from src.backend.hooks.executor import HookSkipped
from src.backend.hooks.executor import HookSoftFailed
from src.backend.hooks.executor import execute_hook
from src.backend.tools.models import ChatFile
from src.backend.tools.models import SearchToolUsage
from src.shared_configs.contextvars import get_current_tenant_id


def setup_logger():
    return _logging.getLogger(__name__)


logger = setup_logger()
ERROR_TYPE_CANCELLED = "cancelled"
APPROX_CHARS_PER_TOKEN = 4

# ---------------------------------------------------------------------------
# Config flags
# ---------------------------------------------------------------------------
DISABLE_VECTOR_DB: bool = False
AUTO_PLACE_AFTER_LATEST_MESSAGE: int = -1
FILE_READER_TOOL_ID = "file_reader"
SEARCH_TOOL_ID = "search"


# ---------------------------------------------------------------------------
# Search / filter stubs
# ---------------------------------------------------------------------------


@dataclass
class BaseFilters:
    source_type: list | None = None
    document_set: list | None = None


# ---------------------------------------------------------------------------
# LLM stubs
# ---------------------------------------------------------------------------


@dataclass
class LLMOverride:
    display_name: str | None = None
    model_version: str | None = None
    model_provider: str | None = None


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------


class _AppCode:
    def __init__(self, code: str) -> None:
        self.code = code

    def __repr__(self) -> str:
        return f"AppErrorCode.{self.code}"


class AppErrorCode:
    QUERY_REJECTED = _AppCode("QUERY_REJECTED")
    INSUFFICIENT_PERMISSIONS = _AppCode("INSUFFICIENT_PERMISSIONS")


class AppError(Exception):
    def __init__(
        self, error_code: _AppCode, detail: str = "", status_code: int = 400
    ) -> None:
        super().__init__(detail)
        self.error_code = error_code
        self.detail = detail
        self.status_code = status_code


def log_app_error(e: Exception) -> None:
    logger.error("AppError: %s", e)


# ---------------------------------------------------------------------------
# Hook data models
# ---------------------------------------------------------------------------


class QueryProcessingPayload(BaseModel):
    query: str
    user_email: str | None = None
    chat_session_id: str | None = None


class QueryProcessingResponse(BaseModel):
    query: str | None = None
    rejection_message: str | None = None


# ---------------------------------------------------------------------------
# Server model stubs
# ---------------------------------------------------------------------------


class ModelResponseSlot(BaseModel):
    message_id: int
    model_name: str


# ---------------------------------------------------------------------------
# DB function stubs (no SQLAlchemy session in this repo)
# ---------------------------------------------------------------------------


def get_chat_session_by_id(
    chat_session_id: Any,
    user_id: Any,
    db_session: Any,
    *,
    eager_load_persona: bool = False,
) -> Any:
    from src.backend.db.models import ChatSession, Persona

    cs = ChatSession(id=chat_session_id, user_id=str(user_id) if user_id else None)
    cs.persona = Persona()
    return cs


def get_or_create_root_message(chat_session_id: Any, db_session: Any) -> Any:
    from src.backend.db.models import ChatMessage

    return ChatMessage(
        id=0, session_id=chat_session_id, chat_session_id=chat_session_id
    )


def create_new_chat_message(
    *,
    chat_session_id: Any,
    parent_message: Any,
    message: str,
    token_count: int,
    message_type: Any,
    files: Any = None,
    db_session: Any,
    commit: bool = True,
) -> Any:
    from src.backend.db.models import ChatMessage

    return ChatMessage(
        id=1,
        session_id=chat_session_id,
        chat_session_id=chat_session_id,
        message=message,
        message_type=message_type,
        token_count=token_count,
    )


def reserve_message_id(
    *,
    db_session: Any,
    chat_session_id: Any,
    parent_message: Any,
    message_type: Any,
    model_display_name: str | None = None,
) -> Any:
    from src.backend.db.models import ChatMessage

    return ChatMessage(
        id=1,
        session_id=chat_session_id,
        chat_session_id=chat_session_id,
        message_type=message_type,
    )


def reserve_multi_model_message_ids(
    *,
    db_session: Any,
    chat_session_id: Any,
    parent_message_id: Any,
    model_display_names: list[str],
) -> list[Any]:
    from src.backend.db.models import ChatMessage

    return [
        ChatMessage(
            id=i + 1, session_id=chat_session_id, chat_session_id=chat_session_id
        )
        for i in range(len(model_display_names))
    ]


def filter_document_set_names_by_user_access(
    *, db_session: Any, document_set_names: Any, user: Any
) -> set[str]:
    return set(document_set_names)


def get_user_files_from_project(
    *, project_id: Any, user_id: Any, db_session: Any
) -> list:
    return []


def get_tools(db_session: Any) -> list:
    return []


# ---------------------------------------------------------------------------
# LLM factory stubs
# ---------------------------------------------------------------------------


def get_llm_for_persona(
    *,
    persona: Any,
    user: Any,
    llm_override: Any = None,
    additional_headers: Any = None,
) -> LLM:
    """Return an LLM instance for the given persona and optional override.

    Resolution order:
    1. Explicit ``llm_override`` (model_provider + model_version fields).
    2. AppSettings defaults loaded from GEN_AI_* environment variables.
    """
    from src.backend.configs.app_configs import load_app_settings

    defaults = load_app_settings().llm

    provider = defaults.model_provider
    model = defaults.model_name
    api_key = defaults.api_key
    api_base = defaults.api_base
    max_tokens = defaults.max_input_tokens

    if llm_override is not None:
        if getattr(llm_override, "model_provider", None):
            provider = llm_override.model_provider
        if getattr(llm_override, "model_version", None):
            model = llm_override.model_version

    config = LLMConfig(
        model_provider=provider,
        model_name=model,
        api_key=api_key,
        api_base=api_base,
        max_input_tokens=max_tokens,
    )
    return OpenAICompatibleLLM(config)


def get_llm_token_counter(llm: Any) -> Callable[[str], int]:
    return lambda text: max(1, len(text) // 4)


def set_llm_mock_response(response: Any) -> Any:
    return None


def reset_llm_mock_response(token: Any) -> None:
    pass


def litellm_exception_to_error_msg(e: Exception, llm: Any) -> tuple[str, str, bool]:
    return str(e), "LLM_ERROR", True


# ---------------------------------------------------------------------------
# File / MIME stubs
# ---------------------------------------------------------------------------


def extract_file_text(
    file: Any, file_name: str = "", break_on_unprocessable: bool = True
) -> str:
    try:
        return file.read().decode("utf-8", errors="ignore")
    except Exception:
        return ""


class _NullFileStore:
    def read_file(self, file_id: str, mode: str = "b") -> Any:
        return io.BytesIO(b"")

    def read_file_record(self, file_id: str) -> None:
        raise FileNotFoundError(file_id)


def get_default_file_store() -> _NullFileStore:
    return _NullFileStore()


def load_in_memory_chat_files(*, user_file_ids: list, db_session: Any) -> list:
    return []


def verify_user_files(
    *, user_files: Any, user_id: Any, db_session: Any, project_id: Any = None
) -> None:
    """Verify that every file in *user_files* is accessible to *user_id*.

    Raises ``AppError(INSUFFICIENT_PERMISSIONS)`` for the first file that
    cannot be verified.  Files with no ``id`` attribute are skipped silently —
    they are transient in-memory files that were never persisted.
    """
    if not user_files:
        return

    for file_ref in user_files:
        file_id = getattr(file_ref, "id", None) or getattr(file_ref, "file_id", None)
        if not file_id:
            continue

        # Try to read the file record from the store; absence means no access.
        try:
            store = get_default_file_store()
            store.read_file_record(str(file_id))
        except FileNotFoundError:
            logger.warning(
                "verify_user_files: file %s not found for user %s (project=%s)",
                file_id,
                user_id,
                project_id,
            )
            raise AppError(
                AppErrorCode.INSUFFICIENT_PERMISSIONS,
                detail=f"File {file_id} is not accessible.",
                status_code=403,
            )


def mime_type_to_chat_file_type(mime_type: str | None) -> Any:
    from src.backend.file_store.models import ChatFileType

    if mime_type and "image" in mime_type:
        return ChatFileType.IMAGE
    if mime_type and "text" in mime_type:
        return ChatFileType.PLAIN_TEXT
    return ChatFileType.OTHER


# ---------------------------------------------------------------------------
# Tool construction stubs
# ---------------------------------------------------------------------------


@dataclass
class SearchToolConfig:
    user_selected_filters: Any = None
    project_id_filter: int | None = None
    persona_id_filter: int | None = None
    bypass_acl: bool = False
    slack_context: Any = None
    enable_slack_search: bool = False


@dataclass
class CustomToolConfig:
    chat_session_id: Any = None
    message_id: Any = None
    additional_headers: dict | None = None
    mcp_headers: dict | None = None


@dataclass
class FileReaderToolConfig:
    user_file_ids: list = field(default_factory=list)
    chat_file_ids: list = field(default_factory=list)


def construct_tools(
    *,
    persona: Any,
    emitter: Any,
    user: Any,
    llm: Any,
    search_tool_config: SearchToolConfig,
    custom_tool_config: CustomToolConfig,
    file_reader_tool_config: FileReaderToolConfig,
    allowed_tool_ids: Any = None,
    search_usage_forcing_setting: Any = None,
) -> dict:
    return {}


def get_cache_backend() -> _InMemoryCache:
    return _InMemoryCache()


# ---------------------------------------------------------------------------
# Telemetry / cost stubs
# ---------------------------------------------------------------------------


def mt_cloud_telemetry(
    *,
    tenant_id: Any = None,
    distinct_id: Any = None,
    event: Any = None,
    properties: Any = None,
) -> None:
    pass


_provider_token_windows: dict[str, list[float]] = {}
_provider_token_lock = __import__("threading").Lock()
_TOKEN_WINDOW_SECONDS: Final[int] = 60
_TOKEN_LIMIT_ENV_VAR: Final[str] = "GEN_AI_TOKEN_LIMIT_PER_MINUTE"


def check_llm_cost_limit_for_provider(
    *, db_session: Any = None, tenant_id: Any = None, llm_provider_api_key: Any = None
) -> None:
    """Enforce a per-provider sliding-window token-request rate limit.

    The limit is read from the ``GEN_AI_TOKEN_LIMIT_PER_MINUTE`` environment
    variable (default: unlimited).  The key is the first 8 chars of the API key
    (or ``"default"`` when no key is configured) so different key holders are
    rate-limited independently.

    Raises ``AppError`` with HTTP 429 when the limit is exceeded.
    """
    import os
    import time

    raw_limit = os.environ.get(_TOKEN_LIMIT_ENV_VAR, "").strip()
    if not raw_limit:
        return
    try:
        limit = int(raw_limit)
    except ValueError:
        logger.warning(
            "check_llm_cost_limit_for_provider: invalid %s=%r — skipping check",
            _TOKEN_LIMIT_ENV_VAR,
            raw_limit,
        )
        return

    key = (str(llm_provider_api_key or "default"))[:8]
    now = time.monotonic()
    cutoff = now - _TOKEN_WINDOW_SECONDS

    with _provider_token_lock:
        window = _provider_token_windows.setdefault(key, [])
        # Evict expired timestamps
        _provider_token_windows[key] = [t for t in window if t >= cutoff]
        count = len(_provider_token_windows[key])
        if count >= limit:
            raise AppError(
                AppErrorCode.QUERY_REJECTED,
                detail=(
                    f"LLM request rate limit of {limit} requests/minute exceeded. "
                    "Please wait and try again."
                ),
                status_code=429,
            )
        _provider_token_windows[key].append(now)


# ---------------------------------------------------------------------------
# Misc
# ---------------------------------------------------------------------------


def log_function_time(print_only: bool = False) -> Any:
    def decorator(fn: Any) -> Any:
        return fn

    return decorator


def _collect_available_file_ids(
    chat_history: list[ChatMessage],
    project_id: int | None,
    user_id: UUID | None,
    db_session: Session,
) -> AvailableFiles:
    """Collect all file IDs the FileReaderTool should be allowed to access.

    Returns *separate* lists for chat-attached files (``file_record`` IDs) and
    project/user files (``user_file`` IDs) so the tool can pick the right
    loader without a try/except fallback."""
    chat_file_ids: set[UUID] = set()
    user_file_ids: set[UUID] = set()

    for msg in chat_history:
        if not msg.files:
            continue
        for fd in msg.files:
            try:
                chat_file_ids.add(UUID(fd["id"]))
            except (ValueError, KeyError):
                pass

    if project_id:
        user_files = get_user_files_from_project(
            project_id=project_id,
            user_id=user_id,
            db_session=db_session,
        )
        for uf in user_files:
            user_file_ids.add(uf.id)

    return AvailableFiles(
        user_file_ids=list(user_file_ids),
        chat_file_ids=list(chat_file_ids),
    )


def _should_enable_slack_search(
    persona: Persona,
    filters: BaseFilters | None,
) -> bool:
    """Determine if Slack search should be enabled.

    Returns True if:
    - Source type filter exists and includes Slack, OR
    - Default persona with no source type filter
    """
    source_types = filters.source_type if filters else None
    return (source_types is not None and DocumentSource.SLACK in source_types) or (
        persona.id == DEFAULT_PERSONA_ID and source_types is None
    )


def _convert_loaded_files_to_chat_files(
    loaded_files: list[ChatLoadedFile],
) -> list[ChatFile]:
    """Convert ChatLoadedFile objects to ChatFile for tool usage (e.g., PythonTool).

    Returns lazy ChatFile objects: ``.content`` materializes via the underlying
    ``loaded_file.content`` only when a tool actually accesses it. Previously
    this function gated on ``len(loaded_file.content) > 0`` to filter out
    zero-byte files, but evaluating ``len(content)`` would force every lazy
    file to materialize and defeat the OOM fix. The guard is dropped; tools
    receive zero-byte content for empty files, which PythonTool handles fine
    (sha256 of empty bytes + upload of an empty body — the LLM will see the
    empty result and react).
    """
    chat_files: list[ChatFile] = []
    for loaded_file in loaded_files:
        filename = loaded_file.filename or f"file_{loaded_file.file_id}"
        # Pull content via a closure so the bytes only flow through one
        # materialization (the ChatLoadedFile's loader), then ride along.
        chat_files.append(
            ChatFile.lazy_from_filename(
                filename=filename,
                loader=lambda lf=loaded_file: lf.content,
            )
        )
    return chat_files


def _deduped_filename(filename: str, seen_filenames: set[str], file_id: str) -> str:
    if filename not in seen_filenames:
        seen_filenames.add(filename)
        return filename

    stem, suffix = os.path.splitext(filename)
    deduped_filename = f"{stem}_{file_id}{suffix}"
    seen_filenames.add(deduped_filename)
    return deduped_filename


def _load_context_user_files_for_tools(
    user_files: list[UserFile],
    existing_filenames: set[str],
) -> list[ChatFile]:
    """Stage tabular project/persona files for code-interpreter as lazy
    ChatFile instances.

    Raw bytes are not read here; each ChatFile carries a loader closure that
    pulls from the file store only when PythonTool actually accesses
    ``.content`` during staging. This avoids loading every project/persona
    file into RAM for chats that never invoke the Python tool.
    """
    if not user_files:
        return []

    chat_files: list[ChatFile] = []
    seen_file_ids: set[str] = set()

    for user_file in user_files:
        if user_file.file_id in seen_file_ids:
            continue
        seen_file_ids.add(user_file.file_id)

        if not mime_type_to_chat_file_type(user_file.file_type).use_metadata_only():
            continue

        filename = _deduped_filename(
            user_file.name or f"file_{user_file.id}",
            existing_filenames,
            str(user_file.id),
        )

        def _load(
            file_id: str = user_file.file_id, user_file_id: UUID = user_file.id
        ) -> bytes:
            # Preserve the pre-lazy degraded-but-functional behavior: if the
            # underlying file is gone or temporarily unreachable, log it and
            # hand PythonTool an empty payload instead of letting the
            # exception propagate out of ChatFile.__getattribute__.
            try:
                return get_default_file_store().read_file(file_id, mode="b").read()
            except Exception as e:
                logger.warning(
                    "Failed to load context file %s for Python execution: %s",
                    user_file_id,
                    e,
                )
                return b""

        chat_files.append(ChatFile.lazy_from_filename(filename=filename, loader=_load))

    return chat_files


def resolve_context_user_files(
    persona: Persona,
    project_id: int | None,
    user_id: UUID | None,
    db_session: Session,
) -> list[UserFile]:
    """Apply the precedence rule to decide which user files to load.

    A custom persona fully supersedes the project.  When a chat uses a
    custom persona, the project is purely organisational — its files are
    never loaded and never made searchable.

    Custom persona → persona's own user_files (may be empty).
    Default persona inside a project → project files.
    Otherwise → empty list.
    """
    if persona.id != DEFAULT_PERSONA_ID:
        return list(persona.user_files) if persona.user_files else []
    if project_id:
        return get_user_files_from_project(
            project_id=project_id,
            user_id=user_id,
            db_session=db_session,
        )
    return []


def _empty_extracted_context_files() -> ExtractedContextFiles:
    return ExtractedContextFiles(
        file_texts=[],
        image_files=[],
        use_as_search_filter=False,
        total_token_count=0,
        file_metadata=[],
        uncapped_token_count=None,
    )


def _extract_text_from_in_memory_file(f: InMemoryChatFile) -> str | None:
    """Extract text content from an InMemoryChatFile.

    PLAIN_TEXT: the content is pre-extracted UTF-8 plaintext stored during
    ingestion — decode directly.
    DOC / CSV / other text types: the content is the original file bytes —
    use extract_file_text which handles encoding detection and format parsing.
    Embedded-image summaries are only generated at index time; chat-time
    extraction is text-only and any image context comes through search.
    """
    try:
        if f.file_type == ChatFileType.PLAIN_TEXT:
            return f.content.decode("utf-8", errors="ignore").replace("\x00", "")

        text_content = extract_file_text(
            file=io.BytesIO(f.content),
            file_name=f.filename or "",
            break_on_unprocessable=False,
        )
        return text_content or None
    except Exception:
        logger.warning("Failed to extract text from file %s", f.file_id, exc_info=True)
        return None


def extract_context_files(
    user_files: list[UserFile],
    llm_max_context_window: int,
    reserved_token_count: int,
    db_session: Session,
    # Because the tokenizer is a generic tokenizer, the token count may be incorrect.
    # to account for this, the maximum context that is allowed for this function is
    # 60% of the LLM's max context window. The other benefit is that for projects with
    # more files, this makes it so that we don't throw away the history too quickly every time.
    max_llm_context_percentage: float = 0.6,
) -> ExtractedContextFiles:
    """Load user files into context if they fit; otherwise flag for search.

    The caller is responsible for deciding *which* user files to pass in
    (project files, persona files, etc.).  This function only cares about
    the all-or-nothing fit check and the actual content loading.

    Args:
        project_id: The project ID to load files from
        user_id: The user ID for authorization
        llm_max_context_window: Maximum tokens allowed in the LLM context window
        reserved_token_count: Number of tokens to reserve for other content
        db_session: Database session
        max_llm_context_percentage: Maximum percentage of the LLM context window to use.
    Returns:
        ExtractedContextFiles containing:
        - List of text content strings from context files (text files only)
        - List of image files from context (ChatLoadedFile objects)
        - Total token count of all extracted files
        - File metadata for context files
        - Uncapped token count of all extracted files
        - File metadata for files that don't fit in context and vector DB is disabled
    """
    # TODO(yuhong): I believe this is not handling all file types correctly.

    if not user_files:
        return _empty_extracted_context_files()

    # Aggregate tokens for the file content that will be added
    # Skip tokens for those with metadata only
    aggregate_tokens = sum(
        uf.token_count or 0
        for uf in user_files
        if not mime_type_to_chat_file_type(uf.file_type).use_metadata_only()
    )
    max_actual_tokens = (
        llm_max_context_window - reserved_token_count
    ) * max_llm_context_percentage

    if aggregate_tokens >= max_actual_tokens:
        use_as_search_filter = not DISABLE_VECTOR_DB
        if DISABLE_VECTOR_DB:
            overflow_tool_metadata = [_build_tool_metadata(uf) for uf in user_files]
        else:
            overflow_tool_metadata = [
                _build_tool_metadata(uf)
                for uf in user_files
                if mime_type_to_chat_file_type(uf.file_type).use_metadata_only()
            ]
        return ExtractedContextFiles(
            file_texts=[],
            image_files=[],
            use_as_search_filter=use_as_search_filter,
            total_token_count=0,
            file_metadata=[],
            uncapped_token_count=aggregate_tokens,
            file_metadata_for_tool=overflow_tool_metadata,
        )

    # Files fit — load them into context
    user_file_map = {uf.file_id: uf for uf in user_files}
    in_memory_files = load_in_memory_chat_files(
        user_file_ids=[uf.id for uf in user_files],
        db_session=db_session,
    )

    file_texts: list[str] = []
    image_files: list[ChatLoadedFile] = []
    file_metadata: list[ContextFileMetadata] = []
    tool_metadata: list[FileToolMetadata] = []
    total_token_count = 0

    for f in in_memory_files:
        uf = user_file_map.get(str(f.file_id))
        filename = f.filename or f"file_{f.file_id}"

        if f.file_type.use_metadata_only():
            # Metadata-only files are not injected as full text.
            # Only the metadata is provided, with LLM using tools
            if not uf:
                logger.error(
                    "File with id=%s in metadata-only path with no associated user file",
                    f.file_id,
                )
                continue
            tool_metadata.append(_build_tool_metadata(uf))
        elif f.file_type.is_text_file():
            text_content = _extract_text_from_in_memory_file(f)
            if not text_content:
                continue
            if not uf:
                logger.warning("No user file for file_id=%s", f.file_id)
                continue
            file_texts.append(text_content)
            file_metadata.append(
                ContextFileMetadata(
                    file_id=str(uf.id),
                    filename=filename,
                    file_content=text_content,
                )
            )
            if uf.token_count:
                total_token_count += uf.token_count
        elif f.file_type == ChatFileType.IMAGE:
            token_count = uf.token_count if uf and uf.token_count else 0
            total_token_count += token_count
            image_files.append(
                ChatLoadedFile(
                    file_id=f.file_id,
                    content=f.content,
                    file_type=f.file_type,
                    filename=f.filename,
                    content_text=None,
                    token_count=token_count,
                )
            )

    return ExtractedContextFiles(
        file_texts=file_texts,
        image_files=image_files,
        use_as_search_filter=False,
        total_token_count=total_token_count,
        file_metadata=file_metadata,
        uncapped_token_count=aggregate_tokens,
        file_metadata_for_tool=tool_metadata,
    )


def _build_tool_metadata(user_file: UserFile) -> FileToolMetadata:
    """Build lightweight FileToolMetadata from a UserFile record.

    Delegates to ``build_file_context`` so that the file ID exposed to the
    LLM is always consistent with what FileReaderTool expects.
    """
    return build_file_context(
        tool_file_id=str(user_file.id),
        filename=user_file.name,
        file_type=mime_type_to_chat_file_type(user_file.file_type),
        approx_char_count=(user_file.token_count or 0) * APPROX_CHARS_PER_TOKEN,
    ).tool_metadata


def determine_search_params(
    persona_id: int,
    project_id: int | None,
    extracted_context_files: ExtractedContextFiles,
) -> SearchParams:
    """Decide which search filter IDs and search-tool usage apply for a chat turn.

    A custom persona fully supersedes the project — project files are never
    searchable and the search tool config is entirely controlled by the
    persona.  The project_id filter is only set for the default persona.

    For the default persona inside a project:
      - Files overflow  → ENABLED  (vector DB scopes to these files)
      - Files fit       → DISABLED (content already in prompt)
      - No files at all → DISABLED (nothing to search)
    """
    is_custom_persona = persona_id != DEFAULT_PERSONA_ID

    project_id_filter: int | None = None
    persona_id_filter: int | None = None
    if extracted_context_files.use_as_search_filter:
        if is_custom_persona:
            persona_id_filter = persona_id
        else:
            project_id_filter = project_id

    search_usage = SearchToolUsage.AUTO
    if not is_custom_persona and project_id:
        has_context_files = bool(extracted_context_files.uncapped_token_count)
        files_loaded_in_context = bool(extracted_context_files.file_texts)

        if extracted_context_files.use_as_search_filter:
            search_usage = SearchToolUsage.ENABLED
        elif files_loaded_in_context or not has_context_files:
            search_usage = SearchToolUsage.DISABLED

    return SearchParams(
        project_id_filter=project_id_filter,
        persona_id_filter=persona_id_filter,
        search_usage=search_usage,
    )


def _resolve_query_processing_hook_result(
    hook_result: QueryProcessingResponse | HookSkipped | HookSoftFailed,
    message_text: str,
) -> str:
    """Apply the Query Processing hook result to the message text.

    Returns the (possibly rewritten) message text, or raises AppError with
    QUERY_REJECTED if the hook signals rejection (query is null or empty).
    HookSkipped and HookSoftFailed are pass-throughs — the original text is
    returned unchanged.
    """
    if isinstance(hook_result, (HookSkipped, HookSoftFailed)):
        return message_text
    if not (hook_result.query and hook_result.query.strip()):
        raise AppError(
            AppErrorCode.QUERY_REJECTED,
            hook_result.rejection_message
            or "The hook extension for query processing did not return a valid query. No rejection reason was provided.",
        )
    return hook_result.query.strip()


def build_chat_turn(
    new_msg_req: SendMessageRequest,
    user: User,
    db_session: Session,
    # None → single-model (persona default LLM); non-empty list → multi-model (one LLM per override)
    llm_overrides: list[LLMOverride] | None,
    *,
    litellm_additional_headers: dict[str, str] | None = None,
    custom_tool_additional_headers: dict[str, str] | None = None,
    mcp_headers: dict[str, str] | None = None,
    bypass_acl: bool = False,
    # Slack context for federated Slack search
    slack_context: SlackContext | None = None,
    # Additional context to include in the chat history, e.g. Slack threads where the
    # conversation cannot be represented by a chain of User/Assistant messages.
    # NOTE: not stored in the database, only passed in to the LLM as context
    additional_context: str | None = None,
) -> Generator[AnswerStreamPart, None, ChatTurnSetup]:
    """Shared setup generator for both single-model and multi-model chat turns.

    Yields the packet(s) the frontend needs for request tracking, then returns an
    immutable ``ChatTurnSetup`` containing everything the execution strategy needs.

    Callers use::

        setup = yield from build_chat_turn(new_msg_req, ..., llm_overrides=...)

    to forward yielded packets upstream while receiving the return value locally.

    Args:
        llm_overrides: ``None`` → single-model (persona default LLM).
                       Non-empty list → multi-model (one LLM per override).
    """
    tenant_id = get_current_tenant_id()
    is_multi = bool(llm_overrides)

    user_id = user.id
    llm_user_identifier = (
        "anonymous_user" if user.is_anonymous else (user.email or str(user_id))
    )

    # ── Session resolution ───────────────────────────────────────────────────
    if not new_msg_req.chat_session_id:
        if not new_msg_req.chat_session_info:
            raise RuntimeError("Must specify a chat session id or chat session info")
        chat_session = create_chat_session_from_request(
            chat_session_request=new_msg_req.chat_session_info,
            user=user,
            db_session=db_session,
        )
        yield CreateChatSessionID(chat_session_id=chat_session.id)
        chat_session = get_chat_session_by_id(
            chat_session_id=chat_session.id,
            user_id=user_id,
            db_session=db_session,
            eager_load_persona=True,
        )
    else:
        chat_session = get_chat_session_by_id(
            chat_session_id=new_msg_req.chat_session_id,
            user_id=user_id,
            db_session=db_session,
            eager_load_persona=True,
        )

    persona = chat_session.persona
    message_text = new_msg_req.message

    user_identity = LLMUserIdentity(
        user_id=llm_user_identifier, session_id=str(chat_session.id)
    )

    # Milestone tracking, most devs using the API don't need to understand this
    mt_cloud_telemetry(
        tenant_id=tenant_id,
        distinct_id=str(user.id) if not user.is_anonymous else tenant_id,
        event=MilestoneRecordType.MULTIPLE_ASSISTANTS,
    )
    mt_cloud_telemetry(
        tenant_id=tenant_id,
        distinct_id=str(user.id) if not user.is_anonymous else tenant_id,
        event=MilestoneRecordType.USER_MESSAGE_SENT,
        properties={
            "origin": new_msg_req.origin.value,
            "has_files": len(new_msg_req.file_descriptors) > 0,
            "has_project": chat_session.project_id is not None,
            "has_persona": persona is not None and persona.id != DEFAULT_PERSONA_ID,
        },
    )

    # Check LLM cost limits then build the LLM instance(s).
    llms: list[LLM] = []
    model_display_names: list[str] = []
    selected_overrides: list[LLMOverride | None] = (
        list(llm_overrides or [])
        if is_multi
        else [new_msg_req.llm_override or chat_session.llm_override]
    )
    for override in selected_overrides:
        llm = get_llm_for_persona(
            persona=persona,
            user=user,
            llm_override=override,
            additional_headers=litellm_additional_headers,
        )
        check_llm_cost_limit_for_provider(
            db_session=db_session,
            tenant_id=tenant_id,
            llm_provider_api_key=llm.config.api_key,
        )
        llms.append(llm)
        model_display_names.append(_build_model_display_name(override, llm))
    token_counter = get_llm_token_counter(llms[0])

    # Verify that the user-specified files actually belong to the user
    verify_user_files(
        user_files=new_msg_req.file_descriptors,
        user_id=user_id,
        db_session=db_session,
        project_id=chat_session.project_id,
    )

    # Re-create linear history of messages
    chat_history = create_chat_history_chain(
        chat_session_id=chat_session.id, db_session=db_session
    )

    # Determine the parent message based on the request:
    # - AUTO_PLACE_AFTER_LATEST_MESSAGE (-1): auto-place after latest message in chain
    # - None or root ID: regeneration from root (first message)
    # - positive int: place after that specific parent message
    root_message = get_or_create_root_message(
        chat_session_id=chat_session.id, db_session=db_session
    )

    if new_msg_req.parent_message_id == AUTO_PLACE_AFTER_LATEST_MESSAGE:
        parent_message = chat_history[-1] if chat_history else root_message
    elif (
        new_msg_req.parent_message_id is None
        or new_msg_req.parent_message_id == root_message.id
    ):
        # Regeneration from root — clear history so we start fresh
        parent_message = root_message
        chat_history = []
    else:
        parent_message = None
        for i in range(len(chat_history) - 1, -1, -1):
            if chat_history[i].id == new_msg_req.parent_message_id:
                parent_message = chat_history[i]
                # Truncate to only messages up to and including the parent
                chat_history = chat_history[: i + 1]
                break

    if parent_message is None:
        raise ValueError(
            "The new message sent is not on the latest mainline of messages"
        )

    # ── Query Processing hook + user message ─────────────────────────────────
    # Skipped on regeneration (parent is USER type): message already exists/was accepted.
    if parent_message.message_type == MessageType.USER:
        user_message = parent_message
    else:
        # New message — run the Query Processing hook before saving to DB.
        # Skipped on regeneration: the message already exists and was accepted previously.
        # Skip for empty/whitespace-only messages — no meaningful query to process,
        # and SendMessageRequest.message has no min_length guard.
        if message_text.strip():
            hook_result = execute_hook(
                db_session=db_session,
                hook_point=HookPoint.QUERY_PROCESSING,
                payload=QueryProcessingPayload(
                    query=message_text,
                    # Pass None for anonymous users or authenticated users without an email
                    # (e.g. some SSO flows). QueryProcessingPayload.user_email is str | None,
                    # so None is accepted and serialised as null in both cases.
                    user_email=None if user.is_anonymous else user.email,
                    chat_session_id=str(chat_session.id),
                ).model_dump(),
                response_type=QueryProcessingResponse,
            )
            message_text = _resolve_query_processing_hook_result(
                hook_result, message_text
            )

        user_message = create_new_chat_message(
            chat_session_id=chat_session.id,
            parent_message=parent_message,
            message=message_text,
            token_count=token_counter(message_text),
            message_type=MessageType.USER,
            files=new_msg_req.file_descriptors,
            db_session=db_session,
            commit=True,
        )
        chat_history.append(user_message)

    # Collect file IDs for the file reader tool *before* summary truncation so
    # that files attached to older (summarized-away) messages are still accessible
    # via the FileReaderTool.
    available_files = _collect_available_file_ids(
        chat_history=chat_history,
        project_id=chat_session.project_id,
        user_id=user_id,
        db_session=db_session,
    )

    # Find applicable summary for the current branch
    summary_message = find_summary_for_branch(db_session, chat_history)
    # Collect file metadata from messages that will be dropped by summary truncation.
    # These become "pre-summarized" file metadata so the forgotten-file mechanism can
    # still tell the LLM about them.
    summarized_file_metadata: dict[str, FileToolMetadata] = {}
    if summary_message and summary_message.last_summarized_message_id:
        cutoff_id = summary_message.last_summarized_message_id
        for msg in chat_history:
            if msg.id > cutoff_id or not msg.files:
                continue
            for fd in msg.files:
                file_id = fd.get("id")
                if not file_id:
                    continue
                summarized_file_metadata[file_id] = FileToolMetadata(
                    file_id=file_id,
                    filename=fd.get("name") or "unknown",
                    # We don't know the exact size without loading the file,
                    # but 0 signals "unknown" to the LLM.
                    approx_char_count=0,
                )
        # Filter chat_history to only messages after the cutoff
        chat_history = [m for m in chat_history if m.id > cutoff_id]

    # Compute skip-clarification flag (cheap, always available)
    skip_clarification = is_last_assistant_message_clarification(chat_history)

    user_memory_context = get_memories(user, db_session)

    # This prompt may come from the Agent or Project. Fetched here (before run_llm_loop)
    # because the inner loop shouldn't need to access the DB-form chat history, but we
    # need it early for token reservation.
    persona_prompt = get_persona_prompt(persona, chat_session)

    # When use_memories is disabled, strip memories from the prompt context but keep
    # user info/preferences. The full context is still passed to the LLM loop for
    # memory tool persistence.
    prompt_memory_context = (
        user_memory_context
        if user.use_memories
        else user_memory_context.without_memories()
    )

    # ── Token reservation ────────────────────────────────────────────────────
    max_reserved_system_prompt_tokens_str = (persona.system_prompt or "") + (
        persona_prompt or ""
    )
    reserved_token_count = calculate_reserved_tokens(
        db_session=db_session,
        persona_system_prompt=max_reserved_system_prompt_tokens_str,
        token_counter=token_counter,
        files=new_msg_req.file_descriptors,
        user_memory_context=prompt_memory_context,
    )

    # Determine which user files to use. A custom persona fully supersedes the project —
    # project files are never loaded or searchable when a custom persona is in play.
    # Only the default persona inside a project uses the project's files.
    context_user_files = resolve_context_user_files(
        persona=persona,
        project_id=chat_session.project_id,
        user_id=user_id,
        db_session=db_session,
    )

    # Use the smallest context window across models for safety (harmless for N=1).
    llm_max_context_window = min(llm.config.max_input_tokens for llm in llms)

    extracted_context_files = extract_context_files(
        user_files=context_user_files,
        llm_max_context_window=llm_max_context_window,
        reserved_token_count=reserved_token_count,
        db_session=db_session,
    )

    search_params = determine_search_params(
        persona_id=persona.id,
        project_id=chat_session.project_id,
        extracted_context_files=extracted_context_files,
    )

    # Also grant access to persona-attached user files for FileReaderTool
    if persona.user_files:
        existing = set(available_files.user_file_ids)
        for uf in persona.user_files:
            if uf.id not in existing:
                available_files.user_file_ids.append(uf.id)

    all_tools = get_tools(db_session)
    tool_id_to_name_map = {tool.id: tool.name for tool in all_tools}

    search_tool_id = next(
        (tool.id for tool in all_tools if tool.in_code_tool_id == SEARCH_TOOL_ID), None
    )

    forced_tool_id = new_msg_req.forced_tool_id
    if (
        search_params.search_usage == SearchToolUsage.DISABLED
        and forced_tool_id is not None
        and search_tool_id is not None
        and forced_tool_id == search_tool_id
    ):
        forced_tool_id = None

    # TODO(nmgarza5): Once summarization is done, we don't need to load all files from the beginning.
    # Load all files needed for this chat chain into memory.
    files = load_all_chat_files(chat_history, db_session)
    # Convert loaded files to ChatFile format for tools like PythonTool
    chat_files_for_tools = _convert_loaded_files_to_chat_files(files)
    chat_files_for_tools.extend(
        _load_context_user_files_for_tools(
            context_user_files,
            {chat_file.filename for chat_file in chat_files_for_tools},
        )
    )

    # ── Reserve assistant message ID(s) → yield to frontend ──────────────────
    if is_multi:
        assert llm_overrides is not None
        reserved_messages = reserve_multi_model_message_ids(
            db_session=db_session,
            chat_session_id=chat_session.id,
            parent_message_id=user_message.id,
            model_display_names=model_display_names,
        )
        yield MultiModelMessageResponseIDInfo(
            user_message_id=user_message.id,
            responses=[
                ModelResponseSlot(message_id=m.id, model_name=name)
                for m, name in zip(reserved_messages, model_display_names)
            ],
        )
    else:
        assistant_response = reserve_message_id(
            db_session=db_session,
            chat_session_id=chat_session.id,
            parent_message=user_message.id,
            message_type=MessageType.ASSISTANT,
            model_display_name=model_display_names[0],
        )
        reserved_messages = [assistant_response]
        yield MessageResponseIDInfo(
            user_message_id=user_message.id,
            reserved_assistant_message_id=assistant_response.id,
        )

    # Convert the chat history into a simple format that is free of any DB objects
    # and is easy to parse for the agent loop.
    has_file_reader_tool = any(
        tool.in_code_tool_id == FILE_READER_TOOL_ID for tool in persona.tools
    )

    chat_history_result = convert_chat_history(
        chat_history=chat_history,
        files=files,
        context_image_files=extracted_context_files.image_files,
        additional_context=additional_context or new_msg_req.additional_context,
        token_counter=token_counter,
        tool_id_to_name_map=tool_id_to_name_map,
    )
    simple_chat_history = chat_history_result.simple_messages

    # Metadata for every text file injected into the history. After context-window
    # truncation drops older messages, the LLM loop compares surviving file_id tags
    # against this map to discover "forgotten" files and provide their metadata to
    # FileReaderTool.
    all_injected_file_metadata: dict[str, FileToolMetadata] = (
        chat_history_result.all_injected_file_metadata if has_file_reader_tool else {}
    )

    # Merge in file metadata from messages dropped by summary truncation. These files
    # are no longer in simple_chat_history so they'd be invisible to the forgotten-file
    # mechanism — they'll always appear as "forgotten" since no surviving message carries
    # their file_id tag.
    if summarized_file_metadata:
        for fid, meta in summarized_file_metadata.items():
            all_injected_file_metadata.setdefault(fid, meta)

    if all_injected_file_metadata:
        logger.debug(
            "FileReader: file metadata for LLM: %s",
            [(fid, m.filename) for fid, m in all_injected_file_metadata.items()],
        )

    if summary_message is not None:
        summary_simple = ChatMessageSimple(
            message=summary_message.message,
            token_count=summary_message.token_count,
            message_type=MessageType.ASSISTANT,
        )
        simple_chat_history.insert(0, summary_simple)

    # ── Stop signal and processing status ────────────────────────────────────
    cache = get_cache_backend()
    reset_cancel_status(chat_session.id, cache)

    def check_is_connected() -> bool:
        return check_stop_signal(chat_session.id, cache)

    set_processing_status(
        chat_session_id=chat_session.id,
        cache=cache,
        value=True,
    )

    # Release any read transaction before the long-running LLM stream.
    # If commit fails here, reset the processing status before propagating —
    # otherwise the chat session appears stuck at "processing" permanently.
    try:
        db_session.commit()
    except Exception:
        set_processing_status(chat_session_id=chat_session.id, cache=cache, value=False)
        raise

    return ChatTurnSetup(
        new_msg_req=new_msg_req,
        chat_session=chat_session,
        persona=persona,
        user_message=user_message,
        user_identity=user_identity,
        llms=llms,
        model_display_names=model_display_names,
        simple_chat_history=simple_chat_history,
        extracted_context_files=extracted_context_files,
        reserved_messages=reserved_messages,
        reserved_token_count=reserved_token_count,
        search_params=search_params,
        all_injected_file_metadata=all_injected_file_metadata,
        available_files=available_files,
        tool_id_to_name_map=tool_id_to_name_map,
        forced_tool_id=forced_tool_id,
        files=files,
        chat_files_for_tools=chat_files_for_tools,
        persona_prompt=persona_prompt,
        user_memory_context=user_memory_context,
        skip_clarification=skip_clarification,
        check_is_connected=check_is_connected,
        cache=cache,
        bypass_acl=bypass_acl,
        slack_context=slack_context,
        custom_tool_additional_headers=custom_tool_additional_headers,
        mcp_headers=mcp_headers,
    )


# Sentinel placed on the merged queue when a model thread finishes.
_MODEL_DONE = object()

# How often the drain loop polls for user-initiated cancellation (stop button).
_CANCEL_POLL_INTERVAL_S: Final[float] = 0.05


def _run_models(
    setup: ChatTurnSetup,
    user: User,
    external_state_container: ChatStateContainer | None = None,
) -> AnswerStream:
    """Stream packets from one or more LLM loops running in parallel worker threads.

    Each model gets its own worker thread, ``Emitter``, and short-lived DB sessions
    opened on demand. Threads write packets to a shared unbounded queue as they are
    produced; the drain loop yields them in arrival order so the caller receives a
    single interleaved stream regardless of how many models are running.

    No DB connection is held across the LLM stream — completion + error handlers
    open their own short sessions when persistence is needed.

    Single-model (N=1) and multi-model (N>1) use the same execution path. Every
    packet is tagged with ``model_index`` by the model's Emitter — ``0`` for N=1,
    ``0``/``1``/``2`` for multi-model.

    Args:
        setup: Fully constructed turn context — LLMs, persona, history, tool config.
        user: Authenticated user making the request.
        external_state_container: Pre-constructed state container for the first model.
            Used by evals and the non-streaming API path so the caller can inspect
            accumulated state (tool calls, answer tokens, citations) after the stream
            is consumed. When ``None`` a fresh container is created automatically.

    Returns:
        Generator yielding ``Packet`` objects as they arrive from worker threads —
        answer tokens, tool output, citations — followed by a terminal ``Packet``
        containing ``OverallStop`` once all models complete (or one containing
        ``OverallStop(stop_reason="user_cancelled")`` if the connection drops).
    """
    n_models = len(setup.llms)

    merged_queue: queue.Queue[tuple[int, Packet | Exception | object]] = queue.Queue()

    state_containers: list[ChatStateContainer] = [
        (
            external_state_container
            if (external_state_container is not None and i == 0)
            else ChatStateContainer()
        )
        for i in range(n_models)
    ]
    model_succeeded: list[bool] = [False] * n_models
    # Set to True when a model raises an exception (distinct from "still running").
    # Used in the stop-button path to avoid calling completion for errored models.
    model_errored: list[bool] = [False] * n_models

    # Set when the drain loop exits early (HTTP disconnect / GeneratorExit).
    # Signals emitters to skip future puts so workers exit promptly.
    drain_done = threading.Event()

    def _run_model(model_idx: int) -> None:
        """Run one LLM loop inside a worker thread, writing packets to ``merged_queue``."""

        model_emitter = Emitter(
            model_idx=model_idx,
            merged_queue=merged_queue,
            drain_done=drain_done,
        )
        sc = state_containers[model_idx]
        model_llm = setup.llms[model_idx]

        try:
            # Each function opens short-lived DB sessions on demand.
            # Do NOT pass a long-lived session here — it would hold a
            # connection for the entire LLM loop (minutes), and cloud
            # infrastructure may drop idle connections.
            thread_tool_dict = construct_tools(
                persona=setup.persona,
                emitter=model_emitter,
                user=user,
                llm=model_llm,
                search_tool_config=SearchToolConfig(
                    user_selected_filters=setup.new_msg_req.internal_search_filters,
                    project_id_filter=setup.search_params.project_id_filter,
                    persona_id_filter=setup.search_params.persona_id_filter,
                    bypass_acl=setup.bypass_acl,
                    slack_context=setup.slack_context,
                    enable_slack_search=_should_enable_slack_search(
                        setup.persona, setup.new_msg_req.internal_search_filters
                    ),
                ),
                custom_tool_config=CustomToolConfig(
                    chat_session_id=setup.chat_session.id,
                    message_id=setup.user_message.id,
                    additional_headers=setup.custom_tool_additional_headers,
                    mcp_headers=setup.mcp_headers,
                ),
                file_reader_tool_config=FileReaderToolConfig(
                    user_file_ids=setup.available_files.user_file_ids,
                    chat_file_ids=setup.available_files.chat_file_ids,
                ),
                allowed_tool_ids=setup.new_msg_req.allowed_tool_ids,
                search_usage_forcing_setting=setup.search_params.search_usage,
            )
            model_tools = [
                tool for tool_list in thread_tool_dict.values() for tool in tool_list
            ]

            if setup.forced_tool_id and setup.forced_tool_id not in {
                tool.id for tool in model_tools
            }:
                raise ValueError(
                    f"Forced tool {setup.forced_tool_id} not found in tools"
                )

            # Per-thread copy: run_llm_loop mutates simple_chat_history in-place.
            run_llm_loop(
                emitter=model_emitter,
                state_container=sc,
                simple_chat_history=list(setup.simple_chat_history),
                tools=model_tools,
                persona_prompt=setup.persona_prompt,
                context_files=setup.extracted_context_files,
                persona=setup.persona,
                user_memory_context=setup.user_memory_context,
                llm=model_llm,
                token_counter=get_llm_token_counter(model_llm),
                forced_tool_id=setup.forced_tool_id,
                user_identity=setup.user_identity,
                chat_session_id=str(setup.chat_session.id),
                chat_files=setup.chat_files_for_tools,
                include_citations=setup.new_msg_req.include_citations,
                all_injected_file_metadata=setup.all_injected_file_metadata,
                inject_memories_in_prompt=user.use_memories,
            )

            model_succeeded[model_idx] = True

        except Exception as e:
            model_errored[model_idx] = True
            merged_queue.put((model_idx, e))

        finally:
            merged_queue.put((model_idx, _MODEL_DONE))

    def _save_errored_message(model_idx: int, context: str) -> None:
        """Save an error message to a reserved ChatMessage that failed during execution."""
        try:
            with get_session_with_current_tenant() as save_db_session:
                msg = save_db_session.get(
                    ChatMessage, setup.reserved_messages[model_idx].id
                )
                if msg is not None:
                    error_text = (
                        "Error from %s: model encountered an error during generation."
                        % setup.model_display_names[model_idx]
                    )
                    msg.message = error_text
                    msg.error = error_text
                    save_db_session.commit()
        except Exception:
            logger.exception(
                "%s error save failed for model %d (%s)",
                context,
                model_idx,
                setup.model_display_names[model_idx],
            )

    # Each worker thread needs its own Context copy — a single Context object
    # cannot be entered concurrently by multiple threads (RuntimeError).
    executor = ThreadPoolExecutor(
        max_workers=n_models, thread_name_prefix="multi-model"
    )
    completion_persisted: bool = False
    try:
        for i in range(n_models):
            ctx = contextvars.copy_context()
            executor.submit(ctx.run, _run_model, i)

        # ── Main thread: merge and yield packets ────────────────────────────
        models_remaining = n_models
        while models_remaining > 0:
            try:
                model_idx, item = merged_queue.get(timeout=_CANCEL_POLL_INTERVAL_S)
            except queue.Empty:
                # Check for user-initiated cancellation every 50 ms.
                if not setup.check_is_connected():
                    # Save state for every model before exiting.
                    # - Succeeded models: full answer (is_connected=True).
                    # - Still-in-flight models: partial answer + "stopped by user".
                    # - Errored models: delete the orphaned reserved message; do NOT
                    #   save "stopped by user" for a model that actually threw an exception.
                    for i in range(n_models):
                        if model_errored[i]:
                            _save_errored_message(i, "stop-button")
                            continue
                        try:
                            succeeded = model_succeeded[i]

                            def _stop_button_is_connected(s: bool = succeeded) -> bool:
                                return s

                            llm_loop_completion_handle(
                                state_container=state_containers[i],
                                is_connected=_stop_button_is_connected,
                                assistant_message=setup.reserved_messages[i],
                                llm=setup.llms[i],
                                reserved_tokens=setup.reserved_token_count,
                            )
                        except Exception:
                            logger.exception(
                                "stop-button completion failed for model %d (%s)",
                                i,
                                setup.model_display_names[i],
                            )
                    yield Packet(
                        placement=Placement(turn_index=0),
                        obj=OverallStop(type="stop", stop_reason="user_cancelled"),
                    )
                    completion_persisted = True
                    return
                continue
            else:
                if item is _MODEL_DONE:
                    models_remaining -= 1
                elif isinstance(item, Exception):
                    # Yield a tagged error for this model but keep the other models running.
                    # Do NOT decrement models_remaining — _run_model's finally always posts
                    # _MODEL_DONE, which is the sole completion signal.
                    error_msg = str(item)
                    stack_trace = "".join(
                        traceback.format_exception(type(item), item, item.__traceback__)
                    )
                    model_llm = setup.llms[model_idx]
                    if model_llm.config.api_key and len(model_llm.config.api_key) > 2:
                        error_msg = error_msg.replace(
                            model_llm.config.api_key, "[REDACTED_API_KEY]"
                        )
                        stack_trace = stack_trace.replace(
                            model_llm.config.api_key, "[REDACTED_API_KEY]"
                        )
                    yield StreamingError(
                        error=error_msg,
                        stack_trace=stack_trace,
                        error_code="MODEL_ERROR",
                        is_retryable=True,
                        details={
                            "model": model_llm.config.model_name,
                            "provider": model_llm.config.model_provider,
                            "model_index": model_idx,
                        },
                    )
                elif isinstance(item, Packet):
                    # model_index already embedded by the model's Emitter in _run_model
                    yield item

        # ── Completion: save each successful model's response ───────────────
        # All model loops have completed (run_llm_loop returned) — no more writes
        # to state_containers. Each model's completion runs inside its own
        # short-lived DB session so no connection is held across the loop.
        for i in range(n_models):
            if not model_succeeded[i]:
                # Model errored — delete its orphaned reserved message.
                _save_errored_message(i, "normal")
                continue
            try:
                llm_loop_completion_handle(
                    state_container=state_containers[i],
                    is_connected=setup.check_is_connected,
                    assistant_message=setup.reserved_messages[i],
                    llm=setup.llms[i],
                    reserved_tokens=setup.reserved_token_count,
                )
            except Exception:
                logger.exception(
                    "normal completion failed for model %d (%s)",
                    i,
                    setup.model_display_names[i],
                )
        completion_persisted = True

    finally:
        if completion_persisted:
            # Normal exit or stop-button exit: completion already persisted.
            # Threads are done (normal path) or can finish in the background (stop-button).
            executor.shutdown(wait=False)
        else:
            # Early exit (GeneratorExit from raw HTTP disconnect, or unhandled
            # exception in the drain loop).
            # 1. Signal emitters to stop — future emit() calls return immediately,
            #    so workers exit their LLM loops promptly.
            drain_done.set()
            # 2. Wait for all workers to finish. Once drain_done is set the Emitter
            #    short-circuits, so workers should exit quickly.
            executor.shutdown(wait=True)
            # 3. All workers are done — complete from the main thread only.
            for i in range(n_models):
                if model_succeeded[i]:
                    try:
                        llm_loop_completion_handle(
                            state_container=state_containers[i],
                            # Model already finished — persist full response.
                            is_connected=lambda: True,
                            assistant_message=setup.reserved_messages[i],
                            llm=setup.llms[i],
                            reserved_tokens=setup.reserved_token_count,
                        )
                    except Exception:
                        logger.exception(
                            "disconnect completion failed for model %d (%s)",
                            i,
                            setup.model_display_names[i],
                        )
                elif model_errored[i]:
                    _save_errored_message(i, "disconnect")
            # 4. Drain buffered packets from memory — no consumer is running.
            while not merged_queue.empty():
                try:
                    merged_queue.get_nowait()
                except queue.Empty:
                    break


def _stream_chat_turn(
    new_msg_req: SendMessageRequest,
    user: User,
    llm_overrides: list[LLMOverride] | None = None,
    litellm_additional_headers: dict[str, str] | None = None,
    custom_tool_additional_headers: dict[str, str] | None = None,
    mcp_headers: dict[str, str] | None = None,
    bypass_acl: bool = False,
    additional_context: str | None = None,
    slack_context: SlackContext | None = None,
    external_state_container: ChatStateContainer | None = None,
) -> AnswerStream:
    """Private implementation for single-model and multi-model chat turn streaming.

    Builds the turn context via ``build_chat_turn`` inside a short-lived DB session,
    then streams packets from ``_run_models`` back to the caller without holding any
    DB connection. Handles setup errors, LLM errors, and cancellation uniformly,
    saving whatever partial state has been accumulated before re-raising or yielding
    a terminal error packet.

    Not called directly — use the public wrappers:
    - ``handle_stream_message_objects`` for single-model (N=1) requests.
    - ``handle_multi_model_stream`` for side-by-side multi-model comparison (N>1).

    Args:
        new_msg_req: The incoming chat request from the user.
        user: Authenticated user; may be anonymous for public personas.
        llm_overrides: ``None`` → single-model (persona default LLM).
            Non-empty list → multi-model (one LLM per override, 2–3 items).
        litellm_additional_headers: Extra headers forwarded to the LLM provider.
        custom_tool_additional_headers: Extra headers for custom tool HTTP calls.
        mcp_headers: Extra headers for MCP tool calls.
        bypass_acl: If ``True``, document ACL checks are skipped (used by Slack bot).
        additional_context: Extra context prepended to the LLM's chat history, not
            stored in the DB (used for Slack thread hydration).
        slack_context: Federated Slack search context passed through to the search tool.
        external_state_container: Optional pre-constructed state container. When
            provided, accumulated state (tool calls, citations, answer tokens) is
            written into it so the caller can inspect the result after streaming.

    Returns:
        Generator yielding ``Packet`` objects — answer tokens, tool output, citations —
        followed by a terminal ``Packet`` containing ``OverallStop``.
    """
    if new_msg_req.mock_llm_response is not None and not INTEGRATION_TESTS_MODE:
        raise ValueError(
            "mock_llm_response can only be used when INTEGRATION_TESTS_MODE=true"
        )

    mock_response_token: Token[str | None] | None = None
    setup: ChatTurnSetup | None = None

    try:
        with get_session_with_current_tenant() as setup_db_session:
            try:
                if (
                    not bypass_acl
                    and new_msg_req.internal_search_filters is not None
                    and new_msg_req.internal_search_filters.document_set is not None
                ):
                    # TODO: this doc-set access check is also enforced in SearchTool.run();
                    # this instance can be removed in a follow-up PR.
                    accessible_names = filter_document_set_names_by_user_access(
                        db_session=setup_db_session,
                        document_set_names=new_msg_req.internal_search_filters.document_set,
                        user=user,
                    )
                    unauthorized = sorted(
                        name
                        for name in new_msg_req.internal_search_filters.document_set
                        if name not in accessible_names
                    )
                    if unauthorized:
                        raise AppError(
                            AppErrorCode.INSUFFICIENT_PERMISSIONS,
                            "User does not have access to document sets: %s"
                            % unauthorized,
                        )

                setup = yield from build_chat_turn(
                    new_msg_req=new_msg_req,
                    user=user,
                    db_session=setup_db_session,
                    llm_overrides=llm_overrides,
                    litellm_additional_headers=litellm_additional_headers,
                    custom_tool_additional_headers=custom_tool_additional_headers,
                    mcp_headers=mcp_headers,
                    bypass_acl=bypass_acl,
                    slack_context=slack_context,
                    additional_context=additional_context,
                )
                setup_db_session.expunge_all()
            except Exception:
                setup_db_session.rollback()
                raise

        if new_msg_req.mock_llm_response is not None:
            mock_response_token = set_llm_mock_response(new_msg_req.mock_llm_response)

        assert setup is not None, (
            "build_chat_turn must complete before _run_models is called"
        )
        yield from _run_models(
            setup=setup,
            user=user,
            external_state_container=external_state_container,
        )

    except AppError as e:
        if e.error_code is not AppErrorCode.QUERY_REJECTED:
            log_app_error(e)
        yield StreamingError(
            error=e.detail,
            error_code=e.error_code.code,
            is_retryable=e.status_code >= 500,
        )
        return

    except ValueError as e:
        logger.exception("Failed to process chat message.")
        yield StreamingError(
            error=str(e),
            error_code="VALIDATION_ERROR",
            is_retryable=True,
        )
        return

    except EmptyLLMResponseError as e:
        stack_trace = traceback.format_exc()
        logger.warning(
            "LLM returned an empty response (provider=%s, model=%s, tool_choice=%s)",
            e.provider,
            e.model,
            e.tool_choice,
        )
        yield StreamingError(
            error=e.client_error_msg,
            stack_trace=stack_trace,
            error_code=e.error_code,
            is_retryable=e.is_retryable,
            details={
                "model": e.model,
                "provider": e.provider,
                "tool_choice": e.tool_choice.value,
            },
        )

    except Exception as e:
        logger.exception("Failed to process chat message due to %s", e)
        stack_trace = traceback.format_exc()

        llm = setup.llms[0] if setup else None
        if llm:
            client_error_msg, error_code, is_retryable = litellm_exception_to_error_msg(
                e, llm
            )
            if llm.config.api_key and len(llm.config.api_key) > 2:
                client_error_msg = client_error_msg.replace(
                    llm.config.api_key, "[REDACTED_API_KEY]"
                )
                stack_trace = stack_trace.replace(
                    llm.config.api_key, "[REDACTED_API_KEY]"
                )
            yield StreamingError(
                error=client_error_msg,
                stack_trace=stack_trace,
                error_code=error_code,
                is_retryable=is_retryable,
                details={
                    "model": llm.config.model_name,
                    "provider": llm.config.model_provider,
                },
            )
        else:
            yield StreamingError(
                error="Failed to initialize the chat. Please check your configuration and try again.",
                stack_trace=stack_trace,
                error_code="INIT_FAILED",
                is_retryable=True,
            )

    finally:
        if mock_response_token is not None:
            reset_llm_mock_response(mock_response_token)
        try:
            if setup is not None:
                set_processing_status(
                    chat_session_id=setup.chat_session.id,
                    cache=setup.cache,
                    value=False,
                )
        except Exception:
            logger.exception("Error in setting processing status")


def handle_stream_message_objects(
    new_msg_req: SendMessageRequest,
    user: User,
    litellm_additional_headers: dict[str, str] | None = None,
    custom_tool_additional_headers: dict[str, str] | None = None,
    mcp_headers: dict[str, str] | None = None,
    bypass_acl: bool = False,
    additional_context: str | None = None,
    slack_context: SlackContext | None = None,
    external_state_container: ChatStateContainer | None = None,
) -> AnswerStream:
    """Single-model streaming entrypoint. For multi-model comparison, use ``handle_multi_model_stream``."""
    yield from _stream_chat_turn(
        new_msg_req=new_msg_req,
        user=user,
        llm_overrides=None,
        litellm_additional_headers=litellm_additional_headers,
        custom_tool_additional_headers=custom_tool_additional_headers,
        mcp_headers=mcp_headers,
        bypass_acl=bypass_acl,
        additional_context=additional_context,
        slack_context=slack_context,
        external_state_container=external_state_container,
    )


def _build_model_display_name(override: LLMOverride | None, llm: LLM) -> str:
    """Build a human-readable display name for the LLM that will answer.

    Falls back to the configured ``llm.config.model_name`` when no override is
    set (default persona LLM) so the usage-metrics export always records the
    actual model used, not an "unknown" sentinel or empty string.
    """
    if override is not None:
        chosen = override.display_name or override.model_version
        if chosen:
            return chosen
    return llm.config.model_name


def handle_multi_model_stream(
    new_msg_req: SendMessageRequest,
    user: User,
    llm_overrides: list[LLMOverride],
    litellm_additional_headers: dict[str, str] | None = None,
    custom_tool_additional_headers: dict[str, str] | None = None,
    mcp_headers: dict[str, str] | None = None,
) -> AnswerStream:
    """Thin wrapper for side-by-side multi-model comparison (2–3 models).

    Validates the override list and delegates to ``_stream_chat_turn``,
    which handles both single-model and multi-model execution via the same path.

    Args:
        new_msg_req: The incoming chat request.
        user: Authenticated user making the request.
        llm_overrides: Exactly 2 or 3 ``LLMOverride`` objects — one per model to run.
        litellm_additional_headers: Extra headers forwarded to each LLM provider.
        custom_tool_additional_headers: Extra headers for custom tool HTTP calls.
        mcp_headers: Extra headers for MCP tool calls.

    Returns:
        Generator yielding interleaved ``Packet`` objects from all models, each tagged
        with ``model_index`` in its placement.
    """
    n_models = len(llm_overrides)
    if n_models < 2 or n_models > 3:
        yield StreamingError(
            error="Multi-model requires 2-3 overrides, got %d" % n_models,
            error_code="VALIDATION_ERROR",
            is_retryable=False,
        )
        return
    yield from _stream_chat_turn(
        new_msg_req=new_msg_req,
        user=user,
        llm_overrides=llm_overrides,
        litellm_additional_headers=litellm_additional_headers,
        custom_tool_additional_headers=custom_tool_additional_headers,
        mcp_headers=mcp_headers,
    )


def llm_loop_completion_handle(
    state_container: ChatStateContainer,
    is_connected: Callable[[], bool],
    assistant_message: ChatMessage,
    llm: LLM,
    reserved_tokens: int,
) -> None:
    # Snapshot all state under the container's lock before any DB write.
    # Worker threads may still be running (e.g. user-cancellation path), so
    # direct attribute access is not thread-safe — use the provided getters.
    answer_tokens = state_container.get_answer_tokens()
    reasoning_tokens = state_container.get_reasoning_tokens()
    citation_to_doc = state_container.get_citation_to_doc()
    tool_calls = state_container.get_tool_calls()
    is_clarification = state_container.get_is_clarification()
    all_search_docs = state_container.get_all_search_docs()
    emitted_citations = state_container.get_emitted_citations()
    pre_answer_processing_time = state_container.get_pre_answer_processing_time()

    completed_normally = is_connected()
    chat_session_id: UUID = assistant_message.chat_session_id
    assistant_message_id: int = assistant_message.id
    if completed_normally:
        if answer_tokens is None:
            raise RuntimeError(
                "LLM run completed normally but did not return an answer."
            )
        final_answer = answer_tokens
    else:
        logger.debug("Chat session %s stopped by user", chat_session_id)
        if answer_tokens:
            final_answer = (
                answer_tokens + " ... \n\nGeneration was stopped by the user."
            )
        else:
            final_answer = "The generation was stopped by the user."

    # Open a short-lived session here rather than holding one across the LLM
    # stream. Re-fetch the ChatMessage so save_chat_turn's mutations are applied
    # on top of current DB state — using merge() would silently overwrite any
    # concurrent writes (admin edits, retries) made between build_chat_turn's
    # commit and this completion handler.
    with get_session_with_current_tenant() as db_session:
        attached_message = db_session.get(ChatMessage, assistant_message_id)
        if attached_message is None:
            raise RuntimeError(
                "ChatMessage %d not found during completion" % assistant_message_id
            )

        save_chat_turn(
            message_text=final_answer,
            reasoning_tokens=reasoning_tokens,
            citation_to_doc=citation_to_doc,
            tool_calls=tool_calls,
            all_search_docs=all_search_docs,
            db_session=db_session,
            assistant_message=attached_message,
            is_clarification=is_clarification,
            emitted_citations=emitted_citations,
            pre_answer_processing_time=pre_answer_processing_time,
        )

        updated_chat_history = create_chat_history_chain(
            chat_session_id=chat_session_id,
            db_session=db_session,
        )
        total_tokens = calculate_total_history_tokens(updated_chat_history)

    compression_params = get_compression_params(
        max_input_tokens=llm.config.max_input_tokens,
        current_history_tokens=total_tokens,
        reserved_tokens=reserved_tokens,
    )
    if compression_params.should_compress:
        compress_chat_history(
            chat_history=updated_chat_history,
            llm=llm,
            compression_params=compression_params,
        )


_CITATION_LINK_START_PATTERN = re.compile(r"\s*\[\[\d+\]\]\(")


def _find_markdown_link_end(text: str, destination_start: int) -> int | None:
    depth = 0
    i = destination_start

    while i < len(text):
        curr = text[i]
        if curr == "\\":
            i += 2
            continue

        if curr == "(":
            depth += 1
        elif curr == ")":
            if depth == 0:
                return i
            depth -= 1

        i += 1

    return None


def remove_answer_citations(answer: str) -> str:
    stripped_parts: list[str] = []
    cursor = 0

    while match := _CITATION_LINK_START_PATTERN.search(answer, cursor):
        stripped_parts.append(answer[cursor : match.start()])
        link_end = _find_markdown_link_end(answer, match.end())
        if link_end is None:
            stripped_parts.append(answer[match.start() :])
            return "".join(stripped_parts)

        cursor = link_end + 1

    stripped_parts.append(answer[cursor:])
    return "".join(stripped_parts)


@log_function_time()
def gather_stream(
    packets: AnswerStream,
) -> ChatBasicResponse:
    answer: str | None = None
    citations: list[CitationInfo] = []
    error_msg: str | None = None
    message_id: int | None = None
    top_documents: list[SearchDoc] = []

    for packet in packets:
        if isinstance(packet, Packet):
            # Handle the different packet object types
            if isinstance(packet.obj, AgentResponseStart):
                # AgentResponseStart contains the final documents
                if packet.obj.final_documents:
                    top_documents = packet.obj.final_documents
            elif isinstance(packet.obj, AgentResponseDelta):
                # AgentResponseDelta contains incremental content updates
                if answer is None:
                    answer = ""
                if packet.obj.content:
                    answer += packet.obj.content
            elif isinstance(packet.obj, CitationInfo):
                # CitationInfo contains citation information
                citations.append(packet.obj)
        elif isinstance(packet, StreamingError):
            error_msg = packet.error
        elif isinstance(packet, MessageResponseIDInfo):
            message_id = packet.reserved_assistant_message_id

    if message_id is None:
        raise ValueError("Message ID is required")

    if answer is None:
        if error_msg is not None:
            answer = ""
        else:
            # This should never be the case as these non-streamed flows do not have a stop-generation signal
            raise RuntimeError("Answer was not generated")

    return ChatBasicResponse(
        answer=answer,
        answer_citationless=remove_answer_citations(answer),
        citation_info=citations,
        message_id=message_id,
        error_msg=error_msg,
        top_documents=top_documents,
    )


@log_function_time()
def gather_stream_full(
    packets: AnswerStream,
    state_container: ChatStateContainer,
) -> ChatFullResponse:
    """
    Aggregate streaming packets and state container into a complete ChatFullResponse.

    This function consumes all packets from the stream and combines them with
    the accumulated state from the ChatStateContainer to build a complete response
    including answer, reasoning, citations, and tool calls.

    Args:
        packets: The stream of packets from handle_stream_message_objects
        state_container: The state container that accumulates tool calls, reasoning, etc.

    Returns:
        ChatFullResponse with all available data
    """
    answer: str | None = None
    citations: list[CitationInfo] = []
    error_msg: str | None = None
    message_id: int | None = None
    top_documents: list[SearchDoc] = []
    chat_session_id: UUID | None = None

    for packet in packets:
        if isinstance(packet, Packet):
            if isinstance(packet.obj, AgentResponseStart):
                if packet.obj.final_documents:
                    top_documents = packet.obj.final_documents
            elif isinstance(packet.obj, AgentResponseDelta):
                if answer is None:
                    answer = ""
                if packet.obj.content:
                    answer += packet.obj.content
            elif isinstance(packet.obj, CitationInfo):
                citations.append(packet.obj)
        elif isinstance(packet, StreamingError):
            error_msg = packet.error
        elif isinstance(packet, MessageResponseIDInfo):
            message_id = packet.reserved_assistant_message_id
        elif isinstance(packet, CreateChatSessionID):
            chat_session_id = packet.chat_session_id

    if message_id is None:
        raise ValueError("Message ID is required")

    # Use state_container for complete answer (handles edge cases gracefully)
    final_answer = state_container.get_answer_tokens() or answer or ""

    # Get reasoning from state container (None when model doesn't produce reasoning)
    reasoning = state_container.get_reasoning_tokens()

    # Convert ToolCallInfo list to ToolCallResponse list
    tool_call_responses = [
        ToolCallResponse(
            tool_name=tc.tool_name,
            tool_arguments=tc.tool_call_arguments,
            tool_result=tc.tool_call_response,
            search_docs=tc.search_docs,
            generated_images=tc.generated_images,
            pre_reasoning=tc.reasoning_tokens,
        )
        for tc in state_container.get_tool_calls()
    ]

    return ChatFullResponse(
        answer=final_answer,
        answer_citationless=remove_answer_citations(final_answer),
        pre_answer_reasoning=reasoning,
        tool_calls=tool_call_responses,
        top_documents=top_documents,
        citation_info=citations,
        message_id=message_id,
        chat_session_id=chat_session_id,
        error_msg=error_msg,
    )
