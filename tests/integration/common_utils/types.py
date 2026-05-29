"""Local type definitions replacing onyx.* / ee.onyx.* imports.

All enums, constants, and Pydantic models that the integration tests need
are defined here so the test suite has no dependency on external packages.
"""

from __future__ import annotations

import os
from enum import Enum
from typing import Any, Generic, TypeVar

from pydantic import BaseModel

# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class UserRole(str, Enum):
    ADMIN = "admin"
    BASIC = "basic"
    CURATOR = "curator"
    GLOBAL_CURATOR = "global_curator"
    LIMITED = "limited"
    EXT_PERM_USER = "ext_perm_user"
    SLACK_USER = "slack_user"


class AccessType(str, Enum):
    PUBLIC = "public"
    PRIVATE = "private"
    SYNC = "sync"


class AccountType(str, Enum):
    STANDARD = "standard"
    EXTERNAL_PERMISSIONED = "external_permissioned"
    SLACK_USER = "slack_user"


class MessageType(str, Enum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"


class QAFeedbackType(str, Enum):
    LIKE = "like"
    DISLIKE = "dislike"
    MIXED = "mixed"


class SessionType(str, Enum):
    CHAT = "chat"
    SLACK = "slack"


# Alias used in reporting
FlowType = SessionType


class IndexingStatus(str, Enum):
    NOT_STARTED = "not_started"
    IN_PROGRESS = "in_progress"
    SUCCESS = "success"
    FAILED = "failed"
    CANCELED = "canceled"
    COMPLETED_WITH_ERRORS = "completed_with_errors"


class DocumentSyncStatus(str, Enum):
    SYNCED = "synced"
    NOT_SYNCED = "not_synced"


class ConnectorCredentialPairStatus(str, Enum):
    ACTIVE = "active"
    PAUSED = "paused"
    DELETING = "deleting"


class InputType(str, Enum):
    SLIM_RETRIEVAL = "slim_retrieval"
    POLL = "poll"
    LOAD_STATE = "load_state"
    EVENT = "event"
    PRUNE = "prune"


class DocumentSource(str, Enum):
    INGESTION_API = "ingestion_api"
    FILE = "file"
    WEB = "web"
    CONFLUENCE = "confluence"
    JIRA = "jira"
    GOOGLE_DRIVE = "google_drive"
    SLACK = "slack"
    GITHUB = "github"
    SHAREPOINT = "sharepoint"
    NOTION = "notion"
    ZENDESK = "zendesk"
    LINEAR = "linear"
    HUBSPOT = "hubspot"
    GMAIL = "gmail"


class SharingScope(str, Enum):
    PUBLIC = "public"
    PRIVATE = "private"


class IndexModelStatus(str, Enum):
    PAST = "past"
    PRESENT = "present"
    FUTURE = "future"


class UserFileStatus(str, Enum):
    ACTIVE = "active"
    DELETED = "deleted"


class ExternalAppType(str, Enum):
    GITHUB = "github"
    JIRA = "jira"
    CONFLUENCE = "confluence"
    SLACK = "slack"
    GOOGLE = "google"


class Permission(str, Enum):
    BASIC_ACCESS = "basic_access"
    FULL_ADMIN_PANEL_ACCESS = "full_admin_panel_access"


class FileOrigin(str, Enum):
    CHAT_UPLOAD = "chat_upload"
    CONNECTOR = "connector"
    GENERATED_REPORT = "generated_report"
    QUERY_HISTORY_CSV = "query_history_csv"


class StreamingType(str, Enum):
    TOOL_RESULT = "tool_result"
    ANSWER = "answer"


class LlmProviderNames(str, Enum):
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    AZURE = "azure"
    BEDROCK = "bedrock"


class ApplicationStatus(str, Enum):
    ACTIVE = "active"
    GRACE_PERIOD = "grace_period"
    GATED_ACCESS = "gated_access"
    SEAT_LIMIT_EXCEEDED = "seat_limit_exceeded"


class TaskStatus(str, Enum):
    PENDING = "pending"
    STARTED = "started"
    SUCCESS = "success"
    FAILURE = "failure"


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

ANONYMOUS_USER_EMAIL = "anonymous@anonymous.invalid"
ANONYMOUS_USER_UUID = "00000000-0000-0000-0000-000000000000"
FASTAPI_USERS_AUTH_COOKIE_NAME = "fastapiusersauth"
DEFAULT_PERSONA_ID = 0
PUBLIC_DOC_PAT = "__public__"

# Redis / Postgres config — read from env (mirrors onyx.configs.app_configs)
REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", "6379"))
REDIS_DB_NUMBER = int(os.getenv("REDIS_DB_NUMBER", "0"))
POSTGRES_HOST = os.getenv("POSTGRES_HOST", "localhost")
POSTGRES_PORT = os.getenv("POSTGRES_PORT", "5432")
POSTGRES_USER = os.getenv("POSTGRES_USER", "postgres")
POSTGRES_PASSWORD = os.getenv("POSTGRES_PASSWORD", "password")

# ---------------------------------------------------------------------------
# Generic paginated response
# ---------------------------------------------------------------------------

T = TypeVar("T")


class PaginatedReturn(BaseModel, Generic[T]):
    items: list[T]
    total_items: int


# ---------------------------------------------------------------------------
# Pydantic models replacing onyx.server.* models used by managers/tests
# ---------------------------------------------------------------------------


class IndexAttemptSnapshot(BaseModel):
    id: int
    status: IndexingStatus | None = None
    new_docs_indexed: int | None = None
    docs_removed_from_index: int | None = None
    total_docs_indexed: int | None = None
    error_msg: str | None = None
    full_exception_trace: str | None = None
    time_started: str | None = None
    time_updated: str


class ConnectorIndexingStatusLite(BaseModel):
    cc_pair_id: int
    connector_id: int
    credential_id: int
    connector_name: str
    last_status: IndexingStatus | None = None
    last_success: str | None = None


class CCPairFullInfo(BaseModel):
    id: int
    name: str | None = None
    connector_id: int
    credential_id: int
    status: ConnectorCredentialPairStatus = ConnectorCredentialPairStatus.ACTIVE
    last_index_attempt_status: IndexingStatus | None = None


class ConnectorCredentialPairIdentifier(BaseModel):
    connector_id: int
    credential_id: int


class ConnectorStatus(BaseModel):
    id: int
    name: str
    source: DocumentSource | str


class ConnectorUpdateRequest(BaseModel):
    connector_specific_config: dict[str, Any]
    input_type: InputType | None = None
    is_public: bool | None = None
    groups: list[int] = []
    name: str | None = None


class CredentialSnapshot(BaseModel):
    id: int
    name: str | None = None
    source: DocumentSource | str
    admin_public: bool
    time_created: str
    time_updated: str


class FileUploadResponse(BaseModel):
    file_paths: list[str]


class FileDescriptor(BaseModel):
    id: str
    name: str | None = None
    type: str | None = None


class APIKeyArgs(BaseModel):
    name: str | None = None
    role: UserRole = UserRole.BASIC


class ChatSessionCreationRequest(BaseModel):
    persona_id: int
    description: str | None = None


class SendMessageRequest(BaseModel):
    message: str
    chat_session_id: str
    parent_message_id: int | None = None
    prompt_id: int | None = None
    search_doc_ids: list[int] | None = None
    file_descriptors: list[FileDescriptor] = []
    regenerate: bool = False
    retrieval_options: dict | None = None
    query_override: str | None = None
    llm_override: dict | None = None


AUTO_PLACE_AFTER_LATEST_MESSAGE = -1


class GeneratedImage(BaseModel):
    url: str | None = None
    revised_prompt: str | None = None


class SavedSearchDoc(BaseModel):
    document_id: str
    chunk_ind: int
    semantic_identifier: str
    link: str | None = None
    score: float | None = None


class SearchDoc(BaseModel):
    document_id: str
    chunk_ind: int
    semantic_identifier: str
    link: str | None = None
    score: float | None = None
    match_highlights: list[str] = []


class LLMOverride(BaseModel):
    model_provider: str | None = None
    model_version: str | None = None
    temperature: float | None = None


class LLMProviderUpsertRequest(BaseModel):
    name: str
    provider: str
    api_key: str | None = None
    api_base: str | None = None
    api_version: str | None = None
    default_model_name: str
    fast_default_model_name: str | None = None
    is_public: bool = True
    groups: list[int] = []
    display_model_names: list[str] | None = None
    deployment_name: str | None = None


class DefaultModel(BaseModel):
    name: str
    provider: str


class ModelConfigurationUpsertRequest(BaseModel):
    default_llm_name: str | None = None
    fast_llm_name: str | None = None


class LLMProviderView(BaseModel):
    id: int
    name: str
    provider: str
    default_model_name: str
    is_public: bool
    groups: list[int] = []


class FullUserSnapshot(BaseModel):
    id: str
    email: str
    role: UserRole
    is_active: bool


class InvitedUserSnapshot(BaseModel):
    email: str


class UserInfo(BaseModel):
    id: str
    email: str
    role: UserRole
    is_active: bool


class AllUsersResponse(BaseModel):
    accepted: list[FullUserSnapshot]
    invited: list[InvitedUserSnapshot]
    total: int


class PersonaUpsertRequest(BaseModel):
    name: str
    description: str = ""
    system_prompt: str = ""
    task_prompt: str = ""
    document_set_ids: list[int] = []
    llm_model_provider_override: str | None = None
    llm_model_version_override: str | None = None
    is_public: bool = True
    groups: list[int] = []
    users: list[str] = []
    label_ids: list[int] = []
    icon_shape: int | None = None
    icon_color: str | None = None


class FullPersonaSnapshot(PersonaUpsertRequest):
    id: int
    default_persona: bool = False


class UserGroup(BaseModel):
    id: int
    name: str
    users: list[FullUserSnapshot] = []
    cc_pairs: list[Any] = []
    is_up_to_date: bool = True


class SendSearchQueryRequest(BaseModel):
    query: str
    filters: dict | None = None
    evaluation_type: str | None = None


class SearchFullResponse(BaseModel):
    top_documents: list[SearchDoc] = []
    llm_chunks_indices: list[int] = []
    filtered_count: int | None = None
    total_count: int | None = None


class UserProjectSnapshot(BaseModel):
    id: int
    name: str
    description: str | None = None


class UserFileSnapshot(BaseModel):
    id: int
    name: str
    status: UserFileStatus = UserFileStatus.ACTIVE


class CategorizedFilesSnapshot(BaseModel):
    projects: list[UserProjectSnapshot] = []
    files: list[UserFileSnapshot] = []


class UpsertExternalAppRequest(BaseModel):
    app_type: ExternalAppType
    name: str
    config: dict


class ExternalAppAdminResponse(BaseModel):
    id: int
    app_type: ExternalAppType
    name: str


class ExternalAppUserResponse(BaseModel):
    id: int
    app_type: ExternalAppType
    name: str
    has_credentials: bool


class UpsertUserCredentialsRequest(BaseModel):
    app_id: int
    credentials: dict


# ---------------------------------------------------------------------------
# Query-history models (re-exported from src for convenience)
# ---------------------------------------------------------------------------

from src.servers.query_history.models import ChatSessionMinimal  # noqa: E402
from src.servers.query_history.models import ChatSessionSnapshot  # noqa: E402

# ---------------------------------------------------------------------------
# Reporting models
# ---------------------------------------------------------------------------

from src.servers.reporting.models import UsageReportMetadata  # noqa: E402

# ---------------------------------------------------------------------------
# License models
# ---------------------------------------------------------------------------

from src.servers.license.models import LicenseSource  # noqa: E402
from src.servers.license.models import PlanType  # noqa: E402


class LicenseMetadata(BaseModel):
    """Minimal license metadata for seat-limit / deactivation tests."""

    tenant_id: str | None = None
    seats: int | None = None
    status: ApplicationStatus = ApplicationStatus.ACTIVE
    source: LicenseSource | None = None


__all__ = [
    "ANONYMOUS_USER_EMAIL",
    "ANONYMOUS_USER_UUID",
    "AUTO_PLACE_AFTER_LATEST_MESSAGE",
    "DEFAULT_PERSONA_ID",
    "FASTAPI_USERS_AUTH_COOKIE_NAME",
    "POSTGRES_HOST",
    "POSTGRES_PASSWORD",
    "POSTGRES_PORT",
    "POSTGRES_USER",
    "PUBLIC_DOC_PAT",
    "REDIS_DB_NUMBER",
    "REDIS_HOST",
    "REDIS_PORT",
    "AccessType",
    "AccountType",
    "AllUsersResponse",
    "APIKeyArgs",
    "ApplicationStatus",
    "CCPairFullInfo",
    "CategorizedFilesSnapshot",
    "ChatSessionCreationRequest",
    "ChatSessionMinimal",
    "ChatSessionSnapshot",
    "ConnectorCredentialPairIdentifier",
    "ConnectorCredentialPairStatus",
    "ConnectorIndexingStatusLite",
    "ConnectorStatus",
    "ConnectorUpdateRequest",
    "CredentialSnapshot",
    "DefaultModel",
    "DocumentSource",
    "DocumentSyncStatus",
    "ExternalAppAdminResponse",
    "ExternalAppType",
    "ExternalAppUserResponse",
    "FileDescriptor",
    "FileOrigin",
    "FileUploadResponse",
    "FlowType",
    "FullPersonaSnapshot",
    "FullUserSnapshot",
    "GeneratedImage",
    "IndexAttemptSnapshot",
    "IndexModelStatus",
    "IndexingStatus",
    "InputType",
    "InvitedUserSnapshot",
    "LicenseMetadata",
    "LicenseSource",
    "LLMOverride",
    "LLMProviderUpsertRequest",
    "LLMProviderView",
    "LlmProviderNames",
    "MessageType",
    "ModelConfigurationUpsertRequest",
    "PaginatedReturn",
    "Permission",
    "PersonaUpsertRequest",
    "PlanType",
    "QAFeedbackType",
    "SavedSearchDoc",
    "SearchDoc",
    "SearchFullResponse",
    "SendMessageRequest",
    "SendSearchQueryRequest",
    "SessionType",
    "SharingScope",
    "StreamingType",
    "TaskStatus",
    "UpsertExternalAppRequest",
    "UpsertUserCredentialsRequest",
    "UsageReportMetadata",
    "UserFileSnapshot",
    "UserFileStatus",
    "UserGroup",
    "UserInfo",
    "UserProjectSnapshot",
    "UserRole",
]
