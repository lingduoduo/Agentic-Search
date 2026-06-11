"""Pydantic models for the documents API.

Enums that exist in Onyx's Postgres layer (AccessType, InputType, etc.) are
defined here as self-contained string enums so these models can be used
without a SQLAlchemy / Postgres dependency.  SQLAlchemy-based ``from_model()``
classmethods from the Onyx originals are omitted; models are plain data
containers only.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime
from enum import Enum
from typing import Any
from typing import Generic
from typing import TypeVar

from pydantic import BaseModel
from pydantic import ConfigDict
from pydantic import Field

from src.internal.configs.constants import DocumentSource


# ---------------------------------------------------------------------------
# Local enum definitions (mirror Onyx equivalents, no DB dependency)
# ---------------------------------------------------------------------------


class AccessType(str, Enum):
    PUBLIC = "public"
    PRIVATE = "private"
    SYNC = "sync"


class InputType(str, Enum):
    LOAD_STATE = "load_state"
    POLL = "poll"
    EVENT = "event"
    SLIM_RETRIEVAL = "slim_retrieval"


class ConnectorCredentialPairStatus(str, Enum):
    ACTIVE = "active"
    PAUSED = "paused"
    DELETING = "deleting"
    INITIAL_INDEXING = "initial_indexing"


class IndexingStatus(str, Enum):
    NOT_STARTED = "not_started"
    IN_PROGRESS = "in_progress"
    SUCCESS = "success"
    FAILED = "failed"
    COMPLETED_WITH_ERRORS = "completed_with_errors"


class PermissionSyncStatus(str, Enum):
    IN_PROGRESS = "in_progress"
    SUCCESS = "success"
    FAILED = "failed"


class ProcessingMode(str, Enum):
    REGULAR = "regular"
    SKIP_INDEXING = "skip_indexing"


class TaskStatus(str, Enum):
    PENDING = "pending"
    STARTED = "started"
    SUCCESS = "success"
    FAILURE = "failure"


class IndexingMode(str, Enum):
    UPDATE = "update"
    REINDEX = "reindex"


# ---------------------------------------------------------------------------
# Document / chunk info
# ---------------------------------------------------------------------------


class DocumentSyncStatus(BaseModel):
    doc_id: str
    last_synced: datetime | None
    last_modified: datetime | None


class DocumentInfo(BaseModel):
    num_chunks: int
    num_tokens: int


class ChunkInfo(BaseModel):
    content: str
    num_tokens: int


# ---------------------------------------------------------------------------
# Connector models
# ---------------------------------------------------------------------------


class ConnectorBase(BaseModel):
    name: str
    source: DocumentSource
    input_type: InputType
    connector_specific_config: dict[str, Any]
    refresh_freq: int | None = None
    prune_freq: int | None = None
    indexing_start: datetime | None = None


class ConnectorUpdateRequest(ConnectorBase):
    access_type: AccessType
    groups: list[int] = Field(default_factory=list)

    def to_connector_base(self) -> ConnectorBase:
        return ConnectorBase(**self.model_dump(exclude={"access_type", "groups"}))


class ConnectorSnapshot(ConnectorBase):
    id: int
    credential_ids: list[int]
    time_created: datetime
    time_updated: datetime
    source: DocumentSource


class IndexedSourcesResponse(BaseModel):
    model_config = ConfigDict(use_enum_values=True)
    sources: list[DocumentSource]


class RunConnectorRequest(BaseModel):
    connector_id: int
    credential_ids: list[int] | None = None
    from_beginning: bool = False


class ConnectorRequestSubmission(BaseModel):
    connector_name: str


class CCPropertyUpdateRequest(BaseModel):
    name: str
    value: str


# ---------------------------------------------------------------------------
# Credential models
# ---------------------------------------------------------------------------


class CredentialSwapRequest(BaseModel):
    new_credential_id: int
    connector_id: int
    access_type: AccessType


class CredentialDataUpdateRequest(BaseModel):
    name: str
    credential_json: dict[str, Any]


class CredentialBase(BaseModel):
    credential_json: dict[str, Any]
    admin_public: bool
    source: DocumentSource
    name: str | None = None
    curator_public: bool = False
    groups: list[int] = Field(default_factory=list)


class CredentialSnapshot(CredentialBase):
    id: int
    user_id: str | None
    user_email: str | None = None
    time_created: datetime
    time_updated: datetime


class ObjectCreationIdResponse(BaseModel):
    id: int
    credential: CredentialSnapshot | None = None


# ---------------------------------------------------------------------------
# Index attempt models
# ---------------------------------------------------------------------------


class IndexAttemptSnapshot(BaseModel):
    id: int
    status: IndexingStatus | None
    from_beginning: bool
    new_docs_indexed: int
    total_docs_indexed: int
    docs_removed_from_index: int
    error_msg: str | None
    error_count: int
    full_exception_trace: str | None
    time_started: str | None
    time_updated: str
    poll_range_start: datetime | None = None
    poll_range_end: datetime | None = None


class DeletionAttemptSnapshot(BaseModel):
    connector_id: int
    credential_id: int
    status: TaskStatus


class IndexAttemptStageMetricSnapshot(BaseModel):
    """Per-stage timing aggregate for a single IndexAttempt."""

    stage: str
    scope: str
    event_count: int
    total_duration_ms: int
    avg_duration_ms: float | None
    std_dev_duration_ms: float | None
    min_duration_ms: int | None
    max_duration_ms: int | None
    time_first_event: datetime | None
    time_last_event: datetime | None


class IndexAttemptStageMetricsResponse(BaseModel):
    index_attempt_id: int
    stages: list[IndexAttemptStageMetricSnapshot]


# ---------------------------------------------------------------------------
# Permission-sync attempt models
# ---------------------------------------------------------------------------


class DocPermissionSyncAttemptSnapshot(BaseModel):
    id: int
    status: PermissionSyncStatus
    error_message: str | None
    full_exception_trace: str | None
    total_docs_synced: int
    docs_with_permission_errors: int
    time_created: str
    time_started: str | None
    time_finished: str | None


class ExternalGroupSyncAttemptSnapshot(BaseModel):
    id: int
    status: PermissionSyncStatus
    error_message: str | None
    full_exception_trace: str | None
    total_users_processed: int
    total_groups_processed: int
    total_group_memberships_synced: int
    time_created: str
    time_started: str | None
    time_finished: str | None


# ---------------------------------------------------------------------------
# CC-pair models
# ---------------------------------------------------------------------------


class ConnectorCredentialPairIdentifier(BaseModel):
    connector_id: int
    credential_id: int


class ConnectorCredentialPairMetadata(BaseModel):
    name: str
    access_type: AccessType
    auto_sync_options: dict[str, Any] | None = None
    groups: list[int] = Field(default_factory=list)
    processing_mode: ProcessingMode = ProcessingMode.REGULAR


class CCStatusUpdateRequest(BaseModel):
    status: ConnectorCredentialPairStatus


class ConnectorCredentialPairDescriptor(BaseModel):
    id: int
    name: str
    connector: ConnectorSnapshot
    credential: CredentialSnapshot
    access_type: AccessType


class CCPairSummary(BaseModel):
    id: int
    name: str
    source: DocumentSource
    access_type: AccessType

    @classmethod
    def from_cc_pair_descriptor(
        cls, descriptor: ConnectorCredentialPairDescriptor
    ) -> "CCPairSummary":
        return cls(
            id=descriptor.id,
            name=descriptor.name,
            source=descriptor.connector.source,
            access_type=descriptor.access_type,
        )


class CCPairFullInfo(BaseModel):
    id: int
    name: str
    status: ConnectorCredentialPairStatus
    in_repeated_error_state: bool
    num_docs_indexed: int
    connector: ConnectorSnapshot
    credential: CredentialSnapshot
    number_of_index_attempts: int
    last_index_attempt_status: IndexingStatus | None
    latest_deletion_attempt: DeletionAttemptSnapshot | None
    access_type: AccessType
    is_editable_for_current_user: bool
    deletion_failure_message: str | None
    indexing: bool
    creator: str | None
    creator_email: str | None
    last_indexed: datetime | None
    last_pruned: datetime | None
    last_full_permission_sync: datetime | None
    overall_indexing_speed: float | None
    latest_checkpoint_description: str | None
    last_permission_sync_attempt_status: PermissionSyncStatus | None
    permission_syncing: bool
    last_permission_sync_attempt_finished: datetime | None
    last_permission_sync_attempt_error_message: str | None
    supports_targeted_reindex: bool


# ---------------------------------------------------------------------------
# Connector status / indexing status models
# ---------------------------------------------------------------------------


class ConnectorStatus(BaseModel):
    cc_pair_id: int
    name: str
    connector: ConnectorSnapshot
    credential: CredentialSnapshot
    access_type: AccessType
    groups: list[int]


class ConnectorIndexingStatus(ConnectorStatus):
    cc_pair_status: ConnectorCredentialPairStatus
    in_repeated_error_state: bool
    owner: str
    last_finished_status: IndexingStatus | None
    last_status: IndexingStatus | None
    last_success: datetime | None
    latest_index_attempt: IndexAttemptSnapshot | None
    docs_indexed: int
    in_progress: bool


class FailedConnectorIndexingStatus(BaseModel):
    cc_pair_id: int
    name: str
    error_msg: str | None
    is_deletable: bool
    connector_id: int
    credential_id: int


class DocsCountOperator(str, Enum):
    GREATER_THAN = ">"
    LESS_THAN = "<"
    EQUAL_TO = "="


class ConnectorIndexingStatusLite(BaseModel):
    cc_pair_id: int
    name: str
    source: DocumentSource
    access_type: AccessType
    cc_pair_status: ConnectorCredentialPairStatus
    in_progress: bool
    in_repeated_error_state: bool
    last_finished_status: IndexingStatus | None
    last_status: IndexingStatus | None
    last_success: datetime | None
    is_editable: bool
    docs_indexed: int
    latest_index_attempt_docs_indexed: int | None


class SourceSummary(BaseModel):
    total_connectors: int
    active_connectors: int
    public_connectors: int
    total_docs_indexed: int


class ConnectorIndexingStatusLiteResponse(BaseModel):
    source: DocumentSource
    summary: SourceSummary
    current_page: int
    total_pages: int
    indexing_statuses: Sequence[ConnectorIndexingStatusLite]


class IndexingStatusRequest(BaseModel):
    secondary_index: bool = False
    source: DocumentSource | None = None
    access_type_filters: list[AccessType] = Field(default_factory=list)
    last_status_filters: list[IndexingStatus] = Field(default_factory=list)
    docs_count_operator: DocsCountOperator | None = None
    docs_count_value: int | None = None
    name_filter: str | None = None
    source_to_page: dict[DocumentSource, int] = Field(default_factory=dict)
    get_all_connectors: bool = False


# ---------------------------------------------------------------------------
# Pagination helpers
# ---------------------------------------------------------------------------

PaginatedType = TypeVar("PaginatedType", bound=BaseModel)


class PaginatedReturn(BaseModel, Generic[PaginatedType]):
    items: list[PaginatedType]
    total_items: int


class CCPairSyncAttemptsResponse(BaseModel, Generic[PaginatedType]):
    """Paginated sync-attempt history.

    ``applicable=False`` means the source does not run this kind of sync;
    the frontend distinguishes this from ``applicable=True, items=[]``
    (no attempts yet).
    """

    applicable: bool
    items: list[PaginatedType]
    total_items: int


# ---------------------------------------------------------------------------
# File upload models
# ---------------------------------------------------------------------------


class FileUploadResponse(BaseModel):
    file_paths: list[str]
    file_names: list[str]
    zip_metadata_file_id: str | None


class ConnectorFileInfo(BaseModel):
    file_id: str
    file_name: str
    file_size: int | None = None
    upload_date: str | None = None


class ConnectorFilesResponse(BaseModel):
    files: list[ConnectorFileInfo]


# ---------------------------------------------------------------------------
# Auth / OAuth models
# ---------------------------------------------------------------------------


class AuthStatus(BaseModel):
    authenticated: bool


class AuthUrl(BaseModel):
    auth_url: str


class GmailCallback(BaseModel):
    state: str
    code: str


class GDriveCallback(BaseModel):
    state: str
    code: str


# ---------------------------------------------------------------------------
# Google credential models
# ---------------------------------------------------------------------------


class GoogleAppWebCredentials(BaseModel):
    client_id: str
    project_id: str
    auth_uri: str
    token_uri: str
    auth_provider_x509_cert_url: str
    client_secret: str
    redirect_uris: list[str]
    javascript_origins: list[str]


class GoogleAppCredentials(BaseModel):
    web: GoogleAppWebCredentials


class GoogleServiceAccountKey(BaseModel):
    type: str
    project_id: str
    private_key_id: str
    private_key: str
    client_email: str
    client_id: str
    auth_uri: str
    token_uri: str
    auth_provider_x509_cert_url: str
    client_x509_cert_url: str
    universe_domain: str


class GoogleServiceAccountCredentialRequest(BaseModel):
    google_primary_admin: str | None = None


# ---------------------------------------------------------------------------
# Misc
# ---------------------------------------------------------------------------


class CeleryTaskStatus(BaseModel):
    id: str
    name: str
    status: TaskStatus
    start_time: datetime | None
    register_time: datetime | None


class StatusResponse(BaseModel):
    success: bool
    message: str
    data: Any = None
