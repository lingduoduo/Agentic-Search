"""Dataclass records for the local Agentic Search metadata store."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

JsonObject = dict[str, Any]
PrincipalType = Literal["public", "user", "group"]
IndexAttemptStatus = Literal["not_started", "in_progress", "success", "failed"]


@dataclass(slots=True)
class ConnectorConfig:
    """Configuration and metadata for a document connector."""

    id: str
    name: str
    source: str
    config: JsonObject = field(default_factory=dict)
    enabled: bool = True
    metadata: JsonObject = field(default_factory=dict)
    created_at: str | None = None
    updated_at: str | None = None


@dataclass(slots=True)
class StoredDocument:
    """Document payload and source metadata managed by the store."""

    id: str
    title: str
    contents: str
    url: str | None = None
    connector_id: str | None = None
    metadata: JsonObject = field(default_factory=dict)
    created_at: str | None = None
    updated_at: str | None = None


@dataclass(slots=True)
class UserRecord:
    """User identity used for access control and chat ownership."""

    id: str
    email: str | None = None
    name: str | None = None
    metadata: JsonObject = field(default_factory=dict)
    created_at: str | None = None
    updated_at: str | None = None


@dataclass(slots=True)
class GroupRecord:
    """Named group and its current user membership."""

    id: str
    name: str
    user_ids: list[str] = field(default_factory=list)
    metadata: JsonObject = field(default_factory=dict)
    created_at: str | None = None
    updated_at: str | None = None


@dataclass(slots=True)
class DocumentPermission:
    """Read permission for a document granted to public, a user, or a group."""

    document_id: str
    principal_type: PrincipalType
    principal_id: str | None = None
    access: str = "read"
    created_at: str | None = None


@dataclass(slots=True)
class ChatSessionRecord:
    """Conversation session metadata."""

    id: str
    user_id: str | None = None
    title: str | None = None
    metadata: JsonObject = field(default_factory=dict)
    created_at: str | None = None
    updated_at: str | None = None


@dataclass(slots=True)
class ChatMessageRecord:
    """Single chat message in a conversation session."""

    id: str
    session_id: str
    role: str
    content: str
    metadata: JsonObject = field(default_factory=dict)
    created_at: str | None = None


@dataclass(slots=True)
class HookRecord:
    """Persisted configuration for an outbound webhook."""

    id: str
    name: str
    hook_point: str
    endpoint_url: str
    api_key: str | None = None
    fail_strategy: str = "soft"
    timeout_seconds: float = 5.0
    is_active: bool = True
    is_reachable: bool | None = None
    creator_id: str | None = None
    created_at: str | None = None
    updated_at: str | None = None


@dataclass(slots=True)
class IndexAttemptRecord:
    """Status and counters for a connector indexing attempt."""

    id: str
    connector_id: str | None = None
    status: IndexAttemptStatus = "not_started"
    total_documents: int = 0
    total_chunks: int = 0
    error: str | None = None
    metadata: JsonObject = field(default_factory=dict)
    created_at: str | None = None
    updated_at: str | None = None
