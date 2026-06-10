"""Shared application constants."""

from __future__ import annotations

from enum import Enum


class MessageType(str, Enum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL_CALL_RESPONSE = "tool_call_response"
    USER_REMINDER = "user_reminder"


class DocumentSource(str, Enum):
    FILE = "file"
    WEB = "web"
    SLACK = "slack"
    CONFLUENCE = "confluence"
    GITHUB = "github"
    GOOGLE_DRIVE = "google_drive"
    NOTION = "notion"
    JIRA = "jira"
    LINEAR = "linear"
    SALESFORCE = "salesforce"
    ZENDESK = "zendesk"
    GMAIL = "gmail"
    INGESTION_API = "ingestion_api"
    NOT_APPLICABLE = "not_applicable"


# Persona IDs
DEFAULT_PERSONA_ID: int = 0


# FileOrigin enum
class FileOrigin(str, Enum):
    CHAT_UPLOAD = "chat_upload"
    CONNECTOR = "connector"
    GENERATED = "generated"
    OTHER = "other"


# Milestone record types (telemetry)
class MilestoneRecordType(str, Enum):
    CHAT = "chat"
    SEARCH = "search"


# Document index and retrieval constants
# Represents a document accessible to all users (no access restriction).
PUBLIC_DOC_PAT = "PUBLIC"

# Separator used when joining multi-part content (title + body, etc.)
RETURN_SEPARATOR = "\n\n"

# Separator used when building index names
INDEX_SEPARATOR = "__"

# Field name for source type in OpenSearch schema
SOURCE_TYPE = "source_type"

# Key for the reindex flag in the key-value store
KV_REINDEX_KEY = "kv_reindex_key"

# Blurb size used for title prefix matching during content cleanup
BLURB_SIZE = 250

# Separator used when combining multiple sections into a large chunk
SECTION_SEPARATOR = "\n\n---\n\n"

# OpenSearch migration constants
OPENSEARCH_MIGRATION_ENABLED_KEY = "opensearch_migration_enabled"
OPENSEARCH_RETRIEVAL_ENABLED_KEY = "opensearch_retrieval_enabled"
