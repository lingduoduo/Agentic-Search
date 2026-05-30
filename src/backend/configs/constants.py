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
    INGESTION_API = "ingestion_api"
    NOT_APPLICABLE = "not_applicable"


# Persona IDs
DEFAULT_PERSONA_ID: int = 0

# Misc constants used across the codebase
TMP_DRALPHA_PERSONA_NAME: str = "__deep_research__"


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
