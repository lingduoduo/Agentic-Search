from __future__ import annotations

from enum import Enum

from pydantic import BaseModel


class FlowType(str, Enum):
    CHAT = "chat"
    SLACK = "slack"


class ChatMessageSkeleton(BaseModel):
    message_id: str
    session_id: str
    user_id: str | None
    flow_type: FlowType
    time_sent: str
    number_of_tokens: int


class UserSkeleton(BaseModel):
    user_id: str
    email: str | None
    is_active: bool


class UsageReportMetadata(BaseModel):
    report_name: str
    requestor_id: str | None
    time_created: str
    period_from: str | None
    period_to: str | None


__all__ = [
    "ChatMessageSkeleton",
    "FlowType",
    "UsageReportMetadata",
    "UserSkeleton",
]
