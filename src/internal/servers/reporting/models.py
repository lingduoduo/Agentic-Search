from __future__ import annotations

from pydantic import BaseModel

from src.internal.servers.query_history.models import SessionType as FlowType


class ChatMessageSkeleton(BaseModel):
    message_id: str
    session_id: str
    user_id: str | None
    flow_type: FlowType
    time_sent: str
    number_of_tokens: int


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
]
