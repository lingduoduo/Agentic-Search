"""Usage report generation.

Adapted from the sampled Onyx ee/onyx/server/reporting/usage_export_generation.py.
Replaces: SQLAlchemy queries, external FileStore, and Celery with
AgenticSearchStore queries and in-memory ZIP assembly.
"""

from __future__ import annotations

import csv
import io
import uuid
import zipfile
from datetime import datetime
from datetime import timezone

from src.db import AgenticSearchStore
from src.servers.reporting.usage_export_models import ChatMessageSkeleton
from src.servers.reporting.usage_export_models import FlowType
from src.servers.reporting.usage_export_models import UsageReportMetadata
from src.servers.reporting.usage_export_models import UserSkeleton


def _generate_chat_messages_csv(
    store: AgenticSearchStore,
    period_from: str | None,
    period_to: str | None,
) -> bytes:
    """Return a CSV of chat messages (one row per message) as bytes."""
    buf = io.StringIO()
    writer = csv.DictWriter(
        buf,
        fieldnames=[
            "message_id",
            "session_id",
            "user_id",
            "flow_type",
            "time_sent",
            "number_of_tokens",
        ],
    )
    writer.writeheader()

    PAGE_SIZE = 100
    page = 0
    while True:
        sessions = store.get_paginated_chat_sessions(
            page_num=page,
            page_size=PAGE_SIZE,
            start_time=period_from,
            end_time=period_to,
        )
        if not sessions:
            break
        for session in sessions:
            flow_type = (
                FlowType(session.metadata.get("flow_type", FlowType.CHAT))
                if session.metadata.get("flow_type") in (FlowType.CHAT, FlowType.SLACK)
                else FlowType.CHAT
            )
            for message in store.list_chat_messages(session.id):
                content = message.content or ""
                skeleton = ChatMessageSkeleton(
                    message_id=message.id,
                    session_id=session.id,
                    user_id=session.user_id,
                    flow_type=flow_type,
                    time_sent=message.created_at or "",
                    number_of_tokens=len(content.split()),
                )
                writer.writerow(
                    {
                        "message_id": skeleton.message_id,
                        "session_id": skeleton.session_id,
                        "user_id": skeleton.user_id,
                        "flow_type": skeleton.flow_type.value,
                        "time_sent": skeleton.time_sent,
                        "number_of_tokens": skeleton.number_of_tokens,
                    }
                )
        if len(sessions) < PAGE_SIZE:
            break
        page += 1

    return buf.getvalue().encode()


def _generate_users_csv(store: AgenticSearchStore) -> bytes:
    """Return a CSV of all users as bytes."""
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=["user_id", "email", "is_active"])
    writer.writeheader()
    for user in store.list_all_users():
        skeleton = UserSkeleton(
            user_id=user.id,
            email=user.email,
            is_active=True,
        )
        writer.writerow(
            {
                "user_id": skeleton.user_id,
                "email": skeleton.email,
                "is_active": skeleton.is_active,
            }
        )
    return buf.getvalue().encode()


def create_new_usage_report(
    store: AgenticSearchStore,
    requestor_id: str | None,
    period_from: str | None,
    period_to: str | None,
) -> UsageReportMetadata:
    """Generate a ZIP usage report and persist it in the store."""
    report_id = str(uuid.uuid4())
    date_prefix = datetime.now(tz=timezone.utc).strftime("%Y-%m-%d")
    report_name = f"{date_prefix}_{report_id}_usage_report.zip"

    chat_csv = _generate_chat_messages_csv(store, period_from, period_to)
    users_csv = _generate_users_csv(store)

    zip_buf = io.BytesIO()
    with zipfile.ZipFile(zip_buf, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("chat_messages.csv", chat_csv)
        zf.writestr("users.csv", users_csv)
    zip_bytes = zip_buf.getvalue()

    store.save_usage_report(
        report_name=report_name,
        data=zip_bytes,
        requestor_id=requestor_id,
        period_from=period_from,
        period_to=period_to,
    )

    return UsageReportMetadata(
        report_name=report_name,
        requestor_id=requestor_id,
        time_created=datetime.now(tz=timezone.utc).isoformat(),
        period_from=period_from,
        period_to=period_to,
    )


__all__ = ["create_new_usage_report"]
