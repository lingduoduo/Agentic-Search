from __future__ import annotations

from enum import Enum

from pydantic import BaseModel

from src.backend.db.models import ChatMessageRecord
from src.backend.db.models import ChatSessionRecord


class MessageType(str, Enum):
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"


class SessionType(str, Enum):
    CHAT = "chat"
    SLACK = "slack"


class QAFeedbackType(str, Enum):
    LIKE = "like"
    DISLIKE = "dislike"
    MIXED = "mixed"


class AbridgedSearchDoc(BaseModel):
    document_id: str
    semantic_identifier: str
    link: str | None


class MessageSnapshot(BaseModel):
    id: str
    message: str
    message_type: MessageType
    documents: list[AbridgedSearchDoc]
    feedback_type: QAFeedbackType | None
    feedback_text: str | None
    time_created: str

    @classmethod
    def from_record(cls, record: ChatMessageRecord) -> "MessageSnapshot":
        feedback = record.metadata.get("feedback")
        if feedback == "like":
            feedback_type: QAFeedbackType | None = QAFeedbackType.LIKE
        elif feedback == "dislike":
            feedback_type = QAFeedbackType.DISLIKE
        else:
            feedback_type = None

        role_map = {
            "user": MessageType.USER,
            "assistant": MessageType.ASSISTANT,
            "system": MessageType.SYSTEM,
        }

        documents = [
            AbridgedSearchDoc(
                document_id=doc.get("id", ""),
                semantic_identifier=doc.get("title", ""),
                link=doc.get("url"),
            )
            for doc in record.metadata.get("documents", [])
        ]
        return cls(
            id=record.id,
            message=record.content,
            message_type=role_map.get(record.role, MessageType.USER),
            documents=documents,
            feedback_type=feedback_type,
            feedback_text=record.metadata.get("feedback_text"),
            time_created=record.created_at or "",
        )


class ChatSessionMinimal(BaseModel):
    id: str
    user_id: str | None
    name: str | None
    first_user_message: str
    first_ai_message: str
    time_created: str
    feedback_type: QAFeedbackType | None
    flow_type: SessionType
    conversation_length: int

    @classmethod
    def from_records(
        cls,
        session: ChatSessionRecord,
        messages: list[ChatMessageRecord],
    ) -> "ChatSessionMinimal":
        first_user = next((m.content for m in messages if m.role == "user"), "")
        first_ai = next((m.content for m in messages if m.role == "assistant"), "")
        feedbacks = [
            m.metadata.get("feedback") for m in messages if m.metadata.get("feedback")
        ]
        if not feedbacks:
            feedback_type: QAFeedbackType | None = None
        elif all(f == "like" for f in feedbacks):
            feedback_type = QAFeedbackType.LIKE
        elif all(f == "dislike" for f in feedbacks):
            feedback_type = QAFeedbackType.DISLIKE
        else:
            feedback_type = QAFeedbackType.MIXED

        flow_type = (
            SessionType(session.metadata.get("flow_type", SessionType.CHAT))
            if session.metadata.get("flow_type")
            in (SessionType.CHAT, SessionType.SLACK)
            else SessionType.CHAT
        )

        non_system = [m for m in messages if m.role != "system"]
        return cls(
            id=session.id,
            user_id=session.user_id,
            name=session.title,
            first_user_message=first_user,
            first_ai_message=first_ai,
            time_created=session.created_at or "",
            feedback_type=feedback_type,
            flow_type=flow_type,
            conversation_length=len(non_system),
        )


class ChatSessionSnapshot(BaseModel):
    id: str
    user_id: str | None
    name: str | None
    messages: list[MessageSnapshot]
    time_created: str
    flow_type: SessionType

    @classmethod
    def from_records(
        cls,
        session: ChatSessionRecord,
        messages: list[ChatMessageRecord],
    ) -> "ChatSessionSnapshot":
        flow_type = (
            SessionType(session.metadata.get("flow_type", SessionType.CHAT))
            if session.metadata.get("flow_type")
            in (SessionType.CHAT, SessionType.SLACK)
            else SessionType.CHAT
        )

        return cls(
            id=session.id,
            user_id=session.user_id,
            name=session.title,
            messages=[
                MessageSnapshot.from_record(m) for m in messages if m.role != "system"
            ],
            time_created=session.created_at or "",
            flow_type=flow_type,
        )


class QuestionAnswerPairSnapshot(BaseModel):
    session_id: str
    message_pair_num: int
    user_message: str
    ai_response: str
    retrieved_documents: list[AbridgedSearchDoc]
    feedback_type: QAFeedbackType | None
    feedback_text: str | None
    user_id: str | None
    time_created: str
    flow_type: SessionType

    @classmethod
    def from_snapshot(
        cls, snapshot: ChatSessionSnapshot
    ) -> list["QuestionAnswerPairSnapshot"]:
        pairs: list[tuple[MessageSnapshot, MessageSnapshot]] = []
        for i in range(1, len(snapshot.messages), 2):
            pairs.append((snapshot.messages[i - 1], snapshot.messages[i]))

        return [
            cls(
                session_id=snapshot.id,
                message_pair_num=idx + 1,
                user_message=user_msg.message,
                ai_response=ai_msg.message,
                retrieved_documents=ai_msg.documents,
                feedback_type=ai_msg.feedback_type,
                feedback_text=ai_msg.feedback_text,
                user_id=snapshot.user_id,
                time_created=user_msg.time_created,
                flow_type=snapshot.flow_type,
            )
            for idx, (user_msg, ai_msg) in enumerate(pairs)
        ]

    def to_csv_row(self) -> dict[str, str | None]:
        return {
            "session_id": self.session_id,
            "message_pair_num": str(self.message_pair_num),
            "user_message": self.user_message,
            "ai_response": self.ai_response,
            "retrieved_documents": "|".join(
                doc.link or doc.semantic_identifier for doc in self.retrieved_documents
            ),
            "feedback_type": self.feedback_type.value if self.feedback_type else "",
            "feedback_text": self.feedback_text or "",
            "user_id": self.user_id,
            "time_created": self.time_created,
            "flow_type": self.flow_type.value,
        }


class PaginatedReturn(BaseModel):
    items: list[ChatSessionMinimal]
    total_items: int


__all__ = [
    "AbridgedSearchDoc",
    "ChatSessionMinimal",
    "ChatSessionSnapshot",
    "MessageSnapshot",
    "MessageType",
    "PaginatedReturn",
    "QAFeedbackType",
    "QuestionAnswerPairSnapshot",
    "SessionType",
]
