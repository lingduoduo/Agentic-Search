"""File-store models used by the chat pipeline."""

from __future__ import annotations
from enum import Enum
from typing import Callable
from pydantic import BaseModel


class ChatFileType(str, Enum):
    IMAGE = "image"
    PLAIN_TEXT = "plain_text"
    PDF = "pdf"
    OTHER = "other"


class InMemoryChatFile(BaseModel):
    """A file held in memory for the duration of a chat turn."""

    file_id: str
    content: bytes = b""
    file_type: ChatFileType = ChatFileType.PLAIN_TEXT
    filename: str | None = None

    class Config:
        arbitrary_types_allowed = True


class FileDescriptor(BaseModel):
    file_id: str
    filename: str | None = None
    content_type: str = "text/plain"


def install_lazy_content_loader(
    inst: InMemoryChatFile, loader: Callable[[], bytes]
) -> None:
    """Attach a lazy loader so content is loaded only on first access."""
    object.__setattr__(inst, "_lazy_loader", loader)
