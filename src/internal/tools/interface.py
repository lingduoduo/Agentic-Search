"""Abstract ChatTool interface for the chat loop."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class ChatTool(ABC):
    """Minimal abstract base for tool implementations used by the chat loop."""

    id: int | None = None
    name: str = ""

    @abstractmethod
    def tool_definition(self) -> dict[str, Any]: ...

    @abstractmethod
    def run(self, **kwargs: Any) -> Any: ...
