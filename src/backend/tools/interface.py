"""Abstract Tool interface."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class Tool(ABC):
    """Minimal abstract base for all tool implementations."""

    id: int | None = None
    name: str = ""

    @abstractmethod
    def tool_definition(self) -> dict[str, Any]: ...

    @abstractmethod
    def run(self, **kwargs: Any) -> Any: ...
