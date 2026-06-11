"""Abstract Tool interface."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from pydantic import BaseModel, Field


class Tool(ABC):
    """Minimal abstract base for all tool implementations."""

    id: int | None = None
    name: str = ""

    @abstractmethod
    def tool_definition(self) -> dict[str, Any]: ...

    @abstractmethod
    def run(self, **kwargs: Any) -> Any: ...


class ToolEntity(BaseModel):
    """API tool entity representing configuration details required for creating a LangChain-compatible tool"""

    id: str = Field(default="", description="ID of the API tool provider")
    name: str = Field(default="", description="Name of the API tool")
    url: str = Field(
        default="", description="URL used to send requests to the API tool"
    )
    method: str = Field(default="get", description="HTTP method used by the API tool")
    description: str = Field(default="", description="Description of the API tool")
    headers: list[dict] = Field(
        default_factory=list, description="HTTP headers used in the API request"
    )
    parameters: list[dict] = Field(
        default_factory=list, description="List of parameters required by the API tool"
    )
