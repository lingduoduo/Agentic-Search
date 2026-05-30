"""Built-in tool name registry and stub classes."""

from __future__ import annotations

from dataclasses import dataclass, field

# Tool names that produce citable documents (trigger citation reminders)
CITEABLE_TOOLS_NAMES: set[str] = {"search", "web_search", "open_url"}

# Tool names that stop the loop after running (e.g. image generation)
STOPPING_TOOLS_NAMES: set[str] = {"image_generation"}

# Placeholder name-to-class map (populated when tool implementations are loaded)
TOOL_NAME_TO_CLASS: dict[str, type] = {}


class SearchTool:
    NAME = "search"
    id: int | None = None


class WebSearchTool:
    NAME = "web_search"
    id: int | None = None
    supports_site_filter: bool = False


class PythonTool:
    NAME = "python"
    id: int | None = None


class OpenURLTool:
    NAME = "open_url"
    id: int | None = None


class ImageGenerationTool:
    NAME = "image_generation"
    id: int | None = None


class MemoryTool:
    NAME = "memory"
    id: int | None = None


@dataclass
class _ParallelToolCallResults:
    tool_responses: list = field(default_factory=list)
    updated_citation_mapping: dict = field(default_factory=dict)


def run_tool_calls(tool_calls, tools, **kwargs) -> _ParallelToolCallResults:
    """Stub — real implementation requires full onyx tool runner."""
    return _ParallelToolCallResults()


def extract_url_snippet_map(docs) -> dict:
    """Return URL → snippet map from search docs."""
    return {doc.link: doc.blurb for doc in docs if getattr(doc, "link", None)}
