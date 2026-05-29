"""SSE streaming packet types for the search API."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel
from pydantic import ConfigDict

from .models import SearchDocWithContent


class SearchQueriesPacket(BaseModel):
    model_config = ConfigDict(frozen=True)

    type: Literal["search_queries"] = "search_queries"
    all_executed_queries: list[str]


class SearchDocsPacket(BaseModel):
    model_config = ConfigDict(frozen=True)

    type: Literal["search_docs"] = "search_docs"
    search_docs: list[SearchDocWithContent]


class SearchErrorPacket(BaseModel):
    model_config = ConfigDict(frozen=True)

    type: Literal["search_error"] = "search_error"
    error: str


class LLMSelectedDocsPacket(BaseModel):
    model_config = ConfigDict(frozen=True)

    type: Literal["llm_selected_docs"] = "llm_selected_docs"
    llm_selected_doc_ids: list[str] | None


__all__ = [
    "LLMSelectedDocsPacket",
    "SearchDocsPacket",
    "SearchErrorPacket",
    "SearchQueriesPacket",
]
