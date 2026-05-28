"""Enums for search-context and answer-generation flows."""

from __future__ import annotations

from enum import StrEnum


class SearchType(StrEnum):
    RETRIEVAL = "retrieval"
    GOOGLE = "google"
    SERPAPI = "serpapi"


class QueryType(StrEnum):
    KEYWORD = "keyword"
    SEMANTIC = "semantic"
    HYBRID = "hybrid"


class AnswerStyle(StrEnum):
    CONCISE = "concise"
    DETAILED = "detailed"
    BULLETS = "bullets"


class AgentBehavior(StrEnum):
    DIRECT = "direct"
    SEARCH_FIRST = "search_first"
    RESEARCH = "research"
