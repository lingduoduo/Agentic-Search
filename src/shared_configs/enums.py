"""Shared enums used across services."""

from enum import StrEnum


class WebContentProviderType(StrEnum):
    DIRECT = "direct"
    WEB_CRAWLER = "web_crawler"


class WebSearchProviderType(StrEnum):
    GOOGLE = "google"
    BING = "bing"
    SERPAPI = "serpapi"


class EmbeddingProvider(StrEnum):
    OPENAI = "openai"
    COHERE = "cohere"
    VOYAGE = "voyage"
    GOOGLE = "google"
    AZURE = "azure"
    LITELLM = "litellm"
    BEDROCK = "bedrock"


class EmbedTextType(StrEnum):
    QUERY = "query"
    PASSAGE = "passage"


class RerankerProvider(StrEnum):
    COHERE = "cohere"
    BEDROCK = "bedrock"
    LITELLM = "litellm"
