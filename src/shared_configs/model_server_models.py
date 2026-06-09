"""Pydantic models for model-server request/response payloads."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel

from shared_configs.enums import EmbeddingProvider
from shared_configs.enums import EmbedTextType
from shared_configs.enums import RerankerProvider

# Type alias — a single embedding vector
Embedding = list[float]


class EmbedRequest(BaseModel):
    model_name: str | None = None
    texts: list[str]
    max_context_length: int = 512
    normalize_embeddings: bool = True
    api_key: str | None = None
    provider_type: EmbeddingProvider | None = None
    text_type: EmbedTextType = EmbedTextType.PASSAGE
    manual_query_prefix: str | None = None
    manual_passage_prefix: str | None = None
    api_url: str | None = None
    api_version: str | None = None
    deployment_name: str | None = None
    reduced_dimension: int | None = None


class EmbedResponse(BaseModel):
    embeddings: list[Embedding]
    model_config: dict[str, Any] = {}


class RerankRequest(BaseModel):
    query: str
    documents: list[str]
    model_name: str
    provider_type: RerankerProvider | None = None
    api_key: str | None = None
    api_url: str | None = None


class RerankResponse(BaseModel):
    scores: list[float]


class IntentRequest(BaseModel):
    query: str
    keyword_percent_threshold: float = 0.1
    semantic_percent_threshold: float = 0.4


class IntentResponse(BaseModel):
    is_keyword: bool
    keywords: list[str]
