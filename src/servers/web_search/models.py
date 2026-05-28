"""Pydantic models for web-search provider administration."""

from __future__ import annotations

try:
    from enum import StrEnum
except ImportError:
    from enum import Enum

    class StrEnum(str, Enum):  # type: ignore[no-redef]
        pass


from typing import Any

from pydantic import BaseModel, Field


class WebSearchProviderType(StrEnum):
    RETRIEVAL = "retrieval"
    GOOGLE = "google"
    SERPAPI = "serpapi"


class WebContentProviderType(StrEnum):
    DIRECT = "direct"


class WebSearchProviderConfig(BaseModel):
    search_url: str | None = Field(
        default=None,
        description="POST /retrieve endpoint for retrieval-backed search.",
    )
    cse_id: str | None = Field(
        default=None,
        description="Google Custom Search engine ID.",
    )
    serp_engine: str = Field(default="google")
    page_size: int = Field(default=5, ge=1, le=10)
    timeout_seconds: int = Field(default=15, ge=1)
    extra: dict[str, Any] = Field(default_factory=dict)


class WebContentProviderConfig(BaseModel):
    timeout_seconds: int = Field(default=15, ge=1)
    max_length: int = Field(default=4000, ge=1)
    user_agent: str | None = None


class WebSearchProviderView(BaseModel):
    id: int
    name: str
    provider_type: WebSearchProviderType
    is_active: bool
    config: WebSearchProviderConfig = Field(default_factory=WebSearchProviderConfig)
    masked_api_key: str | None = Field(
        default=None,
        description="Masked API key for display. None means no key stored.",
    )


class WebSearchProviderUpsertRequest(BaseModel):
    id: int | None = Field(default=None, description="Existing provider ID to update.")
    name: str = Field(min_length=1)
    provider_type: WebSearchProviderType
    config: WebSearchProviderConfig = Field(default_factory=WebSearchProviderConfig)
    api_key: str | None = Field(
        default=None,
        description="API key for the provider. Required only for keyed providers.",
    )
    api_key_changed: bool = Field(
        default=False,
        description="Set true when replacing credentials for an existing provider.",
    )
    activate: bool = Field(
        default=False,
        description="If true, sets this provider as active after upsert.",
    )


class WebContentProviderView(BaseModel):
    id: int
    name: str
    provider_type: WebContentProviderType
    is_active: bool
    config: WebContentProviderConfig = Field(default_factory=WebContentProviderConfig)
    masked_api_key: str | None = None


class WebContentProviderUpsertRequest(BaseModel):
    id: int | None = None
    name: str = Field(min_length=1)
    provider_type: WebContentProviderType
    config: WebContentProviderConfig = Field(default_factory=WebContentProviderConfig)
    api_key: str | None = None
    api_key_changed: bool = False
    activate: bool = False


class WebSearchProviderTestRequest(BaseModel):
    provider_type: WebSearchProviderType
    api_key: str | None = Field(
        default=None,
        description="API key for testing. If omitted, use_stored_key may be true.",
    )
    use_stored_key: bool = Field(
        default=False,
        description="Use the active stored key for this provider type.",
    )
    config: WebSearchProviderConfig = Field(default_factory=WebSearchProviderConfig)
    query: str = Field(default="health check", min_length=1)
    live: bool = Field(
        default=False,
        description="When true, perform a live provider request instead of validation only.",
    )


class WebContentProviderTestRequest(BaseModel):
    provider_type: WebContentProviderType
    api_key: str | None = None
    use_stored_key: bool = False
    config: WebContentProviderConfig = Field(default_factory=WebContentProviderConfig)
    url: str = Field(default="https://example.com", min_length=1)
    live: bool = Field(
        default=False,
        description="When true, fetch the URL instead of validation only.",
    )
