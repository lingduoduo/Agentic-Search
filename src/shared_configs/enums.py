"""Shared enums used across services."""
from enum import StrEnum


class WebContentProviderType(StrEnum):
    DIRECT = "direct"
    WEB_CRAWLER = "web_crawler"


class WebSearchProviderType(StrEnum):
    GOOGLE = "google"
    BING = "bing"
    SERPAPI = "serpapi"
