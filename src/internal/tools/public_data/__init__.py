"""Keyless public data-source tools (reference, markets, geo).

Every tool here reaches a free public API that needs no credentials, so they
work on a clean checkout with no configuration. ``public_data_tools()`` is the
set ``knowledge_base.tool_knowledge_base()`` seeds.

Return contract: the citeable tools (Wikipedia, ArXiv, Wayback) answer with a
JSON array of ``{"title", "content", "url"}`` — the same shape the corpus
search returns, which is what lets their results become source cards. Every
other tool answers with a JSON object of facts. Failures are
``{"error": ...}``; no tool raises.
"""

from __future__ import annotations

from ..base import Tool
from .geo import (
    build_location_tool,
    build_nearby_places_tool,
    build_weather_tool,
)
from .knowledge import (
    build_arxiv_tool,
    build_wayback_tool,
    build_wikipedia_tool,
)
from .market import (
    build_crypto_price_tool,
    build_currency_tool,
    build_stock_quote_tool,
)


def public_data_tools() -> list[Tool]:
    """Build the nine public data-source tools, in catalog order."""
    return [
        build_wikipedia_tool(),
        build_arxiv_tool(),
        build_wayback_tool(),
        build_weather_tool(),
        build_stock_quote_tool(),
        build_crypto_price_tool(),
        build_currency_tool(),
        build_location_tool(),
        build_nearby_places_tool(),
    ]


__all__ = ["public_data_tools"]
