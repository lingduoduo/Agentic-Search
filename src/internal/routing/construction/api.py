"""API Request Construction — NL → allowlisted request params. No request is sent."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass

from src.context.models import ChatMessage

from ..route import RetrieverTarget, RouteDecision
from .base import ConstructedQuery

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ApiSpec:
    name: str
    base_url: str
    params: tuple[str, ...]


_PARAM_PROMPT = """Extract API request parameters from the question as a JSON object.
Allowed parameters: {params}. Include only those explicitly present; omit the rest.
Question: {query}
JSON:""".strip()


class ApiRequestConstructor:
    def __init__(self, llm: object, spec: ApiSpec) -> None:
        self._llm = llm
        self._spec = spec

    def _extract(self, query: str) -> dict:
        prompt = _PARAM_PROMPT.format(params=", ".join(self._spec.params), query=query)
        resp = self._llm.complete([ChatMessage(role="user", content=prompt)])
        text = (getattr(resp, "text", None) or str(resp)).strip()
        if text.startswith("```"):
            text = text.split("```")[1].removeprefix("json").strip()
        raw = json.loads(text)
        allowed = set(self._spec.params)
        return {k: v for k, v in raw.items() if k in allowed and v is not None}

    def construct(self, query: str, route: RouteDecision) -> ConstructedQuery:
        try:
            params = self._extract(query)
        except Exception as exc:
            logger.warning("API param extraction failed: %s", exc)
            params = {}
        return ConstructedQuery(
            RetrieverTarget.API,
            {"endpoint": self._spec.base_url, "params": params},
            query,
        )
