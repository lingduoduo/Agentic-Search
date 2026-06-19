from __future__ import annotations

import logging
import os
import re

from src.context.models import ChatMessage

logger = logging.getLogger(__name__)

_PROMPT = (
    "Generate {n} alternative phrasings of the user's search query. "
    "Keep the meaning identical but vary the wording. "
    "Return each on its own line, no numbering needed.\n\nQuery: {query}"
)

_STRIP = re.compile(r"^\s*(?:\d+[.)]|[-*])\s*")


def _text(resp: object) -> str:
    if isinstance(resp, str):
        return resp
    return getattr(resp, "text", None) or getattr(resp, "content", None) or str(resp)


class MultiQueryGenerator:
    def __init__(self, llm, *, n: int = 3) -> None:
        self._llm = llm
        self._n = n

    def generate(self, query: str) -> list[str]:
        try:
            raw = _text(
                self._llm.complete(
                    [
                        ChatMessage(
                            role="user", content=_PROMPT.format(n=self._n, query=query)
                        )
                    ]
                )
            )
        except Exception as exc:
            logger.warning("multi-query generation failed: %s", exc)
            return []
        out: list[str] = []
        seen = {query.lower()}
        for line in raw.splitlines():
            cleaned = _STRIP.sub("", line).strip()
            if cleaned and cleaned.lower() not in seen:
                seen.add(cleaned.lower())
                out.append(cleaned)
        return out[: self._n]

    @classmethod
    def from_env(cls, llm) -> "MultiQueryGenerator | None":
        if os.environ.get("QT_MULTI_QUERY", "").lower() not in ("1", "true", "yes"):
            return None
        return cls(llm, n=int(os.environ.get("QT_MULTI_QUERY_N", "3")))
