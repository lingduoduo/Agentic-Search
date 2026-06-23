from __future__ import annotations

import logging
import os
import re

from src.context.query_transform import QueryTransformConfig

logger = logging.getLogger(__name__)

ROUTER_LABELS = [
    "decompose",
    "hyde",
    "step_back",
    "keywords",
    "construct_filters",
    "multi_query",
    "rewrite",
]

_QUESTION_WORDS = ("what", "why", "how", "when", "where", "who", "which")
_DATE_RE = re.compile(r"\b(19|20)\d{2}\b")
_RANGE_WORDS = ("after", "before", "since", "between", "from")
_NOISE_WORDS = ("uhh", "umm", "basically", "like", "kinda", "i think", "so ")


def _heuristic(query: str) -> QueryTransformConfig:
    q = query.lower()
    tokens = query.split()
    n = len(tokens)
    has_question = any(w in q for w in _QUESTION_WORDS)
    multi_clause = (" and " in q) or (";" in q) or (", " in q) or n > 18
    has_date = bool(_DATE_RE.search(q)) or any(w in q for w in _RANGE_WORDS)
    short_keyword = n <= 3
    noisy = any(w in q for w in _NOISE_WORDS) or "??" in query
    return QueryTransformConfig(
        decompose=multi_clause,
        hyde=has_question and not short_keyword,
        step_back=has_question and not multi_clause,
        keywords=short_keyword,
        construct_filters=has_date,
        multi_query=not short_keyword and not multi_clause,
        rewrite=(noisy or n > 12) and not short_keyword,
    )


class QueryRouter:
    """Predict the per-query transform set. Learned model with heuristic fallback."""

    def __init__(self, model_path: str | None = None) -> None:
        self._model = None
        if model_path and os.path.exists(model_path):
            try:
                import joblib

                self._model = joblib.load(model_path)
            except Exception as exc:  # pragma: no cover - infra dependent
                logger.warning("router model load failed, using heuristic: %s", exc)

    def predict(self, query: str) -> QueryTransformConfig:
        if self._model is None:
            return _heuristic(query)
        try:
            row = self._model.predict([query])[0]
            flags = {label: bool(row[i]) for i, label in enumerate(ROUTER_LABELS)}
            return QueryTransformConfig(**flags)
        except Exception as exc:
            logger.warning("router predict failed, using heuristic: %s", exc)
            return _heuristic(query)

    @classmethod
    def from_env(cls) -> "QueryRouter | None":
        if os.environ.get("QT_ROUTER", "").lower() not in ("1", "true", "yes"):
            return None
        return cls(model_path=os.environ.get("QT_ROUTER_MODEL_PATH"))
