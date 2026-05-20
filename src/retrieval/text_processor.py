"""Config-driven document preprocessing for retrieval-oriented text handling."""

from __future__ import annotations

import os
import re
from typing import Any, Iterable, Sequence

try:
    import yaml
except ImportError:  # pragma: no cover - optional dependency
    yaml = None


DEFAULT_CONFIG = {
    "dify": {
        "process_rules": {
            "pre_processing": [
                {"id": "remove_extra_spaces", "enabled": True},
                {"id": "remove_urls_emails", "enabled": True},
            ],
            "segmentation": {
                "separator": r"(?<=[.!?])\s+|\n+",
            },
        }
    }
}

_WHITESPACE_PATTERN = re.compile(r"\s+")


class TextProcessor:
    """Normalize and segment plain text or document-like payloads."""

    NON_CONTENT_FIELDS = {
        "id",
        "dataset_id",
        "document_id",
        "segment_id",
        "node_id",
        "document_enabled",
        "segment_enabled",
        "rank",
        "score",
        "bm25_score",
        "recall_score",
        "type",
        "partner",
    }

    def __init__(self, config_path: str | None = None) -> None:
        self.config_path = config_path or os.path.join(
            os.path.dirname(__file__), "config.yaml"
        )
        self.config = self._load_config()["dify"]["process_rules"]
        self._separator_pattern = re.compile(
            self.config.get("segmentation", {}).get("separator", r"(?<=[.!?])\s+|\n+")
        )
        self._url_email_pattern = re.compile(r"https?://\S+|\b\S+@\S+\.\S+\b")
        self._token_pattern = re.compile(r"\b\w+\b")

    def _load_config(self) -> dict[str, Any]:
        if os.path.exists(self.config_path) and yaml is not None:
            with open(self.config_path, "r", encoding="utf-8") as handle:
                loaded = yaml.safe_load(handle)
                if isinstance(loaded, dict):
                    return loaded
        return DEFAULT_CONFIG

    def _is_rule_enabled(self, rule_index: int, *, default: bool = False) -> bool:
        rules = self.config.get("pre_processing", [])
        if rule_index >= len(rules):
            return default
        return bool(rules[rule_index].get("enabled", default))

    def preprocess(self, text: str) -> str:
        """Apply configurable cleanup before tokenization or segmentation."""

        text = str(text or "")
        if self._is_rule_enabled(0, default=True):
            text = _WHITESPACE_PATTERN.sub(" ", text).strip()
        if self._is_rule_enabled(1, default=True):
            text = self._url_email_pattern.sub("", text)
            text = _WHITESPACE_PATTERN.sub(" ", text).strip()
        return text

    def segment(self, text: str) -> list[str]:
        """Split cleaned text into non-empty segments."""

        return [
            segment.strip()
            for segment in self._separator_pattern.split(text)
            if segment and segment.strip()
        ]

    def preprocess_json(self, json_data: dict[str, Any]) -> list[str]:
        """Extract and clean `rec_texts` from a JSON payload."""

        return self._extract_text_values(json_data)

    def _extract_text_values(
        self,
        input_data: Any,
        text_fields: Sequence[str] | None = None,
    ) -> list[str]:
        if isinstance(input_data, str):
            return [input_data]

        if not isinstance(input_data, dict):
            return []

        if text_fields:
            values: Iterable[Any] = (input_data.get(field, "") for field in text_fields)
        elif "rec_texts" in input_data:
            values = input_data.get("rec_texts", [])
        else:
            values = (
                value
                for key, value in input_data.items()
                if key not in self.NON_CONTENT_FIELDS
                and isinstance(value, (str, int, float))
                and not isinstance(value, bool)
            )

        extracted: list[str] = []
        for value in values:
            if value is None:
                continue
            cleaned = self.preprocess(str(value))
            if cleaned:
                extracted.append(cleaned)
        return extracted

    def process(self, input_data: Any) -> list[str]:
        """Unified entry point returning segmented chunks."""

        combined_text = self.normalize_document(input_data)
        return self.segment(combined_text) if combined_text else []

    def normalize_document(
        self,
        input_data: Any,
        text_fields: Sequence[str] | None = None,
    ) -> str:
        """Convert supported inputs into one cleaned document string."""

        return " ".join(
            self._extract_text_values(input_data, text_fields=text_fields)
        ).strip()

    def tokenize(
        self,
        input_data: Any,
        text_fields: Sequence[str] | None = None,
    ) -> list[str]:
        """Normalize and tokenize with one shared retrieval-friendly path."""

        normalized_text = self.normalize_document(
            input_data, text_fields=text_fields
        ).lower()
        return self.tokenize_normalized(normalized_text)

    def tokenize_normalized(self, normalized_text: str) -> list[str]:
        """Tokenize already-normalized text without re-running preprocessing."""

        return self._token_pattern.findall(normalized_text)
