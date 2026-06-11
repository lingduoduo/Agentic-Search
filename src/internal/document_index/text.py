"""Document text normalization, tokenization, and vocabulary utilities."""

from __future__ import annotations

import os
import re
import unicodedata
from collections import Counter
from typing import Any, Iterable, Sequence

try:
    import yaml
except ImportError:  # pragma: no cover - optional dependency
    yaml = None

SOS_token = 0
EOS_token = 1
MAX_LENGTH = 20
_NON_WORD_PATTERN = re.compile(r"[^\w\s]")
_WHITESPACE_PATTERN = re.compile(r"\s+")
_URL_EMAIL_PATTERN = re.compile(r"https?://\S+|\b\S+@\S+\.\S+\b")
_TOKEN_PATTERN = re.compile(r"\b\w+\b")
_DEFAULT_SEPARATOR = r"(?<=[.!?])\s+|\n+"
DEFAULT_CONFIG = {
    "process_rules": {
        "remove_extra_spaces": True,
        "remove_urls_emails": True,
        "separator": _DEFAULT_SEPARATOR,
    }
}


class TextProcessor:
    """Normalize and segment strings or document-like dictionaries."""

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
        rules = self._load_rules(config_path)
        self.remove_extra_spaces = bool(rules.get("remove_extra_spaces", True))
        self.remove_urls_emails = bool(rules.get("remove_urls_emails", True))
        self._separator_pattern = re.compile(
            str(rules.get("separator", _DEFAULT_SEPARATOR))
        )

    @staticmethod
    def _load_rules(config_path: str | None) -> dict[str, Any]:
        if not config_path or not os.path.exists(config_path) or yaml is None:
            return DEFAULT_CONFIG["process_rules"]
        with open(config_path, encoding="utf-8") as handle:
            loaded = yaml.safe_load(handle)
        if not isinstance(loaded, dict):
            return DEFAULT_CONFIG["process_rules"]
        rules = loaded.get("process_rules", loaded.get("dify", {}).get("process_rules"))
        if not isinstance(rules, dict):
            return DEFAULT_CONFIG["process_rules"]

        preprocessing = {
            rule.get("id"): rule.get("enabled")
            for rule in rules.get("pre_processing", [])
            if isinstance(rule, dict)
        }
        segmentation = rules.get("segmentation", {})
        return {
            "remove_extra_spaces": rules.get(
                "remove_extra_spaces",
                preprocessing.get("remove_extra_spaces", True),
            ),
            "remove_urls_emails": rules.get(
                "remove_urls_emails",
                preprocessing.get("remove_urls_emails", True),
            ),
            "separator": rules.get(
                "separator",
                segmentation.get("separator", _DEFAULT_SEPARATOR),
            ),
        }

    def preprocess(self, text: str) -> str:
        """Apply configured cleanup to text."""

        text = str(text or "")
        if self.remove_extra_spaces:
            text = _WHITESPACE_PATTERN.sub(" ", text).strip()
        if self.remove_urls_emails:
            text = _URL_EMAIL_PATTERN.sub("", text)
            text = _WHITESPACE_PATTERN.sub(" ", text).strip()
        return text

    def segment(self, text: str) -> list[str]:
        """Split cleaned text into non-empty segments."""

        return [
            segment.strip()
            for segment in self._separator_pattern.split(text)
            if segment and segment.strip()
        ]

    def _extract_text_values(
        self,
        input_data: Any,
        text_fields: Sequence[str] | None = None,
    ) -> list[str]:
        if isinstance(input_data, str):
            return [self.preprocess(input_data)]
        if not isinstance(input_data, dict):
            return []

        values: Iterable[Any]
        if text_fields:
            values = (input_data.get(field, "") for field in text_fields)
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
        return [
            cleaned
            for value in values
            if value is not None and (cleaned := self.preprocess(str(value)))
        ]

    def normalize_document(
        self,
        input_data: Any,
        text_fields: Sequence[str] | None = None,
    ) -> str:
        """Convert supported input into one cleaned document string."""

        return " ".join(self._extract_text_values(input_data, text_fields)).strip()

    def preprocess_json(self, json_data: dict[str, Any]) -> list[str]:
        return self._extract_text_values(json_data)

    def process(self, input_data: Any) -> list[str]:
        normalized = self.normalize_document(input_data)
        return self.segment(normalized) if normalized else []

    def tokenize(
        self,
        input_data: Any,
        text_fields: Sequence[str] | None = None,
    ) -> list[str]:
        return _TOKEN_PATTERN.findall(
            self.normalize_document(input_data, text_fields).lower()
        )

    @staticmethod
    def tokenize_normalized(normalized_text: str) -> list[str]:
        return _TOKEN_PATTERN.findall(normalized_text)


_TEXT_PROCESSOR = TextProcessor()


class Vocabulary:
    """Track token-to-index mappings and token counts."""

    def __init__(self) -> None:
        self.token2idx: dict[str, int] = {}
        self.token2cnt: dict[str, int] = {}
        self.idx2token: dict[int, str] = {
            SOS_token: "SOS",
            EOS_token: "EOS",
        }
        self.num_token = 2

    def add_tokens(self, seq: str) -> None:
        """Split a sequence on whitespace and add each token."""

        self.add_token_sequence(seq.split())

    def add_token_sequence(self, tokens: Iterable[str]) -> None:
        """Add an existing token sequence without an extra join/split cycle."""

        for token in tokens:
            self.create_vocab(token)

    def create_vocab(self, token: str) -> None:
        """Insert a token if missing, otherwise update its count."""

        if token not in self.token2idx:
            self.token2idx[token] = self.num_token
            self.token2cnt[token] = 1
            self.idx2token[self.num_token] = token
            self.num_token += 1
            return

        self.token2cnt[token] += 1

    def build(
        self,
        sequences: Iterable[Sequence[str] | str],
        *,
        min_freq: int = 1,
    ) -> None:
        """Rebuild the vocabulary from token sequences or whitespace-delimited strings."""

        self.token2idx = {}
        self.token2cnt = {}
        self.idx2token = {
            SOS_token: "SOS",
            EOS_token: "EOS",
        }
        self.num_token = 2

        counts: Counter[str] = Counter()
        for sequence in sequences:
            if isinstance(sequence, str):
                tokens = sequence.split()
            else:
                tokens = [str(token) for token in sequence if str(token)]
            counts.update(tokens)

        for token, count in counts.items():
            if count < min_freq:
                continue
            self.token2idx[token] = self.num_token
            self.token2cnt[token] = count
            self.idx2token[self.num_token] = token
            self.num_token += 1

    def encode(
        self,
        tokens: Sequence[str] | str,
        *,
        max_length: int | None = None,
    ) -> list[int]:
        """Encode tokens using the current vocabulary.

        Unknown tokens map to 0 so the result can be used directly for padding.
        """

        if isinstance(tokens, str):
            normalized_tokens = tokens.split()
        else:
            normalized_tokens = [str(token) for token in tokens if str(token)]

        encoded = [self.token2idx.get(token, 0) for token in normalized_tokens]
        if max_length is not None:
            return encoded[:max_length]
        return encoded


def normalize_text(text: str, *, keep_non_ascii: bool = False) -> str:
    """Normalize a sentence for lightweight token-based indexing."""

    normalized = text.lower()
    if not keep_non_ascii:
        normalized = unicodedata.normalize("NFD", normalized)
        normalized = normalized.encode("ascii", errors="ignore").decode("utf-8")
    normalized = _NON_WORD_PATTERN.sub(" ", normalized)
    return _WHITESPACE_PATTERN.sub(" ", normalized).strip()


def tokenize_text(
    text: str,
    *,
    keep_non_ascii: bool = False,
    max_length: int | None = None,
) -> list[str]:
    """Normalize text and split it into tokens."""

    tokens = normalize_text(text, keep_non_ascii=keep_non_ascii).split()
    if max_length is not None:
        return tokens[:max_length]
    return tokens


def normalize_document(
    input_data: Any,
    *,
    text_fields: Sequence[str] | None = None,
    keep_non_ascii: bool = False,
) -> str:
    """Normalize a plain string or document-like payload into one text string."""

    if isinstance(input_data, str):
        return normalize_text(input_data, keep_non_ascii=keep_non_ascii)

    normalized = _TEXT_PROCESSOR.normalize_document(input_data, text_fields=text_fields)
    return normalize_text(normalized, keep_non_ascii=keep_non_ascii)


def tokenize_document(
    input_data: Any,
    *,
    text_fields: Sequence[str] | None = None,
    keep_non_ascii: bool = False,
    max_length: int | None = None,
) -> list[str]:
    """Tokenize a plain string or structured document payload."""

    tokens = normalize_document(
        input_data,
        text_fields=text_fields,
        keep_non_ascii=keep_non_ascii,
    ).split()
    if max_length is not None:
        return tokens[:max_length]
    return tokens


def build_vocabulary_from_sequences(
    sequences: Iterable[str],
    *,
    keep_non_ascii: bool = False,
    max_length: int = MAX_LENGTH,
) -> Vocabulary:
    """Create a vocabulary from an iterable of text sequences."""

    vocabulary = Vocabulary()
    for sequence in sequences:
        tokens = tokenize_text(
            sequence,
            keep_non_ascii=keep_non_ascii,
            max_length=max_length,
        )
        if tokens:
            vocabulary.add_token_sequence(tokens)
    return vocabulary


def extract_keywords(
    text: Any,
    *,
    limit: int = 10,
    text_fields: Sequence[str] | None = None,
    keep_non_ascii: bool = False,
    max_length: int = MAX_LENGTH,
) -> list[str]:
    """Return the most frequent normalized tokens from a document."""

    if limit < 1:
        return []

    if isinstance(text, str):
        tokens = tokenize_text(
            text,
            keep_non_ascii=keep_non_ascii,
            max_length=max_length,
        )
    else:
        tokens = tokenize_document(
            text,
            text_fields=text_fields,
            keep_non_ascii=keep_non_ascii,
            max_length=max_length,
        )
    counts = Counter(tokens)
    return [token for token, _ in counts.most_common(limit)]
