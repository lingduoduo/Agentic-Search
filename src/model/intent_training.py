"""Utilities for generating and training intent-classifier data."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .intent_classifier import INTENT_LABELS, IntentPipeline, load_training_data
from src.internal.document_index.text import extract_keywords, tokenize_text

INTENTS = tuple(INTENT_LABELS)  # ordering used for sort key
STOPWORDS = frozenset(
    "a an and are as at be by for from how in into is it of on or that the to used using with".split()
)


@dataclass(frozen=True)
class IntentTrainingResult:
    """Summary returned after training and saving an intent classifier."""

    pipeline: IntentPipeline
    num_examples: int
    label_counts: dict[str, int]


def load_corpus(path: Path) -> list[dict[str, Any]]:
    """Load a JSONL corpus file into document dictionaries."""

    documents: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                documents.append(json.loads(line))
    return documents


def load_vocabulary_tokens(path: Path, *, limit: int = 64) -> list[str]:
    """Load the first *limit* vocabulary tokens from a vocabulary metadata file."""

    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle)

    token2idx = payload.get("vocabulary", {}).get("token2idx", {})
    ranked = sorted(token2idx.items(), key=lambda item: item[1])
    return [token for token, _ in ranked[:limit]]


def build_domain_terms(
    document: dict[str, Any], vocabulary_tokens: list[str]
) -> list[str]:
    """Pick representative terms from a document and global vocabulary."""

    keywords = extract_keywords(
        document,
        limit=6,
        text_fields=("title", "contents"),
        max_length=64,
    )
    title_tokens = tokenize_text(document.get("title", ""), max_length=6)

    seen: set[str] = set()
    domain_terms: list[str] = []
    for token in title_tokens + keywords + vocabulary_tokens:
        if len(token) < 3 or token in STOPWORDS or token in seen:
            continue
        seen.add(token)
        domain_terms.append(token)
        if len(domain_terms) >= 6:
            break
    return domain_terms


def build_examples_for_document(
    document: dict[str, Any],
    vocabulary_tokens: list[str],
) -> list[dict[str, Any]]:
    """Build intent-labeled examples for one corpus document."""

    title = document.get("title", "retrieval topic")
    contents = document.get("contents", "")
    terms = build_domain_terms(document, vocabulary_tokens)
    t1 = _pick_term(terms, 0, "retrieval")
    t2 = _pick_term(terms, 1, "search")

    examples = [
        {"text": f"find {title}", "label": "search"},
        {"text": f"look up {t1}", "label": "search"},
        {"text": f"{title}", "label": "search"},
        {"text": f"retrieve the {t2} documentation", "label": "search"},
        {"text": f"what is {title} and how is it used?", "label": "chat"},
        {"text": f"explain {t1} in {title}", "label": "chat"},
        {"text": f"compare {t1} and {t2}", "label": "chat"},
        {"text": f"summarize {title}", "label": "chat"},
        {"text": f"send an email about {title}", "label": "tool"},
        {"text": f"create a ticket for {t1}", "label": "tool"},
        {"text": f"schedule a meeting about {title}", "label": "tool"},
        {"text": f"open a pull request for {t2}", "label": "tool"},
    ]

    for example in examples:
        example["source_doc_id"] = document.get("id")
        example["source_title"] = title
        example["keywords"] = terms[:4]
        example["context_hint"] = contents[:120]
    return examples


def generate_intent_examples(
    *,
    corpus_path: Path,
    vocabulary_path: Path,
) -> list[dict[str, Any]]:
    """Generate intent-training examples from a local corpus and vocabulary."""

    documents = load_corpus(corpus_path)
    vocabulary_tokens = load_vocabulary_tokens(vocabulary_path)

    examples: list[dict[str, Any]] = []
    for document in documents:
        examples.extend(build_examples_for_document(document, vocabulary_tokens))

    examples.sort(
        key=lambda item: (
            INTENTS.index(item["label"]),
            item["source_doc_id"],
            item["text"],
        )
    )
    return examples


def write_intent_examples(examples: list[dict[str, Any]], output_path: Path) -> None:
    """Write intent examples as pretty JSON."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(examples, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def train_intent_classifier(
    *,
    examples_path: Path,
    output_path: Path,
    epochs: int = 10,
    lr: float = 1e-3,
    min_freq: int = 2,
    vocab_size: int = 5000,
    embedding_dim: int = 128,
    hidden_dim: int = 256,
) -> IntentTrainingResult:
    """Train an IntentPipeline from examples and save it to *output_path*."""

    data = load_training_data(str(examples_path))
    if not data:
        raise ValueError(f"No valid examples found in {examples_path}")

    label_counts: dict[str, int] = {}
    for _, label in data:
        label_counts[label] = label_counts.get(label, 0) + 1

    pipeline = IntentPipeline(
        vocab_size=vocab_size,
        embedding_dim=embedding_dim,
        hidden_dim=hidden_dim,
    )
    pipeline.train(data, epochs=epochs, lr=lr, min_freq=min_freq)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    pipeline.save(str(output_path), dataset_fingerprint="untracked")
    return IntentTrainingResult(
        pipeline=pipeline,
        num_examples=len(data),
        label_counts=label_counts,
    )


def _pick_term(terms: list[str], index: int, fallback: str) -> str:
    if not terms:
        return fallback
    return terms[index % len(terms)]
