"""Generate intent-classification training examples from the local corpus."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from src.agent_loop.intent_classifier import INTENT_LABELS
from src.search.vocabulary import extract_keywords, tokenize_text

INTENTS = tuple(INTENT_LABELS)  # ordering used for sort key
_STOPWORDS = frozenset(
    "a an and are as at be by for from how in into is it of on or that the to used using with".split()
)


def _load_corpus(path: Path) -> list[dict[str, Any]]:
    documents: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            documents.append(json.loads(line))
    return documents


def _load_vocabulary_tokens(path: Path, *, limit: int = 64) -> list[str]:
    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle)

    token2idx = payload.get("vocabulary", {}).get("token2idx", {})
    ranked = sorted(token2idx.items(), key=lambda item: item[1])
    return [token for token, _ in ranked[:limit]]


def _build_domain_terms(document: dict[str, Any], vocabulary_tokens: list[str]) -> list[str]:
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
        if len(token) < 3 or token in _STOPWORDS or token in seen:
            continue
        seen.add(token)
        domain_terms.append(token)
        if len(domain_terms) >= 6:
            break
    return domain_terms


def _pick_term(terms: list[str], index: int, fallback: str) -> str:
    if not terms:
        return fallback
    return terms[index % len(terms)]


def _build_examples_for_document(
    document: dict[str, Any],
    vocabulary_tokens: list[str],
) -> list[dict[str, Any]]:
    title = document.get("title", "retrieval topic")
    contents = document.get("contents", "")
    terms = _build_domain_terms(document, vocabulary_tokens)
    t1 = _pick_term(terms, 0, "retrieval")
    t2 = _pick_term(terms, 1, "search")
    t3 = _pick_term(terms, 2, "ranking")

    examples = [
        {
            "text": f"What is {title} and how is it used in retrieval systems?",
            "label": "qa",
        },
        {
            "text": f"How does {t1} affect {t2} performance in {title}?",
            "label": "qa",
        },
        {
            "text": f"Show me the section about {title}.",
            "label": "navigate",
        },
        {
            "text": f"Take me to the document that explains {t1} and {t2}.",
            "label": "navigate",
        },
        {
            "text": f"Recommend the best approach for {t1} and {t2} in a search pipeline.",
            "label": "recommendation",
        },
        {
            "text": f"Which should I use for {t1}: {title} or another option for {t3}?",
            "label": "recommendation",
        },
        {
            "text": f"I need to buy a tool or service for {t1}; what should I purchase for a {t2} stack?",
            "label": "purchase",
        },
        {
            "text": f"Which paid product should I choose for {title} if my budget is limited but I still need strong {t3}?",
            "label": "purchase",
        },
    ]

    for example in examples:
        example["source_doc_id"] = document.get("id")
        example["source_title"] = title
        example["keywords"] = terms[:4]
        example["context_hint"] = contents[:120]
    return examples


def generate_examples(
    *,
    corpus_path: Path,
    vocabulary_path: Path,
) -> list[dict[str, Any]]:
    documents = _load_corpus(corpus_path)
    vocabulary_tokens = _load_vocabulary_tokens(vocabulary_path)

    examples: list[dict[str, Any]] = []
    for document in documents:
        examples.extend(_build_examples_for_document(document, vocabulary_tokens))

    examples.sort(key=lambda item: (INTENTS.index(item["label"]), item["source_doc_id"], item["text"]))
    return examples


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate sample intent-training data.")
    parser.add_argument("--corpus", default="data/corpus.jsonl", help="Path to corpus JSONL")
    parser.add_argument(
        "--vocabulary",
        default="data/vocabulary_corpus.json",
        help="Path to vocabulary metadata JSON",
    )
    parser.add_argument(
        "--output",
        default="data/intent_examples.sample.json",
        help="Path to output JSON examples",
    )
    args = parser.parse_args()

    examples = generate_examples(
        corpus_path=Path(args.corpus),
        vocabulary_path=Path(args.vocabulary),
    )

    output_path = Path(args.output)
    output_path.write_text(json.dumps(examples, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {len(examples)} examples to {output_path}")


if __name__ == "__main__":
    main()
