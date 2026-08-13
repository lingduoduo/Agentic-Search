"""Parity between our WordPiece and HuggingFace's, over the real corpora.

Skipped where transformers is unavailable, which includes the unit-test CI job.
The tokenizer's own behavior is covered dependency-free in test_wordpiece.py.
"""

import json
from pathlib import Path

import pytest

from src.internal.document_index.text import normalize_text
from src.model.intent_pretrained import DEFAULT_MODEL, load_pretrained_bundle
from src.model.wordpiece import UNK_ID

DATA = Path(__file__).resolve().parents[2] / "data"


def _bundle():
    if not (DATA / "intent_pretrained" / "vocab.txt").exists():
        pytest.skip(
            "run `python -m src.model.intent_training embeddings "
            f"--output {DATA / 'intent_pretrained'}`"
        )
    return load_pretrained_bundle(DATA / "intent_pretrained")


def _corpus_texts() -> list[str]:
    texts = [
        record["text"]
        for record in json.loads(
            (DATA / "intent_eval_queries.json").read_text(encoding="utf-8")
        )
    ]
    texts += [
        record["text"]
        for record in json.loads(
            (DATA / "intent_out_of_scope.json").read_text(encoding="utf-8")
        )
    ]
    texts += [
        record["text"]
        for record in json.loads(
            (DATA / "intent_examples.json").read_text(encoding="utf-8")
        )
    ]
    return texts


def test_our_wordpiece_matches_huggingface_on_normalized_text():
    transformers = pytest.importorskip("transformers")
    bundle = _bundle()
    reference = transformers.AutoTokenizer.from_pretrained(DEFAULT_MODEL)

    for text in _corpus_texts():
        normalized = normalize_text(text)
        expected = reference(normalized, add_special_tokens=False)["input_ids"]
        assert bundle.vocabulary.encode(text) == expected, text


def test_every_evaluation_query_and_probe_decomposes_without_unknowns():
    """The direct refutation of the 47%-unread measurement from #509."""
    bundle = _bundle()

    for text in _corpus_texts():
        ids = bundle.vocabulary.encode(text)
        assert ids, text
        assert UNK_ID not in ids, text
