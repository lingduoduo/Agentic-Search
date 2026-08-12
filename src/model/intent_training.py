"""Utilities for generating and training intent-classifier data."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import sys
import tempfile
import time
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .intent_classifier import INTENT_LABELS, IntentPipeline, load_training_data
from .intent_data import (
    IntentDatasetSplit,
    IntentEvalQuery,
    IntentExample,
    load_intent_eval_queries,
    load_intent_examples,
    load_out_of_scope_probes,
    split_intent_examples,
)
from .intent_evaluation import (
    IntentPredictionRecord,
    PromotionCriteria,
    PromotionDecision,
    authoritative_routes_match,
    calibration_report,
    compare_for_promotion,
    compose_candidate_cascade,
    evaluate_intent_predictions,
    out_of_scope_abstention_rate,
    realistic_accuracy_report,
    select_confidence_threshold,
)
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


@dataclass(frozen=True)
class IntentTrainingConfig:
    """Inputs, hyperparameters, and promotion gates for one offline run."""

    examples_path: Path
    output_dir: Path
    baseline_path: Path
    out_of_scope_path: Path | None = None
    eval_queries_path: Path | None = None
    seed: int = 17
    # `train_batched` takes one full-batch step per epoch, so 10 epochs is 10
    # optimizer steps — measurably untrained (realistic accuracy 0.333 versus
    # 0.567 at 300). See docs/training-and-evaluation.md.
    epochs: int = 300
    lr: float = 1e-3
    # 1, not 2: dropping singletons was measured to cost more vocabulary than
    # the unknown-token signal it buys. On the committed dataset min_freq=2
    # left only 4 of 341 training rows containing an unknown word while
    # shrinking the vocabulary 147 -> 135, and realistic accuracy fell from
    # 0.600 to 0.500 with the out-of-scope margin going negative.
    min_freq: int = 1
    vocab_size: int = 5000
    embedding_dim: int = 128
    hidden_dim: int = 256
    train_fraction: float = 0.70
    validation_fraction: float = 0.15
    min_tool_precision: float = 0.95
    max_high_confidence_errors: int = 0
    require_macro_f1_non_decreasing: bool = True
    require_llm_fallback_reduction: bool = True
    require_latency_improvement: bool = True


@dataclass(frozen=True)
class IntentTrainingRun:
    """Paths and promotion result produced by an offline training run."""

    checkpoint_path: Path
    split_manifest_path: Path
    evaluation_report_path: Path
    selected_threshold: float
    promotion: PromotionDecision


# Neutral fillers: ordinary English that carries no intent by itself. Each of
# these slots is used by exactly two frames per label, so their tokens train
# under all three classes and stay class-neutral. A request built only from
# words like these pools toward the centre and scores low confidence, which is
# the only way a three-label model can decline anything.
_ROLES = ("the platform team", "my manager", "the on-call engineer", "the legal team")
_TIMES = ("last quarter", "this morning", "the last release", "yesterday")
_ARTIFACTS = ("the design doc", "the runbook", "the invoice", "the summary")
_OPENERS = ("quick one", "hey", "I'm not sure", "when you get a chance")
# Cue fillers: action verbs are genuine tool evidence and are deliberately
# unbalanced — they appear under `tool` only.
_ACTIONS = ("share", "send", "post", "file")

# (frame_id, template, label, tags). Slots: {title}, {t1}, {t2} carry document
# terms; {role}, {time}, {artifact}, {opener} are neutral; {action} is a cue.
# Every frame includes a document term so no two documents produce equal text.
# Frames tagged `ambiguous` are phrased so `_regex_route` defers on them: that
# region is the only one the learned model ever decides.
_FRAMES: tuple[tuple[str, str, str, tuple[str, ...]], ...] = (
    ("s-find", "find {title}", "search", ("direct",)),
    ("s-lookup", "look up {t1}", "search", ("paraphrase",)),
    ("s-bare", "{title}", "search", ("short",)),
    (
        "s-where-artifact",
        "where is {artifact} for {t1}",
        "search",
        ("ambiguous", "question"),
    ),
    (
        "s-we-need-time",
        "we need the {t1} numbers from {time}",
        "search",
        ("ambiguous", "statement"),
    ),
    (
        "s-opener-anyone",
        "{opener}, does anyone have {artifact} for {t2}",
        "search",
        ("ambiguous", "question"),
    ),
    (
        "s-role-sent",
        "the {t1} page {role} sent {time}",
        "search",
        ("ambiguous", "statement"),
    ),
    (
        "s-official-role",
        "{opener} the official {t1} reference {role} uses",
        "search",
        ("ambiguous", "statement"),
    ),
    (
        "s-multi",
        "I need background on {t2} and the latest benchmark table",
        "search",
        ("multi_intent",),
    ),
    ("c-what-is", "what is {title} and how is it used?", "chat", ("direct",)),
    ("c-explain", "explain {t1} in {title}", "chat", ("paraphrase",)),
    ("c-compare", "compare {t1} and {t2}", "chat", ("comparison",)),
    (
        "c-artifact-say",
        "I wonder what {artifact} says about {t1}",
        "chat",
        ("ambiguous", "statement"),
    ),
    (
        "c-what-changed",
        "not sure what changed about {t1} since {time}",
        "chat",
        ("ambiguous", "statement"),
    ),
    (
        "c-opener-understand",
        "{opener}, help me understand what {t1} is doing under the hood",
        "chat",
        ("ambiguous", "polite"),
    ),
    (
        "c-role-why",
        "curious why {role} uses {t1} instead of {t2} {time}",
        "chat",
        ("ambiguous", "statement"),
    ),
    (
        "c-opener-artifact",
        "{opener} is {artifact} still right about {t1} for {role}",
        "chat",
        ("ambiguous", "question"),
    ),
    ("t-email", "send an email about {title}", "tool", ("direct",)),
    ("t-ticket", "create a ticket for {t1}", "tool", ("direct",)),
    ("t-pr", "open a pull request for {t2}", "tool", ("direct",)),
    (
        "t-artifact-action",
        "{action} {artifact} for {t1}",
        "tool",
        ("ambiguous", "imperative"),
    ),
    (
        "t-schedule-time",
        "schedule the {t1} review for {time}",
        "tool",
        ("ambiguous", "imperative"),
    ),
    (
        "t-opener-push",
        "{opener}, please {action} the {t1} summary to the shared channel",
        "tool",
        ("ambiguous", "polite"),
    ),
    (
        "t-role-invite",
        "the {title} rollout needs a calendar invite for {role} before {time}",
        "tool",
        ("ambiguous", "statement"),
    ),
    (
        "t-opener-can-you",
        "{opener} can you {action} {artifact} to {role}",
        "tool",
        ("ambiguous", "question"),
    ),
    (
        "t-multi",
        "the {t1} contract needs a review summary and a sign-off note",
        "tool",
        ("multi_intent",),
    ),
)


_HARD_CASES: tuple[tuple[str, str, str, tuple[str, ...]], ...] = (
    (
        "hard-secure-email-chat",
        "Explain how to send an email securely",
        "chat",
        ("hard_negative",),
    ),
    (
        "hard-secure-email-tool",
        "Send the security summary to the project owner",
        "tool",
        ("paraphrase",),
    ),
    (
        "hard-duplicate-documents-chat",
        "Discuss approaches to finding duplicate documents",
        "chat",
        ("hard_negative",),
    ),
    (
        "hard-duplicate-documents-search",
        "Locate the duplicate-document policy",
        "search",
        ("paraphrase",),
    ),
)


def load_corpus(path: Path) -> list[dict[str, Any]]:
    """Load a JSONL corpus file into document dictionaries."""

    documents: list[dict[str, Any]] = []
    document_identities: set[str] = set()
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if line:
                document = json.loads(line)
                if not isinstance(document, Mapping):
                    raise ValueError(
                        f"Corpus record at line {line_number} must be an object"
                    )
                _validate_optional_corpus_field(document, "id", (str, int), line_number)
                _validate_optional_corpus_field(document, "title", str, line_number)
                _validate_optional_corpus_field(document, "contents", str, line_number)
                identity = _corpus_document_identity(document)
                if identity in document_identities:
                    raise ValueError(
                        "Corpus contains duplicate document identity "
                        f"{identity!r} at line {line_number}"
                    )
                document_identities.add(identity)
                normalized_document = dict(document)
                normalized_document["_intent_document_id"] = identity
                documents.append(normalized_document)
    if not documents:
        raise ValueError(f"Corpus contains no documents: {path}")
    return documents


def load_vocabulary_tokens(path: Path, *, limit: int = 64) -> list[str]:
    """Load the first *limit* vocabulary tokens from a vocabulary metadata file."""

    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle)

    if not isinstance(payload, Mapping):
        raise ValueError("Vocabulary metadata must contain an object")
    vocabulary = payload.get("vocabulary", {})
    if not isinstance(vocabulary, Mapping):
        raise ValueError("Vocabulary metadata 'vocabulary' must be an object")
    token2idx = vocabulary.get("token2idx", {})
    if not isinstance(token2idx, Mapping) or any(
        not isinstance(token, str)
        or not token
        or isinstance(index, bool)
        or not isinstance(index, int)
        for token, index in token2idx.items()
    ):
        raise ValueError("Vocabulary metadata 'token2idx' must map tokens to integers")
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
    *,
    document_index: int = 0,
) -> list[dict[str, Any]]:
    """Build one intent-labeled example per frame for a corpus document."""

    title = document.get("title", "retrieval topic")
    contents = document.get("contents", "")
    terms = build_domain_terms(document, vocabulary_tokens)
    slots = {
        "title": title,
        "t1": _pick_term(terms, 0, "retrieval"),
        "t2": _pick_term(terms, 1, "search"),
        # Fillers rotate by document so each chatter word appears many times
        # across the dataset rather than once.
        "role": _ROLES[document_index % len(_ROLES)],
        "time": _TIMES[document_index % len(_TIMES)],
        "artifact": _ARTIFACTS[document_index % len(_ARTIFACTS)],
        "opener": _OPENERS[document_index % len(_OPENERS)],
        "action": _ACTIONS[document_index % len(_ACTIONS)],
    }

    document_id = str(document.get("_intent_document_id", document.get("id", title)))
    examples: list[dict[str, Any]] = []
    for frame_id, template, label, tags in _FRAMES:
        examples.append(
            {
                "id": f"corpus:{document_id}:{frame_id}",
                "text": template.format(**slots),
                "label": label,
                # Grouping by frame keeps one phrasing pattern inside one split.
                "source": f"frame:{frame_id}",
                "tags": list(tags),
                "source_doc_id": document.get("id", document_id),
                "source_title": title,
                "keywords": terms[:4],
                "context_hint": contents[:120],
            }
        )
    return examples


def generate_intent_examples(
    *,
    corpus_path: Path,
    vocabulary_path: Path | None = None,
) -> list[dict[str, Any]]:
    """Generate stable, source-grouped examples from a local corpus."""

    documents = load_corpus(corpus_path)
    vocabulary_tokens = (
        load_vocabulary_tokens(vocabulary_path) if vocabulary_path is not None else []
    )

    examples: list[dict[str, Any]] = []
    for document_index, document in enumerate(documents):
        examples.extend(
            build_examples_for_document(
                document, vocabulary_tokens, document_index=document_index
            )
        )
    _inject_hard_cases(examples)

    examples.sort(
        key=lambda item: (
            INTENTS.index(item["label"]),
            item["source"],
            item["id"],
            item["text"],
        )
    )
    _validate_generated_examples(examples)
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


def run_intent_training(config: IntentTrainingConfig) -> IntentTrainingRun:
    """Train, evaluate, compare, and atomically emit a candidate artifact set."""

    _validate_training_config(config)
    examples = load_intent_examples(Path(config.examples_path))
    eval_queries = (
        load_intent_eval_queries(Path(config.eval_queries_path))
        if config.eval_queries_path is not None
        else ()
    )
    if eval_queries:
        _validate_eval_queries_are_held_out(eval_queries, examples)
    split = split_intent_examples(
        examples,
        seed=config.seed,
        train_fraction=config.train_fraction,
        validation_fraction=config.validation_fraction,
    )
    baseline_records = _load_baseline_records(Path(config.baseline_path), split.test)

    pipeline = IntentPipeline(
        vocab_size=config.vocab_size,
        embedding_dim=config.embedding_dim,
        hidden_dim=config.hidden_dim,
    )
    training_data = [
        (tokenize_text(example.text), example.label) for example in split.train
    ]
    pipeline.train(
        training_data,
        epochs=config.epochs,
        lr=config.lr,
        min_freq=config.min_freq,
        seed=config.seed,
    )

    validation_records = _predict_model_eligible_examples(pipeline, split.validation)
    out_of_scope_confidences = _predict_out_of_scope_confidences(pipeline, config)
    # Out-of-scope abstention no longer constrains selection: with the gate
    # demoted, requiring it here would keep blocking coverage for a bar this
    # model family cannot clear.
    selected_threshold = select_confidence_threshold(
        validation_records,
        tool_precision_min=config.min_tool_precision,
        max_high_confidence_errors=config.max_high_confidence_errors,
    )
    model_records = _predict_examples(pipeline, split.test)
    candidate_records = compose_candidate_cascade(
        model_records, baseline_records, threshold=selected_threshold
    )
    candidate_report = evaluate_intent_predictions(
        candidate_records,
        threshold=selected_threshold,
        authoritative_routes_unchanged=authoritative_routes_match(
            candidate_records, baseline_records
        ),
        out_of_scope_abstention=(
            out_of_scope_abstention_rate(out_of_scope_confidences, selected_threshold)
            if out_of_scope_confidences
            else None
        ),
    )
    baseline_report = evaluate_intent_predictions(
        baseline_records, threshold=selected_threshold
    )
    criteria = PromotionCriteria(
        min_tool_precision=config.min_tool_precision,
        max_high_confidence_errors=config.max_high_confidence_errors,
        require_macro_f1_non_decreasing=config.require_macro_f1_non_decreasing,
        require_llm_fallback_reduction=config.require_llm_fallback_reduction,
        require_latency_improvement=config.require_latency_improvement,
    )
    promotion = compare_for_promotion(candidate_report, baseline_report, criteria)
    calibration = calibration_report(
        validation_records,
        selected_threshold=selected_threshold,
        out_of_scope_confidences=out_of_scope_confidences,
    )
    realistic = (
        realistic_accuracy_report(
            _predict_examples(
                pipeline,
                [
                    IntentExample(
                        id=query.id,
                        text=query.text,
                        label=query.label,
                        source="eval",
                    )
                    for query in eval_queries
                ],
            ),
            threshold=selected_threshold,
        )
        if eval_queries
        else None
    )

    output_dir = Path(config.output_dir)
    checkpoint_path = output_dir / "intent_model.pt"
    split_manifest_path = output_dir / "split_manifest.json"
    evaluation_report_path = output_dir / "evaluation_report.json"
    manifest = _split_manifest(split)
    report = {
        "labels": list(INTENT_LABELS),
        "hyperparameters": _hyperparameters(config),
        "dataset_fingerprint": split.fingerprint,
        "selected_threshold": selected_threshold,
        "split_label_counts": {
            name: manifest["splits"][name]["label_counts"]
            for name in ("train", "validation", "test")
        },
        "calibration": calibration,
        "realistic_accuracy": realistic,
        "candidate_metrics": candidate_report.to_dict(),
        "baseline_metrics": baseline_report.to_dict(),
        "promotion": promotion.to_dict(),
    }

    manifest_text = _serialize_json(manifest)
    report_text = _serialize_json(report)
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_artifact_set(
        pipeline=pipeline,
        dataset_fingerprint=split.fingerprint,
        promoted_min_confidence=(selected_threshold if promotion.promotable else None),
        checkpoint_path=checkpoint_path,
        split_manifest_path=split_manifest_path,
        split_manifest_text=manifest_text,
        evaluation_report_path=evaluation_report_path,
        evaluation_report_text=report_text,
    )

    return IntentTrainingRun(
        checkpoint_path=checkpoint_path,
        split_manifest_path=split_manifest_path,
        evaluation_report_path=evaluation_report_path,
        selected_threshold=selected_threshold,
        promotion=promotion,
    )


def _inject_hard_cases(examples: list[dict[str, Any]]) -> None:
    """Replace four direct examples with independently authored boundary pairs."""

    if not examples:
        return
    first_document_id = examples[0].get("source_doc_id")
    used_indices: set[int] = set()
    for example_id, text, label, tags in _HARD_CASES:
        index = next(
            index
            for index, example in enumerate(examples)
            if index not in used_indices
            and example["label"] == label
            and example.get("source_doc_id") == first_document_id
        )
        used_indices.add(index)
        source = (
            "manual:secure-email"
            if example_id.startswith("hard-secure-email")
            else "manual:duplicate-documents"
        )
        examples[index] = {
            **examples[index],
            "id": example_id,
            "text": text,
            "source": source,
            "tags": list(tags),
        }


def _validate_eval_queries_are_held_out(
    queries: Sequence[IntentEvalQuery], examples: Sequence[IntentExample]
) -> None:
    """Reject an evaluation set the generator already produces verbatim."""
    trained = {example.text.casefold().strip() for example in examples}
    if all(query.text.casefold().strip() in trained for query in queries):
        raise ValueError(
            "Evaluation queries all appear verbatim in the training examples, so "
            "the set cannot measure generalization."
        )


def _predict_examples(
    pipeline: IntentPipeline, examples: Sequence[IntentExample]
) -> tuple[IntentPredictionRecord, ...]:
    records: list[IntentPredictionRecord] = []
    for example in examples:
        started = time.perf_counter()
        prediction = pipeline.predict_text(example.text)
        latency_ms = (time.perf_counter() - started) * 1000.0
        records.append(
            IntentPredictionRecord(
                example_id=example.id,
                expected=example.label,
                predicted=prediction.intent,
                confidence=prediction.confidence,
                latency_ms=latency_ms,
                mechanism="model",
            )
        )
    return tuple(records)


def _predict_out_of_scope_confidences(
    pipeline: IntentPipeline, config: IntentTrainingConfig
) -> tuple[float, ...]:
    """Score requests the router should decline, if probes were supplied."""
    if config.out_of_scope_path is None:
        return ()
    probes = load_out_of_scope_probes(Path(config.out_of_scope_path))
    return tuple(pipeline.predict_text(text).confidence for _, text in probes)


def _predict_model_eligible_examples(
    pipeline: IntentPipeline, examples: Sequence[IntentExample]
) -> tuple[IntentPredictionRecord, ...]:
    """Predict only examples not already owned by the regex route."""
    from src.internal.servers.web.intent_routing import _regex_route

    eligible = tuple(
        example for example in examples if _regex_route(example.text) is None
    )
    if not eligible:
        raise ValueError(
            "Validation split has no model-eligible examples after regex routing"
        )
    return _predict_examples(pipeline, eligible)


def _load_baseline_records(
    path: Path, test_examples: Sequence[IntentExample]
) -> tuple[IntentPredictionRecord, ...]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"Invalid baseline predictions JSON in {path}: {exc.msg}"
        ) from exc
    if not isinstance(payload, list):
        raise ValueError("Baseline predictions JSON must contain a list of records")

    required_fields = {
        "example_id",
        "expected",
        "predicted",
        "confidence",
        "latency_ms",
        "mechanism",
    }
    by_id: dict[str, IntentPredictionRecord] = {}
    for index, item in enumerate(payload):
        if not isinstance(item, Mapping) or set(item) != required_fields:
            raise ValueError(
                f"Baseline prediction at index {index} must have exactly "
                f"{sorted(required_fields)!r}"
            )
        example_id = _nonempty_string(item["example_id"], "example_id", index)
        expected = _supported_label(item["expected"], "expected", index)
        predicted = _supported_label(item["predicted"], "predicted", index)
        confidence = _finite_number(item["confidence"], "confidence", index)
        latency_ms = _finite_number(item["latency_ms"], "latency_ms", index)
        mechanism = _nonempty_string(item["mechanism"], "mechanism", index)
        if mechanism not in {"regex", "classifier", "rule_based"}:
            raise ValueError(
                f"Baseline prediction at index {index} has invalid mechanism"
            )
        if not 0.0 <= confidence <= 1.0:
            raise ValueError(
                f"Baseline prediction at index {index} has invalid confidence"
            )
        if latency_ms < 0.0:
            raise ValueError(
                f"Baseline prediction at index {index} has invalid latency_ms"
            )
        if example_id in by_id:
            raise ValueError(f"Duplicate baseline prediction id: {example_id!r}")
        by_id[example_id] = IntentPredictionRecord(
            example_id=example_id,
            expected=expected,
            predicted=predicted,
            confidence=confidence,
            latency_ms=latency_ms,
            mechanism=mechanism,
        )

    expected_by_id = {example.id: example.label for example in test_examples}
    if set(by_id) != set(expected_by_id):
        missing = sorted(set(expected_by_id) - set(by_id))
        extra = sorted(set(by_id) - set(expected_by_id))
        raise ValueError(
            "Baseline prediction IDs must exactly match test IDs; "
            f"missing={missing!r}, extra={extra!r}"
        )
    for example_id, expected in expected_by_id.items():
        if by_id[example_id].expected != expected:
            raise ValueError(
                f"Baseline expected label does not match test example {example_id!r}"
            )
    return tuple(by_id[example.id] for example in test_examples)


def generate_baseline_predictions(
    *,
    examples_path: Path,
    fallback_predictions_path: Path,
    output_path: Path,
    seed: int = 17,
) -> tuple[IntentPredictionRecord, ...]:
    """Generate held-out regex records and merge captured ambiguous fallbacks."""
    examples = load_intent_examples(examples_path)
    split = split_intent_examples(examples, seed=seed)
    captured = _load_captured_fallbacks(fallback_predictions_path)

    from src.internal.servers.web.intent_routing import _regex_route

    records: list[IntentPredictionRecord] = []
    ambiguous: dict[str, IntentExample] = {}
    for example in split.test:
        started = time.perf_counter()
        strategy = _regex_route(example.text)
        latency_ms = (time.perf_counter() - started) * 1_000
        if strategy is None:
            ambiguous[example.id] = example
            continue
        records.append(
            IntentPredictionRecord(
                example_id=example.id,
                expected=example.label,
                predicted=strategy.value,
                confidence=1.0,
                latency_ms=latency_ms,
                mechanism="regex",
            )
        )

    if set(captured) != set(ambiguous):
        missing = sorted(set(ambiguous) - set(captured))
        extra = sorted(set(captured) - set(ambiguous))
        raise ValueError(
            "Captured fallback predictions must exactly match ambiguous held-out "
            f"IDs; missing={missing!r}, extra={extra!r}"
        )
    for example_id, example in ambiguous.items():
        record = captured[example_id]
        if record.expected != example.label:
            raise ValueError(
                f"Captured fallback expected label does not match {example_id!r}"
            )
        records.append(record)

    ordered = tuple(sorted(records, key=lambda record: record.example_id))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps([record.__dict__ for record in ordered], indent=2) + "\n",
        encoding="utf-8",
    )
    return ordered


def _load_captured_fallbacks(path: Path) -> dict[str, IntentPredictionRecord]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"Invalid fallback predictions JSON in {path}: {exc.msg}"
        ) from exc
    if not isinstance(payload, list):
        raise ValueError("Fallback predictions JSON must contain a list")
    records: dict[str, IntentPredictionRecord] = {}
    required = {
        "example_id",
        "expected",
        "predicted",
        "confidence",
        "latency_ms",
        "mechanism",
    }
    for index, item in enumerate(payload):
        if not isinstance(item, Mapping) or set(item) != required:
            raise ValueError(
                f"Fallback prediction at index {index} must have exactly "
                f"{sorted(required)!r}"
            )
        mechanism = _nonempty_string(item["mechanism"], "mechanism", index)
        if mechanism not in {"classifier", "rule_based"}:
            raise ValueError(
                "Captured fallback mechanism must be classifier or rule_based"
            )
        example_id = _nonempty_string(item["example_id"], "example_id", index)
        if example_id in records:
            raise ValueError(f"Duplicate fallback prediction id: {example_id!r}")
        confidence = _finite_number(item["confidence"], "confidence", index)
        latency_ms = _finite_number(item["latency_ms"], "latency_ms", index)
        if not 0.0 <= confidence <= 1.0 or latency_ms < 0.0:
            raise ValueError(f"Fallback prediction at index {index} is invalid")
        records[example_id] = IntentPredictionRecord(
            example_id=example_id,
            expected=_supported_label(item["expected"], "expected", index),
            predicted=_supported_label(item["predicted"], "predicted", index),
            confidence=confidence,
            latency_ms=latency_ms,
            mechanism=mechanism,
        )
    return records


def _split_manifest(split: IntentDatasetSplit) -> dict[str, Any]:
    return {
        "seed": split.seed,
        "dataset_fingerprint": split.fingerprint,
        "splits": {
            name: {
                "size": len(examples),
                "label_counts": dict(
                    sorted(Counter(example.label for example in examples).items())
                ),
                "example_ids": [example.id for example in examples],
                "sources": sorted({example.source for example in examples}),
            }
            for name, examples in (
                ("train", split.train),
                ("validation", split.validation),
                ("test", split.test),
            )
        },
    }


def _hyperparameters(config: IntentTrainingConfig) -> dict[str, Any]:
    return {
        "seed": config.seed,
        "epochs": config.epochs,
        "lr": config.lr,
        "min_freq": config.min_freq,
        "vocab_size": config.vocab_size,
        "embedding_dim": config.embedding_dim,
        "hidden_dim": config.hidden_dim,
        "train_fraction": config.train_fraction,
        "validation_fraction": config.validation_fraction,
        "min_tool_precision": config.min_tool_precision,
        "max_high_confidence_errors": config.max_high_confidence_errors,
        "require_macro_f1_non_decreasing": (config.require_macro_f1_non_decreasing),
        "require_llm_fallback_reduction": config.require_llm_fallback_reduction,
        "require_latency_improvement": config.require_latency_improvement,
        "eval_queries": (
            str(config.eval_queries_path) if config.eval_queries_path else None
        ),
    }


def _validate_training_config(config: IntentTrainingConfig) -> None:
    integer_fields = {
        "epochs": config.epochs,
        "min_freq": config.min_freq,
        "vocab_size": config.vocab_size,
        "embedding_dim": config.embedding_dim,
        "hidden_dim": config.hidden_dim,
    }
    invalid = [name for name, value in integer_fields.items() if value <= 0]
    if invalid:
        raise ValueError(f"Training values must be positive: {', '.join(invalid)}")
    if config.hidden_dim < 2:
        raise ValueError("hidden_dim must be at least 2")
    if not math.isfinite(config.lr) or config.lr <= 0:
        raise ValueError("lr must be a positive finite number")
    if not 0.0 <= config.min_tool_precision <= 1.0:
        raise ValueError("min_tool_precision must be between 0 and 1")
    if config.max_high_confidence_errors < 0:
        raise ValueError("max_high_confidence_errors must be non-negative")


def _serialize_json(payload: Mapping[str, Any]) -> str:
    return json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"


def _temporary_sibling(path: Path, *, suffix: str = ".tmp") -> tuple[int, Path]:
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=suffix, dir=path.parent
    )
    return descriptor, Path(temporary_name)


def _write_artifact_set(
    *,
    pipeline: IntentPipeline,
    dataset_fingerprint: str,
    promoted_min_confidence: float | None,
    checkpoint_path: Path,
    split_manifest_path: Path,
    split_manifest_text: str,
    evaluation_report_path: Path,
    evaluation_report_text: str,
) -> None:
    staged: list[tuple[Path, Path]] = []
    backups: dict[Path, Path] = {}
    publication_started = False
    try:
        checkpoint_descriptor, temporary_checkpoint = _temporary_sibling(
            checkpoint_path
        )
        os.close(checkpoint_descriptor)
        staged.append((temporary_checkpoint, checkpoint_path))
        pipeline.save(
            str(temporary_checkpoint),
            dataset_fingerprint=dataset_fingerprint,
            promoted_min_confidence=promoted_min_confidence,
        )

        for path, serialized in (
            (split_manifest_path, split_manifest_text),
            (evaluation_report_path, evaluation_report_text),
        ):
            descriptor, temporary_path = _temporary_sibling(path)
            staged.append((temporary_path, path))
            with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
                handle.write(serialized)

        for _, final_path in staged:
            if final_path.exists():
                descriptor, backup_path = _temporary_sibling(
                    final_path, suffix=".rollback"
                )
                os.close(descriptor)
                backups[final_path] = backup_path
                shutil.copyfile(final_path, backup_path)

        publication_started = True
        for temporary_path, final_path in staged:
            temporary_path.replace(final_path)
    except BaseException:
        if publication_started:
            for _, final_path in staged:
                backup_path = backups.get(final_path)
                if backup_path is not None:
                    backup_path.replace(final_path)
                else:
                    final_path.unlink(missing_ok=True)
        for temporary_path, _ in staged:
            temporary_path.unlink(missing_ok=True)
        for backup_path in backups.values():
            backup_path.unlink(missing_ok=True)
        raise
    else:
        for backup_path in backups.values():
            backup_path.unlink(missing_ok=True)


def _validate_optional_corpus_field(
    document: Mapping[str, object],
    field: str,
    expected_type: type | tuple[type, ...],
    line_number: int,
) -> None:
    if field not in document:
        return
    value = document[field]
    if (
        value is None
        or isinstance(value, bool)
        or not isinstance(value, expected_type)
        or (isinstance(value, str) and not value.strip())
    ):
        raise ValueError(f"Corpus record at line {line_number} has invalid {field}")


def _corpus_document_identity(document: Mapping[str, object]) -> str:
    if "id" in document:
        return str(document["id"])
    canonical = json.dumps(
        document, ensure_ascii=False, separators=(",", ":"), sort_keys=True
    )
    digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    return f"sha256:{digest}"


def _validate_generated_examples(examples: Sequence[Mapping[str, object]]) -> None:
    if not examples:
        raise ValueError("Generated intent examples must not be empty")
    ids: set[str] = set()
    labels_by_text: dict[str, str] = {}
    for index, example in enumerate(examples):
        if not isinstance(example, Mapping):
            raise ValueError(
                f"Generated intent example at index {index} must be an object"
            )
        values: dict[str, str] = {}
        for field in ("id", "text", "label", "source"):
            value = example.get(field)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(
                    f"Generated intent example at index {index} has invalid {field}"
                )
            values[field] = value
        if values["label"] not in INTENT_LABELS:
            raise ValueError(
                f"Generated intent example at index {index} has invalid label"
            )
        if values["id"] in ids:
            raise ValueError(f"Duplicate generated intent example id: {values['id']!r}")
        normalized_text = values["text"].casefold().strip()
        previous_label = labels_by_text.get(normalized_text)
        if previous_label is not None and previous_label != values["label"]:
            raise ValueError(
                "Generated intent examples contain conflicting duplicate text "
                f"{values['text']!r}"
            )
        tags = example.get("tags", ())
        if not isinstance(tags, (list, tuple)) or any(
            not isinstance(tag, str) or not tag.strip() for tag in tags
        ):
            raise ValueError(
                f"Generated intent example at index {index} has invalid tags"
            )
        ids.add(values["id"])
        labels_by_text[normalized_text] = values["label"]


def _nonempty_string(value: object, field: str, index: int) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Baseline prediction at index {index} has invalid {field}")
    return value


def _supported_label(value: object, field: str, index: int) -> str:
    label = _nonempty_string(value, field, index)
    if label not in INTENT_LABELS:
        raise ValueError(f"Baseline prediction at index {index} has invalid {field}")
    return label


def _finite_number(value: object, field: str, index: int) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"Baseline prediction at index {index} has invalid {field}")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"Baseline prediction at index {index} has invalid {field}")
    return result


class _IntentArgumentParser(argparse.ArgumentParser):
    """Argument parser whose invalid invocations use the workflow error code."""

    def error(self, message: str) -> None:
        self.print_usage(sys.stderr)
        raise ValueError(f"argument error: {message}")


def _build_parser() -> argparse.ArgumentParser:
    parser = _IntentArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    generate = subparsers.add_parser("generate", help="generate intent examples")
    generate.add_argument("--corpus", required=True, type=Path)
    generate.add_argument("--vocabulary", type=Path)
    generate.add_argument("--output", required=True, type=Path)

    baseline = subparsers.add_parser(
        "baseline", help="generate held-out baseline predictions"
    )
    baseline.add_argument("--examples", required=True, type=Path)
    baseline.add_argument("--fallback-predictions", required=True, type=Path)
    baseline.add_argument("--output", required=True, type=Path)
    baseline.add_argument("--seed", type=int, default=17)

    train = subparsers.add_parser("train", help="train and evaluate a candidate")
    train.add_argument("--examples", required=True, type=Path)
    train.add_argument("--baseline", required=True, type=Path)
    train.add_argument("--out-of-scope", type=Path)
    train.add_argument("--eval-queries", type=Path)
    train.add_argument("--output-dir", required=True, type=Path)
    train.add_argument("--seed", type=int, default=17)
    train.add_argument("--epochs", type=int, default=300)
    train.add_argument("--lr", type=float, default=1e-3)
    train.add_argument("--min-freq", type=int, default=1)
    train.add_argument("--vocab-size", type=int, default=5000)
    train.add_argument("--embedding-dim", type=int, default=128)
    train.add_argument("--hidden-dim", type=int, default=256)
    train.add_argument("--train-fraction", type=float, default=0.70)
    train.add_argument("--validation-fraction", type=float, default=0.15)
    train.add_argument("--min-tool-precision", type=float, default=0.95)
    train.add_argument("--max-high-confidence-errors", type=int, default=0)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the generation or offline train/evaluate command."""

    try:
        parser = _build_parser()
        args = parser.parse_args(argv)
        if args.command == "generate":
            examples = generate_intent_examples(
                corpus_path=args.corpus, vocabulary_path=args.vocabulary
            )
            _validate_generated_examples(examples)
            write_intent_examples(examples, args.output)
            return 0

        if args.command == "baseline":
            generate_baseline_predictions(
                examples_path=args.examples,
                fallback_predictions_path=args.fallback_predictions,
                output_path=args.output,
                seed=args.seed,
            )
            return 0

        run = run_intent_training(
            IntentTrainingConfig(
                examples_path=args.examples,
                baseline_path=args.baseline,
                out_of_scope_path=args.out_of_scope,
                eval_queries_path=args.eval_queries,
                output_dir=args.output_dir,
                seed=args.seed,
                epochs=args.epochs,
                lr=args.lr,
                min_freq=args.min_freq,
                vocab_size=args.vocab_size,
                embedding_dim=args.embedding_dim,
                hidden_dim=args.hidden_dim,
                train_fraction=args.train_fraction,
                validation_fraction=args.validation_fraction,
                min_tool_precision=args.min_tool_precision,
                max_high_confidence_errors=args.max_high_confidence_errors,
            )
        )
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"intent training failed: {exc}", file=sys.stderr)
        return 1
    return 0 if run.promotion.promotable else 2


def _pick_term(terms: list[str], index: int, fallback: str) -> str:
    if not terms:
        return fallback
    return terms[index % len(terms)]


if __name__ == "__main__":
    raise SystemExit(main())
