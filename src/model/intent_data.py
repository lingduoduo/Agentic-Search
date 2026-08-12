"""Validated intent-training examples and deterministic source-grouped splits."""

from __future__ import annotations

import hashlib
import json
import random
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path

from .intent_classifier import INTENT_LABELS


@dataclass(frozen=True)
class IntentExample:
    id: str
    text: str
    label: str
    source: str
    tags: tuple[str, ...] = ()


@dataclass(frozen=True)
class IntentDatasetSplit:
    train: tuple[IntentExample, ...]
    validation: tuple[IntentExample, ...]
    test: tuple[IntentExample, ...]
    seed: int
    fingerprint: str


def load_intent_examples(path: Path) -> list[IntentExample]:
    """Load and validate a JSON list of intent-training examples."""
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid intent examples JSON in {path}: {exc.msg}") from exc

    if not isinstance(payload, list):
        raise ValueError("Intent examples JSON must contain a list of records")

    examples: list[IntentExample] = []
    ids: set[str] = set()
    labels_by_text: dict[str, str] = {}
    for index, record in enumerate(payload):
        if not isinstance(record, Mapping):
            raise ValueError(f"Intent example at index {index} must be an object")

        fields = {
            field: _required_text(record, field, index)
            for field in ("id", "text", "label", "source")
        }
        if fields["label"] not in INTENT_LABELS:
            raise ValueError(f"Unknown intent label: {fields['label']!r}")
        if fields["id"] in ids:
            raise ValueError(f"Duplicate intent example id: {fields['id']!r}")

        normalized_text = fields["text"].casefold().strip()
        previous_label = labels_by_text.get(normalized_text)
        if previous_label is not None and previous_label != fields["label"]:
            raise ValueError(
                "Found conflicting labels for duplicate text "
                f"{fields['text']!r}: {previous_label!r} and {fields['label']!r}"
            )

        ids.add(fields["id"])
        labels_by_text[normalized_text] = fields["label"]
        examples.append(
            IntentExample(
                id=fields["id"],
                text=fields["text"],
                label=fields["label"],
                source=fields["source"],
                tags=_tags(record, index),
            )
        )
    return examples


def split_intent_examples(
    examples: Iterable[IntentExample],
    *,
    seed: int = 17,
    train_fraction: float = 0.70,
    validation_fraction: float = 0.15,
) -> IntentDatasetSplit:
    """Make a reproducible, label-stratified split without source leakage."""
    _validate_fractions(train_fraction, validation_fraction)
    examples = tuple(examples)
    expected_labels = tuple(INTENT_LABELS)
    observed_labels = {example.label for example in examples}
    unknown_labels = observed_labels - set(expected_labels)
    if unknown_labels:
        raise ValueError(
            f"Unknown intent labels in examples: {sorted(unknown_labels)!r}"
        )

    missing_labels = set(expected_labels) - observed_labels
    if missing_labels:
        raise ValueError(
            f"Cannot split intent examples; missing labels: {sorted(missing_labels)!r}"
        )

    groups = _source_groups(examples)
    sources_by_label = {
        label: {
            source
            for source, group in groups.items()
            if any(example.label == label for example in group)
        }
        for label in expected_labels
    }
    too_few_groups = [
        label for label, sources in sources_by_label.items() if len(sources) < 3
    ]
    if too_few_groups:
        raise ValueError(
            "Each label needs at least three independent source groups; "
            "add more independent source groups before splitting "
            f"({', '.join(too_few_groups)})."
        )

    allocations = {
        label: _split_group_counts(len(sources), train_fraction, validation_fraction)
        for label, sources in sources_by_label.items()
    }
    assigned: list[list[IntentExample]] = [[], [], []]
    actual = {label: [0, 0, 0] for label in expected_labels}
    sources = sorted(groups)
    random.Random(seed).shuffle(sources)

    for source in sources:
        group = groups[source]
        group_labels = {example.label for example in group}
        split_index = max(
            range(3),
            key=lambda index: sum(
                allocations[label][index] - actual[label][index]
                for label in group_labels
            ),
        )
        assigned[split_index].extend(group)
        for label in group_labels:
            actual[label][split_index] += 1

    if any(
        actual[label][index] == 0 for label in expected_labels for index in range(3)
    ):
        raise ValueError(
            "Unable to place every label in every split; add more independent "
            "source groups before splitting."
        )

    return IntentDatasetSplit(
        train=tuple(assigned[0]),
        validation=tuple(assigned[1]),
        test=tuple(assigned[2]),
        seed=seed,
        fingerprint=_fingerprint(examples),
    )


def load_out_of_scope_probes(path: Path) -> tuple[tuple[str, str], ...]:
    """Load unlabeled requests the router should decline to serve.

    Probes carry no intent label: the three-label taxonomy has no way to say
    "none of these", so out-of-scope safety is measured as abstention below the
    serving threshold rather than as a prediction.
    """
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"Invalid out-of-scope probe JSON in {path}: {exc.msg}"
        ) from exc

    if not isinstance(payload, list):
        raise ValueError("Out-of-scope probe JSON must contain a list of records")

    probes: list[tuple[str, str]] = []
    ids: set[str] = set()
    for index, record in enumerate(payload):
        if not isinstance(record, Mapping):
            raise ValueError(f"Out-of-scope probe at index {index} must be an object")
        probe_id = _required_text(record, "id", index)
        text = _required_text(record, "text", index)
        if probe_id in ids:
            raise ValueError(f"Duplicate out-of-scope probe id: {probe_id!r}")
        ids.add(probe_id)
        probes.append((probe_id, text))
    if not probes:
        raise ValueError(f"Out-of-scope probe file contains no records: {path}")
    return tuple(probes)


def _required_text(record: Mapping[str, object], field: str, index: int) -> str:
    value = record.get(field)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Intent example at index {index} has empty {field!r}")
    return value


def _tags(record: Mapping[str, object], index: int) -> tuple[str, ...]:
    value = record.get("tags", ())
    if not isinstance(value, (list, tuple)) or any(
        not isinstance(tag, str) or not tag.strip() for tag in value
    ):
        raise ValueError(f"Intent example at index {index} has invalid tags")
    return tuple(value)


def _validate_fractions(train_fraction: float, validation_fraction: float) -> None:
    test_fraction = 1.0 - train_fraction - validation_fraction
    if not all(
        0.0 < fraction < 1.0
        for fraction in (train_fraction, validation_fraction, test_fraction)
    ):
        raise ValueError(
            "Split fractions must each be greater than zero and sum to one"
        )


def _source_groups(
    examples: tuple[IntentExample, ...],
) -> dict[str, tuple[IntentExample, ...]]:
    grouped: dict[str, list[IntentExample]] = {}
    for example in examples:
        if not isinstance(example.source, str) or not example.source.strip():
            raise ValueError("Intent examples must have a nonempty source")
        grouped.setdefault(example.source, []).append(example)
    return {
        source: tuple(
            sorted(group, key=lambda item: (item.id, item.text, item.label, item.tags))
        )
        for source, group in grouped.items()
    }


def _split_group_counts(
    total: int, train_fraction: float, validation_fraction: float
) -> tuple[int, int, int]:
    counts = [1, 1, 1]
    targets = (
        total * train_fraction,
        total * validation_fraction,
        total * (1.0 - train_fraction - validation_fraction),
    )
    for _ in range(total - sum(counts)):
        index = max(
            range(3), key=lambda candidate: targets[candidate] - counts[candidate]
        )
        counts[index] += 1
    return tuple(counts)  # type: ignore[return-value]


def _fingerprint(examples: tuple[IntentExample, ...]) -> str:
    records = sorted(
        (
            {
                "id": example.id,
                "label": example.label,
                "source": example.source,
                "tags": list(example.tags),
                "text": example.text,
            }
            for example in examples
        ),
        key=lambda record: json.dumps(record, ensure_ascii=False, sort_keys=True),
    )
    canonical = json.dumps(
        records, ensure_ascii=False, separators=(",", ":"), sort_keys=True
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()
