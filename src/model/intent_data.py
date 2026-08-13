"""Validated intent inputs: canonical anchors, labelled examples, eval queries.

``load_canonical_examples`` reads the curated set the router is built from.
The labelled-example loader predates it and survives the retired classifier
because ``intent_seed`` still reads it to draft canonical candidates. The
train/validation/test split went with the classifier — nothing splits, and
nothing here trains anything.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from .intent_knn import CanonicalExample
from .intent_taxonomy import INTENT_LABELS, validate_modules


@dataclass(frozen=True)
class IntentExample:
    id: str
    text: str
    label: str
    source: str
    tags: tuple[str, ...] = ()


@dataclass(frozen=True)
class IntentEvalQuery:
    """One hand-authored request used only to measure realistic accuracy."""

    id: str
    text: str
    label: str
    modules: tuple[str, ...] = ()


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


def load_intent_eval_queries(path: Path) -> tuple[IntentEvalQuery, ...]:
    """Load the fixed realistic-evaluation set.

    This set is never trained on and never split: it is an instrument, so it is
    validated strictly and used exactly as authored.
    """
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"Invalid intent evaluation query JSON in {path}: {exc.msg}"
        ) from exc

    if not isinstance(payload, list):
        raise ValueError("Intent evaluation query JSON must contain a list of records")

    queries: list[IntentEvalQuery] = []
    ids: set[str] = set()
    kind = "Intent evaluation query"
    for index, record in enumerate(payload):
        if not isinstance(record, Mapping):
            raise ValueError(f"{kind} at index {index} must be an object")
        query_id = _required_text(record, "id", index, kind=kind)
        text = _required_text(record, "text", index, kind=kind)
        label = _required_text(record, "label", index, kind=kind)
        if label not in INTENT_LABELS:
            raise ValueError(f"Unknown intent label: {label!r}")
        if query_id in ids:
            raise ValueError(f"Duplicate intent evaluation query id: {query_id!r}")
        modules = _modules(record, index, label, kind=kind, required=False)
        ids.add(query_id)
        queries.append(
            IntentEvalQuery(id=query_id, text=text, label=label, modules=modules)
        )

    if not queries:
        raise ValueError(f"Intent evaluation query file contains no records: {path}")
    return tuple(queries)


def load_canonical_examples(path: Path) -> tuple[CanonicalExample, ...]:
    """Load and validate the curated examples that make up the routing index.

    These examples *are* the model, so validation is strict: a mislabeled or
    duplicated record changes routing directly, with no training run in between
    to average the mistake away.
    """
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"Invalid canonical example JSON in {path}: {exc.msg}"
        ) from exc

    if not isinstance(payload, list):
        raise ValueError("Canonical example JSON must contain a list of records")

    kind = "Canonical example"
    examples: list[CanonicalExample] = []
    ids: set[str] = set()
    texts: set[str] = set()
    for index, record in enumerate(payload):
        if not isinstance(record, Mapping):
            raise ValueError(f"{kind} at index {index} must be an object")
        identifier = _required_text(record, "id", index, kind=kind)
        text = _required_text(record, "text", index, kind=kind)
        route = _required_text(record, "route", index, kind=kind)
        if route not in INTENT_LABELS:
            raise ValueError(f"Unknown intent label: {route!r}")
        if identifier in ids:
            raise ValueError(f"Duplicate canonical example id: {identifier!r}")
        normalized = text.casefold().strip()
        if normalized in texts:
            # Duplicated text doubles that point's pull on every nearby query.
            raise ValueError(f"Duplicate canonical example text: {text!r}")
        modules = _modules(record, index, route, kind=kind, required=True)
        ids.add(identifier)
        texts.add(normalized)
        examples.append(
            CanonicalExample(id=identifier, text=text, route=route, modules=modules)
        )

    if not examples:
        raise ValueError(f"Canonical example file contains no records: {path}")
    return tuple(examples)


def _modules(
    record: Mapping[str, object],
    index: int,
    route: str,
    *,
    kind: str,
    required: bool,
) -> tuple[str, ...]:
    """Read and validate a record's modules against its route."""
    value = record.get("modules")
    if value is None:
        if required:
            raise ValueError(f"{kind} at index {index} has no 'modules'")
        return ()
    if not isinstance(value, (list, tuple)) or any(
        not isinstance(module, str) for module in value
    ):
        raise ValueError(f"{kind} at index {index} has invalid modules")
    modules = tuple(value)
    validate_modules(route, modules)
    return modules


def _required_text(
    record: Mapping[str, object],
    field: str,
    index: int,
    *,
    kind: str = "Intent example",
) -> str:
    value = record.get(field)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{kind} at index {index} has empty {field!r}")
    return value


def _tags(record: Mapping[str, object], index: int) -> tuple[str, ...]:
    value = record.get("tags", ())
    if not isinstance(value, (list, tuple)) or any(
        not isinstance(tag, str) or not tag.strip() for tag in value
    ):
        raise ValueError(f"Intent example at index {index} has invalid tags")
    return tuple(value)
