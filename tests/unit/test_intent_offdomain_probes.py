"""Invariants for the off-domain in-scope probe set.

These probes exist to measure one thing the other eval sets cannot: how the
router behaves on in-scope requests phrased in vocabulary the canonical set does
not contain. 47% of the canonical anchors carry IR/ML vocabulary because they
were curated from this project's own examples, and #511 measured 13 of 16
off-domain probes abstaining rather than routing.

They are authored **before** any anchor is added in response to them
(docs/superpowers/specs/2026-08-14-intent-canonical-coverage-design.md). That
ordering is the entire methodological point: a probe set written after the
anchors, or adjusted once its score is known, measures the curation rather than
the router. These tests guard the properties that make the set worth trusting;
they deliberately assert nothing about accuracy, which belongs in the evaluation
report.

No encoder is needed for any of this, so unlike the pinned bars these run in the
fast CI job.
"""

import json
from collections import Counter
from pathlib import Path

import pytest

from src.model.pre_training.intents.model import INTENT_LABELS, modules_for_route

PROBES_PATH = (
    Path(__file__).resolve().parents[2] / "data" / "intent_offdomain_probes.json"
)

# Vocabulary the canonical set is saturated with. A probe containing any of it
# is not off-domain, which makes it useless for the one measurement this file
# exists to support.
_IN_DOMAIN_VOCABULARY = {
    "index",
    "indexes",
    "indexing",
    "retrieval",
    "retrieve",
    "embedding",
    "embeddings",
    "corpus",
    "rerank",
    "reranker",
    "vector",
    "semantic",
    "token",
    "tokens",
    "chunk",
    "chunking",
    "encoder",
    "faiss",
    "bm25",
    "rag",
    "llm",
    "dataset",
    "relevance",
}

_MIN_PROBES = 40
_MIN_PER_ROUTE = 12


@pytest.fixture(scope="module")
def probes():
    if not PROBES_PATH.exists():
        pytest.skip(f"probe set not present: {PROBES_PATH}")
    return json.loads(PROBES_PATH.read_text(encoding="utf-8"))


def test_probe_set_is_large_enough_to_say_anything(probes):
    """#511's 16 probes gave a 13/16 headline with an interval too wide to act on.

    A bigger set is the point of authoring a new one, so a floor is asserted
    rather than left to whoever edits the file next.
    """
    assert len(probes) >= _MIN_PROBES


def test_every_probe_is_schema_valid_and_uniquely_identified(probes):
    assert [p["id"] for p in probes] == list(dict.fromkeys(p["id"] for p in probes))
    for probe in probes:
        assert probe["label"] in INTENT_LABELS, probe["id"]
        assert probe["modules"], probe["id"]
        allowed = modules_for_route(probe["label"])
        assert all(module in allowed for module in probe["modules"]), probe["id"]


def test_routes_are_balanced_so_no_route_dominates_the_measurement(probes):
    """An unbalanced probe set measures its majority route, not the router."""
    counts = Counter(p["label"] for p in probes)
    assert set(counts) == set(INTENT_LABELS)
    assert min(counts.values()) >= _MIN_PER_ROUTE


def test_no_probe_uses_the_vocabulary_the_canonical_set_is_saturated_with(probes):
    """The one property that makes these probes off-domain at all.

    Without it the set silently degrades into more of the same in-domain
    queries, and would report a flattering number for exactly the weakness it
    was built to expose.
    """
    offenders = {
        probe["id"]: sorted(
            word
            for word in probe["text"].casefold().replace("'", " ").split()
            if word.strip(",.?") in _IN_DOMAIN_VOCABULARY
        )
        for probe in probes
    }
    assert not {k: v for k, v in offenders.items() if v}


def test_probes_do_not_duplicate_the_canonical_set_or_the_other_eval_sets(probes):
    """A probe that restates an anchor measures the anchor.

    Exact-text only here — the cosine near-duplicate check needs an encoder and
    lives with the pinned bars. One probe was replaced during authoring for
    scoring 0.9323 against "forward the invoice to finance", which is the same
    request one word apart.
    """
    data = PROBES_PATH.parent
    existing: set[str] = set()
    for name in (
        "intent_canonical.json",
        "intent_eval_queries.json",
        "intent_eval_hard.json",
    ):
        path = data / name
        if not path.exists():
            pytest.skip(f"{name} not present")
        existing |= {
            record["text"].casefold().strip()
            for record in json.loads(path.read_text(encoding="utf-8"))
        }

    assert not [p["id"] for p in probes if p["text"].casefold().strip() in existing]
