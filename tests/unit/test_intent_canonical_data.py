"""Balance guards on the committed canonical routing set.

The canonical examples *are* the model, and ``data/`` is gitignored, so
``data/intent_canonical.json`` is force-added and a later hand-edit would not
show up in ``git status``. Without these assertions such an edit could unbalance
the routes or starve a module below ``MIN_MODULE_SUPPORT`` — at which point the
module is silently never emitted, with no other symptom.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import pytest

from src.model.intent_data import load_canonical_examples
from src.model.intent_knn import MIN_MODULE_SUPPORT
from src.model.intent_taxonomy import INTENT_LABELS, SEMANTIC_MODULES

CANONICAL = Path(__file__).resolve().parents[2] / "data" / "intent_canonical.json"

# Headroom above MIN_MODULE_SUPPORT: a module that lands on the floor is one
# deletion away from being unemittable, which is not a state worth shipping.
_MIN_SEMANTIC_SUPPORT = 15
_MIN_ROUTE_SHARE = 0.25
_MAX_ROUTE_SHARE = 0.40
# Two canonical points this close are one point wearing two ids: the pair
# doubles its pull on every nearby query without adding coverage.
_MAX_INTERNAL_COSINE = 0.90


@pytest.fixture(scope="module")
def examples():
    if not CANONICAL.exists():
        pytest.skip(f"canonical example file is missing: {CANONICAL}")
    return load_canonical_examples(CANONICAL)


def test_the_committed_file_loads_through_the_canonical_loader(examples):
    assert len(examples) >= 1


def test_the_set_stays_in_the_curated_size_band(examples):
    """Wide enough that appending an anchor is a normal edit, not a test break.

    Extending the set is the documented way to change routing behavior, so a
    ceiling sitting on the current count would block the one workflow the
    operator guide tells people to use.
    """
    assert 260 <= len(examples) <= 400


def test_no_route_dominates_the_index(examples):
    counts = Counter(example.route for example in examples)
    assert set(counts) == set(INTENT_LABELS)
    for route, count in counts.items():
        share = count / len(examples)
        assert _MIN_ROUTE_SHARE <= share <= _MAX_ROUTE_SHARE, (route, count)


def test_every_semantic_module_keeps_headroom_above_the_support_floor(examples):
    counts = Counter(module for example in examples for module in example.modules)
    assert _MIN_SEMANTIC_SUPPORT > MIN_MODULE_SUPPORT
    thin = {
        module: counts[module]
        for module in SEMANTIC_MODULES
        if counts[module] < _MIN_SEMANTIC_SUPPORT
    }
    assert not thin, thin


def test_bare_entity_stays_capped(examples):
    """A form label, routed at cascade step 2 before this model ever runs."""
    counts = Counter(module for example in examples for module in example.modules)
    assert 8 <= counts["bare_entity"] <= 12


def test_no_two_canonical_examples_are_near_duplicates(examples):
    pytest.importorskip("sentence_transformers")
    import numpy as np

    from src.model.intent_encoder import encode_texts

    vectors = encode_texts([example.text for example in examples])
    similarities = vectors @ vectors.T
    np.fill_diagonal(similarities, -1.0)
    row, column = np.unravel_index(int(np.argmax(similarities)), similarities.shape)
    assert float(similarities[row, column]) < _MAX_INTERNAL_COSINE, (
        examples[row].text,
        examples[column].text,
    )
