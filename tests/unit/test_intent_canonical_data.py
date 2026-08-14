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
#
# Re-derived for intfloat/e5-small-v2 (2026-08-13), because a raw-cosine bar is
# encoder-specific. Measured over the same 280 anchors, 39,060 pairs:
#
#   encoder      mean    sd      median  p99     p99.9   max
#   MiniLM-L6    0.0931  0.1064  0.0810  0.4407  0.6401  0.8404
#   e5-small-v2  0.7775  0.0266  0.7764  0.8524  0.8865  0.9340
#
# e5 compresses every pair into a narrow high band, so the old 0.90 sits at
# +7.6 sd above MiniLM's mean (0.06 clear of its maximum) but only +4.6 sd
# above e5's, *below* e5's maximum -- it would now flag 17 pairs, almost all of
# them merely on-topic. Re-applying the rule the old constant encoded (a hair
# above a clean set's maximum) lands near 0.98 under e5, which is no bar at
# all; 0.95 is tighter and is already this repo's duplicate threshold in the
# same units under the same encoder (LEAKAGE_COSINE in intent_index_cli).
#
# The measured maximum, 0.9340 -- "there was a policy about retaining user
# transcripts" against "what does the retention policy say about transcripts",
# both route `search` -- is a genuine near-duplicate: two phrasings of one
# request, which is exactly what this test exists to catch. The fix belongs in
# the canonical set, not in this constant, and the change that re-derived this
# ceiling was forbidden from editing that set.
#
# So be clear about what that costs: **this test no longer catches that pair**,
# and no test does. It is tracked in prose only -- PR #512's body and
# .superpowers/sdd/2026-08-13-intent-encoder-e5/task-3-report.md -- as an open
# data follow-up. Whoever de-duplicates it should re-measure this distribution
# afterwards and tighten this constant if the new maximum allows.
_MAX_INTERNAL_COSINE = 0.95


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
    """A form label, routed at cascade step 2 before this model ever runs.

    The floor is MIN_MODULE_SUPPORT (10), not 8: a count between 8 and 9
    would pass this test yet fall below the support floor, silently dropping
    the module from `_emit_modules`' candidate set.
    """
    counts = Counter(module for example in examples for module in example.modules)
    assert MIN_MODULE_SUPPORT <= counts["bare_entity"] <= 12


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
