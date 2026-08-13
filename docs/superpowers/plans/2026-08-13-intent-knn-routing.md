# Intent routing by nearest canonical example — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the trained intent MLP with cosine similarity against ~270 curated canonical routing examples encoded by MiniLM, and grow the evaluation instrument from 30 to ~220 in-domain queries.

**Architecture:** A route is decided by the mean of the top-3 cosine similarities between the query vector and the canonical examples of each route. Two thresholds gate the decision — absolute confidence (out-of-scope) and route margin (ambiguous) — and either failure abstains to the existing LLM classifier. Route-scoped, multi-label modules refine the answer for diagnostics but never decide the route. All index math is numpy-only; the encoder sits behind a one-function seam.

**Tech Stack:** Python 3, numpy, scikit-learn (metrics only), sentence-transformers (encoding), pytest.

**Spec:** `docs/superpowers/specs/2026-08-13-intent-knn-routing-design.md`

## Global Constraints

- `INTENT_LABELS = ["chat", "search", "tool"]` is unchanged.
- `src/model/intent_taxonomy.py`, `src/model/intent_knn.py`, and `src/model/intent_data.py` must import with **no torch, no transformers, no sentence-transformers**. The unit-test CI job installs none of them, and this repo has twice shipped collection failures from unguarded imports (#356, re-fixed in #418). `sentence_transformers` is imported **function-locally**, in `src/model/intent_encoder.py` only.
- `route_request` in `src/internal/servers/web/intent_routing.py` is **not modified**. Neither is the regex cascade, the clarification path, any dispatcher, or the frontend.
- `predict_route(query, *, settings) -> IntentModelDecision | None` keeps its exact signature and return type.
- The encoder is **lazily loaded** — never at app lifespan. Web `TestClient` tests already hang on lifespan model loads (`examples/run_web_integration_tests.sh` exists for this reason).
- Encoder model is `sentence-transformers/all-MiniLM-L6-v2`, 384 dimensions. Vectors are stored L2-normalized as `float32`.
- The 14 modules are route-scoped and fixed by this plan. 13 are `kind="intent"`; `bare_entity` is `kind="form"` and is excluded from module macro-F1.
- An example carries exactly one route and **one or more** modules.
- `data/` is gitignored — anything tracked under it needs `git add -f`.
- Torch/sentence-transformers-importing tests need `pytest.importorskip`.
- Work happens on branch `feat/intent-knn-routing` (already created off `main` at `1a6084d`). One commit per task. Open a PR after the last task; **never merge it**.
- Lint before each commit: `ruff check . --fix && ruff format .`
- **Nothing is deleted until the new path is measured.** The MLP machinery is removed in Task 9, after Task 8 produces numbers.

**The taxonomy, fixed:**

| route | module | kind |
|---|---|---|
| search | `lookup_document` | intent |
| search | `lookup_fact` | intent |
| search | `current_info` | intent |
| search | `bare_entity` | **form** |
| chat | `explain` | intent |
| chat | `summarize` | intent |
| chat | `compare` | intent |
| chat | `generate` | intent |
| chat | `converse` | intent |
| tool | `create` | intent |
| tool | `send` | intent |
| tool | `schedule` | intent |
| tool | `modify` | intent |
| tool | `execute` | intent |

`ACTION_MODULES = {"create", "send", "schedule", "modify", "execute"}` — used only for composite detection.

## File structure

| file | responsibility |
|---|---|
| `src/model/intent_taxonomy.py` (new) | `INTENT_LABELS`, the 14 modules, kinds, route-of-module lookup. Zero dependencies. |
| `src/model/intent_knn.py` (new) | `CanonicalExample`, `KnnDecision`, `IntentIndex`: scoring, thresholds, composite detection, minimum support, npz save/load. numpy only. |
| `src/model/intent_encoder.py` (new) | `encode_texts(texts) -> np.ndarray`. The only module that imports sentence-transformers, function-locally. |
| `src/model/intent_index_cli.py` (new) | `seed` / `build` / `evaluate` commands. |
| `src/model/intent_data.py` (modify) | `load_canonical_examples`; `modules` on eval queries. |
| `src/model/intent_evaluation.py` (modify) | `module_metrics_report`; import `INTENT_LABELS` from the taxonomy. |
| `src/internal/servers/web/ml_intent.py` (modify) | `predict_route` backed by the index. |
| `src/internal/configs/app_configs.py` (modify) | `intent_index_path`, `intent_min_route_margin`, `intent_min_module_score`. |
| `data/intent_canonical.json` (new, tracked with `-f`) | ~270 canonical examples. |
| `data/intent_eval_hard.json` (new, tracked with `-f`) | ~40 hard queries. |
| `data/intent_eval_queries.json` (modify) | 30 → ~180 queries, all gaining `modules`. |
| `docs/training-and-evaluation.md` (modify) | operator workflow and measured results. |

---

### Task 1: The taxonomy

**Files:**
- Create: `src/model/intent_taxonomy.py`
- Test: `tests/unit/test_intent_taxonomy.py` (new)

**Interfaces:**
- Consumes: nothing.
- Produces: `INTENT_LABELS: tuple[str, ...]`; `MODULES: dict[str, ModuleSpec]`; `@dataclass(frozen=True) ModuleSpec(name: str, route: str, kind: str)`; `ACTION_MODULES: frozenset[str]`; `SEMANTIC_MODULES: tuple[str, ...]`; `modules_for_route(route: str) -> tuple[str, ...]`; `route_of_module(module: str) -> str`; `validate_modules(route: str, modules: Sequence[str]) -> None`.

- [ ] **Step 1: Write the failing tests**

Create `tests/unit/test_intent_taxonomy.py`:

```python
import pytest

from src.model.intent_taxonomy import (
    ACTION_MODULES,
    INTENT_LABELS,
    MODULES,
    SEMANTIC_MODULES,
    modules_for_route,
    route_of_module,
    validate_modules,
)


def test_routes_are_the_three_serving_labels():
    assert INTENT_LABELS == ("chat", "search", "tool")


def test_taxonomy_has_fourteen_modules_thirteen_of_them_semantic():
    assert len(MODULES) == 14
    assert len(SEMANTIC_MODULES) == 13
    assert "bare_entity" not in SEMANTIC_MODULES


def test_bare_entity_is_a_form_label_not_a_semantic_intent():
    """It describes utterance shape, so it must not enter macro-F1."""
    assert MODULES["bare_entity"].kind == "form"
    assert all(MODULES[name].kind == "intent" for name in SEMANTIC_MODULES)


def test_every_module_belongs_to_exactly_one_route():
    for name, spec in MODULES.items():
        assert spec.route in INTENT_LABELS
        assert name in modules_for_route(spec.route)
        assert route_of_module(name) == spec.route


def test_routes_partition_the_modules():
    covered = [name for route in INTENT_LABELS for name in modules_for_route(route)]
    assert sorted(covered) == sorted(MODULES)


def test_action_modules_are_exactly_the_tool_modules():
    """Composite detection keys off these; drift would silently break it."""
    assert ACTION_MODULES == frozenset(modules_for_route("tool"))


def test_validate_accepts_multiple_modules_from_the_same_route():
    validate_modules("search", ["current_info", "lookup_fact"])


def test_validate_rejects_a_module_from_another_route():
    with pytest.raises(ValueError, match="summarize"):
        validate_modules("search", ["lookup_fact", "summarize"])


def test_validate_rejects_an_unknown_module():
    with pytest.raises(ValueError, match="nonsense"):
        validate_modules("search", ["nonsense"])


def test_validate_rejects_an_empty_module_list():
    with pytest.raises(ValueError, match="at least one"):
        validate_modules("search", [])


def test_validate_rejects_duplicate_modules():
    with pytest.raises(ValueError, match="duplicate"):
        validate_modules("search", ["lookup_fact", "lookup_fact"])


def test_route_of_unknown_module_is_an_error():
    with pytest.raises(KeyError):
        route_of_module("nonsense")
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `pytest tests/unit/test_intent_taxonomy.py -v`
Expected: FAIL at collection — `ModuleNotFoundError: No module named 'src.model.intent_taxonomy'`.

- [ ] **Step 3: Implement the taxonomy**

Create `src/model/intent_taxonomy.py`:

```python
"""The two-level intent taxonomy: three routes, each with its own modules.

Modules are route-scoped rather than orthogonal primitives. At a few hundred
canonical examples a small hierarchical taxonomy is labelable consistently and
explainable; orthogonal primitives are the right end state at several thousand.

The module names are not invented. Each is drawn from a regex cue already used
by ``src/internal/servers/web/intent_routing.py``, so the taxonomy describes
distinctions the router already makes.

This module imports nothing. Every other intent module depends on it.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

INTENT_LABELS: tuple[str, ...] = ("chat", "search", "tool")


@dataclass(frozen=True)
class ModuleSpec:
    """One module: its route, and whether it names an intent or an utterance form."""

    name: str
    route: str
    kind: str


_SPECS: tuple[ModuleSpec, ...] = (
    # search — _SEARCH_LOOKUP_RE, _CURRENCY_RE, _is_bare_lookup
    ModuleSpec("lookup_document", "search", "intent"),
    ModuleSpec("lookup_fact", "search", "intent"),
    ModuleSpec("current_info", "search", "intent"),
    # A form label, not an intent: "OpenAI" is a bare entity, "OpenAI CEO" is a
    # fact lookup. It is excluded from module macro-F1, and _is_bare_lookup
    # routes such queries at cascade step 2, before this model ever runs.
    ModuleSpec("bare_entity", "search", "form"),
    # chat — _CHAT_START_RE, _GENERATIVE_RE
    ModuleSpec("explain", "chat", "intent"),
    ModuleSpec("summarize", "chat", "intent"),
    ModuleSpec("compare", "chat", "intent"),
    ModuleSpec("generate", "chat", "intent"),
    ModuleSpec("converse", "chat", "intent"),
    # tool — _TOOL_ACTION_RE, _TOOL_OBJECT_RE
    ModuleSpec("create", "tool", "intent"),
    ModuleSpec("send", "tool", "intent"),
    ModuleSpec("schedule", "tool", "intent"),
    ModuleSpec("modify", "tool", "intent"),
    ModuleSpec("execute", "tool", "intent"),
)

MODULES: dict[str, ModuleSpec] = {spec.name: spec for spec in _SPECS}

SEMANTIC_MODULES: tuple[str, ...] = tuple(
    spec.name for spec in _SPECS if spec.kind == "intent"
)

# Composite detection keys off these: a runner-up route whose best module is an
# action is the signature of "find X and book it".
ACTION_MODULES: frozenset[str] = frozenset(
    spec.name for spec in _SPECS if spec.route == "tool"
)


def modules_for_route(route: str) -> tuple[str, ...]:
    """Every module belonging to *route*, in taxonomy order."""
    return tuple(spec.name for spec in _SPECS if spec.route == route)


def route_of_module(module: str) -> str:
    """The route *module* belongs to. Raises KeyError if unknown."""
    return MODULES[module].route


def validate_modules(route: str, modules: Sequence[str]) -> None:
    """Check a label's modules: nonempty, known, unique, and all in *route*."""
    if not modules:
        raise ValueError(f"Route {route!r} needs at least one module")
    seen: set[str] = set()
    for module in modules:
        if module in seen:
            raise ValueError(f"Found duplicate module {module!r} for route {route!r}")
        seen.add(module)
        spec = MODULES.get(module)
        if spec is None:
            raise ValueError(f"Unknown module {module!r}")
        if spec.route != route:
            raise ValueError(
                f"Module {module!r} belongs to route {spec.route!r}, not {route!r}"
            )
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `pytest tests/unit/test_intent_taxonomy.py -v`
Expected: PASS, all 11.

- [ ] **Step 5: Commit**

```bash
ruff check . --fix && ruff format .
git add src/model/intent_taxonomy.py tests/unit/test_intent_taxonomy.py
git commit -m "feat(intent): add the route-scoped module taxonomy"
```

---

### Task 2: The index — scoring, thresholds, composite detection

**Files:**
- Create: `src/model/intent_knn.py`
- Test: `tests/unit/test_intent_knn.py` (new)

**Interfaces:**
- Consumes: `INTENT_LABELS`, `ACTION_MODULES`, `modules_for_route`, `validate_modules` from Task 1.
- Produces:
  - `@dataclass(frozen=True) CanonicalExample(id: str, text: str, route: str, modules: tuple[str, ...])`
  - `@dataclass(frozen=True) KnnDecision(route: str, confidence: float, margin: float, modules: tuple[str, ...], composite: bool, abstained: bool, abstain_reason: str | None)`
  - `class IntentIndex(examples: Sequence[CanonicalExample], vectors: np.ndarray, encoder: str, fingerprint: str)` with properties `size: int`, `encoder: str`, `fingerprint: str`; methods `route_scores(vector) -> dict[str, float]`, `module_scores(vector) -> dict[str, float]`, `decide(vector, *, min_confidence, min_margin, min_module_score) -> KnnDecision`, `low_support_modules(minimum: int = MIN_MODULE_SUPPORT) -> tuple[str, ...]`, `save(path: Path) -> None`, classmethod `load(path: Path) -> IntentIndex`.
  - `TOP_K = 3`, `MIN_MODULE_SUPPORT = 10`, `INDEX_FILENAME = "index.npz"`
- Task 4 calls `IntentIndex(...)` and `save`. Task 5 calls `load` and `decide`.

Vectors are assumed L2-normalized on the way in; the constructor rejects any row whose norm is not 1 within `1e-3`, because a silently unnormalized row turns cosine into an arbitrary dot product and there is no later symptom.

- [ ] **Step 1: Write the failing tests**

Create `tests/unit/test_intent_knn.py`:

```python
"""Index behavior over hand-built orthogonal vectors.

Deliberately no encoder: the unit-test CI job installs neither torch nor
sentence-transformers, and all routing math must be exercisable without them.
Canonical vectors sit on the basis axes so every cosine is exact.
"""

from pathlib import Path

import numpy as np
import pytest

from src.model.intent_knn import (
    MIN_MODULE_SUPPORT,
    CanonicalExample,
    IntentIndex,
)

# search -> e0, chat -> e1, tool -> e2. Cosines are then exactly readable.
_AXIS = {"search": 0, "chat": 1, "tool": 2}
_MODULE = {"search": "lookup_fact", "chat": "explain", "tool": "schedule"}


def _unit(*components: float) -> np.ndarray:
    vector = np.array(components, dtype=np.float32)
    return vector / np.linalg.norm(vector)


def _index(per_route: int = 3, **kwargs) -> IntentIndex:
    examples, rows = [], []
    for route, axis in _AXIS.items():
        for position in range(per_route):
            examples.append(
                CanonicalExample(
                    id=f"{route}-{position}",
                    text=f"{route} example {position}",
                    route=route,
                    modules=(_MODULE[route],),
                )
            )
            rows.append(np.eye(3, dtype=np.float32)[axis])
    return IntentIndex(
        examples=examples,
        vectors=np.stack(rows),
        encoder="test-encoder",
        fingerprint="sha256:test",
        **kwargs,
    )


def _decide(index: IntentIndex, vector, **overrides):
    thresholds = {"min_confidence": 0.5, "min_margin": 0.05, "min_module_score": 0.4}
    return index.decide(vector, **{**thresholds, **overrides})


def test_query_on_a_route_axis_picks_that_route_with_full_confidence():
    decision = _decide(_index(), _unit(1, 0, 0))

    assert decision.route == "search"
    assert decision.confidence == pytest.approx(1.0)
    assert decision.margin == pytest.approx(1.0)
    assert decision.abstained is False


def test_route_score_is_the_mean_of_the_top_three_not_of_all_examples():
    """One near neighbor must not be diluted by distant same-route examples."""
    examples = [
        CanonicalExample(f"s-{i}", f"t{i}", "search", ("lookup_fact",))
        for i in range(5)
    ]
    examples += [CanonicalExample("c-0", "t", "chat", ("explain",))]
    vectors = np.stack(
        [_unit(1, 0, 0)] * 3 + [_unit(0, 0, 1)] * 2 + [_unit(0, 1, 0)]
    )
    index = IntentIndex(examples, vectors, "test-encoder", "sha256:test")

    scores = index.route_scores(_unit(1, 0, 0))

    # Top 3 of search are the three exact matches; the two distant ones do not
    # drag the mean down.
    assert scores["search"] == pytest.approx(1.0)


def test_fewer_than_three_examples_averages_what_exists():
    examples = [CanonicalExample("s-0", "t", "search", ("lookup_fact",))]
    examples += [CanonicalExample("c-0", "t", "chat", ("explain",))]
    examples += [CanonicalExample("t-0", "t", "tool", ("schedule",))]
    vectors = np.stack([_unit(1, 0, 0), _unit(0, 1, 0), _unit(0, 0, 1)])
    index = IntentIndex(examples, vectors, "test-encoder", "sha256:test")

    assert index.route_scores(_unit(1, 0, 0))["search"] == pytest.approx(1.0)


def test_equidistant_query_abstains_on_margin_not_on_confidence():
    """Ambiguity and out-of-scope are different failures with different causes."""
    decision = _decide(_index(), _unit(1, 1, 0), min_confidence=0.5)

    assert decision.confidence == pytest.approx(0.7071, abs=1e-3)
    assert decision.margin == pytest.approx(0.0, abs=1e-6)
    assert decision.abstained is True
    assert decision.abstain_reason == "margin_below_threshold"


def test_query_far_from_everything_abstains_on_confidence():
    """Cosine is unnormalized across routes, so 'far from all' is expressible."""
    decision = _decide(_index(), _unit(1, 1, 1), min_confidence=0.7)

    assert decision.confidence == pytest.approx(0.5774, abs=1e-3)
    assert decision.abstained is True
    assert decision.abstain_reason == "confidence_below_threshold"


def test_confidence_is_checked_before_margin():
    """A query far from everything is out of scope, not merely ambiguous."""
    decision = _decide(_index(), _unit(1, 1, 1), min_confidence=0.7, min_margin=0.9)

    assert decision.abstain_reason == "confidence_below_threshold"


def test_a_confident_decision_still_reports_its_route_and_modules():
    """Abstention defers routing; it never blanks the diagnostics."""
    decision = _decide(_index(), _unit(1, 1, 0), min_confidence=0.5)

    assert decision.route == "search"
    assert decision.modules == ("lookup_fact",)


def test_modules_are_multi_label_above_the_module_threshold():
    examples = [
        CanonicalExample("s-0", "t", "search", ("lookup_fact", "current_info")),
        CanonicalExample("s-1", "t", "search", ("lookup_document",)),
        CanonicalExample("c-0", "t", "chat", ("explain",)),
        CanonicalExample("t-0", "t", "tool", ("schedule",)),
    ]
    vectors = np.stack(
        [_unit(1, 0, 0), _unit(0, 0.2, 1), _unit(0, 1, 0), _unit(0, 0, 1)]
    )
    index = IntentIndex(examples, vectors, "test-encoder", "sha256:test")

    decision = _decide(index, _unit(1, 0, 0), min_module_score=0.5)

    assert set(decision.modules) == {"lookup_fact", "current_info"}


def test_modules_fall_back_to_the_single_best_when_none_clear_the_bar():
    decision = _decide(_index(), _unit(1, 0, 0), min_module_score=0.99999)

    assert decision.modules == ("lookup_fact",)


def test_modules_come_only_from_the_chosen_route():
    decision = _decide(_index(), _unit(1, 0, 0), min_module_score=0.0)

    assert decision.modules == ("lookup_fact",)


def test_module_threshold_cannot_change_the_route():
    """The module knob is diagnostic; it must never move routing."""
    routes = {
        _decide(_index(), _unit(1, 0, 0), min_module_score=score).route
        for score in (0.0, 0.5, 0.99)
    }

    assert routes == {"search"}


def test_composite_fires_when_a_close_runner_up_is_an_action():
    """The 'find the best place and book it' signature."""
    decision = _decide(_index(), _unit(1, 0, 0.98), min_margin=0.0)

    assert decision.route == "search"
    assert decision.composite is True


def test_composite_does_not_fire_for_a_close_non_action_runner_up():
    decision = _decide(_index(), _unit(1, 0.98, 0), min_margin=0.0)

    assert decision.composite is False


def test_composite_does_not_fire_when_the_runner_up_is_distant():
    decision = _decide(_index(), _unit(1, 0, 0), min_margin=0.05)

    assert decision.composite is False


def test_low_support_modules_are_reported():
    index = _index(per_route=3)

    assert set(index.low_support_modules()) >= {"lookup_fact", "explain", "schedule"}
    assert MIN_MODULE_SUPPORT == 10


def test_a_decision_always_carries_a_module_even_when_all_are_thin():
    """Support filtering must not leave a decision with no module at all."""
    index = _index(per_route=3)
    assert "lookup_fact" in index.low_support_modules()

    decision = _decide(index, _unit(1, 0, 0), min_module_score=0.0)

    assert decision.modules == ("lookup_fact",)


def test_a_thin_module_is_dropped_when_a_supported_one_exists():
    """Support filtering bites only when it has something better to offer."""
    examples, rows = [], []
    for position in range(12):
        examples.append(
            CanonicalExample(f"s-{position}", "t", "search", ("lookup_fact",))
        )
        rows.append(_unit(1, 0, 0))
    examples.append(CanonicalExample("s-thin", "t", "search", ("current_info",)))
    rows.append(_unit(1, 0, 0))
    examples.append(CanonicalExample("c-0", "t", "chat", ("explain",)))
    rows.append(_unit(0, 1, 0))
    examples.append(CanonicalExample("t-0", "t", "tool", ("schedule",)))
    rows.append(_unit(0, 0, 1))
    index = IntentIndex(examples, np.stack(rows), "test-encoder", "sha256:x")

    decision = _decide(index, _unit(1, 0, 0), min_module_score=0.0)

    assert decision.modules == ("lookup_fact",)
    assert "current_info" in index.low_support_modules()


def test_constructor_rejects_unnormalized_vectors():
    """An unnormalized row turns cosine into an arbitrary dot product."""
    examples = [CanonicalExample("s-0", "t", "search", ("lookup_fact",))]

    with pytest.raises(ValueError, match="normalized"):
        IntentIndex(
            examples,
            np.array([[2.0, 0.0, 0.0]], dtype=np.float32),
            "test-encoder",
            "sha256:test",
        )


def test_constructor_rejects_a_row_count_that_disagrees_with_the_examples():
    examples = [CanonicalExample("s-0", "t", "search", ("lookup_fact",))]

    with pytest.raises(ValueError, match="rows"):
        IntentIndex(examples, np.eye(3, dtype=np.float32), "test-encoder", "sha256:x")


def test_constructor_rejects_modules_outside_the_example_route():
    examples = [CanonicalExample("s-0", "t", "search", ("summarize",))]

    with pytest.raises(ValueError, match="summarize"):
        IntentIndex(
            examples,
            np.eye(3, dtype=np.float32)[:1],
            "test-encoder",
            "sha256:x",
        )


def test_round_trip_preserves_examples_vectors_and_provenance(tmp_path: Path):
    index = _index()
    path = tmp_path / "index.npz"
    index.save(path)

    reloaded = IntentIndex.load(path)

    assert reloaded.size == index.size
    assert reloaded.encoder == "test-encoder"
    assert reloaded.fingerprint == "sha256:test"
    before = _decide(index, _unit(1, 0, 0))
    after = _decide(reloaded, _unit(1, 0, 0))
    assert after.route == before.route
    assert after.confidence == pytest.approx(before.confidence)
    assert after.modules == before.modules


def test_load_reports_a_missing_index_by_name(tmp_path: Path):
    with pytest.raises(FileNotFoundError, match="index.npz"):
        IntentIndex.load(tmp_path / "index.npz")
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `pytest tests/unit/test_intent_knn.py -v`
Expected: FAIL at collection — `ModuleNotFoundError: No module named 'src.model.intent_knn'`.

- [ ] **Step 3: Implement the index**

Create `src/model/intent_knn.py`:

```python
"""Route a request by similarity to curated canonical examples.

The route is the argmax of a per-route score: the mean of the top-3 cosine
similarities among that route's canonical examples. Two thresholds gate the
result, because two different things go wrong. A low absolute score means
nothing canonical resembles the request — out of scope. A low margin between
the best and second-best route means two routes fit equally well — ambiguous.
Either one abstains, and the caller falls through to the LLM classifier.

Cosine is deliberately not normalized across routes. A softmax head sums to one
by construction and so cannot express "none of these", which is why the previous
model's out-of-scope separation was only +0.059.

This module imports numpy and nothing heavier: all routing math must run in a
CI job that installs neither torch nor sentence-transformers.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .intent_taxonomy import (
    ACTION_MODULES,
    INTENT_LABELS,
    modules_for_route,
    validate_modules,
)

TOP_K = 3
MIN_MODULE_SUPPORT = 10
INDEX_FILENAME = "index.npz"

_NORM_TOLERANCE = 1e-3


@dataclass(frozen=True)
class CanonicalExample:
    """One curated routing example: the unit of the index."""

    id: str
    text: str
    route: str
    modules: tuple[str, ...]


@dataclass(frozen=True)
class KnnDecision:
    """A route decision with its diagnostics and, if any, its abstention."""

    route: str
    confidence: float
    margin: float
    modules: tuple[str, ...]
    composite: bool
    abstained: bool
    abstain_reason: str | None


class IntentIndex:
    """Canonical example vectors and the scoring rules over them."""

    def __init__(
        self,
        examples: Sequence[CanonicalExample],
        vectors: np.ndarray,
        encoder: str,
        fingerprint: str,
    ) -> None:
        if vectors.ndim != 2 or vectors.shape[0] != len(examples):
            raise ValueError(
                "Index rows must equal the example count: "
                f"{vectors.shape} against {len(examples)} examples"
            )
        norms = np.linalg.norm(vectors, axis=1)
        if not np.allclose(norms, 1.0, atol=_NORM_TOLERANCE):
            raise ValueError(
                "Index vectors must be L2-normalized; cosine similarity is "
                "otherwise an arbitrary dot product with no later symptom"
            )
        for example in examples:
            validate_modules(example.route, example.modules)

        self._examples = tuple(examples)
        self._vectors = vectors.astype(np.float32)
        self._encoder = encoder
        self._fingerprint = fingerprint
        self._route_rows = {
            route: np.array(
                [i for i, e in enumerate(self._examples) if e.route == route],
                dtype=np.int64,
            )
            for route in INTENT_LABELS
        }
        self._module_rows = {
            module: np.array(
                [i for i, e in enumerate(self._examples) if module in e.modules],
                dtype=np.int64,
            )
            for module in (m for route in INTENT_LABELS for m in modules_for_route(route))
        }

    @property
    def size(self) -> int:
        return len(self._examples)

    @property
    def encoder(self) -> str:
        return self._encoder

    @property
    def fingerprint(self) -> str:
        return self._fingerprint

    @property
    def examples(self) -> tuple[CanonicalExample, ...]:
        return self._examples

    def _top_k_mean(self, similarities: np.ndarray, rows: np.ndarray) -> float:
        """Mean of the highest TOP_K similarities among *rows*, or 0.0 if empty.

        Taking a top-k mean rather than the mean of all rows keeps one close
        neighbor from being diluted by distant same-route examples, and keeps
        one outlier from carrying the route the way a bare max would.
        """
        if rows.size == 0:
            return 0.0
        selected = similarities[rows]
        if selected.size > TOP_K:
            selected = np.partition(selected, -TOP_K)[-TOP_K:]
        return float(selected.mean())

    def _similarities(self, vector: np.ndarray) -> np.ndarray:
        return self._vectors @ np.asarray(vector, dtype=np.float32)

    def route_scores(self, vector: np.ndarray) -> dict[str, float]:
        """Per-route top-3 mean cosine."""
        similarities = self._similarities(vector)
        return {
            route: self._top_k_mean(similarities, rows)
            for route, rows in self._route_rows.items()
        }

    def module_scores(self, vector: np.ndarray) -> dict[str, float]:
        """Per-module top-3 mean cosine, over every module in the taxonomy."""
        similarities = self._similarities(vector)
        return {
            module: self._top_k_mean(similarities, rows)
            for module, rows in self._module_rows.items()
        }

    def low_support_modules(self, minimum: int = MIN_MODULE_SUPPORT) -> tuple[str, ...]:
        """Modules with too few examples to score meaningfully."""
        return tuple(
            module
            for module, rows in self._module_rows.items()
            if rows.size < minimum
        )

    def decide(
        self,
        vector: np.ndarray,
        *,
        min_confidence: float,
        min_margin: float,
        min_module_score: float,
    ) -> KnnDecision:
        """Score *vector*, apply both routing thresholds, and report modules."""
        routes = self.route_scores(vector)
        ranked = sorted(routes.items(), key=lambda item: item[1], reverse=True)
        (best_route, confidence), (runner_up, runner_up_score) = ranked[0], ranked[1]
        margin = confidence - runner_up_score

        # Confidence is checked first: a request far from everything is out of
        # scope, which is a more useful thing to say than "ambiguous".
        abstain_reason: str | None = None
        if confidence < min_confidence:
            abstain_reason = "confidence_below_threshold"
        elif margin < min_margin:
            abstain_reason = "margin_below_threshold"

        modules = self._emit_modules(vector, best_route, min_module_score)
        return KnnDecision(
            route=best_route,
            confidence=confidence,
            margin=margin,
            modules=modules,
            composite=self._is_composite(vector, runner_up, margin, min_margin),
            abstained=abstain_reason is not None,
            abstain_reason=abstain_reason,
        )

    def _emit_modules(
        self, vector: np.ndarray, route: str, min_module_score: float
    ) -> tuple[str, ...]:
        """Every well-supported module of *route* scoring at or above the bar.

        Falls back to the single best so a decision always carries a module.
        This is multi-label because real requests carry several intents at once:
        "compare the current prices of BTC and ETH" is both current_info and
        lookup_fact.
        """
        low_support = set(self.low_support_modules())
        scores = self.module_scores(vector)
        candidates = {
            module: scores[module]
            for module in modules_for_route(route)
            if module not in low_support
        }
        if not candidates:
            candidates = {
                module: scores[module] for module in modules_for_route(route)
            }
        emitted = tuple(
            module
            for module, score in candidates.items()
            if score >= min_module_score
        )
        if emitted:
            return emitted
        return (max(candidates, key=lambda module: candidates[module]),)

    def _is_composite(
        self, vector: np.ndarray, runner_up: str, margin: float, min_margin: float
    ) -> bool:
        """True when a close runner-up route is an action.

        This is the signature of a request that needs two steps — "find the best
        Italian place near the office and book it for 7". Nothing acts on the
        flag yet; it is recorded so the plan-aware router can be designed
        against measured data rather than guesses.
        """
        if margin >= min_margin:
            return False
        scores = self.module_scores(vector)
        candidates = modules_for_route(runner_up)
        if not candidates:
            return False
        best = max(candidates, key=lambda module: scores[module])
        return best in ACTION_MODULES

    def save(self, path: Path) -> None:
        """Write vectors and labels to a single npz."""
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            path,
            vectors=self._vectors,
            examples=np.array(
                json.dumps(
                    [
                        {
                            "id": e.id,
                            "text": e.text,
                            "route": e.route,
                            "modules": list(e.modules),
                        }
                        for e in self._examples
                    ]
                )
            ),
            encoder=np.array(self._encoder),
            fingerprint=np.array(self._fingerprint),
        )

    @classmethod
    def load(cls, path: Path) -> "IntentIndex":
        """Read an index written by ``save``."""
        if not path.exists():
            raise FileNotFoundError(
                f"Intent index is missing {path.name}: {path}. Run "
                "`python -m src.model.intent_index_cli build` to create it."
            )
        payload = np.load(path, allow_pickle=False)
        records = json.loads(str(payload["examples"]))
        examples = [
            CanonicalExample(
                id=record["id"],
                text=record["text"],
                route=record["route"],
                modules=tuple(record["modules"]),
            )
            for record in records
        ]
        return cls(
            examples=examples,
            vectors=payload["vectors"],
            encoder=str(payload["encoder"]),
            fingerprint=str(payload["fingerprint"]),
        )
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `pytest tests/unit/test_intent_knn.py -v`
Expected: PASS, all 22.

- [ ] **Step 5: Prove the module imports with no ML dependency**

```bash
python - <<'PY'
import sys

class _Block:
    def find_module(self, name, path=None):
        if name.split(".")[0] in {"torch", "transformers", "sentence_transformers"}:
            raise ModuleNotFoundError(f"blocked: {name}")
        return None

sys.meta_path.insert(0, _Block())
from src.model.intent_knn import IntentIndex
from src.model.intent_taxonomy import MODULES
print("imported without torch:", IntentIndex, len(MODULES))
PY
```
Expected: prints the class and `14`. A `ModuleNotFoundError` means something in the import chain pulls torch — fix it now, because CI will hit the same wall.

- [ ] **Step 6: Commit**

```bash
ruff check . --fix && ruff format .
git add src/model/intent_knn.py tests/unit/test_intent_knn.py
git commit -m "feat(intent): score routes by nearest canonical example"
```

---

### Task 3: Canonical and eval loaders

**Files:**
- Modify: `src/model/intent_data.py`
- Test: `tests/unit/test_intent_data.py`

**Interfaces:**
- Consumes: `validate_modules`, `INTENT_LABELS` from Task 1; `CanonicalExample` from Task 2.
- Produces: `load_canonical_examples(path: Path) -> tuple[CanonicalExample, ...]`; `IntentEvalQuery` gains `modules: tuple[str, ...] = ()`; `load_intent_eval_queries` reads and validates `modules` when present.
- Task 4 calls `load_canonical_examples`; Task 7 reads `IntentEvalQuery.modules`.

`IntentExample` and `load_intent_examples` are **not** modified — `data/intent_examples.json` keeps its current schema and stays the curation source.

- [ ] **Step 1: Write the failing tests**

Add to `tests/unit/test_intent_data.py`:

```python
def _canonical_records():
    return [
        {
            "id": "canon-001",
            "text": "what is the current price of bitcoin",
            "route": "search",
            "modules": ["current_info", "lookup_fact"],
        },
        {
            "id": "canon-002",
            "text": "summarize the Q3 earnings report",
            "route": "chat",
            "modules": ["summarize"],
        },
    ]


def _write_json(tmp_path, name, payload):
    path = tmp_path / name
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_canonical_examples_round_trip_with_multiple_modules(tmp_path):
    path = _write_json(tmp_path, "canonical.json", _canonical_records())

    examples = load_canonical_examples(path)

    assert len(examples) == 2
    assert examples[0].route == "search"
    assert examples[0].modules == ("current_info", "lookup_fact")
    assert examples[1].modules == ("summarize",)


def test_canonical_rejects_a_module_from_another_route(tmp_path):
    records = _canonical_records()
    records[0]["modules"] = ["current_info", "summarize"]
    path = _write_json(tmp_path, "canonical.json", records)

    with pytest.raises(ValueError, match="summarize"):
        load_canonical_examples(path)


def test_canonical_rejects_an_empty_module_list(tmp_path):
    records = _canonical_records()
    records[0]["modules"] = []
    path = _write_json(tmp_path, "canonical.json", records)

    with pytest.raises(ValueError, match="at least one"):
        load_canonical_examples(path)


def test_canonical_rejects_a_missing_modules_field(tmp_path):
    records = _canonical_records()
    del records[0]["modules"]
    path = _write_json(tmp_path, "canonical.json", records)

    with pytest.raises(ValueError, match="modules"):
        load_canonical_examples(path)


def test_canonical_rejects_an_unknown_route(tmp_path):
    records = _canonical_records()
    records[0]["route"] = "browse"
    path = _write_json(tmp_path, "canonical.json", records)

    with pytest.raises(ValueError, match="browse"):
        load_canonical_examples(path)


def test_canonical_rejects_duplicate_ids(tmp_path):
    records = _canonical_records()
    records[1]["id"] = records[0]["id"]
    path = _write_json(tmp_path, "canonical.json", records)

    with pytest.raises(ValueError, match="Duplicate"):
        load_canonical_examples(path)


def test_canonical_rejects_duplicate_text(tmp_path):
    """Duplicated text silently doubles a point's pull on every query."""
    records = _canonical_records()
    records[1]["text"] = records[0]["text"].upper()
    path = _write_json(tmp_path, "canonical.json", records)

    with pytest.raises(ValueError, match="[Dd]uplicate"):
        load_canonical_examples(path)


def test_canonical_rejects_an_empty_file(tmp_path):
    path = _write_json(tmp_path, "canonical.json", [])

    with pytest.raises(ValueError, match="no records"):
        load_canonical_examples(path)


def test_eval_queries_carry_modules_when_present(tmp_path):
    path = _write_json(
        tmp_path,
        "eval.json",
        [
            {
                "id": "eval-001",
                "text": "find the Q3 earnings report",
                "label": "search",
                "modules": ["lookup_document"],
            }
        ],
    )

    queries = load_intent_eval_queries(path)

    assert queries[0].modules == ("lookup_document",)


def test_eval_queries_without_modules_still_load(tmp_path):
    """The legacy 30 queries predate modules; loading must not break."""
    path = _write_json(
        tmp_path,
        "eval.json",
        [{"id": "eval-001", "text": "find the report", "label": "search"}],
    )

    assert load_intent_eval_queries(path)[0].modules == ()


def test_eval_queries_reject_a_module_from_another_route(tmp_path):
    path = _write_json(
        tmp_path,
        "eval.json",
        [
            {
                "id": "eval-001",
                "text": "find the report",
                "label": "search",
                "modules": ["summarize"],
            }
        ],
    )

    with pytest.raises(ValueError, match="summarize"):
        load_intent_eval_queries(path)
```

Add `import json` and extend the existing import of `src.model.intent_data` to include `load_canonical_examples` at the top of the test file.

- [ ] **Step 2: Run the tests to verify they fail**

Run: `pytest tests/unit/test_intent_data.py -v`
Expected: FAIL — `ImportError: cannot import name 'load_canonical_examples'`.

- [ ] **Step 3: Implement the loaders**

In `src/model/intent_data.py`, replace the import of `INTENT_LABELS`:

```python
from .intent_knn import CanonicalExample
from .intent_taxonomy import INTENT_LABELS, validate_modules
```

Add `modules` to `IntentEvalQuery`:

```python
@dataclass(frozen=True)
class IntentEvalQuery:
    """One hand-authored request used only to measure realistic accuracy."""

    id: str
    text: str
    label: str
    modules: tuple[str, ...] = ()
```

Add the canonical loader:

```python
def load_canonical_examples(path: Path) -> tuple[CanonicalExample, ...]:
    """Load and validate the curated examples that make up the routing index.

    These examples *are* the model, so validation is strict: a mislabeled or
    duplicated record changes routing directly, with no training run in between
    to average the mistake away.
    """
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid canonical example JSON in {path}: {exc.msg}") from exc

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
```

In `load_intent_eval_queries`, after the label check, read modules and pass them through:

```python
        modules = _modules(record, index, label, kind=kind, required=False)
        ids.add(query_id)
        queries.append(
            IntentEvalQuery(id=query_id, text=text, label=label, modules=modules)
        )
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `pytest tests/unit/test_intent_data.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
ruff check . --fix && ruff format .
git add src/model/intent_data.py tests/unit/test_intent_data.py
git commit -m "feat(intent): load canonical examples and module-labeled eval queries"
```

---

### Task 4: Encoder seam and index build

**Files:**
- Create: `src/model/intent_encoder.py`
- Create: `src/model/intent_index_cli.py`
- Test: `tests/unit/test_intent_encoder.py` (new), `tests/unit/test_intent_index_cli.py` (new)

**Interfaces:**
- Consumes: `load_canonical_examples` (Task 3), `IntentIndex`, `INDEX_FILENAME` (Task 2).
- Produces:
  - `intent_encoder.DEFAULT_ENCODER = "sentence-transformers/all-MiniLM-L6-v2"`; `encode_texts(texts: Sequence[str], *, model_name: str = DEFAULT_ENCODER) -> np.ndarray` returning L2-normalized `float32` of shape `(len(texts), 384)`; `encoder_dimension(model_name) -> int`.
  - `intent_index_cli.build_index(canonical_path: Path, output_dir: Path, *, model_name: str, encode=encode_texts) -> IntentIndex`; `check_leakage(index: IntentIndex, eval_texts: Sequence[str], eval_vectors: np.ndarray) -> list[str]`; `main(argv) -> int` with subcommands `seed`, `build`, `evaluate`.
  - `LEAKAGE_COSINE = 0.95`
- Task 5 loads the index this writes; Task 7 uses `evaluate`.

The `encode` parameter on `build_index` exists so the build is testable without an encoder. It is not a production knob.

**Two deliberate forward references.** `main`'s `seed` branch imports `intent_seed` (Task 5) and its `evaluate` branch imports `intent_index_eval` (Task 8). Both imports are **function-local**, so the module imports cleanly and this task's tests — which exercise only `build` — pass before either module exists. Running `seed` or `evaluate` before their task lands raises `ModuleNotFoundError`; that is expected and is why the plan orders them this way rather than stubbing.

- [ ] **Step 1: Write the failing tests**

Create `tests/unit/test_intent_encoder.py`:

```python
"""The encoder seam. Real encoding is covered where the model is installed."""

import numpy as np
import pytest

from src.model.intent_encoder import DEFAULT_ENCODER


def test_default_encoder_is_the_minilm_the_index_is_built_with():
    assert DEFAULT_ENCODER == "sentence-transformers/all-MiniLM-L6-v2"


def test_encode_returns_normalized_float32_rows():
    pytest.importorskip("sentence_transformers")
    from src.model.intent_encoder import encode_texts

    vectors = encode_texts(["find the runbook", "send an email to the team"])

    assert vectors.shape == (2, 384)
    assert vectors.dtype == np.float32
    np.testing.assert_allclose(np.linalg.norm(vectors, axis=1), 1.0, atol=1e-5)


def test_encoding_is_deterministic():
    """Routing must not flip between requests for a fixed query."""
    pytest.importorskip("sentence_transformers")
    from src.model.intent_encoder import encode_texts

    first = encode_texts(["find the runbook"])
    second = encode_texts(["find the runbook"])

    np.testing.assert_allclose(first, second, atol=1e-6)


def test_word_order_changes_the_vector():
    """The whole point of the encoder: a bag of embeddings could not do this."""
    pytest.importorskip("sentence_transformers")
    from src.model.intent_encoder import encode_texts

    vectors = encode_texts(
        ["how to send an email", "send an email to how"]
    )

    assert float(vectors[0] @ vectors[1]) < 0.99
```

Create `tests/unit/test_intent_index_cli.py`:

```python
import json
from pathlib import Path

import numpy as np
import pytest

from src.model import intent_index_cli
from src.model.intent_knn import INDEX_FILENAME, IntentIndex

_AXIS = {"search": 0, "chat": 1, "tool": 2}
_MODULE = {"search": "lookup_fact", "chat": "explain", "tool": "schedule"}


def _fake_encode(texts, *, model_name="test-encoder"):
    """Encode by the route word each text starts with, onto a basis axis."""
    rows = []
    for text in texts:
        axis = _AXIS.get(text.split()[0], 0)
        rows.append(np.eye(3, dtype=np.float32)[axis])
    return np.stack(rows)


def _canonical(tmp_path: Path, count: int = 3) -> Path:
    records = [
        {
            "id": f"{route}-{position}",
            "text": f"{route} canonical text {position}",
            "route": route,
            "modules": [_MODULE[route]],
        }
        for route in _AXIS
        for position in range(count)
    ]
    path = tmp_path / "canonical.json"
    path.write_text(json.dumps(records), encoding="utf-8")
    return path


def test_build_writes_a_loadable_index(tmp_path):
    output = tmp_path / "index"

    index = intent_index_cli.build_index(
        _canonical(tmp_path), output, model_name="test-encoder", encode=_fake_encode
    )

    assert index.size == 9
    reloaded = IntentIndex.load(output / INDEX_FILENAME)
    assert reloaded.size == 9
    assert reloaded.encoder == "test-encoder"


def test_build_fingerprints_the_canonical_file(tmp_path):
    """A stale index against an edited canonical file must be detectable."""
    canonical = _canonical(tmp_path)
    first = intent_index_cli.build_index(
        canonical, tmp_path / "a", model_name="test-encoder", encode=_fake_encode
    )

    records = json.loads(canonical.read_text(encoding="utf-8"))
    records[0]["text"] = "search canonical text edited"
    canonical.write_text(json.dumps(records), encoding="utf-8")
    second = intent_index_cli.build_index(
        canonical, tmp_path / "b", model_name="test-encoder", encode=_fake_encode
    )

    assert first.fingerprint != second.fingerprint


def test_leakage_check_flags_an_eval_query_identical_to_a_canonical_example(tmp_path):
    index = intent_index_cli.build_index(
        _canonical(tmp_path), tmp_path / "index",
        model_name="test-encoder", encode=_fake_encode,
    )
    texts = ["search canonical text 0"]

    leaks = intent_index_cli.check_leakage(index, texts, _fake_encode(texts))

    assert leaks and "search canonical text 0" in leaks[0]


def test_leakage_check_flags_a_near_duplicate_above_the_cosine_bar(tmp_path):
    """With kNN the index IS the model, so overlap manufactures accuracy."""
    index = intent_index_cli.build_index(
        _canonical(tmp_path), tmp_path / "index",
        model_name="test-encoder", encode=_fake_encode,
    )
    texts = ["search something else entirely"]

    leaks = intent_index_cli.check_leakage(index, texts, _fake_encode(texts))

    assert leaks


def test_leakage_check_passes_for_a_genuinely_distinct_query(tmp_path):
    index = intent_index_cli.build_index(
        _canonical(tmp_path), tmp_path / "index",
        model_name="test-encoder", encode=_fake_encode,
    )
    texts = ["unrelated"]
    vectors = np.array([[0.0, 0.6, 0.8]], dtype=np.float32)

    assert intent_index_cli.check_leakage(index, texts, vectors) == []


def test_build_command_reports_low_support_modules(tmp_path, capsys, monkeypatch):
    monkeypatch.setattr(intent_index_cli, "encode_texts", _fake_encode)
    output = tmp_path / "index"

    exit_code = intent_index_cli.main(
        ["build", "--canonical", str(_canonical(tmp_path)), "--output", str(output)]
    )

    assert exit_code == 0
    assert "low support" in capsys.readouterr().out.lower()


def test_build_command_reports_a_missing_canonical_file(tmp_path, capsys):
    exit_code = intent_index_cli.main(
        ["build", "--canonical", str(tmp_path / "missing.json"),
         "--output", str(tmp_path / "index")]
    )

    assert exit_code == 1
    assert "missing.json" in capsys.readouterr().err
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `pytest tests/unit/test_intent_encoder.py tests/unit/test_intent_index_cli.py -v`
Expected: FAIL at collection — `ModuleNotFoundError: No module named 'src.model.intent_encoder'`.

- [ ] **Step 3: Implement the encoder seam**

Create `src/model/intent_encoder.py`:

```python
"""The only place a sentence encoder is loaded.

The import is function-local on purpose. Every other intent module must stay
importable in a CI job that installs neither torch nor sentence-transformers,
and this repo has twice shipped collection failures from unguarded imports
(#356, re-fixed in #418). Keeping the dependency behind one function is what
makes the rest of the routing path testable without it.
"""

from __future__ import annotations

from collections.abc import Sequence
from functools import lru_cache

import numpy as np

DEFAULT_ENCODER = "sentence-transformers/all-MiniLM-L6-v2"


@lru_cache(maxsize=2)
def _model(model_name: str):
    """Load and cache the encoder. Loading costs seconds; encoding costs ms."""
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(model_name, device="cpu")


def encode_texts(
    texts: Sequence[str], *, model_name: str = DEFAULT_ENCODER
) -> np.ndarray:
    """Encode *texts* to L2-normalized float32 rows.

    Normalizing here means every consumer can treat a dot product as a cosine,
    and the index constructor can reject anything that is not normalized.
    """
    vectors = _model(model_name).encode(
        list(texts),
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    return np.asarray(vectors, dtype=np.float32)


def encoder_dimension(model_name: str = DEFAULT_ENCODER) -> int:
    """The encoder's output width."""
    return int(_model(model_name).get_sentence_embedding_dimension())
```

- [ ] **Step 4: Implement the build CLI**

Create `src/model/intent_index_cli.py`:

```python
"""Build and evaluate the canonical routing index.

Three commands. ``seed`` proposes module labels for the existing training
examples so the canonical set can be curated rather than authored from nothing.
``build`` encodes the canonical set into an index. ``evaluate`` scores an index
against the held-out query sets.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from .intent_data import load_canonical_examples
from .intent_encoder import DEFAULT_ENCODER, encode_texts
from .intent_knn import INDEX_FILENAME, IntentIndex

LEAKAGE_COSINE = 0.95


def _fingerprint(path: Path) -> str:
    """Hash the canonical file so a stale index is detectable."""
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return f"sha256:{digest}"


def build_index(
    canonical_path: Path,
    output_dir: Path,
    *,
    model_name: str = DEFAULT_ENCODER,
    encode=encode_texts,
) -> IntentIndex:
    """Encode the canonical examples and write the index.

    ``encode`` is injectable so the build is testable without an encoder; it is
    not a production knob.
    """
    examples = load_canonical_examples(canonical_path)
    vectors = encode([example.text for example in examples], model_name=model_name)
    index = IntentIndex(
        examples=examples,
        vectors=vectors,
        encoder=model_name,
        fingerprint=_fingerprint(canonical_path),
    )
    index.save(output_dir / INDEX_FILENAME)
    return index


def check_leakage(
    index: IntentIndex, eval_texts: Sequence[str], eval_vectors: np.ndarray
) -> list[str]:
    """Report evaluation queries that duplicate a canonical example.

    With nearest-neighbor routing the index *is* the model, so an eval query
    that also sits in the index scores against itself and manufactures accuracy
    that no user would ever see.
    """
    canonical_texts = [example.text for example in index.examples]
    normalized = {text.casefold().strip(): text for text in canonical_texts}
    similarities = eval_vectors @ index._vectors.T  # noqa: SLF001 — same package
    leaks: list[str] = []
    for position, text in enumerate(eval_texts):
        exact = normalized.get(text.casefold().strip())
        if exact is not None:
            leaks.append(f"{text!r} exactly matches canonical {exact!r}")
            continue
        best = int(np.argmax(similarities[position]))
        score = float(similarities[position][best])
        if score >= LEAKAGE_COSINE:
            leaks.append(
                f"{text!r} is {score:.3f} similar to canonical "
                f"{canonical_texts[best]!r}"
            )
    return leaks


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    seed = subparsers.add_parser(
        "seed", help="propose module labels for existing training examples"
    )
    seed.add_argument("--examples", required=True, type=Path)
    seed.add_argument("--output", required=True, type=Path)

    build = subparsers.add_parser("build", help="encode the canonical set")
    build.add_argument("--canonical", required=True, type=Path)
    build.add_argument("--output", required=True, type=Path)
    build.add_argument("--model", default=DEFAULT_ENCODER)

    evaluate = subparsers.add_parser("evaluate", help="score an index")
    evaluate.add_argument("--index", required=True, type=Path)
    evaluate.add_argument("--eval-queries", required=True, type=Path)
    evaluate.add_argument("--hard-queries", type=Path)
    evaluate.add_argument("--out-of-scope", type=Path)
    evaluate.add_argument("--canonical", required=True, type=Path)
    evaluate.add_argument("--output", required=True, type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run a seed, build, or evaluate command."""
    try:
        args = _build_parser().parse_args(argv)
        if args.command == "seed":
            from .intent_seed import write_seed_canonical

            count = write_seed_canonical(args.examples, args.output)
            print(f"wrote {count} proposed canonical examples to {args.output}")
            return 0
        if args.command == "build":
            index = build_index(args.canonical, args.output, model_name=args.model)
            print(f"built index of {index.size} examples at {args.output}")
            low = index.low_support_modules()
            print(f"low support modules ({len(low)}): {', '.join(low) or 'none'}")
            return 0

        from .intent_index_eval import run_index_evaluation

        report = run_index_evaluation(
            index_path=args.index,
            eval_queries_path=args.eval_queries,
            hard_queries_path=args.hard_queries,
            out_of_scope_path=args.out_of_scope,
            canonical_path=args.canonical,
            output_path=args.output,
        )
        print(json.dumps(report["headline"], indent=2))
        return 0
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"intent index command failed: {exc}", file=sys.stderr)
        return 1
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `pytest tests/unit/test_intent_encoder.py tests/unit/test_intent_index_cli.py -v`
Expected: PASS. The four encoder tests skip unless sentence-transformers is installed.

- [ ] **Step 6: Commit**

```bash
ruff check . --fix && ruff format .
git add src/model/intent_encoder.py src/model/intent_index_cli.py \
  tests/unit/test_intent_encoder.py tests/unit/test_intent_index_cli.py
git commit -m "feat(intent): add the encoder seam and index build command"
```

---

### Task 5: Curate the canonical set

**Files:**
- Create: `src/model/intent_seed.py`
- Create: `data/intent_canonical.json` (tracked with `git add -f`)
- Test: `tests/unit/test_intent_seed.py` (new)

**Interfaces:**
- Consumes: `load_intent_examples` from `intent_data`; the taxonomy from Task 1.
- Produces: `propose_modules(text: str, route: str) -> tuple[str, ...]`; `write_seed_canonical(examples_path: Path, output_path: Path) -> int`. Called by `intent_index_cli.main` under the `seed` command.

The seeder proposes; a human disposes. It exists so curating ~270 examples is a review pass over machine proposals rather than 270 acts of authorship, and it keys off the same regex cues the taxonomy was derived from — so its proposals agree with the router by construction.

- [ ] **Step 1: Write the failing tests**

Create `tests/unit/test_intent_seed.py`:

```python
import json
from pathlib import Path

from src.model.intent_seed import propose_modules, write_seed_canonical


def test_imperative_send_proposes_the_send_module():
    assert "send" in propose_modules("send the runbook to the team", "tool")


def test_currency_cue_proposes_current_info():
    assert "current_info" in propose_modules(
        "what is the current price of bitcoin", "search"
    )


def test_document_noun_proposes_document_lookup():
    assert "lookup_document" in propose_modules(
        "find the onboarding doc for new engineers", "search"
    )


def test_summarize_verb_proposes_the_summarize_module():
    assert propose_modules("summarize the Q3 earnings report", "chat") == ("summarize",)


def test_a_proposal_may_carry_several_modules():
    """Real requests carry more than one intent at once."""
    proposed = propose_modules(
        "explain why the reranker got slower and compare it to the old one", "chat"
    )

    assert set(proposed) == {"explain", "compare"}


def test_every_proposal_is_valid_for_its_route():
    from src.model.intent_taxonomy import modules_for_route

    for route in ("chat", "search", "tool"):
        for text in ("something entirely unmatched by any cue", "the thing"):
            proposed = propose_modules(text, route)
            assert proposed, (route, text)
            assert set(proposed) <= set(modules_for_route(route))


def test_write_seed_produces_a_file_the_canonical_loader_accepts(tmp_path: Path):
    from src.model.intent_data import load_canonical_examples

    examples = [
        {"id": f"e-{i}", "text": t, "label": lbl, "source": "s"}
        for i, (t, lbl) in enumerate(
            [
                ("find the onboarding doc", "search"),
                ("summarize the outage postmortem", "chat"),
                ("send the runbook to the team", "tool"),
            ]
        )
    ]
    source = tmp_path / "examples.json"
    source.write_text(json.dumps(examples), encoding="utf-8")
    output = tmp_path / "canonical.json"

    count = write_seed_canonical(source, output)

    assert count == 3
    assert len(load_canonical_examples(output)) == 3
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `pytest tests/unit/test_intent_seed.py -v`
Expected: FAIL at collection — `ModuleNotFoundError: No module named 'src.model.intent_seed'`.

- [ ] **Step 3: Implement the seeder**

Create `src/model/intent_seed.py`:

```python
"""Propose module labels for existing training examples.

Curating ~270 canonical examples is a review pass over machine proposals, not
270 acts of authorship. The cues below are the same ones
``src/internal/servers/web/intent_routing.py`` already routes on, which is why
the taxonomy has these fourteen modules and no others — so a proposal agrees
with the router by construction, and a disagreement is worth looking at.

Every proposal is a draft. The committed canonical file is the reviewed result.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from .intent_data import load_intent_examples
from .intent_taxonomy import modules_for_route

_CUES: dict[str, tuple[re.Pattern[str], ...]] = {
    "current_info": (
        re.compile(
            r"\b(latest|current|recent|news|price|stock|weather|today|now|"
            r"this week|right now)\b",
            re.IGNORECASE,
        ),
    ),
    "lookup_document": (
        re.compile(
            r"\b(doc|document|report|runbook|postmortem|checklist|spec|readme|"
            r"guide|notes|deck|spreadsheet|page|wiki|policy)\b",
            re.IGNORECASE,
        ),
    ),
    "lookup_fact": (
        re.compile(
            r"\b(which|who|when|where|how many|how much|what is the|value of|"
            r"number|setting|config|version)\b",
            re.IGNORECASE,
        ),
    ),
    "summarize": (re.compile(r"\b(summari[sz]e|tl;?dr|recap|overview of)\b", re.I),),
    "compare": (
        re.compile(r"\b(compare|versus|vs\.?|difference between|better than)\b", re.I),
    ),
    "generate": (
        re.compile(
            r"\b(write|draft|translate|rephrase|reword|rewrite|brainstorm|"
            r"compose|generate)\b",
            re.IGNORECASE,
        ),
    ),
    "converse": (
        re.compile(r"\b(hello|hi there|thanks|thank you|joke|poem|haiku)\b", re.I),
    ),
    "explain": (
        re.compile(r"\b(explain|why|how does|how do|what is|describe|tell me about)\b", re.I),
    ),
    "create": (re.compile(r"\b(create|open|file|add|new)\b", re.IGNORECASE),),
    "send": (re.compile(r"\b(send|email|notify|post|message|share)\b", re.IGNORECASE),),
    "schedule": (
        re.compile(r"\b(schedule|book|remind|calendar|meeting|invite)\b", re.I),
    ),
    "modify": (
        re.compile(r"\b(update|change|delete|remove|cancel|close|rename|edit)\b", re.I),
    ),
    "execute": (
        re.compile(r"\b(run|execute|deploy|trigger|invoke|rerun|kick off)\b", re.I),
    ),
}

# Used when no cue fires, so every example still gets a valid starting label.
_DEFAULT_MODULE = {
    "search": "lookup_fact",
    "chat": "explain",
    "tool": "execute",
}


def propose_modules(text: str, route: str) -> tuple[str, ...]:
    """Propose one or more modules for *text* within *route*.

    Multi-label by design: "compare the current prices of BTC and ETH" is
    genuinely both a comparison and a request for current information.
    """
    candidates = tuple(
        module
        for module in modules_for_route(route)
        if module in _CUES and any(cue.search(text) for cue in _CUES[module])
    )
    return candidates or (_DEFAULT_MODULE[route],)


def write_seed_canonical(examples_path: Path, output_path: Path) -> int:
    """Write a proposed canonical file from labeled training examples."""
    examples = load_intent_examples(examples_path)
    records = [
        {
            "id": f"canon-{position:03d}",
            "text": example.text,
            "route": example.label,
            "modules": list(propose_modules(example.text, example.label)),
        }
        for position, example in enumerate(examples, start=1)
    ]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(records, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return len(records)
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `pytest tests/unit/test_intent_seed.py -v`
Expected: PASS, all 7.

- [ ] **Step 5: Generate the draft and curate it down to ~270**

```bash
python -m src.model.intent_index_cli seed \
  --examples data/intent_examples.json \
  --output data/intent_canonical.draft.json
python -c "
import json
from collections import Counter
records = json.load(open('data/intent_canonical.draft.json'))
print('drafted', len(records))
print('routes', Counter(r['route'] for r in records))
print('modules', Counter(m for r in records for m in r['modules']))
"
```

Now curate the draft into `data/intent_canonical.json` by hand. The targets:

| property | target |
|---|---|
| total records | 260–280 |
| per route | ~90 each |
| per semantic module | ≥ 20, ideally ~33 |
| `bare_entity` | ~10 and no more |
| module assignments per record | ~1.6 average |

Curation rules, in order:

1. **Drop generated-looking records.** The 520 came from a corpus generator; anything that reads like a template rather than something a person would type is the first to go.
2. **Correct the proposals.** The seeder is regex-based and will be wrong — especially `lookup_fact` versus `lookup_document` ("what was revenue in the Q3 report" is a fact lookup even though "report" appears) and `explain` versus `summarize`.
3. **Add the second module where it is genuinely there.** A record is multi-label when two intents are both present, not when two cues happen to match.
4. **Cap `bare_entity` at ~10.** `_is_bare_lookup` routes these at cascade step 2, before this model runs, so more examples buy nothing and dilute the search route.
5. **Keep phrasing varied.** Near-duplicates get rejected by the loader, but merely similar phrasings still cluster and waste index capacity.

Verify the result:

```bash
python -c "
import json
from collections import Counter
from pathlib import Path
from src.model.intent_data import load_canonical_examples
examples = load_canonical_examples(Path('data/intent_canonical.json'))
routes = Counter(e.route for e in examples)
modules = Counter(m for e in examples for m in e.modules)
print('total', len(examples))
print('routes', dict(routes))
print('modules', dict(sorted(modules.items())))
print('avg modules/record', sum(modules.values()) / len(examples))
thin = [m for m, c in modules.items() if c < 10 and m != 'bare_entity']
print('BELOW SUPPORT:', thin or 'none')
"
```

Every semantic module must be at or above 10. If one is thin, add examples for it from the draft rather than lowering the bar.

- [ ] **Step 6: Commit**

```bash
rm -f data/intent_canonical.draft.json
ruff check . --fix && ruff format .
git add src/model/intent_seed.py tests/unit/test_intent_seed.py
git add -f data/intent_canonical.json   # data/ is gitignored
git commit -m "feat(intent): curate the canonical routing example set"
```

---

### Task 6: Serve the index behind predict_route

**Files:**
- Modify: `src/internal/servers/web/ml_intent.py`
- Modify: `src/internal/configs/app_configs.py:161-168,192-195,281-284`
- Test: `tests/unit/test_ml_intent.py` (new), `tests/unit/test_intent_routing.py`

**Interfaces:**
- Consumes: `IntentIndex.load`, `KnnDecision` (Task 2); `encode_texts` (Task 4).
- Produces: `IntentModelDecision` gains `modules: tuple[str, ...] = ()` and `composite: bool = False`. `predict_route(query, *, settings) -> IntentModelDecision | None` keeps its signature. `load_intent_index(settings) -> IntentIndex | None` replaces `load_intent_model`.
- Settings: `intent_index_path: Path | None`, `intent_min_route_margin: float = 0.05`, `intent_min_module_score: float = 0.45`. `intent_model_path` is removed; `intent_model_min_confidence` is kept and becomes the cosine bar.

**Abstention asymmetry, deliberate:** a confidence abstention returns the decision and lets `route_request`'s existing `confidence < threshold` comparison handle it, unchanged. A margin abstention returns `None` after recording its own `intent_model` capture stage, because `route_request` has no margin concept and must not gain one. Both end at the LLM classifier.

- [ ] **Step 1: Write the failing tests**

Create `tests/unit/test_ml_intent.py`:

```python
"""Serving adapter behavior, exercised without an encoder."""

from pathlib import Path

import numpy as np
import pytest

from src.internal.configs import AppSettings
from src.internal.servers.web import ml_intent
from src.internal.servers.web.intent_routing import RouteStrategy
from src.model.intent_knn import INDEX_FILENAME, CanonicalExample, IntentIndex

_AXIS = {"search": 0, "chat": 1, "tool": 2}
_MODULE = {"search": "lookup_fact", "chat": "explain", "tool": "schedule"}


def _write_index(tmp_path: Path) -> Path:
    examples, rows = [], []
    for route, axis in _AXIS.items():
        for position in range(12):
            examples.append(
                CanonicalExample(
                    f"{route}-{position}", f"{route} {position}", route,
                    (_MODULE[route],),
                )
            )
            rows.append(np.eye(3, dtype=np.float32)[axis])
    directory = tmp_path / "index"
    IntentIndex(examples, np.stack(rows), "test-encoder", "sha256:x").save(
        directory / INDEX_FILENAME
    )
    return directory


def _settings(tmp_path: Path, **overrides) -> AppSettings:
    return AppSettings(
        intent_index_path=_write_index(tmp_path),
        intent_model_min_confidence=0.5,
        intent_min_route_margin=0.05,
        intent_min_module_score=0.4,
        **overrides,
    )


@pytest.fixture(autouse=True)
def _clear_cache():
    ml_intent._INTENT_INDEXES.clear()
    yield
    ml_intent._INTENT_INDEXES.clear()


def _on_axis(route: str, monkeypatch):
    vector = np.eye(3, dtype=np.float32)[_AXIS[route]][None, :]
    monkeypatch.setattr(ml_intent, "encode_texts", lambda texts: vector)


def test_confident_query_returns_its_route_and_modules(tmp_path, monkeypatch):
    _on_axis("search", monkeypatch)

    decision = ml_intent.predict_route("anything", settings=_settings(tmp_path))

    assert decision is not None
    assert decision.strategy is RouteStrategy.SEARCH
    assert decision.confidence == pytest.approx(1.0)
    assert decision.modules == ("lookup_fact",)
    assert decision.composite is False
    assert decision.latency_ms >= 0.0


def test_low_confidence_is_returned_for_route_request_to_abstain_on(
    tmp_path, monkeypatch
):
    """route_request already compares confidence to threshold; do not duplicate."""
    monkeypatch.setattr(
        ml_intent, "encode_texts",
        lambda texts: np.array([[0.577, 0.577, 0.577]], dtype=np.float32),
    )

    decision = ml_intent.predict_route("anything", settings=_settings(tmp_path))

    assert decision is not None
    assert decision.confidence < decision.threshold


def test_low_margin_defers_by_returning_none(tmp_path, monkeypatch):
    """route_request has no margin concept, so this abstention happens here."""
    monkeypatch.setattr(
        ml_intent, "encode_texts",
        lambda texts: np.array([[0.707, 0.707, 0.0]], dtype=np.float32),
    )

    assert ml_intent.predict_route("anything", settings=_settings(tmp_path)) is None


def test_missing_index_path_defers_without_raising(tmp_path):
    settings = AppSettings(intent_index_path=None)

    assert ml_intent.predict_route("anything", settings=settings) is None


def test_unreadable_index_defers_and_is_not_retried(tmp_path, monkeypatch, caplog):
    settings = AppSettings(intent_index_path=tmp_path / "absent")

    assert ml_intent.predict_route("anything", settings=settings) is None
    assert ml_intent.predict_route("anything", settings=settings) is None


def test_encoder_failure_defers_rather_than_failing_the_request(tmp_path, monkeypatch):
    def _boom(texts):
        raise RuntimeError("no model")

    monkeypatch.setattr(ml_intent, "encode_texts", _boom)

    assert ml_intent.predict_route("anything", settings=_settings(tmp_path)) is None


def test_index_is_loaded_once_and_cached(tmp_path, monkeypatch):
    _on_axis("search", monkeypatch)
    settings = _settings(tmp_path)
    loads = {"count": 0}
    original = IntentIndex.load

    def _counting_load(path):
        loads["count"] += 1
        return original(path)

    monkeypatch.setattr(IntentIndex, "load", staticmethod(_counting_load))

    ml_intent.predict_route("a", settings=settings)
    ml_intent.predict_route("b", settings=settings)

    assert loads["count"] == 1


def test_composite_query_is_flagged(tmp_path, monkeypatch):
    monkeypatch.setattr(
        ml_intent, "encode_texts",
        lambda texts: np.array([[0.71, 0.0, 0.70]], dtype=np.float32),
    )
    settings = _settings(tmp_path)
    object.__setattr__(settings, "intent_min_route_margin", 0.0)

    decision = ml_intent.predict_route("anything", settings=settings)

    assert decision is not None
    assert decision.composite is True
```

Add to `tests/unit/test_intent_routing.py`:

```python
def test_route_request_is_unchanged_by_a_none_returning_model(monkeypatch):
    """A margin abstention must look exactly like having no model at all."""
    from src.internal.servers.web import intent_routing

    monkeypatch.setattr(intent_routing, "predict_route", lambda q, settings=None: None)
    calls = []

    class _LLM:
        def complete(self, messages, temperature=0.0):
            calls.append(messages)
            return "search"

    decision = intent_routing.route_request(
        "where does the reranker timeout live",
        llm=_LLM(),
        explicit_source=False,
    )

    assert decision.strategy is intent_routing.RouteStrategy.SEARCH
    assert calls, "the LLM classifier must still be consulted"
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `pytest tests/unit/test_ml_intent.py -v`
Expected: FAIL — `TypeError: AppSettings.__init__() got an unexpected keyword argument 'intent_index_path'`.

- [ ] **Step 3: Update the settings**

In `src/internal/configs/app_configs.py`, replace the `intent_model_path` field with:

```python
    intent_index_path: Path | None = None
    intent_model_min_confidence: float = 0.6
    intent_min_route_margin: float = 0.05
    intent_min_module_score: float = 0.45
```

Extend the existing validation block that checks `intent_model_min_confidence` to cover both new floats:

```python
        for name in (
            "intent_model_min_confidence",
            "intent_min_route_margin",
            "intent_min_module_score",
        ):
            value = getattr(self, name)
            if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be a probability between 0 and 1")
```

In the loader, read the new environment variables and rename the path one:

```python
    intent_min_route_margin = get_env_float(
        "AGENTIC_SEARCH_INTENT_MIN_ROUTE_MARGIN", 0.05
    )
    intent_min_module_score = get_env_float(
        "AGENTIC_SEARCH_INTENT_MIN_MODULE_SCORE", 0.45
    )
    intent_index_path_value = get_env_str("AGENTIC_SEARCH_INTENT_INDEX_PATH", "")
```

and in the `AppSettings(...)` construction:

```python
        intent_index_path=(
            Path(intent_index_path_value) if intent_index_path_value else None
        ),
        intent_model_min_confidence=intent_model_min_confidence,
        intent_min_route_margin=intent_min_route_margin,
        intent_min_module_score=intent_min_module_score,
```

Keep the existing env var name for `intent_model_min_confidence` — only its units change, and that is documented in Task 9.

- [ ] **Step 4: Rewrite the serving adapter**

Replace the body of `src/internal/servers/web/ml_intent.py`:

```python
"""Lazy adapter: canonical-example index -> RouteStrategy for route_query."""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

from src.internal.configs import AppSettings, load_app_settings
from src.internal.servers.web import request_capture as _capture
from src.internal.servers.web.intent_routing import RouteStrategy
from src.model.intent_encoder import encode_texts

logger = logging.getLogger(__name__)

_INTENT_INDEXES: dict[Path, object | None] = {}

_ROUTE_VALUES = {s.value for s in RouteStrategy}


@dataclass(frozen=True)
class IntentModelDecision:
    """A valid route prediction with serving diagnostics."""

    strategy: RouteStrategy
    confidence: float
    threshold: float
    latency_ms: float
    modules: tuple[str, ...] = ()
    composite: bool = False


def intent_min_confidence(settings: AppSettings | None = None) -> float:
    """Return the configured similarity threshold.

    This is a cosine similarity, not a softmax probability. The two live on
    different scales, so a value carried over from the previous model is
    meaningless here.
    """
    return (settings or load_app_settings()).intent_model_min_confidence


def load_intent_index(settings: AppSettings | None = None) -> object | None:
    """Load the configured index lazily, caching by resolved path.

    Loading is lazy rather than done at app startup: the web TestClient suite
    already hangs on lifespan model loads, and routing degrades safely to the
    LLM classifier while the encoder warms.
    """
    resolved = settings or load_app_settings()
    configured = resolved.intent_index_path
    if configured is None:
        return None
    directory = configured.resolve()
    if directory in _INTENT_INDEXES:
        return _INTENT_INDEXES[directory]
    try:
        from src.model.intent_knn import INDEX_FILENAME, IntentIndex

        index = IntentIndex.load(directory / INDEX_FILENAME)
    except Exception:
        logger.exception("intent-index: load failed — similarity routing disabled")
        _INTENT_INDEXES[directory] = None
    else:
        low_support = index.low_support_modules()
        if low_support:
            logger.warning(
                "intent-index: modules below support, not emitted: %s",
                ", ".join(low_support),
            )
        _INTENT_INDEXES[directory] = index
    return _INTENT_INDEXES[directory]


def predict_route(
    query: str, *, settings: AppSettings | None = None
) -> IntentModelDecision | None:
    """Return a supported route decision, or None to defer to the classifier.

    Confidence abstention is *not* handled here: the decision is returned and
    ``route_request`` applies its existing confidence-versus-threshold rule
    unchanged. Margin abstention has no equivalent there, so it returns None
    after recording its own capture stage. Both paths end at the LLM classifier.
    """
    resolved = settings or load_app_settings()
    index = load_intent_index(resolved)
    if index is None:
        return None
    start = perf_counter()
    try:
        vector = encode_texts([query])[0]
        decision = index.decide(
            vector,
            min_confidence=resolved.intent_model_min_confidence,
            min_margin=resolved.intent_min_route_margin,
            min_module_score=resolved.intent_min_module_score,
        )
    except Exception:
        logger.exception("intent-index: predict failed — deferring")
        return None
    latency_ms = (perf_counter() - start) * 1_000

    if decision.route not in _ROUTE_VALUES:
        return None
    confidence = float(decision.confidence)
    if not math.isfinite(confidence) or not 0.0 <= confidence <= 1.0:
        logger.warning("intent-index: invalid confidence — deferring")
        return None

    if decision.abstain_reason == "margin_below_threshold":
        _capture.record_stage(
            "intent_model",
            "evaluation",
            {
                "predicted_intent": decision.route,
                "confidence": confidence,
                "margin": float(decision.margin),
                "abstained": True,
                "fallback_reason": "margin_below_threshold",
                "composite": decision.composite,
                "latency_ms": latency_ms,
            },
        )
        return None

    return IntentModelDecision(
        strategy=RouteStrategy(decision.route),
        confidence=confidence,
        threshold=resolved.intent_model_min_confidence,
        latency_ms=latency_ms,
        modules=decision.modules,
        composite=decision.composite,
    )
```

- [ ] **Step 5: Carry modules and composite into telemetry**

In `src/internal/servers/web/intent_routing.py`, inside `route_request`, extend `model_detail` — this is the only edit to that file, and it adds keys without changing any decision:

```python
        model_detail = {
            "predicted_intent": model_choice.strategy.value,
            "confidence": model_choice.confidence,
            "threshold": model_choice.threshold,
            "abstained": abstained,
            "fallback_reason": "model_below_threshold" if abstained else None,
            "latency_ms": model_choice.latency_ms,
            "modules": list(model_choice.modules),
            "composite": model_choice.composite,
        }
```

- [ ] **Step 6: Run the tests**

Run: `pytest tests/unit/test_ml_intent.py tests/unit/test_intent_routing.py -v`
Expected: PASS.

- [ ] **Step 7: Prove the routing path imports with no encoder installed**

```bash
python - <<'PY'
import sys

class _Block:
    def find_module(self, name, path=None):
        if name.split(".")[0] in {"torch", "transformers", "sentence_transformers"}:
            raise ModuleNotFoundError(f"blocked: {name}")
        return None

sys.meta_path.insert(0, _Block())
from src.internal.servers.web import ml_intent
print("serving path imports without an encoder:", ml_intent.predict_route)
PY
```
Expected: prints the function. A failure means the `sentence_transformers` import escaped `intent_encoder._model`.

- [ ] **Step 8: Commit**

```bash
ruff check . --fix && ruff format .
git add src/internal/servers/web/ml_intent.py src/internal/configs/app_configs.py \
  src/internal/servers/web/intent_routing.py \
  tests/unit/test_ml_intent.py tests/unit/test_intent_routing.py
git commit -m "feat(intent): route by canonical-example similarity at serving time"
```

---

### Task 7: Author and review the evaluation set

**Files:**
- Modify: `data/intent_eval_queries.json` (30 → ~180, all gaining `modules`)
- Create: `data/intent_eval_hard.json` (~40)
- Test: `tests/unit/test_intent_eval_data.py` (new)

**Interfaces:**
- Consumes: `load_intent_eval_queries` (Task 3); `check_leakage`, `build_index` (Task 4).
- Produces: two validated data files. Task 8 measures against them.

**This task contains a blocking review gate.** The labels authored here define what "correct routing" means for every later change, so they are reviewed by the maintainer before any bar is pinned.

- [ ] **Step 1: Write the failing tests**

Create `tests/unit/test_intent_eval_data.py`:

```python
"""Structural guarantees for the evaluation instrument itself.

These test the measuring device, not the model. A silently degraded eval set
would make every later number meaningless.
"""

from collections import Counter
from pathlib import Path

import pytest

from src.model.intent_data import load_intent_eval_queries
from src.model.intent_taxonomy import SEMANTIC_MODULES

DATA = Path(__file__).resolve().parents[2] / "data"
BULK = DATA / "intent_eval_queries.json"
HARD = DATA / "intent_eval_hard.json"


def _queries(path: Path):
    if not path.exists():
        pytest.skip(f"{path.name} not authored yet")
    return load_intent_eval_queries(path)


def test_bulk_set_is_large_enough_for_one_query_to_not_dominate():
    queries = _queries(BULK)

    assert len(queries) >= 170


def test_bulk_set_is_balanced_across_routes():
    counts = Counter(query.label for query in _queries(BULK))

    assert min(counts.values()) >= 0.25 * sum(counts.values())


def test_every_bulk_query_carries_at_least_one_module():
    missing = [query.id for query in _queries(BULK) if not query.modules]

    assert missing == []


def test_the_original_thirty_queries_survive_unchanged():
    """Continuity with the pinned 0.733 depends on these exact queries."""
    ids = {query.id for query in _queries(BULK)}
    legacy = {query.id for query in _queries(BULK) if query.id.startswith("eval-0")}

    assert len(legacy) >= 30
    assert legacy <= ids


def test_hard_slice_exists_and_is_sized_for_triplets():
    queries = _queries(HARD)

    assert 34 <= len(queries) <= 46


def test_hard_slice_is_built_from_minimal_triplets():
    """Same entity across routes: it isolates the boundary from entity difficulty."""
    queries = _queries(HARD)
    groups = Counter(
        query.id.rsplit("-", 1)[0] for query in queries if query.id.startswith("hard-")
    )
    triplets = [group for group, count in groups.items() if count == 3]

    assert len(triplets) >= 10


def test_each_triplet_spans_more_than_one_route():
    queries = _queries(HARD)
    by_group: dict[str, set[str]] = {}
    for query in queries:
        if query.id.startswith("hard-"):
            by_group.setdefault(query.id.rsplit("-", 1)[0], set()).add(query.label)
    multi = [group for group, routes in by_group.items() if len(routes) > 1]

    assert len(multi) >= 10


def test_every_semantic_module_appears_in_the_evaluation_sets():
    """A module never evaluated is a module with no evidence behind it."""
    seen = {
        module
        for path in (BULK, HARD)
        for query in _queries(path)
        for module in query.modules
    }

    assert set(SEMANTIC_MODULES) - seen == set()


def test_no_query_text_is_repeated_across_the_two_sets():
    texts = [q.text.casefold().strip() for q in _queries(BULK)]
    texts += [q.text.casefold().strip() for q in _queries(HARD)]

    assert len(texts) == len(set(texts))
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `pytest tests/unit/test_intent_eval_data.py -v`
Expected: the hard-slice tests SKIP (`intent_eval_hard.json` absent); the bulk tests FAIL on size (30 < 170) and modules.

- [ ] **Step 3: Add modules to the existing 30 queries**

Keep every existing `id`, `text`, and `label` byte-identical — the `0.733` comparison depends on them — and add a `modules` array to each. Example:

```json
{
  "id": "eval-001",
  "text": "where did we land on the index rebuild last week",
  "label": "search",
  "modules": ["lookup_fact"]
}
```

- [ ] **Step 4: Author ~150 more bulk queries**

Append to `data/intent_eval_queries.json` with ids `eval-031` upward. Requirements:

- ~60 per route, so no route can be gamed by a majority guess.
- Every semantic module appears at least 8 times across the whole bulk set.
- Written as a person would type them — lowercase, elliptical, sometimes without a verb. The existing 30 are the register to match: *"anyone remember which dashboard has the p95 numbers"*, not *"Please retrieve the dashboard containing p95 metrics."*
- ~30% multi-module, where a second intent is genuinely present.
- Do **not** copy or lightly reword any canonical example — Step 6 rejects that mechanically.

- [ ] **Step 5: Author the hard slice as minimal triplets**

Create `data/intent_eval_hard.json` with ~12 triplets plus ~4 composite queries. Ids are `hard-<group>-<n>`, so a group is recoverable from the id. Each triplet holds the entity constant and varies only the route:

```json
[
  {
    "id": "hard-earnings-1",
    "text": "find the Q3 earnings report",
    "label": "search",
    "modules": ["lookup_document"]
  },
  {
    "id": "hard-earnings-2",
    "text": "what was revenue in the Q3 earnings report",
    "label": "search",
    "modules": ["lookup_fact"]
  },
  {
    "id": "hard-earnings-3",
    "text": "summarize the Q3 earnings report",
    "label": "chat",
    "modules": ["summarize"]
  },
  {
    "id": "hard-composite-1",
    "text": "find the best italian place near the office and book it for 7",
    "label": "tool",
    "modules": ["schedule"]
  }
]
```

Build the other groups on the same principle, varying which boundary is probed:

| group | boundary probed |
|---|---|
| earnings report | `lookup_document` / `lookup_fact` / `summarize` |
| runbook | find it / read a value from it / explain it |
| deploy | check status / explain the process / run it |
| oncall rotation | look up who / explain how it works / change it |
| p95 dashboard | locate / read a number / compare two |
| customer email | find it / draft one / send one |
| pricing page | find it / what it says now / rewrite it |
| incident | find the ticket / why it happened / file a new one |
| reranker config | which file / what the timeout is / change it |
| meeting notes | find them / summarize them / schedule the next one |
| vendor contract | find it / when it renews / cancel it |
| index rebuild | find the discussion / how it works / trigger it |

Composite queries pair a lookup with an action in one sentence and are labeled with their primary route.

- [ ] **Step 6: Verify structure and prove there is no leakage**

```bash
pytest tests/unit/test_intent_eval_data.py -v

python - <<'PY'
from pathlib import Path

from src.model.intent_data import load_intent_eval_queries
from src.model.intent_encoder import encode_texts
from src.model.intent_index_cli import build_index, check_leakage

index = build_index(Path("data/intent_canonical.json"), Path("data/intent_index"))
texts = [q.text for q in load_intent_eval_queries(Path("data/intent_eval_queries.json"))]
texts += [q.text for q in load_intent_eval_queries(Path("data/intent_eval_hard.json"))]
leaks = check_leakage(index, texts, encode_texts(texts))
print(f"{len(leaks)} leaks of {len(texts)} queries")
for leak in leaks:
    print(" -", leak)
PY
```
Expected: `0 leaks`. **Any leak must be fixed by rewriting the eval query**, never by deleting the canonical example — the canonical set is the model and shrinking it to pass a check is exactly backwards.

- [ ] **Step 7: Maintainer review — BLOCKING**

Present the two files for review. Do not proceed to Task 8 until the maintainer has reviewed and corrected the labels. Report:

- the counts per route and per module for both sets,
- the full hard slice, since that is where labeling judgment is hardest,
- every query whose label you were unsure of, called out explicitly rather than buried.

- [ ] **Step 8: Commit**

```bash
ruff check . --fix && ruff format .
git add tests/unit/test_intent_eval_data.py
git add -f data/intent_eval_queries.json data/intent_eval_hard.json
git commit -m "test(intent): grow the eval instrument to 220 module-labeled queries"
```

---

### Task 8: Measure, tune thresholds, decide

**Files:**
- Create: `src/model/intent_index_eval.py`
- Modify: `src/model/intent_evaluation.py`
- Test: `tests/unit/test_intent_evaluation.py`

**Interfaces:**
- Consumes: everything from Tasks 1-7.
- Produces: `intent_evaluation.module_metrics_report(records) -> dict[str, Any]` over `ModulePredictionRecord`; `@dataclass(frozen=True) ModulePredictionRecord(example_id: str, expected: tuple[str, ...], predicted: tuple[str, ...], route_correct: bool)`; `intent_index_eval.run_index_evaluation(...) -> dict[str, Any]` writing `evaluation_report.json`.

`IntentPredictionRecord`, `evaluate_intent_predictions`, `realistic_accuracy_report`, `select_confidence_threshold`, `calibration_report`, and `out_of_scope_abstention_rate` are **model-agnostic already** — they take prediction records, not a model. They are reused unchanged.

- [ ] **Step 1: Write the failing tests**

Add to `tests/unit/test_intent_evaluation.py`:

```python
def test_module_macro_f1_excludes_the_form_label():
    """bare_entity describes utterance shape, not intent; it would distort F1."""
    from src.model.intent_evaluation import ModulePredictionRecord, module_metrics_report

    records = [
        ModulePredictionRecord("a", ("lookup_fact",), ("lookup_fact",), True),
        ModulePredictionRecord("b", ("bare_entity",), ("lookup_fact",), True),
    ]

    report = module_metrics_report(records)

    assert "bare_entity" not in report["per_module_metrics"]
    assert report["per_module_metrics"]["lookup_fact"]["recall"] == pytest.approx(1.0)


def test_joint_accuracy_requires_route_and_exact_module_set():
    from src.model.intent_evaluation import ModulePredictionRecord, module_metrics_report

    records = [
        ModulePredictionRecord("a", ("lookup_fact",), ("lookup_fact",), True),
        # right route, extra module -> not joint-correct
        ModulePredictionRecord(
            "b", ("lookup_fact",), ("lookup_fact", "current_info"), True
        ),
        # right modules, wrong route -> not joint-correct
        ModulePredictionRecord("c", ("explain",), ("explain",), False),
    ]

    report = module_metrics_report(records)

    assert report["joint_accuracy"] == pytest.approx(1 / 3)


def test_module_set_order_does_not_affect_joint_accuracy():
    from src.model.intent_evaluation import ModulePredictionRecord, module_metrics_report

    records = [
        ModulePredictionRecord(
            "a", ("lookup_fact", "current_info"), ("current_info", "lookup_fact"), True
        )
    ]

    assert module_metrics_report(records)["joint_accuracy"] == pytest.approx(1.0)


def test_module_report_records_how_many_queries_carried_gold_modules():
    """The legacy 30 predate modules; the metric must say what it covered."""
    from src.model.intent_evaluation import ModulePredictionRecord, module_metrics_report

    records = [
        ModulePredictionRecord("a", ("lookup_fact",), ("lookup_fact",), True),
        ModulePredictionRecord("b", (), ("explain",), True),
    ]

    report = module_metrics_report(records)

    assert report["scored_queries"] == 1
    assert report["total_queries"] == 2
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `pytest tests/unit/test_intent_evaluation.py -v`
Expected: FAIL — `ImportError: cannot import name 'ModulePredictionRecord'`.

- [ ] **Step 3: Add module metrics**

In `src/model/intent_evaluation.py`, switch the import:

```python
from .intent_taxonomy import INTENT_LABELS, SEMANTIC_MODULES
```

and append:

```python
@dataclass(frozen=True)
class ModulePredictionRecord:
    """One query's gold and predicted module sets, plus its route outcome."""

    example_id: str
    expected: tuple[str, ...]
    predicted: tuple[str, ...]
    route_correct: bool


def module_metrics_report(
    records: Iterable[ModulePredictionRecord],
) -> dict[str, Any]:
    """Per-module precision/recall/F1, macro-F1, and joint accuracy.

    Only the thirteen semantic modules are scored. ``bare_entity`` names an
    utterance form rather than an intent, and averaging it in would distort the
    macro number. Queries with no gold modules — the original thirty predate the
    taxonomy — are excluded from scoring and counted separately, so the report
    never implies coverage it does not have.
    """
    records = tuple(records)
    scored = tuple(record for record in records if record.expected)

    per_module: dict[str, dict[str, float]] = {}
    f1_values: list[float] = []
    for module in SEMANTIC_MODULES:
        true_positive = sum(
            1 for r in scored if module in r.expected and module in r.predicted
        )
        false_positive = sum(
            1 for r in scored if module not in r.expected and module in r.predicted
        )
        false_negative = sum(
            1 for r in scored if module in r.expected and module not in r.predicted
        )
        precision = (
            true_positive / (true_positive + false_positive)
            if true_positive + false_positive
            else 0.0
        )
        recall = (
            true_positive / (true_positive + false_negative)
            if true_positive + false_negative
            else 0.0
        )
        f1 = (
            2 * precision * recall / (precision + recall)
            if precision + recall
            else 0.0
        )
        per_module[module] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": true_positive + false_negative,
        }
        f1_values.append(f1)

    joint = sum(
        1
        for r in scored
        if r.route_correct and set(r.expected) == set(r.predicted)
    )
    return {
        "total_queries": len(records),
        "scored_queries": len(scored),
        "per_module_metrics": per_module,
        "macro_f1": sum(f1_values) / len(SEMANTIC_MODULES),
        "joint_accuracy": joint / len(scored) if scored else 0.0,
    }
```

- [ ] **Step 4: Implement the evaluation run**

Create `src/model/intent_index_eval.py`:

```python
"""Score a built index against the held-out query sets.

Reuses the existing report machinery: IntentPredictionRecord and the metric
functions in intent_evaluation take prediction records rather than a model, so
none of it needed to change to score a different kind of model.
"""

from __future__ import annotations

import json
from pathlib import Path
from time import perf_counter
from typing import Any

from .intent_data import load_intent_eval_queries, load_out_of_scope_probes
from .intent_encoder import encode_texts
from .intent_evaluation import (
    IntentPredictionRecord,
    ModulePredictionRecord,
    module_metrics_report,
    realistic_accuracy_report,
)
from .intent_index_cli import check_leakage
from .intent_knn import INDEX_FILENAME, IntentIndex

LEGACY_PREFIX = "eval-0"


def _predict(index, queries, thresholds):
    records, modules, latencies = [], [], []
    texts = [query.text for query in queries]
    vectors = encode_texts(texts)
    for query, vector in zip(queries, vectors):
        start = perf_counter()
        decision = index.decide(vector, **thresholds)
        latencies.append((perf_counter() - start) * 1_000)
        records.append(
            IntentPredictionRecord(
                example_id=query.id,
                expected=query.label,
                predicted=decision.route,
                confidence=decision.confidence,
                latency_ms=latencies[-1],
                mechanism="index",
            )
        )
        modules.append(
            ModulePredictionRecord(
                example_id=query.id,
                expected=tuple(query.modules),
                predicted=tuple(decision.modules),
                route_correct=decision.route == query.label,
            )
        )
    return records, modules, vectors


def run_index_evaluation(
    *,
    index_path: Path,
    eval_queries_path: Path,
    hard_queries_path: Path | None,
    out_of_scope_path: Path | None,
    canonical_path: Path,
    output_path: Path,
) -> dict[str, Any]:
    """Score the index and write the evaluation report."""
    index = IntentIndex.load(index_path / INDEX_FILENAME)
    thresholds = {
        "min_confidence": 0.0,
        "min_margin": 0.0,
        "min_module_score": 0.45,
    }

    bulk = load_intent_eval_queries(eval_queries_path)
    bulk_records, bulk_modules, bulk_vectors = _predict(index, bulk, thresholds)

    leaks = check_leakage(index, [q.text for q in bulk], bulk_vectors)
    if leaks:
        raise ValueError(
            f"{len(leaks)} evaluation queries leak into the canonical set; "
            f"first: {leaks[0]}"
        )

    legacy = tuple(r for r in bulk_records if r.example_id.startswith(LEGACY_PREFIX))
    report: dict[str, Any] = {
        "index": {
            "size": index.size,
            "encoder": index.encoder,
            "fingerprint": index.fingerprint,
            "canonical": str(canonical_path),
            "low_support_modules": list(index.low_support_modules()),
        },
        "bulk": realistic_accuracy_report(bulk_records, threshold=0.0),
        "legacy_30": realistic_accuracy_report(legacy, threshold=0.0),
        "modules": module_metrics_report(bulk_modules),
    }

    if hard_queries_path is not None:
        hard = load_intent_eval_queries(hard_queries_path)
        hard_records, hard_modules, _ = _predict(index, hard, thresholds)
        report["hard"] = realistic_accuracy_report(hard_records, threshold=0.0)
        report["hard_modules"] = module_metrics_report(hard_modules)

    if out_of_scope_path is not None:
        probes = load_out_of_scope_probes(out_of_scope_path)
        probe_vectors = encode_texts([text for _, text in probes])
        probe_confidences = [
            index.decide(vector, **thresholds).confidence for vector in probe_vectors
        ]
        in_scope = [record.confidence for record in bulk_records]
        report["out_of_scope"] = {
            "probes": len(probes),
            "mean_in_scope_confidence": sum(in_scope) / len(in_scope),
            "mean_out_of_scope_confidence": (
                sum(probe_confidences) / len(probe_confidences)
            ),
            "separation_margin": (
                sum(in_scope) / len(in_scope)
                - sum(probe_confidences) / len(probe_confidences)
            ),
        }

    report["headline"] = {
        "bulk_accuracy": report["bulk"]["accuracy"],
        "legacy_30_accuracy": report["legacy_30"]["accuracy"],
        "hard_accuracy": report.get("hard", {}).get("accuracy"),
        "module_macro_f1": report["modules"]["macro_f1"],
        "joint_accuracy": report["modules"]["joint_accuracy"],
        "separation_margin": report.get("out_of_scope", {}).get("separation_margin"),
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return report
```

- [ ] **Step 5: Build the real index and measure**

```bash
python -m src.model.intent_index_cli build \
  --canonical data/intent_canonical.json --output data/intent_index

python -m src.model.intent_index_cli evaluate \
  --index data/intent_index \
  --eval-queries data/intent_eval_queries.json \
  --hard-queries data/intent_eval_hard.json \
  --out-of-scope data/intent_out_of_scope.json \
  --canonical data/intent_canonical.json \
  --output data/intent_index/evaluation_report.json
```

- [ ] **Step 6: Tune the two routing thresholds**

```bash
python - <<'PY'
from pathlib import Path

from src.model.intent_data import load_intent_eval_queries, load_out_of_scope_probes
from src.model.intent_encoder import encode_texts
from src.model.intent_knn import INDEX_FILENAME, IntentIndex

index = IntentIndex.load(Path("data/intent_index") / INDEX_FILENAME)
queries = load_intent_eval_queries(Path("data/intent_eval_queries.json"))
probes = load_out_of_scope_probes(Path("data/intent_out_of_scope.json"))
query_vectors = encode_texts([q.text for q in queries])
probe_vectors = encode_texts([t for _, t in probes])

print(f"{'conf':>6} {'margin':>7} {'cover':>7} {'acc|cov':>8} {'oos_defer':>10}")
for min_confidence in (0.30, 0.35, 0.40, 0.45, 0.50, 0.55):
    for min_margin in (0.02, 0.05, 0.08, 0.12):
        thresholds = dict(
            min_confidence=min_confidence,
            min_margin=min_margin,
            min_module_score=0.45,
        )
        served, correct = 0, 0
        for query, vector in zip(queries, query_vectors):
            decision = index.decide(vector, **thresholds)
            if not decision.abstained:
                served += 1
                correct += decision.route == query.label
        deferred = sum(
            index.decide(v, **thresholds).abstained for v in probe_vectors
        )
        coverage = served / len(queries)
        accuracy = correct / served if served else 0.0
        print(
            f"{min_confidence:>6.2f} {min_margin:>7.2f} {coverage:>7.3f} "
            f"{accuracy:>8.3f} {deferred / len(probes):>10.3f}"
        )
PY
```

Pick the pair with the highest **served accuracy at coverage ≥ 0.60**, breaking ties toward higher out-of-scope deferral. Record both values; they become the defaults in Task 9.

Then measure serving latency end to end, including encoding:

```bash
python - <<'PY'
import statistics, time
from pathlib import Path

from src.model.intent_encoder import encode_texts
from src.model.intent_knn import INDEX_FILENAME, IntentIndex

index = IntentIndex.load(Path("data/intent_index") / INDEX_FILENAME)
query = "where did we land on the index rebuild last week"
thresholds = dict(min_confidence=0.4, min_margin=0.05, min_module_score=0.45)
for _ in range(10):
    index.decide(encode_texts([query])[0], **thresholds)
timings = []
for _ in range(200):
    start = time.perf_counter()
    index.decide(encode_texts([query])[0], **thresholds)
    timings.append((time.perf_counter() - start) * 1000)
print(f"p50 {statistics.median(timings):.2f}ms  p95 {sorted(timings)[190]:.2f}ms")
PY
```

- [ ] **Step 7: Apply the decision rule**

The spec fixes this in advance. Compare bulk-180 route accuracy against:

| bulk accuracy | verdict |
|---|---|
| ≥ 0.80 | clears the promotion bar; wire it live and say so in the PR |
| 0.75 – 0.80 | real improvement over `0.733`; artifact stays dark |
| < 0.75 | **hard stop** |

**If bulk accuracy is below 0.75, stop and report** — bulk, legacy-30, and hard accuracy, per-module F1, joint accuracy, the separation margin, and the latency. Do not tune further. That would be a new spec, exactly as it was for #510.

Also confirm: separation margin ≥ `0.25`, p95 latency ≤ `25ms`, and no module below support.

- [ ] **Step 8: Run the tests**

Run: `pytest tests/unit/test_intent_evaluation.py -v`
Expected: PASS.

- [ ] **Step 9: Commit**

```bash
ruff check . --fix && ruff format .
git add src/model/intent_evaluation.py src/model/intent_index_eval.py \
  tests/unit/test_intent_evaluation.py
git commit -m "feat(intent): score the canonical index and its module labels"
```

---

### Task 9: Remove the MLP, pin the bars, document

**Files:**
- Delete: `src/model/intent_classifier.py`, `src/model/intent_training.py`, `src/model/wordpiece.py`, `src/model/intent_pretrained.py`
- Delete: `tests/unit/test_intent_classifier.py`, `test_intent_training.py`, `test_intent_pretrained.py`, `test_wordpiece.py`, `test_wordpiece_parity.py`
- Modify: `src/model/intent_index_cli.py`, `src/internal/configs/app_configs.py`, `docs/training-and-evaluation.md`
- Test: `tests/unit/test_intent_index_eval.py` (new)

**Interfaces:**
- Consumes: the measured values from Task 8.
- Produces: pinned floors `_BULK_ACCURACY_FLOOR`, `_SEPARATION_MARGIN_FLOOR`; tuned defaults for `intent_model_min_confidence`, `intent_min_route_margin`.

Deletion happens last, after the replacement is measured. Nothing is removed on the strength of an expectation.

- [ ] **Step 1: Write the pinned-bar tests**

Create `tests/unit/test_intent_index_eval.py`:

```python
"""The measured bars. Raise a floor when a run beats it; never lower one
without recording why in the commit message."""

import functools
from pathlib import Path

import pytest

DATA = Path(__file__).resolve().parents[2] / "data"

# Measured on the first canonical-index run. Replace with Task 8's values.
_BULK_ACCURACY_FLOOR = 0.00
_SEPARATION_MARGIN_FLOOR = 0.00
_P95_LATENCY_CEILING_MS = 25.0


@functools.lru_cache(maxsize=1)
def _report():
    pytest.importorskip("sentence_transformers")
    from src.model.intent_index_eval import run_index_evaluation

    index = DATA / "intent_index"
    if not (index / "index.npz").exists():
        pytest.skip(
            "run `python -m src.model.intent_index_cli build --canonical "
            f"data/intent_canonical.json --output {index}` to measure the bars"
        )
    return run_index_evaluation(
        index_path=index,
        eval_queries_path=DATA / "intent_eval_queries.json",
        hard_queries_path=DATA / "intent_eval_hard.json",
        out_of_scope_path=DATA / "intent_out_of_scope.json",
        canonical_path=DATA / "intent_canonical.json",
        output_path=index / "evaluation_report.json",
    )


def test_index_holds_the_bulk_accuracy_bar():
    assert _report()["bulk"]["accuracy"] >= _BULK_ACCURACY_FLOOR


def test_out_of_scope_requests_score_below_in_scope_requests():
    assert _report()["out_of_scope"]["separation_margin"] >= _SEPARATION_MARGIN_FLOOR


def test_every_module_has_enough_canonical_support_to_be_emitted():
    assert _report()["index"]["low_support_modules"] == []


def test_evaluation_refuses_to_score_a_leaking_evaluation_set():
    """The guard must fail loudly, not quietly inflate accuracy."""
    pytest.importorskip("sentence_transformers")
    assert _report()["bulk"]["total_queries"] >= 170
```

- [ ] **Step 2: Pin the measured values**

Replace the three constants above with Task 8's measured numbers: floors set about `0.02` below what was measured, and the latency ceiling at `25.0`.

Run: `pytest tests/unit/test_intent_index_eval.py -v`
Expected: PASS, no longer skipped (with sentence-transformers installed).

- [ ] **Step 3: Set the tuned thresholds as defaults**

In `src/internal/configs/app_configs.py`, replace the placeholder defaults with the pair chosen in Task 8 Step 6, in both the dataclass field defaults and the `get_env_float` fallbacks:

```python
    intent_model_min_confidence: float = 0.45   # <- replace with the tuned value
    intent_min_route_margin: float = 0.05       # <- replace with the tuned value
```

- [ ] **Step 4: Delete the MLP machinery**

```bash
git rm src/model/intent_classifier.py src/model/intent_training.py \
  src/model/wordpiece.py src/model/intent_pretrained.py
git rm tests/unit/test_intent_classifier.py tests/unit/test_intent_training.py \
  tests/unit/test_intent_pretrained.py tests/unit/test_wordpiece.py \
  tests/unit/test_wordpiece_parity.py
rm -rf data/intent_pretrained
```

Then find every remaining reference and repoint or remove it:

```bash
grep -rn "intent_classifier\|intent_training\|intent_pretrained\|wordpiece\|IntentPipeline\|intent_model_path" \
  src tests examples docs --include="*.py" --include="*.md" --include="*.sh"
```

Expected remaining work: `src/model/__init__.py` re-exports, `src/model/intent_data.py`'s docstring, and `docs/training-and-evaluation.md`. Each must be updated — a stale import here fails collection for the whole suite.

- [ ] **Step 5: Run the whole suite**

Run: `pytest`
Expected: PASS. Failures are almost certainly stale imports of the deleted modules; the grep above finds them.

- [ ] **Step 6: Prove the CI import discipline one final time**

```bash
python - <<'PY'
import sys

class _Block:
    def find_module(self, name, path=None):
        if name.split(".")[0] in {"torch", "transformers", "sentence_transformers"}:
            raise ModuleNotFoundError(f"blocked: {name}")
        return None

sys.meta_path.insert(0, _Block())
from src.internal.servers.web import ml_intent
from src.model import intent_data, intent_knn, intent_taxonomy
print("clean:", ml_intent.predict_route, intent_knn.IntentIndex,
      len(intent_taxonomy.MODULES), intent_data.load_canonical_examples)
PY
```
Expected: prints all four. Then confirm the suite itself is clean under the same block:

```bash
pytest tests/unit/test_intent_taxonomy.py tests/unit/test_intent_knn.py \
  tests/unit/test_intent_data.py tests/unit/test_intent_seed.py -v
```

- [ ] **Step 7: Update the operator documentation**

In `docs/training-and-evaluation.md`, replace the intent-model section:

- Replace the `embeddings` → `baseline` → `train` workflow with:

```bash
python -m src.model.intent_index_cli seed \
  --examples data/intent_examples.json --output data/intent_canonical.draft.json
python -m src.model.intent_index_cli build \
  --canonical data/intent_canonical.json --output data/intent_index
python -m src.model.intent_index_cli evaluate \
  --index data/intent_index \
  --eval-queries data/intent_eval_queries.json \
  --hard-queries data/intent_eval_hard.json \
  --out-of-scope data/intent_out_of_scope.json \
  --canonical data/intent_canonical.json \
  --output data/intent_index/evaluation_report.json
```

- Add a paragraph explaining the change:

> The router no longer trains a classifier. It compares the request against ~270 curated canonical examples encoded by MiniLM and takes the route whose nearest examples are closest — the mean of the top-3 cosine similarities per route. Two thresholds gate the answer, because two different things go wrong: a low absolute similarity means nothing canonical resembles the request (out of scope), while a small margin between the best and second-best route means two routes fit equally well (ambiguous). Either abstains to the LLM classifier. This replaced a softmax head that could not express "none of these" at all — probabilities sum to one by construction, which is why the previous out-of-scope separation was only `+0.059`.

- Add the extension workflow, which is the operator-facing change that matters most:

> To change routing behavior, edit `data/intent_canonical.json` and rebuild. There is no training run. Always **append, rebuild, re-measure**: a badly-phrased canonical example becomes a bad attractor for every nearby query, and the only thing that catches it is the evaluation report.

- Document the units trap explicitly:

> `AGENTIC_SEARCH_INTENT_MODEL_MIN_CONFIDENCE` is now a **cosine similarity**, not a softmax probability. A value carried over from the previous model is meaningless — the two live on different scales. Re-tune it with the sweep in the plan rather than reusing the old number.

- Record the measured results from Task 8: bulk-180 accuracy against the previous `0.733` on the legacy 30, hard-slice accuracy, module macro-F1, joint accuracy, the separation margin against `+0.059`, and p50/p95 latency against `0.16ms`/`0.43ms` — stating plainly that latency is a deliberate regression bought with accuracy and out-of-scope safety.
- Note that `data/intent_index/` is a regenerable local artifact under gitignored `data/`, and that the canonical and evaluation JSON files are tracked.

- [ ] **Step 8: Commit and open the PR**

```bash
ruff check . --fix && ruff format .
git branch --show-current   # must be feat/intent-knn-routing
git add -A
git commit -m "feat(intent): retire the intent MLP for canonical-example routing"
git push -u origin feat/intent-knn-routing
gh pr create --title "feat(intent): route by nearest canonical example instead of a trained MLP" --body "..."
```

The PR body must state: bulk-180 / legacy-30 / hard-slice route accuracy, with `0.733` as the before; module macro-F1 and joint accuracy; the out-of-scope separation margin against `+0.059`; p50/p95 latency against `0.16ms`/`0.43ms`, named as an accepted regression; the tuned confidence and margin thresholds and that the confidence unit changed from probability to cosine; the canonical set size and per-module support; which verdict of the decision rule was reached; the full list of deleted modules including #510's wordpiece work; and **"Please review before merging."** Link the spec and this plan.

---

## Verification summary

| Spec requirement | Task |
|---|---|
| Route-scoped module taxonomy, 13 semantic + `bare_entity` as form | 1 |
| `ACTION_MODULES` for composite detection | 1 |
| Multi-label modules per example | 1, 3, 5 |
| Top-3 mean cosine per route; route never derived from modules | 2 |
| `τ_conf` for out-of-scope, `τ_margin` for ambiguity, confidence checked first | 2 |
| `τ_module` cannot change the route | 2 |
| Minimum support of 10, `low_support` reporting | 2, 6 |
| Composite detection recorded, not acted on | 2, 6 |
| Index math importable and testable with no torch/transformers | 2 (Step 5), 6 (Step 7), 9 (Step 6) |
| `sentence_transformers` imported function-locally only | 4 |
| Canonical set ~270, curated from existing examples, no external data | 5 |
| `bare_entity` capped at ~10 | 5 |
| Leakage guard: exact match or cosine > 0.95 fails the build | 4, 7, 8 |
| `predict_route` signature and `route_request` unchanged | 6 |
| Lazy encoder load, never at lifespan | 6 |
| `intent_index_path`, `τ_margin`, `τ_module` settings | 6, 9 |
| Bulk-180 the only promotion gate; legacy-30 for continuity | 7, 8 |
| Hard slice from minimal triplets | 7 |
| Maintainer reviews eval labels before bars are pinned | 7 (Step 7) |
| Module macro-F1 excludes the form label; joint accuracy is exact set match | 8 |
| Thresholds tuned, recorded in `evaluation_report.json` | 8 |
| Decision rule with a hard stop below 0.75 | 8 (Step 7) |
| Separation margin ≥ 0.25, p95 ≤ 25ms | 8, 9 |
| MLP machinery deleted only after measurement | 9 |
| `intent_evaluation.py` retargeted, not deleted | 8 |
| Units trap documented | 9 |
