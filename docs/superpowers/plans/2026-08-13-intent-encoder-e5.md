# Intent encoder swap: MiniLM → e5-small-v2 — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Swap the intent router's sentence encoder from all-MiniLM-L6-v2 to intfloat/e5-small-v2, re-tune its three hyperparameters on data that is not the test set, and re-measure against the promotion bar.

**Architecture:** The encoder sits behind a one-function seam (`intent_encoder.encode_texts`) built for exactly this swap. E5 requires a `"query: "` prefix that is applied inside that function and derived from the model name, so no call site can omit it and the index's existing encoder-match check covers it for free. Dimensions stay 384, so the index format, the scoring rules, the canonical set, and the serving cascade are all untouched.

**Tech Stack:** Python 3, numpy, scikit-learn (metrics), sentence-transformers (encoding only), pytest.

**Spec:** `docs/superpowers/specs/2026-08-13-intent-encoder-e5-design.md`

## Global Constraints

- **`intfloat/e5-small-v2` is already a repo dependency** (dense-retrieval model list) and 384-dimensional — the same width as the shipped encoder, so `index.npz`'s format does not change.
- **The prefix is `"query: "`, applied symmetrically to canonical anchors and to queries**, inside `encode_texts`. Never passed by a caller.
- **An unknown model name must raise, not default to an empty prefix.** Silent prefix omission is the failure mode this contract exists to prevent — it degrades vectors without erroring.
- `src/model/intent_taxonomy.py`, `intent_knn.py`, `intent_data.py`, and `src/internal/servers/web/ml_intent.py` must stay importable with **no torch, no transformers, no sentence-transformers**. `sentence_transformers` is imported function-locally in `intent_encoder.py` only. This repo has twice shipped collection failures from unguarded imports (#356, re-fixed in #418).
- **`route_request` in `src/internal/servers/web/intent_routing.py` is not modified.** Neither is any dispatcher, the frontend, `IntentIndex`'s scoring rules, the taxonomy, `data/intent_canonical.json`, or either evaluation file.
- **No hyperparameter may be chosen on the test slice.** Tuning uses the contaminated `legacy-30` plus a seeded, route-stratified 40-query slice of the clean 151. The remaining 111 clean queries plus `hard-40` are the test set and are untouched until Task 3's final measurement.
- **Leave-one-out is not a selector for `k`.** It is biased — measured on the shipped index it climbs to 0.7643 at k=25 while clean-slice accuracy has already turned down to 0.6755. It stays a reported diagnostic.
- The artifact ships **dark**: `intent_index_path` stays unset by default. Promotion is a separate change.
- `data/` is gitignored — anything tracked under it needs `git add -f`.
- Lint before each commit: `ruff check . --fix && ruff format .`
- One commit per task on branch `feat/intent-encoder-e5` (already created, already rebased onto `main` after #511 merged as `06942d3`). Open a PR after the last task; **never merge it**.

**Measured reference values** — from probes on #511's committed index and instrument. Every one was fitted (k chosen after seeing results), so they are the expectation to beat, not a target to reproduce:

| encoder | k | clean-151 | bulk-181 | hard-40 | OOS AUC | Cohen's d | p95 |
|---|---|---|---|---|---|---|---|
| all-MiniLM-L6-v2 (shipped) | 3 | 0.6225 | 0.6519 | 0.6250 | 0.868 | 1.49 | 6.1ms |
| e5-small-v2 | 8 | 0.8079 | 0.8232 | 0.7250 | 0.872 | — | 12.0ms |
| e5-small-v2 | 15 | 0.8146 | 0.8287 | 0.7500 | 0.871 | — | 12.0ms |

## File structure

| file | responsibility |
|---|---|
| `src/model/intent_encoder.py` | `MODEL_PREFIXES`, `prefix_for()`, prefix applied inside `encode_texts`, `DEFAULT_ENCODER` flipped |
| `src/model/intent_eval_split.py` (new) | the tuning/test split — seeded, route-stratified, pure Python |
| `src/model/intent_evaluation.py` | `separability_report()` — AUC, Cohen's d, raw margin |
| `src/model/intent_index_eval.py` | 3-D sweep on the tuning slice; test-slice reporting |
| `src/internal/configs/app_configs.py` | re-tuned defaults for `intent_top_k`, `intent_model_min_confidence`, `intent_min_route_margin` |
| `tests/unit/test_intent_index_eval.py` | re-pinned bars |
| `docs/training-and-evaluation.md` | workflow, results, the units change, the fitted-number caveat |

---

### Task 1: The prefix contract and the encoder swap

**Files:**
- Modify: `src/model/intent_encoder.py`
- Test: `tests/unit/test_intent_encoder.py`, `tests/unit/test_ml_intent.py`

**Interfaces:**
- Consumes: nothing new.
- Produces: `MODEL_PREFIXES: dict[str, str]`; `prefix_for(model_name: str) -> str` (raises `ValueError` for an unregistered model); `DEFAULT_ENCODER = "intfloat/e5-small-v2"`. `encode_texts`'s signature is unchanged — the prefix is applied internally.

**Why the prefix is derived rather than stored:** the index already records the encoder name, and `load_intent_index` already rejects a mismatch. Making the prefix a pure function of the model name means the existing check covers it with no change to `index.npz`'s format. A stale MiniLM index is 384-dimensional too, so it would otherwise load and score silently — this is the change that makes that check earn its keep.

- [ ] **Step 1: Write the failing tests**

Add to `tests/unit/test_intent_encoder.py`:

```python
def test_default_encoder_is_e5_small():
    from src.model.intent_encoder import DEFAULT_ENCODER

    assert DEFAULT_ENCODER == "intfloat/e5-small-v2"


def test_the_default_encoder_has_a_registered_prefix():
    from src.model.intent_encoder import DEFAULT_ENCODER, prefix_for

    assert prefix_for(DEFAULT_ENCODER) == "query: "


def test_an_unregistered_model_raises_rather_than_using_no_prefix():
    """A silently missing prefix degrades e5 vectors without erroring."""
    from src.model.intent_encoder import prefix_for

    with pytest.raises(ValueError, match="prefix"):
        prefix_for("some/unregistered-model")


def test_minilm_is_still_registered_with_an_empty_prefix():
    """Old indexes are rejected by name, but the mapping must stay honest."""
    from src.model.intent_encoder import prefix_for

    assert prefix_for("sentence-transformers/all-MiniLM-L6-v2") == ""


def test_encode_applies_the_prefix():
    """The whole contract: encode_texts('x') must equal raw encode('query: x')."""
    pytest.importorskip("sentence_transformers")
    import numpy as np

    from src.model.intent_encoder import DEFAULT_ENCODER, _model, encode_texts

    through_seam = encode_texts(["find the runbook"])
    raw = _model(DEFAULT_ENCODER).encode(
        ["query: find the runbook"],
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )

    np.testing.assert_allclose(through_seam, raw.astype(np.float32), atol=1e-5)


def test_the_prefix_actually_changes_the_vector():
    """Proves the assertion above is not vacuous."""
    pytest.importorskip("sentence_transformers")
    import numpy as np

    from src.model.intent_encoder import DEFAULT_ENCODER, _model, encode_texts

    unprefixed = _model(DEFAULT_ENCODER).encode(
        ["find the runbook"],
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )

    assert float(encode_texts(["find the runbook"])[0] @ unprefixed[0]) < 0.999


def test_encoded_width_is_unchanged_at_384():
    """Same width as the previous encoder, so index.npz's format is unchanged."""
    pytest.importorskip("sentence_transformers")

    assert encode_texts(["find the runbook"]).shape == (1, 384)
```

Add to `tests/unit/test_ml_intent.py`:

```python
def test_an_index_built_with_the_previous_encoder_is_rejected(tmp_path, monkeypatch):
    """e5-small is also 384-d, so a stale index would otherwise score silently."""
    import numpy as np

    from src.model.intent_knn import INDEX_FILENAME, CanonicalExample, IntentIndex

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
    directory = tmp_path / "stale"
    IntentIndex(
        examples,
        np.stack(rows),
        "sentence-transformers/all-MiniLM-L6-v2",
        "sha256:x",
    ).save(directory / INDEX_FILENAME)

    settings = AppSettings(intent_index_path=directory)

    assert ml_intent.predict_route("anything", settings=settings) is None
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `pytest tests/unit/test_intent_encoder.py tests/unit/test_ml_intent.py -v`
Expected: FAIL — `ImportError: cannot import name 'prefix_for'`, and the default-encoder assertion fails on the old value.

- [ ] **Step 3: Implement the prefix contract**

In `src/model/intent_encoder.py`, replace `DEFAULT_ENCODER` and add the prefix machinery above `_model`:

```python
DEFAULT_ENCODER = "intfloat/e5-small-v2"

# E5 models are trained with instruction prefixes and degrade *silently*
# without them — no error, just worse vectors. The prefix is therefore a
# property of the model, derived from its name rather than passed by callers,
# so no call site can omit it. Deriving rather than storing it also means the
# index needs no new field: it already records the encoder name, and
# ml_intent.load_intent_index already rejects a mismatch, which covers the
# prefix for free. That matters here because e5-small-v2 is also 384-wide, so
# an index built with the previous encoder would otherwise load and score
# without any error at all.
#
# Both sides of the comparison use "query: ": this is symmetric short-text
# similarity, not the asymmetric query/passage retrieval "passage: " is for.
MODEL_PREFIXES: dict[str, str] = {
    "intfloat/e5-small-v2": "query: ",
    "intfloat/e5-base-v2": "query: ",
    "sentence-transformers/all-MiniLM-L6-v2": "",
}


def prefix_for(model_name: str) -> str:
    """The instruction prefix *model_name* requires.

    Raises rather than defaulting to "": an unregistered model is far more
    likely to be one whose prefix nobody looked up than one that genuinely
    needs none, and guessing wrong is invisible.
    """
    try:
        return MODEL_PREFIXES[model_name]
    except KeyError:
        raise ValueError(
            f"No instruction prefix registered for encoder {model_name!r}. "
            f"Add it to MODEL_PREFIXES — encoders that need a prefix degrade "
            f"silently without one. Known: {sorted(MODEL_PREFIXES)}"
        ) from None
```

Then apply it inside `encode_texts`:

```python
    prefix = prefix_for(model_name)
    vectors = _model(model_name).encode(
        [prefix + text for text in texts],
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `pytest tests/unit/test_intent_encoder.py tests/unit/test_ml_intent.py -v`
Expected: PASS. The first run downloads `intfloat/e5-small-v2` (~130MB) if it is not cached.

- [ ] **Step 5: Prove the routing path still imports with no ML dependency**

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
from src.model.intent_encoder import DEFAULT_ENCODER, prefix_for
print("clean:", ml_intent.predict_route, DEFAULT_ENCODER, repr(prefix_for(DEFAULT_ENCODER)))
PY
```
Expected: prints all four. `prefix_for` must work with sentence-transformers blocked — it is pure data.

- [ ] **Step 6: Commit**

```bash
ruff check . --fix && ruff format .
git add src/model/intent_encoder.py tests/unit/test_intent_encoder.py tests/unit/test_ml_intent.py
git commit -m "feat(intent): read queries with e5-small-v2 and its instruction prefix"
```

---

### Task 2: The tuning/test split and scale-free separability

**Files:**
- Create: `src/model/intent_eval_split.py`
- Modify: `src/model/intent_evaluation.py`, `src/model/intent_index_eval.py`
- Test: `tests/unit/test_intent_eval_split.py` (new), `tests/unit/test_intent_evaluation.py`

**Interfaces:**
- Consumes: `IntentEvalQuery` from `intent_data`; `LEGACY_PREFIX` semantics from `intent_index_eval`.
- Produces:
  - `intent_eval_split.split_eval_queries(queries, *, slice_size=40, seed=17) -> EvalSplit` where `@dataclass(frozen=True) EvalSplit(tuning: tuple, test: tuple)`
  - `intent_evaluation.separability_report(in_scope: Sequence[float], out_of_scope: Sequence[float]) -> dict[str, Any]` returning `auc`, `cohens_d`, `raw_margin`, `max_out_of_scope`, `min_in_scope`, `counts`
  - `intent_index_eval` sweeps `top_k × min_confidence × min_margin` on the tuning slice only

- [ ] **Step 1: Write the failing tests**

Create `tests/unit/test_intent_eval_split.py`:

```python
from collections import Counter

import pytest

from src.model.intent_data import IntentEvalQuery
from src.model.intent_eval_split import split_eval_queries


def _queries(n_clean: int = 151, n_legacy: int = 30):
    routes = ("chat", "search", "tool")
    legacy = [
        IntentEvalQuery(f"eval-{routes[i % 3]}-{i:02d}", f"legacy {i}", routes[i % 3], ())
        for i in range(n_legacy)
    ]
    clean = [
        IntentEvalQuery(f"bulk-{i:03d}", f"clean {i}", routes[i % 3], ())
        for i in range(n_clean)
    ]
    return legacy + clean


def test_every_legacy_query_goes_to_tuning():
    """They are contaminated, so they are worthless as a gate and free to tune on."""
    split = split_eval_queries(_queries())

    assert all(q.id.startswith("eval-") for q in split.tuning if q.id.startswith("eval-"))
    assert not any(q.id.startswith("eval-") for q in split.test)


def test_the_tuning_slice_takes_the_requested_number_of_clean_queries():
    split = split_eval_queries(_queries(), slice_size=40)

    clean_tuning = [q for q in split.tuning if q.id.startswith("bulk-")]
    assert len(clean_tuning) == 40
    assert len(split.test) == 151 - 40


def test_the_split_is_a_partition_with_no_overlap():
    split = split_eval_queries(_queries())

    ids_tuning = {q.id for q in split.tuning}
    ids_test = {q.id for q in split.test}
    assert not (ids_tuning & ids_test)
    assert len(ids_tuning) + len(ids_test) == 181


def test_the_clean_tuning_slice_is_route_stratified():
    split = split_eval_queries(_queries(), slice_size=39)

    counts = Counter(q.label for q in split.tuning if q.id.startswith("bulk-"))
    assert set(counts) == {"chat", "search", "tool"}
    assert max(counts.values()) - min(counts.values()) <= 1


def test_the_split_is_deterministic_for_a_seed():
    a = split_eval_queries(_queries(), seed=17)
    b = split_eval_queries(_queries(), seed=17)

    assert [q.id for q in a.test] == [q.id for q in b.test]


def test_a_different_seed_gives_a_different_split():
    a = split_eval_queries(_queries(), seed=17)
    b = split_eval_queries(_queries(), seed=18)

    assert [q.id for q in a.test] != [q.id for q in b.test]


def test_a_slice_larger_than_the_clean_set_is_rejected():
    with pytest.raises(ValueError, match="slice_size"):
        split_eval_queries(_queries(n_clean=10), slice_size=40)
```

Add to `tests/unit/test_intent_evaluation.py`:

```python
def test_separability_auc_is_one_for_perfectly_separated_scores():
    from src.model.intent_evaluation import separability_report

    report = separability_report([0.8, 0.9, 0.85], [0.1, 0.2, 0.15])

    assert report["auc"] == pytest.approx(1.0)
    assert report["cohens_d"] > 0


def test_separability_auc_is_half_for_identical_distributions():
    from src.model.intent_evaluation import separability_report

    report = separability_report([0.5, 0.6, 0.7], [0.5, 0.6, 0.7])

    assert report["auc"] == pytest.approx(0.5)


def test_raw_margin_can_shrink_while_separability_improves():
    """The whole reason the bar changed units: raw margin is scale-dependent."""
    from src.model.intent_evaluation import separability_report

    wide = separability_report([0.9, 0.5], [0.4, 0.0])          # margin 0.5
    tight = separability_report([0.81, 0.80], [0.79, 0.78])     # margin 0.02

    assert tight["raw_margin"] < wide["raw_margin"]
    assert tight["auc"] >= wide["auc"]


def test_separability_reports_the_overlap_bounds():
    from src.model.intent_evaluation import separability_report

    report = separability_report([0.4, 0.9], [0.1, 0.5])

    assert report["max_out_of_scope"] == pytest.approx(0.5)
    assert report["min_in_scope"] == pytest.approx(0.4)


def test_separability_rejects_an_empty_group():
    from src.model.intent_evaluation import separability_report

    with pytest.raises(ValueError, match="empty"):
        separability_report([], [0.1])
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `pytest tests/unit/test_intent_eval_split.py tests/unit/test_intent_evaluation.py -v`
Expected: FAIL at collection for the split module; `ImportError` for `separability_report`.

- [ ] **Step 3: Implement the split**

Create `src/model/intent_eval_split.py`:

```python
"""The tuning/test split that keeps hyperparameters off the reported number.

Three hyperparameters need values — top_k and the two abstention thresholds —
and choosing any of them on the queries used to report accuracy inflates that
accuracy. The split spends the cheapest data first: the legacy queries are
already contaminated (the canonical set was iterated against them during
curation), so they are worthless as a gate and free to tune on, which
preserves the clean queries as an untouched test set.

Pure Python: no numpy, no encoder. The split must be reproducible anywhere.
"""

from __future__ import annotations

import random
from collections.abc import Sequence
from dataclasses import dataclass

LEGACY_PREFIX = "eval-"
DEFAULT_SLICE_SIZE = 40
DEFAULT_SEED = 17


@dataclass(frozen=True)
class EvalSplit:
    """Queries hyperparameters may see, and queries they may not."""

    tuning: tuple
    test: tuple


def split_eval_queries(
    queries: Sequence,
    *,
    slice_size: int = DEFAULT_SLICE_SIZE,
    seed: int = DEFAULT_SEED,
) -> EvalSplit:
    """Split into (tuning, test).

    Tuning is every legacy query plus a seeded, route-stratified *slice_size*
    sample of the clean ones. Test is the remaining clean queries.
    """
    legacy = [q for q in queries if q.id.startswith(LEGACY_PREFIX)]
    clean = [q for q in queries if not q.id.startswith(LEGACY_PREFIX)]
    if slice_size >= len(clean):
        raise ValueError(
            f"slice_size {slice_size} leaves no test queries: only "
            f"{len(clean)} clean queries available"
        )

    by_route: dict[str, list] = {}
    for query in clean:
        by_route.setdefault(query.label, []).append(query)

    rng = random.Random(seed)
    sampled: list = []
    # Round-robin across routes so the slice is stratified even when
    # slice_size is not divisible by the number of routes.
    pools = {
        route: rng.sample(group, len(group)) for route, group in sorted(by_route.items())
    }
    while len(sampled) < slice_size:
        for route in sorted(pools):
            if len(sampled) == slice_size:
                break
            if pools[route]:
                sampled.append(pools[route].pop())

    chosen = {query.id for query in sampled}
    return EvalSplit(
        tuning=tuple(legacy + sampled),
        test=tuple(q for q in clean if q.id not in chosen),
    )
```

- [ ] **Step 4: Implement scale-free separability**

Append to `src/model/intent_evaluation.py`:

```python
def separability_report(
    in_scope: Sequence[float], out_of_scope: Sequence[float]
) -> dict[str, Any]:
    """How well in-scope and out-of-scope confidences separate.

    Reported scale-free, because raw margin is not comparable across encoders.
    e5 compresses cosine similarities into a narrow high band: measured against
    the same anchors it scores a *smaller* raw margin than MiniLM (0.0401 vs
    0.1188) while being clearly better separated (AUC 0.927 vs 0.868). A bar in
    raw cosine units would reject the better model, so the bar is AUC and the
    raw margin is reported as encoder-specific context only.
    """
    from sklearn.metrics import roc_auc_score

    in_scope = tuple(float(value) for value in in_scope)
    out_of_scope = tuple(float(value) for value in out_of_scope)
    if not in_scope or not out_of_scope:
        raise ValueError("separability needs a non-empty group on both sides")

    in_mean = sum(in_scope) / len(in_scope)
    out_mean = sum(out_of_scope) / len(out_of_scope)

    def _variance(values: tuple[float, ...], mean: float) -> float:
        return sum((value - mean) ** 2 for value in values) / max(len(values), 1)

    pooled = ((_variance(in_scope, in_mean) + _variance(out_of_scope, out_mean)) / 2) ** 0.5
    labels = [1] * len(in_scope) + [0] * len(out_of_scope)
    return {
        "auc": float(roc_auc_score(labels, list(in_scope) + list(out_of_scope))),
        "cohens_d": float((in_mean - out_mean) / pooled) if pooled else 0.0,
        "raw_margin": in_mean - out_mean,
        "max_out_of_scope": max(out_of_scope),
        "min_in_scope": min(in_scope),
        "counts": {"in_scope": len(in_scope), "out_of_scope": len(out_of_scope)},
    }
```

- [ ] **Step 5: Sweep three hyperparameters on the tuning slice only**

In `src/model/intent_index_eval.py`, replace `_select_thresholds`'s two-dimensional sweep with a three-dimensional one over `_SWEEP_TOP_K × _SWEEP_MIN_CONFIDENCES × _SWEEP_MIN_MARGINS`, evaluated on the **tuning** queries. Keep the existing selection rule — highest served accuracy at coverage ≥ `_MIN_COVERAGE`, ties broken toward higher out-of-scope deferral — and record every row.

`run_index_evaluation` changes shape:

1. Load all eval queries, call `split_eval_queries`, and log the two sizes.
2. Sweep on `split.tuning`; the winner is `(top_k, min_confidence, min_margin)`.
3. Report on `split.test` and on `hard_40`, at the selected hyperparameters, plus `legacy_30` and the tuning slice **labelled as tuned-on**.
4. Use `separability_report` for out-of-scope, on the test slice.
5. Write the selected values, the full sweep, the split's seed and sizes, and the separability block into `evaluation_report.json`.

The report's `headline` carries `test_slice_accuracy`, `test_slice_size`, `hard_accuracy`, `out_of_scope_auc`, `cohens_d`, `leave_one_out_accuracy`, and the three selected hyperparameters.

- [ ] **Step 6: Run the tests**

Run: `pytest tests/unit/test_intent_eval_split.py tests/unit/test_intent_evaluation.py tests/unit/test_intent_index_eval.py -v`
Expected: PASS. Update any existing `test_intent_index_eval.py` assertion that referenced the old `clean_151` keys.

- [ ] **Step 7: Commit**

```bash
ruff check . --fix && ruff format .
git add src/model/intent_eval_split.py src/model/intent_evaluation.py \
  src/model/intent_index_eval.py tests/unit/test_intent_eval_split.py \
  tests/unit/test_intent_evaluation.py tests/unit/test_intent_index_eval.py
git commit -m "feat(intent): tune hyperparameters off the test slice, score separability scale-free"
```

---

### Task 3: Rebuild, measure, decide, document

**Files:**
- Modify: `src/internal/configs/app_configs.py`, `tests/unit/test_intent_index_eval.py`, `docs/training-and-evaluation.md`, `docs/configuration.md`
- Rebuild: `data/intent_index/` (gitignored artifact)

**Interfaces:**
- Consumes: everything from Tasks 1-2.
- Produces: re-tuned defaults; re-pinned floors; the PR.

- [ ] **Step 1: Rebuild the index with the new encoder**

```bash
python -m src.model.intent_index_cli build \
  --canonical data/intent_canonical.json --output data/intent_index
```
Expected: 280 examples, `low support modules (0): none`. The index is now e5-small-v2 at 384 dimensions.

Confirm the stale-index guard fires by pointing the router at an index built before this change, if one is still around — or trust Task 1's unit test, which covers it.

- [ ] **Step 2: Tune and measure**

```bash
python -m src.model.intent_index_cli evaluate \
  --index data/intent_index \
  --eval-queries data/intent_eval_queries.json \
  --hard-queries data/intent_eval_hard.json \
  --out-of-scope data/intent_out_of_scope.json \
  --canonical data/intent_canonical.json \
  --output data/intent_index/evaluation_report.json
```

Record from the report: the selected `(top_k, min_confidence, min_margin)`; test-slice accuracy and size; `hard_40`; out-of-scope AUC and Cohen's d; leave-one-out; and the tuning-slice accuracy, labelled as tuned-on and **not** to be quoted as a result.

- [ ] **Step 3: Measure end-to-end latency**

```bash
python - <<'PY'
import statistics, time
from pathlib import Path

from src.model.intent_encoder import encode_texts
from src.model.intent_knn import INDEX_FILENAME, IntentIndex

index = IntentIndex.load(Path("data/intent_index") / INDEX_FILENAME)
query = "where did we land on the index rebuild last week"
thresholds = dict(min_confidence=0.30, min_margin=0.02, min_module_score=0.45)
for _ in range(10):
    index.decide(encode_texts([query])[0], **thresholds)
timings = []
for _ in range(200):
    start = time.perf_counter()
    index.decide(encode_texts([query])[0], **thresholds)
    timings.append((time.perf_counter() - start) * 1000)
timings.sort()
print(f"p50 {statistics.median(timings):.2f}ms  p95 {timings[190]:.2f}ms")
PY
```
Substitute the selected thresholds. Expected around p95 12ms; the bar is 25ms.

- [ ] **Step 4: Apply the decision rule**

Against **test-slice** route accuracy — never the tuning slice, never `legacy_30`:

| test-slice accuracy | verdict |
|---|---|
| ≥ 0.80 | clears the promotion bar; say so in the PR, artifact still ships dark |
| 0.75 – 0.80 | real improvement over `0.6225`; artifact stays dark |
| < 0.75 | **HARD STOP** — report every number and stop |

Also confirm out-of-scope AUC ≥ `0.90`, p95 ≤ `25ms`, and no module below support.

**Expect the number to be lower than the fitted `0.8287`.** That is what a fitted figure does when you stop fitting it, and it is the point of the split. If the hard stop trips, report and stop; do not tune further, do not re-split with a different seed, and do not touch the canonical set or either eval file. Re-rolling the seed until the number improves is the same error this task exists to correct.

- [ ] **Step 5: Set the tuned values as defaults**

In `src/internal/configs/app_configs.py`, set `intent_top_k`, `intent_model_min_confidence`, and `intent_min_route_margin` to the selected values, in both the dataclass fields and the `get_env_float`/`get_env_int` fallbacks. Mirror them in `default_config.py`.

- [ ] **Step 6: Re-pin the bars**

In `tests/unit/test_intent_index_eval.py`, replace the floors with the measured values minus ~0.02, and replace the out-of-scope floor with an **AUC** floor rather than a raw-margin one. Keep the latency ceiling at `25.0`. Note in a comment that these run only where sentence-transformers is installed — they still skip on CI.

- [ ] **Step 7: Run the whole suite**

Run: `pytest`
Expected: PASS. Then re-run the no-ML-dependency import proof from Task 1 Step 5.

- [ ] **Step 8: Document**

In `docs/training-and-evaluation.md`:

- Update the encoder name, the prefix contract, and the rebuild requirement — **every existing index is invalidated**, and because e5-small-v2 is also 384-wide the guard that catches it is the encoder-name check, not a dimension mismatch.
- Replace the measured-results block with the new figures, keeping the MiniLM numbers as the "before" column.
- Correct the out-of-scope bar's units: AUC and Cohen's d, with raw margin retained as encoder-specific context that must not be compared across encoders. Explain why, with the measured inversion (e5 scores a smaller raw margin while separating better).
- Describe the tuning/test split, its seed and sizes, and state plainly that the reported number comes from queries no hyperparameter ever saw — and that the earlier `0.8287` was fitted.
- Update the units note: `AGENTIC_SEARCH_INTENT_MODEL_MIN_CONFIDENCE` is a cosine, and its scale has changed again with the encoder, so a value carried from the MiniLM configuration is meaningless.
- Keep the `TOP_K` correction section, adding that k was re-tuned on the split rather than read off a table.

In `docs/configuration.md`, update the three defaults and `AGENTIC_SEARCH_INTENT_TOP_K`.

- [ ] **Step 9: Commit and open the PR**

```bash
ruff check . --fix && ruff format .
git branch --show-current   # must be feat/intent-encoder-e5
git log --oneline origin/main..HEAD   # must show only this branch's commits
git push -u origin feat/intent-encoder-e5
gh pr create --title "feat(intent): route with e5-small-v2 instead of MiniLM" --body "..."
```

The PR body must state: test-slice accuracy with `0.6225` as the before, and the test slice's size; the selected hyperparameters and that they were tuned on the legacy-30 plus a 40-query slice, never on the reported queries; out-of-scope AUC and Cohen's d against `0.868`/`1.49`, and that the bar changed units and why; p95 latency against `6.1ms`; leave-one-out as a diagnostic, with the note that it is a biased selector for k; which decision-rule band was reached; that the artifact still ships dark and promotion is a separate change; that every existing index is invalidated; and **"Please review before merging."** Link the spec and this plan. Do NOT merge.

---

## Verification summary

| Spec requirement | Task |
|---|---|
| e5-small-v2 as the default encoder | 1 |
| `"query: "` applied symmetrically, inside the seam | 1 |
| Unregistered model raises rather than defaulting to no prefix | 1 |
| Stale 384-d index rejected by encoder name | 1 |
| 384 dimensions, index format unchanged | 1 (Step 1 test) |
| No hyperparameter chosen on the test slice | 2, 3 |
| Legacy-30 spent on tuning, clean remainder held out | 2 |
| Leave-one-out reported, never a selector | 2, 3 |
| Out-of-scope bar in AUC, not raw margin | 2, 3 |
| Three-dimensional sweep recorded | 2 |
| Decision rule on the test slice, hard stop below 0.75 | 3 (Step 4) |
| p95 ≤ 25ms | 3 (Step 3) |
| Ships dark; promotion separate | 3 |
| Fitted-number caveat stated | 3 (Step 8) |
| Cascade, taxonomy, canonical set, eval files untouched | all |
