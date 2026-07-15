# Intent Classifier → `{chat, search, tool}` Route Consolidation — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Retarget the trained MLP intent classifier to `{chat, search, tool}` and insert it into `route_query` as a confidence-gated step that replaces the LLM classification when a model is configured, migrating the CLI consumer to the new labels.

**Architecture:** One label taxonomy (`RouteStrategy` values) drives the classifier, the live web router, and the CLI. A lazy `ml_intent` adapter (gate-embedder pattern: graceful `None`) bridges `src/model` → `route_query`. Ships dark: no model path ⇒ routing is byte-for-byte today's regex→LLM behavior.

**Tech Stack:** Python 3.12, PyTorch (lazy-imported), FastAPI, pytest.

## Global Constraints

- Label taxonomy is exactly `["chat", "search", "tool"]`, in `RouteStrategy` declaration order (CHAT, SEARCH, TOOL); class indices align with the enum.
- Ships dark: with `AGENTIC_SEARCH_INTENT_MODEL_PATH` unset, `predict_route` returns `None` and `route_query` behaves exactly as today. The LLM stays the low-confidence fallback even when a model is loaded.
- Torch/`IntentPipeline` are lazy-imported inside functions, never at module import time (keeps web import + hermetic tests torch-free). No test loads a real `.pt`.
- `T` = `float(os.environ.get("AGENTIC_SEARCH_INTENT_MODEL_MIN_CONFIDENCE", "0.6"))`.
- Run `ruff check <files> --fix && ruff format <files>` before each commit (pre-commit hook; if it reformats and aborts, `git add -A` and re-run the same commit).
- Branch: `feat/intent-classifier-route-consolidation` (spec already committed there).

---

### Task 1: Retarget classifier labels + `resolve_search_settings` policy

**Files:**
- Modify: `src/model/intent_classifier.py` (`INTENT_LABELS` line 9; `resolve_search_settings` policy ~line 212)
- Modify: `tests/unit/test_intent_classifier.py` (retarget label assertions + policy cases)

**Interfaces:**
- Produces: `INTENT_LABELS == ["chat", "search", "tool"]`; `resolve_search_settings` policy keyed by the new labels. `IntentPrediction`, `IntentPipeline` signatures unchanged.

- [ ] **Step 1: Update the label list and policy**

In `src/model/intent_classifier.py`, change:

```python
INTENT_LABELS: list[str] = ["chat", "search", "tool"]
```

and replace the `policy` dict inside `resolve_search_settings` with:

```python
    policy: dict[str, tuple[int, int, bool, bool]] = {
        "chat": (topk, max_search_limit, require_evidence, allow_internal_knowledge),
        "search": (max(topk, 8), max(max_search_limit, 3), True, False),
        "tool": (topk, max_search_limit, require_evidence, allow_internal_knowledge),
    }
```

(The `.get(prediction.intent, (topk, max_search_limit, require_evidence, allow_internal_knowledge))` fallback and the `min_confidence` gate stay as-is.)

- [ ] **Step 2: Update the classifier tests**

In `tests/unit/test_intent_classifier.py`, retarget any assertions that reference the old labels. Replace the `INTENT_LABELS` assertion and any `resolve_search_settings` label cases:

```python
def test_intent_labels_are_route_strategy_values():
    from src.model.intent_classifier import INTENT_LABELS

    assert INTENT_LABELS == ["chat", "search", "tool"]


def test_resolve_search_settings_search_is_retrieval_heavy():
    from src.model.intent_classifier import IntentPrediction, resolve_search_settings

    t, s, r, a, meta = resolve_search_settings(
        IntentPrediction(intent="search", confidence=0.9),
        topk=5,
        max_search_limit=2,
        require_evidence=False,
        allow_internal_knowledge=True,
    )
    assert (t, s, r, a) == (8, 3, True, False)
    assert meta["intent_policy_applied"] is True


def test_resolve_search_settings_chat_passthrough():
    from src.model.intent_classifier import IntentPrediction, resolve_search_settings

    t, s, r, a, _meta = resolve_search_settings(
        IntentPrediction(intent="chat", confidence=0.9),
        topk=5,
        max_search_limit=2,
        require_evidence=False,
        allow_internal_knowledge=True,
    )
    assert (t, s, r, a) == (5, 2, False, True)


def test_low_confidence_leaves_settings_unchanged():
    from src.model.intent_classifier import IntentPrediction, resolve_search_settings

    t, s, r, a, meta = resolve_search_settings(
        IntentPrediction(intent="search", confidence=0.1),
        topk=5,
        max_search_limit=2,
        require_evidence=False,
        allow_internal_knowledge=True,
    )
    assert (t, s, r, a) == (5, 2, False, True)
    assert meta["intent_policy_applied"] is False
```

Update any other test in the file that trains on / asserts the old recsys labels to use `{chat, search, tool}`.

- [ ] **Step 3: Run the classifier tests**

Run: `python -m pytest tests/unit/test_intent_classifier.py -q`
Expected: PASS.

- [ ] **Step 4: Commit**

```bash
ruff check src/model/intent_classifier.py tests/unit/test_intent_classifier.py --fix && ruff format src/model/intent_classifier.py tests/unit/test_intent_classifier.py
git add src/model/intent_classifier.py tests/unit/test_intent_classifier.py
git commit -m "feat(intent): retarget classifier to {chat,search,tool} labels + policy"
```

---

### Task 2: Regenerate synthetic training templates for the new labels

**Files:**
- Modify: `src/model/intent_training.py` (`build_examples_for_document` ~lines 76-129)
- Test: `tests/unit/test_intent_training.py` (create if absent; else append)

**Interfaces:**
- Consumes: `INTENT_LABELS` (Task 1). `INTENTS = tuple(INTENT_LABELS)` continues to drive the sort key.
- Produces: `build_examples_for_document(doc, vocab_tokens)` yields examples labeled only `{chat, search, tool}`.

- [ ] **Step 1: Write the failing test**

Create/append `tests/unit/test_intent_training.py`:

```python
from src.model.intent_training import build_examples_for_document


def test_build_examples_emit_only_route_labels():
    doc = {"id": "d1", "title": "FAISS", "contents": "vector index library"}
    examples = build_examples_for_document(doc, ["vector", "index", "ranking"])
    labels = {e["label"] for e in examples}
    assert labels <= {"chat", "search", "tool"}
    assert labels == {"chat", "search", "tool"}  # all three represented
```

- [ ] **Step 2: Run to verify it fails**

Run: `python -m pytest tests/unit/test_intent_training.py -q`
Expected: FAIL — current templates emit `qa/navigate/recommendation/purchase`.

- [ ] **Step 3: Rewrite the templates**

In `src/model/intent_training.py`, replace the `examples = [...]` list inside `build_examples_for_document` with `{chat, search, tool}` templates seeded from title/terms:

```python
    examples = [
        {"text": f"find {title}", "label": "search"},
        {"text": f"look up {t1}", "label": "search"},
        {"text": f"{title}", "label": "search"},
        {"text": f"retrieve the {t2} documentation", "label": "search"},
        {"text": f"what is {title} and how is it used?", "label": "chat"},
        {"text": f"explain {t1} in {title}", "label": "chat"},
        {"text": f"compare {t1} and {t2}", "label": "chat"},
        {"text": f"summarize {title}", "label": "chat"},
        {"text": f"send an email about {title}", "label": "tool"},
        {"text": f"create a ticket for {t1}", "label": "tool"},
        {"text": f"schedule a meeting about {title}", "label": "tool"},
        {"text": f"open a pull request for {t2}", "label": "tool"},
    ]
```

(The trailing loop that attaches `source_doc_id`/`source_title`/`keywords`/`context_hint` stays unchanged.)

- [ ] **Step 4: Run to verify it passes**

Run: `python -m pytest tests/unit/test_intent_training.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
ruff check src/model/intent_training.py tests/unit/test_intent_training.py --fix && ruff format src/model/intent_training.py tests/unit/test_intent_training.py
git add src/model/intent_training.py tests/unit/test_intent_training.py
git commit -m "feat(intent): regenerate synthetic training data for {chat,search,tool}"
```

---

### Task 3: `ml_intent.py` — lazy loader + `predict_route`

**Files:**
- Create: `src/internal/servers/web/ml_intent.py`
- Test: `tests/unit/servers/web/test_ml_intent.py` (new)

**Interfaces:**
- Consumes: `IntentPipeline` (`src/model/intent_classifier`, `.load(path)` / `.predict_text(query) -> IntentPrediction`); `RouteStrategy` (`src/internal/servers/web/intent_routing`).
- Produces:
  - `intent_min_confidence() -> float`
  - `load_intent_model() -> IntentPipeline | None` (lazy singleton, graceful)
  - `predict_route(query: str) -> tuple[RouteStrategy, float] | None`

- [ ] **Step 1: Write the failing tests**

Create `tests/unit/servers/web/test_ml_intent.py`:

```python
import src.internal.servers.web.ml_intent as ml_intent
from src.internal.servers.web.intent_routing import RouteStrategy
from src.model.intent_classifier import IntentPrediction


class _StubPipeline:
    def __init__(self, intent, confidence):
        self._pred = IntentPrediction(intent=intent, confidence=confidence)

    def predict_text(self, _query):
        return self._pred


def test_min_confidence_default_and_override(monkeypatch):
    monkeypatch.delenv("AGENTIC_SEARCH_INTENT_MODEL_MIN_CONFIDENCE", raising=False)
    assert ml_intent.intent_min_confidence() == 0.6
    monkeypatch.setenv("AGENTIC_SEARCH_INTENT_MODEL_MIN_CONFIDENCE", "0.75")
    assert ml_intent.intent_min_confidence() == 0.75


def test_no_model_path_returns_none(monkeypatch):
    monkeypatch.delenv("AGENTIC_SEARCH_INTENT_MODEL_PATH", raising=False)
    monkeypatch.setattr(ml_intent, "_INTENT_MODEL", None)
    assert ml_intent.load_intent_model() is None
    assert ml_intent.predict_route("find FAISS") is None


def test_predict_route_maps_label_and_confidence(monkeypatch):
    monkeypatch.setattr(ml_intent, "load_intent_model", lambda: _StubPipeline("search", 0.91))
    result = ml_intent.predict_route("find FAISS")
    assert result == (RouteStrategy.SEARCH, 0.91)


def test_predict_route_unknown_label_returns_none(monkeypatch):
    monkeypatch.setattr(ml_intent, "load_intent_model", lambda: _StubPipeline("purchase", 0.99))
    assert ml_intent.predict_route("buy a thing") is None


def test_predict_route_swallows_predict_errors(monkeypatch):
    class _Boom:
        def predict_text(self, _q):
            raise RuntimeError("boom")

    monkeypatch.setattr(ml_intent, "load_intent_model", lambda: _Boom())
    assert ml_intent.predict_route("anything") is None
```

- [ ] **Step 2: Run to verify they fail**

Run: `python -m pytest tests/unit/servers/web/test_ml_intent.py -q`
Expected: FAIL — `ModuleNotFoundError: ...ml_intent`.

- [ ] **Step 3: Create the adapter**

Create `src/internal/servers/web/ml_intent.py`:

```python
"""Lazy adapter: trained intent classifier -> RouteStrategy for route_query."""

from __future__ import annotations

import logging
import os

from src.internal.servers.web.intent_routing import RouteStrategy

logger = logging.getLogger(__name__)

_INTENT_MODEL: object | None = None  # None=unset, False=failed/absent, pipeline=loaded

_ROUTE_VALUES = {s.value for s in RouteStrategy}


def intent_min_confidence() -> float:
    return float(os.environ.get("AGENTIC_SEARCH_INTENT_MODEL_MIN_CONFIDENCE", "0.6"))


def load_intent_model():
    """Lazy singleton trained intent classifier; None when unavailable."""
    global _INTENT_MODEL
    if _INTENT_MODEL is not None:
        return _INTENT_MODEL or None
    path = os.environ.get("AGENTIC_SEARCH_INTENT_MODEL_PATH", "").strip()
    if not path:
        _INTENT_MODEL = False
        return None
    try:
        from src.model.intent_classifier import IntentPipeline

        _INTENT_MODEL = IntentPipeline.load(path)
    except Exception:
        logger.exception("intent-model: load failed — ML routing disabled")
        _INTENT_MODEL = False
        return None
    return _INTENT_MODEL


def predict_route(query: str) -> "tuple[RouteStrategy, float] | None":
    """(RouteStrategy, confidence) from the trained model, or None to defer."""
    model = load_intent_model()
    if model is None:
        return None
    try:
        pred = model.predict_text(query)
    except Exception:
        logger.exception("intent-model: predict failed — deferring")
        return None
    if pred.intent not in _ROUTE_VALUES:
        return None
    return RouteStrategy(pred.intent), float(pred.confidence)
```

- [ ] **Step 4: Run to verify they pass**

Run: `python -m pytest tests/unit/servers/web/test_ml_intent.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
ruff check src/internal/servers/web/ml_intent.py tests/unit/servers/web/test_ml_intent.py --fix && ruff format src/internal/servers/web/ml_intent.py tests/unit/servers/web/test_ml_intent.py
git add src/internal/servers/web/ml_intent.py tests/unit/servers/web/test_ml_intent.py
git commit -m "feat(intent): lazy ml_intent adapter (loader + predict_route)"
```

---

### Task 4: Insert the confidence-gated model step into `route_query`

**Files:**
- Modify: `src/internal/servers/web/intent_routing.py` (imports; `route_query` body, after the confident-`_regex_route` return)
- Test: `tests/unit/servers/web/test_agent_router.py` (append)

**Interfaces:**
- Consumes: `predict_route`, `intent_min_confidence` (Task 3). `route_query` signature is unchanged.
- Produces: `route_query` returns the model route when `predict_route` yields `confidence >= intent_min_confidence()`; otherwise unchanged. New intent stage `mechanism="model"`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/unit/servers/web/test_agent_router.py`:

```python
import src.internal.servers.web.intent_routing as ir


class _SpyLLM:
    def __init__(self):
        self.called = False

    def complete(self, *_a, **_k):
        self.called = True
        return "chat"


def test_route_query_uses_model_when_confident(monkeypatch):
    monkeypatch.setattr(ir, "predict_route", lambda q: (RouteStrategy.SEARCH, 0.9))
    llm = _SpyLLM()
    strategy = ir.route_query(
        "some ambiguous phrasing", llm=llm, has_local_model=True, explicit_source=False
    )
    assert strategy is RouteStrategy.SEARCH
    assert llm.called is False  # model replaced the LLM step


def test_route_query_defers_to_llm_when_model_low_confidence(monkeypatch):
    monkeypatch.setattr(ir, "predict_route", lambda q: (RouteStrategy.SEARCH, 0.3))
    llm = _SpyLLM()
    strategy = ir.route_query(
        "some ambiguous phrasing", llm=llm, has_local_model=True, explicit_source=False
    )
    assert llm.called is True  # low confidence -> LLM fallback
    assert strategy is RouteStrategy.CHAT  # spy LLM returns "chat"


def test_route_query_no_model_is_unchanged(monkeypatch):
    monkeypatch.setattr(ir, "predict_route", lambda q: None)
    llm = _SpyLLM()
    strategy = ir.route_query(
        "some ambiguous phrasing", llm=llm, has_local_model=True, explicit_source=False
    )
    assert llm.called is True  # no model -> today's behavior
    assert strategy is RouteStrategy.CHAT


def test_regex_still_wins_over_model(monkeypatch):
    called = {"model": False}

    def _spy(_q):
        called["model"] = True
        return (RouteStrategy.CHAT, 0.99)

    monkeypatch.setattr(ir, "predict_route", _spy)
    # "find X" matches the anchored search regex -> returns before predict_route.
    strategy = ir.route_query(
        "find the Q3 revenue report", llm=None, has_local_model=False, explicit_source=False
    )
    assert strategy is RouteStrategy.SEARCH
    assert called["model"] is False
```

- [ ] **Step 2: Run to verify they fail**

Run: `python -m pytest tests/unit/servers/web/test_agent_router.py -k "model or regex_still_wins" -v`
Expected: FAIL — `route_query` has no model step / `ir.predict_route` attribute missing.

- [ ] **Step 3: Wire the step**

In `src/internal/servers/web/intent_routing.py`, add the import near the other imports:

```python
from src.internal.servers.web.ml_intent import intent_min_confidence, predict_route
```

In `route_query`, insert directly after the confident-`_regex_route` block (after `return regex_choice`) and before `if llm is not None:`:

```python
    model_choice = predict_route(query)
    if model_choice is not None:
        strategy, confidence = model_choice
        if confidence >= intent_min_confidence():
            _record_intent("model", strategy, {"confidence": confidence})
            return strategy
```

- [ ] **Step 4: Run to verify they pass**

Run: `python -m pytest tests/unit/servers/web/test_agent_router.py -q`
Expected: PASS (existing + new). Existing tests pass because `predict_route` returns `None` by default (no `AGENTIC_SEARCH_INTENT_MODEL_PATH`), leaving the cascade unchanged.

- [ ] **Step 5: Guard against an import cycle**

`ml_intent` imports `RouteStrategy` from `intent_routing`, and `intent_routing` imports from `ml_intent`. Confirm no cycle at import time:

Run: `python -c "import src.internal.servers.web.intent_routing; import src.internal.servers.web.ml_intent; print('ok')"`
Expected: `ok`. (Both import cleanly: `ml_intent` only needs `RouteStrategy`, defined before the `route_query`-level import is exercised; the `intent_routing` → `ml_intent` import is a top-level `from ... import`, so if a cycle appears, move that import inside `route_query`.)

- [ ] **Step 6: Commit**

```bash
ruff check src/internal/servers/web/intent_routing.py tests/unit/servers/web/test_agent_router.py --fix && ruff format src/internal/servers/web/intent_routing.py tests/unit/servers/web/test_agent_router.py
git add src/internal/servers/web/intent_routing.py tests/unit/servers/web/test_agent_router.py
git commit -m "feat(intent): confidence-gated ML route step in route_query (LLM fallback)"
```

---

### Task 5: Migrate the CLI model-route map to the new labels

**Files:**
- Modify: `examples/run_agentic_search.py` (`_resolve_model_route`, `route_by_intent` ~line 217)
- Test: `tests/unit/test_run_agentic_search.py` (retarget the intent cases)

**Interfaces:**
- Consumes: `IntentPrediction` with `{chat, search, tool}` intents.
- Produces: `_resolve_model_route` maps `search→fast`, `chat→balanced`, `tool→reasoning`, unknown→`base`.

- [ ] **Step 1: Update the mapping**

In `examples/run_agentic_search.py`, replace the `route_by_intent` dict in `_resolve_model_route`:

```python
    route_by_intent = {
        "search": "fast",
        "chat": "balanced",
        "tool": "reasoning",
    }
```

- [ ] **Step 2: Retarget the CLI tests**

In `tests/unit/test_run_agentic_search.py`, update the `IntentPrediction(intent=...)` cases that referenced the old labels to the new taxonomy and expected tiers:

```python
    assert _resolve_model_route(args, IntentPrediction(intent="search", confidence=0.8)).route == "fast"
    assert _resolve_model_route(args, IntentPrediction(intent="chat", confidence=0.8)).route == "balanced"
    assert _resolve_model_route(args, IntentPrediction(intent="tool", confidence=0.8)).route == "reasoning"
```

Update the low-confidence case (`confidence=0.2`) to use a new label (e.g. `intent="search"`); its expected `route == "base"` is unchanged.

- [ ] **Step 3: Run the CLI tests**

Run: `python -m pytest tests/unit/test_run_agentic_search.py -q`
Expected: PASS.

- [ ] **Step 4: Full regression + import sanity**

Run: `python -c "import src.internal.servers.web.app"` (OK) and
`python -m pytest tests/unit/test_intent_classifier.py tests/unit/test_intent_training.py tests/unit/servers/web/test_ml_intent.py tests/unit/servers/web/test_agent_router.py tests/unit/test_run_agentic_search.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
ruff check examples/run_agentic_search.py tests/unit/test_run_agentic_search.py --fix && ruff format examples/run_agentic_search.py tests/unit/test_run_agentic_search.py
git add examples/run_agentic_search.py tests/unit/test_run_agentic_search.py
git commit -m "feat(intent): migrate CLI model-route map to {chat,search,tool}"
```

---

## Self-Review

**Spec coverage:** label retarget + policy → Task 1; synthetic templates → Task 2; `ml_intent` loader/predict_route (graceful None, lazy torch) → Task 3; confidence-gated `route_query` step + ships-dark + regex-precedence → Task 4; CLI migration → Task 5. Config env vars are read in `ml_intent` (Task 3) and `intent_min_confidence` (Task 3); observability `mechanism="model"` → Task 4. All spec sections covered.

**Placeholder scan:** every step has concrete code, exact paths, exact commands, expected output. No TBD/TODO.

**Type consistency:** `predict_route(query) -> tuple[RouteStrategy, float] | None` (Task 3) is unpacked as `strategy, confidence` in Task 4. `RouteStrategy(pred.intent)` is valid because `INTENT_LABELS` (Task 1) equals the enum values. `load_intent_model` singleton sentinel (`None`/`False`/pipeline) matches the `gate_embedder` pattern. `resolve_search_settings` returns the existing 5-tuple; only the policy map keys change. `route_query` signature is untouched (still `has_local_model` on this branch — the ML step is inserted in the body, so it does not conflict with the separate `has_local_model`-removal PR beyond a trivial context overlap).

**Ships-dark verification:** Task 4 Step 4 explicitly asserts existing router tests stay green because `predict_route` returns `None` without `AGENTIC_SEARCH_INTENT_MODEL_PATH`.
