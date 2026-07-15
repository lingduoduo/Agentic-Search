# Generated Context Pack

# Intent Classifier Route Consolidation

## Sources

- [Specification: 2026-07-07-intent-classifier-route-consolidation-design.md](../specs/2026-07-07-intent-classifier-route-consolidation-design.md)
- [Plan: 2026-07-07-intent-classifier-route-consolidation.md](../plans/2026-07-07-intent-classifier-route-consolidation.md)

## Specification Context

### Goal

Retarget the trained classifier to `{chat, search, tool}` and insert it into the
`route_query` cascade as a fast, confidence-gated step that replaces the LLM
classification when a model is available. Migrate the CLI consumer to the new
labels. One label taxonomy, one model, serving both surfaces.

### Non-goals

- Not changing the MLP architecture (embedding → mean-pool → FC×3 → softmax).
- Not changing regex routing, `_rule_based_route`, or `_infer_intent_from_output`.
- Not adding distillation (auto-labeling a corpus with the regex+LLM router) —
  training data stays synthetic-template-based, as the repo already does.
  Distillation is a documented follow-up.
- Not introducing new external dependencies.

### Testing

- **`ml_intent`**: no env path → `load_intent_model`/`predict_route` return
  `None`; injected stub pipeline → `predict_route` maps label→`RouteStrategy` +
  confidence; unknown label → `None`; `predict_text` raising → `None`;
  `intent_min_confidence` default + override. No test loads a real `.pt`.
- **`route_query`**: high-confidence model → returns model route, LLM **not**
  called; low-confidence model → LLM fallback; no model → today's behavior
  unchanged; a confident regex match still short-circuits before `predict_route`.
- **`intent_classifier`**: `INTENT_LABELS == ["chat","search","tool"]`; a tiny
  train→predict round-trip returns one of the three; `resolve_search_settings`
  returns the specified policy per new label.
- **`intent_training`**: `build_examples_for_document` yields only
  `{chat, search, tool}` labels.
- **CLI**: `_resolve_model_route` maps each new label to the specified tier.
- Full suite green; ruff clean.

## Implementation Plan Context

### Global Constraints

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

_[Section compacted.]_

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

_[Section compacted.]_

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

_[Section compacted.]_

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

_[Section compacted.]_

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

_[Section compacted.]_

## Context Boundary

This pack summarizes its linked sources. Consult those documents for complete details; no implementation status is inferred here.
