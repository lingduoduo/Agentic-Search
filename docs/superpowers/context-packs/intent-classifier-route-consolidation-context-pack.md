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

## Implementation Plan Context

### Task 1: Retarget classifier labels + `resolve_search_settings` policy

**Files:**
- Modify: `src/model/intent_classifier.py` (`INTENT_LABELS` line 9; `resolve_search_settings` policy ~line 212)
- Modify: `tests/unit/test_intent_classifier.py` (retarget label assertions + policy cases)

**Interfaces:**
- Produces: `INTENT_LABELS == ["chat", "search", "tool"]`; `resolve_search_settings` policy keyed by the new labels. `IntentPrediction`, `IntentPipeline` signatures unchanged.

- [ ] **Step 1: Update the label list and policy**

In `src/model/intent_classifier.py`, change:

and replace the `policy` dict inside `resolve_search_settings` with:

…

### Task 2: Regenerate synthetic training templates for the new labels

**Files:**
- Modify: `src/model/intent_training.py` (`build_examples_for_document` ~lines 76-129)
- Test: `tests/unit/test_intent_training.py` (create if absent; else append)

**Interfaces:**
- Consumes: `INTENT_LABELS` (Task 1). `INTENTS = tuple(INTENT_LABELS)` continues to drive the sort key.
- Produces: `build_examples_for_document(doc, vocab_tokens)` yields examples labeled only `{chat, search, tool}`.

- [ ] **Step 1: Write the failing test**

Create/append `tests/unit/test_intent_training.py`:

- [ ] **Step 2: Run to verify it fails**

Run: `python -m pytest tests/unit/test_intent_training.py -q`
Expected: FAIL — current templates emit `qa/navigate/recommendation/purchase`.

…

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

- [ ] **Step 2: Run to verify they fail**

…

## Context Boundary

This pack summarizes its linked sources. Consult those documents for complete details; no implementation status is inferred here.
