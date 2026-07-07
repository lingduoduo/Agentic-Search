# Consolidate intent: one trained `{chat, search, tool}` classifier — design

## Problem

There are two disconnected "intent" stacks:

1. **Web routing** — `src/internal/servers/web/intent_routing.py`: regex cues + a
   prompted-LLM classifier (`classify_route`) choose `RouteStrategy` ∈
   `{chat, search, tool}` for every `/api/agent` request. The LLM classifier
   costs an up-front LLM call per query that regex can't decide.
2. **Trained MLP** — `src/model/intent_classifier.py`: a PyTorch embedding-MLP
   trained to classify `{purchase, navigate, qa, recommendation}`. It is wired
   only into the CLI example (`examples/run_agentic_search.py`) for model-tier
   routing + search-settings policy. It never touches the live web path.

The trained model and the live router use different label taxonomies and never
meet. We want **one** trained classifier that produces the web routing labels
`{chat, search, tool}` and serves the live router — replacing the per-query LLM
classification with a fast local model, while keeping the LLM as a fallback.

## Goal

Retarget the trained classifier to `{chat, search, tool}` and insert it into the
`route_query` cascade as a fast, confidence-gated step that replaces the LLM
classification when a model is available. Migrate the CLI consumer to the new
labels. One label taxonomy, one model, serving both surfaces.

## Non-goals

- Not changing the MLP architecture (embedding → mean-pool → FC×3 → softmax).
- Not changing regex routing, `_rule_based_route`, or `_infer_intent_from_output`.
- Not adding distillation (auto-labeling a corpus with the regex+LLM router) —
  training data stays synthetic-template-based, as the repo already does.
  Distillation is a documented follow-up.
- Not introducing new external dependencies.

## Cascade (after)

```
route_query(query, *, llm, explicit_source):
  1. explicit_source                       -> SEARCH        (mechanism="explicit_source")
  2. _regex_route(query) confident?         -> that route    (mechanism="regex")
  3. predict_route(query) with conf >= T?   -> that route    (mechanism="model")   ← NEW
  4. llm present?    classify_route(query)  -> that route    (mechanism="classifier")
       (on error)    _rule_based_route      -> that route    (mechanism="rule_based")
  5. no llm          _rule_based_route      -> that route    (mechanism="rule_based")
```

`T` = `AGENTIC_SEARCH_INTENT_MODEL_MIN_CONFIDENCE` (float, default `0.6`).

**Safety property — ships dark.** When no model path is configured (the
default), `predict_route` returns `None`, step 3 is skipped, and `route_query`
behaves **byte-for-byte as today** (regex → LLM). The LLM remains the
low-confidence fallback even when a model is loaded. So this changes routing
only after an operator sets `AGENTIC_SEARCH_INTENT_MODEL_PATH` to a trained
checkpoint.

## Components

### 1. `src/model/intent_classifier.py` — retarget labels
- `INTENT_LABELS = ["chat", "search", "tool"]` — same order as the
  `RouteStrategy` enum declaration (CHAT, SEARCH, TOOL), so class indices align.
- `num_classes` follows `len(INTENT_LABELS)` (already parameterized).
- Rewrite `resolve_search_settings`'s policy map for the new labels:
  - `chat`: passthrough `(topk, max_search_limit, require_evidence, allow_internal_knowledge)` — synthesis, internal knowledge allowed.
  - `search`: `(max(topk, 8), max(max_search_limit, 3), True, False)` — retrieval-heavy, require evidence, no internal knowledge.
  - `tool`: passthrough — action path needs no extra retrieval.
  - Confidence gate (`min_confidence=0.6`) unchanged.

### 2. `src/model/intent_training.py` — regenerate synthetic data
- Rewrite `build_examples_for_document` to emit `{chat, search, tool}` examples
  templated around each corpus doc's title/terms, seeded from the routing cue
  vocabulary already used in `intent_routing.py`. Illustrative templates:
  - `search`: `find {title}`, `look up {t1}`, `{title}` (bare lookup), `retrieve the {t2} docs`
  - `chat`: `what is {title}?`, `explain {t1}`, `compare {t1} and {t2}`, `summarize {title}`
  - `tool`: `send an email about {title}`, `create a ticket for {t1}`, `schedule a meeting about {title}`, `open a PR for {t2}`
- `INTENTS = tuple(INTENT_LABELS)` continues to drive the deterministic sort key
  (now 3 labels).

### 3. `src/internal/servers/web/ml_intent.py` — new lazy adapter
Mirrors the `gate_embedder` pattern (lazy singleton, graceful `None`):
- `intent_min_confidence() -> float` — reads `AGENTIC_SEARCH_INTENT_MODEL_MIN_CONFIDENCE`, default `0.6`.
- `load_intent_model() -> IntentPipeline | None` — module-level singleton;
  reads `AGENTIC_SEARCH_INTENT_MODEL_PATH`; `None` when unset/empty; imports
  `IntentPipeline` lazily and `IntentPipeline.load(path)`; on any exception logs
  and caches a failed sentinel → `None` (never retries, never raises).
- `predict_route(query) -> tuple[RouteStrategy, float] | None` —
  `None` when no model; else `pred = model.predict_text(query)`, map
  `pred.intent` → `RouteStrategy(pred.intent)` (valid because labels == enum
  values), return `(strategy, pred.confidence)`. Unknown label or a raised
  `predict_text` → `None` (fall through to the LLM). Torch stays out of the
  import path (lazy import inside the loader).

### 4. `src/internal/servers/web/intent_routing.py` — insert the step
- Import `predict_route`, `intent_min_confidence` from `ml_intent`.
- In `route_query`, after the confident-`_regex_route` return and before the
  `llm` branch:
  ```python
  model_choice = predict_route(query)
  if model_choice is not None:
      strategy, confidence = model_choice
      if confidence >= intent_min_confidence():
          _record_intent("model", strategy, {"confidence": confidence})
          return strategy
  ```
  Low confidence or `None` falls through to the existing LLM / rule-based steps.

### 5. `examples/run_agentic_search.py` — migrate the CLI
- `_resolve_model_route`'s `route_by_intent` remapped to the new labels:
  `{"search": "fast", "chat": "balanced", "tool": "reasoning"}`. Unknown → `base`.

## Configuration

- `AGENTIC_SEARCH_INTENT_MODEL_PATH` — path to a trained `.pt`; unset ⇒ ML step
  off (default). Read via `os.environ` in `ml_intent` (matching the
  `gate_embedder` env pattern); documented in `default_config.py`.
- `AGENTIC_SEARCH_INTENT_MODEL_MIN_CONFIDENCE` — float, default `0.6`.

## Observability

`route_query` records a `mechanism="model"` intent stage with `{confidence}`
when the ML step decides, so the Request Inspector distinguishes
`intent · model` from `intent · regex` / `intent · classifier`.

## Testing

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

## Success criteria

- With `AGENTIC_SEARCH_INTENT_MODEL_PATH` unset, `/api/agent` routing is
  identical to today (verified by the no-model route_query test).
- With a trained `.pt` set, a query the model is confident about routes without
  any LLM call, and low-confidence queries still reach the LLM.
- One taxonomy (`{chat, search, tool}`, aligned to `RouteStrategy`) drives the
  classifier, the web router, and the CLI. No recsys labels remain.
- The Request Inspector shows `intent · model` with its confidence.
