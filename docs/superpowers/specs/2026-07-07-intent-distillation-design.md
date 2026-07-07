# Intent router distillation — design

## Problem

The trained intent classifier (`src/model/intent_classifier.py`, now
`{chat, search, tool}`) is trained by `intent_training.py` on **synthetic
templates** built around corpus document titles/terms. A prototype showed this
teaches the bag-of-embeddings MLP to key on **topic tokens** rather than intent
cues: e.g. `"compare dense and sparse retrieval"` and `"retrieval augmented
generation overview"` both route to **search** (fooled by "retrieval"/"sparse")
when they are clearly **chat**. A confident-but-wrong prediction bypasses the
LLM (per the confidence gate) and misroutes.

## Goal

Add a **distillation** path: label a real/broad query set with the *current*
router (regex → LLM classifier → rule-based) and train the MLP on those labels,
so it learns the router's intent decisions (verb/structure) instead of topic
words. The prototype proved this fixes the exact failures (distilled 8/8 vs
topic-templated 6/8 on the probe set), including a non-verbatim generalization
(`compare X` → chat regardless of topic).

## Non-goals

- Not changing the classifier architecture, `route_query`, `ml_intent`, or the
  ships-dark gate (this only produces a better `.pt`).
- Not changing the existing synthetic `intent_training.py` path (kept as an
  offline-friendly alternative).
- Not auto-scheduling retraining or adding a training service.

## Architecture

A new `src/model/intent_distillation.py` module. Its core is teacher-driven
labeling that reuses the router's own decision functions:

```
label = _regex_route(q)  (high precision)
        else classify_route(q, llm)  (ambiguous tail, when an LLM is given)
        else _rule_based_route(q)    (offline fallback)
```

The router helpers (`_regex_route`, `_rule_based_route`, `classify_route`) are
imported **lazily inside `label_query`** so the training module has no
import-time dependency on the web layer.

Data flow: `queries → build_distillation_examples → [{text, label}] →
train_intent_classifier → .pt`. Query sources: a file (one query per line, or a
JSON list) and the SQLite store's logged user messages.

## Components

### `src/model/intent_distillation.py`
- `label_query(query, *, llm=None) -> tuple[str, str]` — returns `(label,
  teacher)` where `teacher ∈ {"regex", "llm", "rule_based"}`. Lazy-imports the
  router helpers. On `classify_route` error, falls to `_rule_based_route`.
- `build_distillation_examples(queries, *, llm=None) -> list[dict]` — maps each
  query to `{"text": query, "label": label}` (drops empties).
- `DistillResult` (frozen dataclass): `pipeline`, `num_examples`,
  `label_counts: dict[str,int]`, `teacher_counts: dict[str,int]`.
- `distill_and_train(queries, *, output_path, examples_path, llm=None,
  epochs=15, lr=1e-3, min_freq=1) -> DistillResult` — label → write the examples
  JSON → `train_intent_classifier(...)` → return the result plus teacher_counts.
- `load_queries_from_file(path) -> list[str]` — accepts a `.txt` (one query per
  line, blanks skipped) or a `.json` list of strings/`{text|question}` dicts.

### `src/internal/db/store.py`
- `get_user_query_texts(limit: int | None = None) -> list[str]` — distinct
  non-empty `chat_messages.content WHERE role='user'`, newest first, optional
  cap. Provides a real logged-query corpus.

### CLI — `python -m src.model.intent_distillation`
- `--queries-file PATH` and/or `--from-db SQLITE_PATH` (union of sources; dedup,
  order-preserving) → `--output model.pt` (`--examples-out examples.json`
  optional, defaults beside the model).
- Optional LLM teacher for the ambiguous tail: `--vllm-url` + `--model`
  (+ `--api-key`/env) build an `OpenAICompatibleLLM`; omitted ⇒ regex +
  rule-based only (fully offline).
- `--epochs` / `--lr` passthrough. Prints `num_examples`, `label_counts`,
  `teacher_counts`, and the saved path.

## Error handling

- Empty query list → `distill_and_train` raises `ValueError` (nothing to train).
- `classify_route` raising per-query → that query falls to `_rule_based_route`
  (never aborts the batch).
- `load_queries_from_file` on a missing file → `FileNotFoundError`; malformed
  JSON → `ValueError`.

## Testing

- `label_query`: regex-labelable query → `("search"/"chat"/"tool", "regex")`
  with no LLM touched; a query regex defers on + a **fake LLM** → `(label,
  "llm")`; same query, no LLM → `(label, "rule_based")`; a fake LLM that raises
  → falls to `"rule_based"`.
- `build_distillation_examples`: returns `{text,label}` per query; drops blanks.
- `distill_and_train`: a tiny query list (no LLM) trains and writes a `.pt` that
  `IntentPipeline.load` reloads and predicts one of `{chat,search,tool}`;
  `teacher_counts` sums to `num_examples`; empty list → `ValueError`.
- `load_queries_from_file`: `.txt` and `.json` forms; missing file raises.
- `get_user_query_texts`: seed user + assistant messages; returns only distinct
  user contents, newest first, honoring `limit`.
- No test requires an LLM or loads a real external model; the CLI test runs
  offline (regex+rule_based teacher) on a temp file and asserts a `.pt` is
  produced. Full suite green; ruff clean.

## Success criteria

- `python -m src.model.intent_distillation --queries-file q.txt --output m.pt`
  produces a loadable classifier whose labels come from the router, with printed
  teacher/label counts.
- On the prototype probe set, a distilled model routes `"compare dense and
  sparse retrieval"` and `"retrieval augmented generation overview"` to **chat**
  (the topic-templated model's failures), demonstrated by a test or the CLI.
- The distillation module has no import-time dependency on the web server layer
  (router helpers imported lazily).
- Existing synthetic `intent_training.py` path and the router remain unchanged.
