# Intent routing by nearest canonical example — Design

**Status:** proposed
**Branch:** `feat/intent-knn-routing` (off `main` at `1a6084d`)
**Supersedes the model step of:** #506–#510

## Problem

The trained intent model is not promotable and the instrument that judges it cannot
tell whether it ever will be.

Three measurements from `main` state the problem exactly:

| measurement | value | why it blocks |
|---|---|---|
| realistic accuracy | `0.733` | the promotion bar is `0.75` — a gap of **one query** |
| out-of-scope separation margin | `+0.059` | too narrow to abstain on out-of-scope requests |
| evaluation set size | **30 queries** | one query is 3.3 accuracy points |

#510 removed out-of-vocabulary entirely: token coverage went from 47% to 100% and
accuracy from `0.567` to `0.733`. The remaining gap is not a vocabulary problem.
`docs/training-and-evaluation.md` already names the real one:

> a bag of frozen MiniLM input embeddings discards word order, so running the full
> MiniLM encoder behind the same tokenizer is the next step.

The out-of-scope margin has a separate and more fundamental cause. Softmax outputs
sum to 1 by construction, so a fully out-of-scope request still produces a
confident-looking number. A softmax head cannot express "none of these" — the
`+0.059` margin is an artifact of the output layer, not a property that more
training data would fix.

## Approach

Replace the trained MLP with **similarity against a curated set of canonical
routing examples**, encoded by a real sentence encoder.

```
query
  ↓ MiniLM encoder (all-MiniLM-L6-v2, in-process)
query vector (384-d, L2-normalized)
  ↓ cosine against ~270 canonical example vectors
per-route and per-module scores
  ↓ two thresholds
route + modules, or abstain → LLM classifier
```

This addresses both failures with one change. A real encoder reads word order, which
is the accuracy ceiling. Cosine similarity is unnormalized across routes, which makes
"far from everything" directly representable — the out-of-scope signal stops being a
softmax artifact and becomes a measurement.

It also removes a training run from the loop. Extending the router becomes a JSON
edit plus a rebuild, not a training job with promotion gates over a 23MB checkpoint.

### Rejected alternatives

**Keep the frozen wordpiece bag, change only the head.** Near-zero cost and no new
dependency, but it preserves the exact representation the docs identify as the
ceiling. Cheap, and expected to move nothing.

**Encoder behind the retrieval server.** The hybrid retrieval server already loads
e5; an `/embed` endpoint would keep the web backend light. Rejected: it puts a
network hop on every auto-routed request and makes routing depend on a second
process being up, to save a dependency the web backend already has (torch is
imported today by `IntentPipeline.load`).

**Keep both the MLP and kNN in the cascade.** Two model paths, two artifacts, two
sets of promotion gates, and double the surface a routing bug can hide in — to save
one LLM call in a narrow band.

**Orthogonal NLU++-style intent primitives.** More expressive and the right end
state, but much harder to label consistently at ~270 examples. Revisit at several
thousand.

## Scope

**In scope:** the model step of the routing cascade, the canonical example set, the
evaluation instrument, and the deletion of the MLP machinery it replaces.

**Explicitly out of scope:** plan-aware / multi-route output
(`{primary_route, steps[]}`) and any dispatcher change to execute a multi-step plan.
That is a separate spec. This design *detects* composite requests and records the
signal; it does not act on it.

## Design

### 1. The cascade — one step swapped, nothing else moves

```
explicit_source → regex rules → [kNN step] → LLM classifier → heuristic / clarify
```

`route_request` in `src/internal/servers/web/intent_routing.py` is **unchanged**.
`ml_intent.predict_route(query, *, settings) -> IntentModelDecision | None` keeps its
exact signature and return type, so telemetry, the `intent_model` request-capture
stage, the abstention path, and the clarification path all keep working untouched.
Only what sits behind `predict_route` changes.

`IntentModelDecision` gains two optional fields — `modules: tuple[str, ...]` and
`composite: bool` — that only flow into telemetry and captures. No routing behavior
reads them in this round.

### 2. Label taxonomy — two levels, route-scoped, multi-label

Routes stay `chat | search | tool`. Each route is subdivided into modules, borrowing
NLU++'s module idea while keeping modules scoped to one route each. The taxonomy is
not invented: it is extracted from the regex cues already proven in
`intent_routing.py`.

| route | modules | kind | derived from |
|---|---|---|---|
| **search** | `lookup_document`, `lookup_fact`, `current_info` | intent | `_SEARCH_LOOKUP_RE`, `_CURRENCY_RE` |
| **search** | `bare_entity` | **form** | `_is_bare_lookup` |
| **chat** | `explain`, `summarize`, `compare`, `generate`, `converse` | intent | `_CHAT_START_RE`, `_GENERATIVE_RE` |
| **tool** | `create`, `send`, `schedule`, `modify`, `execute` | intent | `_TOOL_ACTION_RE`, `_TOOL_OBJECT_RE` |

14 modules: 13 semantic intents plus one form label.

**An example carries one route and one or more modules.** Forcing exactly one loses
real signal — *"compare the current prices of BTC and ETH"* is genuinely both
`current_info` and `lookup_fact`.

**`bare_entity` is a routing signal, not a semantic intent.** It describes utterance
*form* (`"OpenAI"`) where the others describe intent (`"OpenAI CEO"` →
`lookup_fact`, `"OpenAI latest"` → `current_info`). This has a mechanical
consequence: `_is_bare_lookup` fires at cascade step 2 and returns SEARCH
deterministically **before** `predict_route` is called, so `bare_entity` sees no live
traffic through the model. It is therefore capped at ~10 examples (enough for
support, no more), tagged `kind: "form"`, and **excluded from module macro-F1** —
averaging a form label into a semantic-intent metric distorts it.

Modules are hierarchical rather than orthogonal at this data scale because the
taxonomy stays small, explainable, and grounded in cues the router already relies on.

### 3. The index — `src/model/intent_knn.py`

An `IntentIndex` holds an L2-normalized `float32` matrix (N×384), per-example route
and module labels, the encoder name, and a fingerprint of the canonical file.

Scoring, given a normalized query vector:

- **Route score** `s(r)` = mean of the **top-3** cosines among examples labeled route *r*
- **Module score** `s(m)` = mean of the top-3 cosines among examples carrying module *m*

Route scoring pools ~90 examples where a module pools ~33, so the route number is
substantially more stable. **The route decision therefore comes from `s(r)`, never
derived from module scores.** Modules refine the decision; they do not make it.

**Decision:**

```
route      = argmax_r s(r)
confidence = s(best route)
margin     = s(best route) − s(runner-up route)
modules    = { m in best route : s(m) ≥ τ_module }  or  {argmax_m s(m)} if empty
```

**Two thresholds gate routing, because two different things go wrong:**

| condition | meaning | action |
|---|---|---|
| `confidence < τ_conf` | nothing canonical resembles this | out of scope → abstain |
| `margin < τ_margin` | two routes score alike | ambiguous → abstain |

Abstaining returns `None`, and the cascade falls through to the LLM classifier —
byte-identical to how an abstaining MLP behaves today.

**`τ_module` never gates routing.** It only selects which modules are reported, so a
wrong module cannot cause a misroute. The third threshold adds diagnostic resolution
without adding routing risk.

**Minimum support:** a module with fewer than 10 canonical examples is never emitted
(flagged `low_support`) and the build warns. A thin module otherwise produces
confident-looking noise.

**Composite detection** (the bridge to the deferred plan-aware spec): when the
runner-up route is within `τ_margin` *and* its best module is an action module
(`create`, `send`, `schedule`, `modify`, `execute`), set `composite = True`. This is
the signature of *"find the best Italian place near the office and book it for 7"*.
This round records the flag in telemetry and abstains to the LLM; the later spec
turns the same flag into an emitted plan. The detection logic gets built and measured
here without building any part of the plan contract.

**Cost:** one MiniLM encode (~5–15ms CPU) plus a 270×384 matmul (~10µs). The matmul
is free; the encoder is the entire latency budget.

### 4. Canonical set — `data/intent_canonical.json`, ~270 records, committed

```json
{
  "id": "canon-041",
  "text": "what is the current price of bitcoin",
  "route": "search",
  "modules": ["current_info", "lookup_fact"]
}
```

Curated entirely from the existing 520 hand-authored records in
`data/intent_examples.json` — the ones that read like real product queries, balanced
across routes and modules. **No external corpora.** NLU++ and IntentGrasp
contributed the module idea and the hard-slice idea respectively; neither
contributes data, so neither imposes a license obligation.

Target balance: ~90 per route; ~33 assignments per semantic module (270 examples at
~1.6 modules each ≈ 430 assignments over 13 semantic modules); `bare_entity` ~10.

Out-of-scope probes stay **out** of the index — inside it they would act as
attractors pulling queries toward a route. They remain eval-only, used to tune
`τ_conf`.

### 5. Evaluation

| set | file | size | role |
|---|---|---|---|
| legacy-30 | `data/intent_eval_queries.json` | 30 | continuity — directly comparable to the pinned `0.733` |
| bulk | same file, extended | ~180 | **the promotion gate** |
| hard slice | `data/intent_eval_hard.json` | ~40 | diagnostic |
| out-of-scope | `data/intent_out_of_scope.json` | 24 | `τ_conf` tuning and margin |

Eval records gain the same `modules` field as canonical records.

**The hard slice is built from ~12 minimal triplets** (36 queries) plus ~4 composite
queries — the triplets holding the same entity across different routes and modules:

> "Find the Q3 earnings report" → `search / lookup_document`
> "What was revenue in the Q3 earnings report?" → `search / lookup_fact`
> "Summarize the Q3 earnings report" → `chat / summarize`

Holding the entity constant isolates the boundary from entity difficulty. Assorted
hard queries would confound the two. The composite queries are tagged `composite` and labeled with their
primary route.

**Authoring:** all ~190 new queries are drafted and labeled by the implementer, then
**reviewed and corrected by the maintainer in one pass before any number is pinned**.
The review is a gate, not a formality — these queries define what "correct routing"
means for every subsequent change.

**Leakage guard, enforced not assumed.** Canonical examples come only from
`intent_examples.json`; eval queries are separately authored and must never appear in
the index. The build command **fails** on normalized exact match *or* cosine > 0.95
between any eval query and any canonical example. With kNN the index *is* the model,
so an unguarded overlap would silently manufacture accuracy.

### 6. Metrics and promotion gates

| metric | today | bar |
|---|---|---|
| **route accuracy, bulk-180** | — | **≥ 0.80** — the only promotion gate |
| route accuracy, legacy-30 | `0.733` | reported for continuity, no gate |
| route accuracy, hard-40 | — | reported, no gate |
| module macro-F1 (13 semantic modules) | — | reported, no gate on first run |
| joint accuracy (route correct **and** module set exact match) | — | reported, no gate on first run |
| out-of-scope separation margin | `+0.059` | **≥ 0.25** |
| p95 routing latency | `0.43ms` | **≤ 25ms** — explicit, accepted regression |

Joint accuracy uses **exact set match** on modules — strict and honest. Module
macro-F1 supplies the partial credit.

No bar is set on module or joint accuracy in this round: gating a 14-way multi-label
metric on 190 queries would be gating on noise. The first run establishes the
baseline, and later work raises floors from measured values.

**Hard stop:** if bulk-180 route accuracy is below `0.75`, report every metric and
**stop**. No further tuning — that would be a new spec, as it was for #510.

### 7. What is deleted, and what survives

Deleting code merged as recently as #510 is a real cost and is stated in full.

| deleted | reason |
|---|---|
| `src/model/intent_classifier.py` | the MLP and checkpoint v4 are replaced |
| `src/model/intent_training.py` | replaced by an index `build` command |
| `src/model/wordpiece.py` | MiniLM's own tokenizer runs inside the encoder |
| `src/model/intent_pretrained.py` and `data/intent_pretrained/` | the frozen bundle has no consumer |
| the corresponding test modules | — |

| kept | note |
|---|---|
| `src/model/intent_evaluation.py` | **retargeted, not deleted** — promotion gates, calibration, and the `evaluation_report.json` contract all still apply to an index |
| `src/model/intent_data.py` | loaders, validation, fingerprints; extended for `modules` |
| `src/internal/servers/web/intent_routing.py` | cascade, clarification, capture, telemetry — untouched |
| `data/intent_examples.json` | the source the canonical set is curated from |

`INTENT_LABELS` moves to `intent_knn.py`. Net roughly −1,500 / +700 LOC.

### 8. Serving integration and dependency discipline

The unit-test CI job installs **neither torch nor transformers**. This repo has
shipped collection failures from unguarded imports twice (#356, re-fixed in #418), so
the boundary is structural:

- **All index math is numpy-only** — cosine, top-3 means, thresholds, abstention,
  composite detection, minimum support. Fully unit-tested against hand-built vectors
  with no encoder present.
- The encoder sits behind a narrow `encode_texts(texts) -> np.ndarray` seam with
  `sentence_transformers` imported **function-locally**, matching the existing
  discipline in `extract_pretrained_bundle`.
- `ml_intent` degrades to `None` when the encoder or the index is unavailable, so the
  router falls through to the LLM classifier rather than failing. Proven by the
  existing `sys.meta_path` import-blocking check.

**Lazy load, no lifespan change.** Web `TestClient` tests already hang when lifespan
loads a model (`examples/run_web_integration_tests.sh` exists for this reason).
Adding a second lifespan model load would make that worse. The first auto-routed
request pays the ~2s model load instead; this is a deliberate trade.

### 9. Configuration

| setting | replaces | note |
|---|---|---|
| `intent_index_path` | `intent_model_path` | directory holding the built index |
| `intent_model_min_confidence` | — | now **`τ_conf`** |
| `intent_min_route_margin` | new | `τ_margin` |
| `intent_min_module_score` | new | `τ_module`, diagnostic only |

**Trap to be explicit about: `τ_conf` is now a cosine, not a softmax probability.**
The currently configured value is meaningless in the new units. Thresholds are tuned
on the validation split and recorded in `evaluation_report.json`; shipping the old
number unchanged would silently make the model over-fire.

The artifact stays **dark** (not wired to live routing) unless the bulk-180 gate is
cleared, matching how #509 and #510 handled promotion.

## Risks

| risk | mitigation |
|---|---|
| Thresholds in new units silently over-fire | tuned on validation, recorded in the report, called out in docs |
| Canonical/eval leakage manufactures accuracy | build fails on exact match or cosine > 0.95 |
| A badly-phrased canonical example becomes a bad attractor | every rebuild re-runs the full eval; "append → rebuild → re-measure" is the only supported workflow |
| Encoder load adds ~90MB RSS and ~2s to the first routed request | lazy, no lifespan change; latency bar is p95 steady-state |
| 190 eval queries is still small | the only honest mitigation is saying so; the hard slice targets the informative gap |
| Modules thin out per category | minimum support of 10, `low_support` flag, build warning |
| Deleting #510's work | it is merged and recoverable from history; `intent_evaluation.py` and the operational discipline survive |

## Success criteria

1. Bulk-180 route accuracy **≥ 0.80**, or a hard stop with full numbers below `0.75`.
2. Out-of-scope separation margin **≥ 0.25**.
3. p95 routing latency **≤ 25ms**.
4. Module macro-F1 and joint accuracy measured and recorded, no bar.
5. The index core imports and its tests pass with torch, transformers, and
   sentence-transformers all blocked.
6. `route_request`'s cascade, clarification, telemetry, and capture behavior are
   unchanged.
7. Maintainer has reviewed and corrected all ~190 eval labels before any bar is
   pinned.
