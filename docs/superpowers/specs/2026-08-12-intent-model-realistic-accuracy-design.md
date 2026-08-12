# Intent model: accuracy on realistic phrasing — design

**Date:** 2026-08-12
**Status:** Approved
**Follows:** #506 (training/promotion pipeline), #507 (out-of-scope probes, calibration, telemetry), #508 (route clarification)

## Problem

The learned intent route ships dark. The pipeline, gates, telemetry, and
clarification path are all on `main` and working, but no checkpoint is
configured, so every route is still decided by the deterministic rules, the LLM
classifier, or a clarification.

The model cannot be promoted, and measurement shows why. Training data is 400
templated examples generated from 20 corpus documents, which yields a
**126-token vocabulary**. Three consequences compound:

- Out-of-vocabulary tokens map to `0`, the padding id that masked-mean pooling
  masks out. All seven fully-out-of-vocabulary probes produce the identical
  output — `search`, confidence `0.343` — because they pool to the zero vector
  and the logits are only the bias terms. The model cannot distinguish "I read
  this" from "I saw nothing".
- Confidence is inversely related to evidence. Fewer surviving tokens means
  less averaging, sharper logits, higher softmax. An out-of-scope query scored
  `0.785` while genuine in-domain queries scored `0.41`–`0.48`.
- On five realistic in-scope queries written by hand, the model scored **3/5**,
  and its highest confidence (`0.631`) was on a wrong answer.

The templated test split cannot detect any of this: its coverage is exactly
`1.00` by construction, so it measures memorization of the generator.

### What this design explicitly rejects

Expanding the vocabulary from the retrieval corpus was measured and does not
work. Adding all 20 documents grows the vocabulary from 126 to 588 tokens and
changes realistic in-scope coverage **not at all** (0.590 before and after),
while raising out-of-scope coverage from 0.169 to 0.259. The corpus contributes
domain nouns that in-scope queries already contained; what real queries lack is
ordinary English — *we, need, on, file, where, from, last* — which out-of-scope
chatter is equally made of.

The conclusion is that vocabulary must grow from **authored phrasing**, because
a token needs training signal. A token present in the vocabulary but never seen
during training carries a random embedding, which produces confident nonsense
rather than abstention.

## Goal

Make the model classify realistic phrasing well enough to be worth promoting,
and measure that on an instrument that can actually detect it.

## Scope

### In scope

- Frame-based generation, so authored phrasing diversity grows the vocabulary.
- A hand-authored realistic evaluation set the generator never produces.
- A `realistic_accuracy` block in the evaluation report.
- A dedicated out-of-vocabulary embedding, distinct from padding.
- Demoting the out-of-scope gate to a reported metric.

### Out of scope

- Changing the representation away from bag-of-embeddings (no sentence encoder,
  no character n-grams).
- Changing `INTENT_LABELS`, the runtime cascade, the clarification path, or any
  dispatcher.
- Distilling labels from stored user queries or adapting a public intent
  dataset.
- Promoting a checkpoint. This design makes promotion possible; activating an
  artifact stays a separate operator decision.

## Components

### Frame-based generation

`build_examples_for_document` currently emits 20 fixed templates per document.
It becomes a frame set: each frame is a phrasing pattern with slots, carrying an
intent and a register (imperative, question, statement, polite). Slot fillers
supply generic English — roles, time expressions, artifacts, actions — alongside
the document's own terms.

The point is the function words. Frames such as "where is the {artifact} from
{time}" and "we need the {artifact} on file" introduce *where, is, from, we,
need, on, file* with training signal attached, which is exactly the vocabulary
the measurement showed missing.

Existing hard-negative and multi-intent cases are retained unchanged. Generated
`source` values keep grouping every frame/document combination so derived
paraphrases cannot cross a split boundary.

### Realistic evaluation set

A new `data/intent_eval_queries.json`, tracked the same way as
`data/intent_out_of_scope.json` (the `data/` directory is git-ignored; these
files are force-added). Each record has a stable id, text, and gold label. It is
**never trained on and never split** — it is a fixed instrument.

Its queries must be written independently of the frame set: if a frame produces
it, it cannot measure generalization. `load_intent_eval_queries` mirrors
`load_out_of_scope_probes`, rejecting unknown labels, empty fields, and
duplicate ids.

The evaluation report gains `realistic_accuracy`: accuracy and per-label
precision, recall, and F1 over that set, at the selected threshold. This is the
number the project moves; the recorded baseline is 3/5 on the five queries used
during diagnosis, which the new set supersedes.

### Out-of-vocabulary embedding

`Vocabulary.encode` maps unknown tokens to `0`. Pooling masks `0`. Together
those silently delete every unknown word.

The shared `Vocabulary` class is **not** changed: `src/internal/document_index/cli.py`
also uses it, and altering its encoding contract would reach into the indexing
pipeline for no benefit here. Instead `IntentPipeline` — which owns its own
`Vocabulary` instance and calls `encode` at exactly two sites — remaps the
result before padding.

Unknown tokens take index `1`. That index is free by construction: `build`
assigns real tokens starting at `2` (`SOS_token = 0`, `EOS_token = 1`), and
`encode` emits `0` only for a token it did not recognise, so every `0` it
returns is unambiguously an unknown word. Index `1` therefore gets its own
trained embedding, and an unread word contributes a learned "I do not know
this" direction instead of vanishing. Padding keeps `0` and stays masked, so
batching still cannot change a prediction.

An empty token sequence, which the two call sites currently floor to `[0]`,
becomes `[1]` for the same reason — no tokens read is a fact about the input,
not padding.

This changes the preprocessing contract, so the checkpoint format goes to
version `3`, recording `unknown_id` alongside the existing `padding_id` and
`pooling`. Version 2 is rejected with an explicit retraining message, as version
1 already is — a version 2 checkpoint's embeddings were trained with unknown
words deleted, so reusing them under the new encoding would silently change
what every index means.

### Promotion gates

`tool_precision_minimum` and `high_confidence_errors_maximum` are unchanged;
both remain meaningful and both measure the model's own covered predictions.

`out_of_scope_abstention_minimum` is **demoted from a gate to a reported
metric**. The evidence is that 100% abstention is unachievable for this model
family at any useful coverage, and because confidence falls as evidence rises,
any meaningful threshold rejects good queries before bad ones. Keeping an
unachievable gate would permanently block promotion while teaching nobody
anything.

The abstention rate is still computed and reported at the selected threshold, so
a regression remains visible. Out-of-scope safety is provided by the LLM
classifier fallback and the clarification path from #508, not by the model —
this is stated plainly in the operator documentation.

## Error handling

- Generation fails before writing when a frame yields empty text, a duplicate
  id, or a label outside `INTENT_LABELS`.
- The evaluation set is validated on load; a malformed or empty file raises
  before training begins.
- An evaluation set whose queries all appear verbatim in the generated training
  data raises, because such a set cannot measure generalization.
- A version 2 checkpoint fails to load with a retraining message and never
  reinterprets its class indices.
- Realistic accuracy is reported, never silently defaulted; if the set is
  missing, the report records `null` rather than a score.

## Verification

### Unit tests

- Frames produce unique stable ids and keep source grouping intact.
- Generated examples still use only `chat`, `search`, and `tool`.
- Generated vocabulary contains function words absent from the corpus, which is
  the property the corpus experiment showed missing.
- The evaluation loader rejects unknown labels, empty text, and duplicate ids.
- An evaluation query that appears verbatim in the training data is rejected.
- An unknown token changes the pooled vector, proving it is no longer dropped.
- Padding still contributes nothing, so batching cannot change a prediction.
- A version 2 checkpoint raises the migration error.

### Workflow tests

- The training command reports `realistic_accuracy` and, when no evaluation set
  is configured, records `null`.
- `out_of_scope_abstention` appears in the report and in no gate.
- A candidate failing tool precision is still not promotable.

### Regression bar

Realistic accuracy is recorded in the report and pinned by a test at or above
the value the first frame-based run achieves. The templated test split remains
in the report for continuity but is no longer treated as evidence of real
performance.

## Success criteria

- Realistic accuracy is measured on a held-out, hand-authored set the generator
  never produces, and is reported by the standard command.
- The learned vocabulary covers ordinary function words with training signal
  behind them.
- An unknown token is visible to the model rather than silently dropped.
- No unachievable gate blocks promotion, and every remaining gate is one the
  model can meaningfully pass or fail.
- The runtime cascade, clarification path, and dispatchers are unchanged.

## Risks and mitigations

- **Authored data measures only phrasings we thought of.** Accepted and stated:
  the evaluation set is written independently of the frame set, and the report
  labels the number as realistic-authored, not production.
- **More synthetic volume can overfit the frame style.** Group splits by frame
  and document so a frame's outputs cannot span splits, and keep the evaluation
  set outside generation entirely.
- **Demoting the out-of-scope gate lowers a safety bar.** Deliberate, approved,
  and compensated: the metric is still reported, the fallback and clarification
  paths remain, and the operator documentation states plainly that the model is
  not the out-of-scope defense.
- **A larger vocabulary with thin per-token signal.** Frames reuse slot fillers
  across documents so each new function word appears many times rather than
  once.
