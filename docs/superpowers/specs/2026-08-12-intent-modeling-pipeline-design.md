# Operational intent-modeling pipeline — design

**Date:** 2026-08-12
**Status:** Approved

## Problem

The web application's automatic route supports an optional trained intent
classifier, but the modeling lifecycle is incomplete. The current classifier
uses the correct route taxonomy:

```python
INTENT_LABELS: list[str] = ["chat", "search", "tool"]
```

However, the committed `data/intent_examples.json` still contains the legacy
`purchase`, `navigate`, `qa`, and `recommendation` labels. There is no supported
command that generates compatible data, creates reproducible splits, trains and
evaluates a checkpoint, compares it with the existing router, and emits an
artifact suitable for deployment. Consequently, the model route normally ships
dark and ambiguous requests continue to the LLM classifier or rule fallback.

## Goal

Make the existing three-label `IntentPipeline` operational from training through
serving, while proving that enabling a candidate improves the automatic routing
workflow rather than merely adding another model.

The runtime contract remains:

```text
query
  -> IntentPipeline
  -> IntentPrediction(intent="chat|search|tool", confidence=...)
  -> existing chat/search/tool dispatcher
```

## Scope

### In scope

- Preserve `INTENT_LABELS = ["chat", "search", "tool"]`.
- Generate and validate compatible labeled examples.
- Produce deterministic train, validation, and test splits.
- Train, evaluate, save, and load intent-model checkpoints.
- Validate checkpoint taxonomy and configuration before serving.
- Configure the model artifact and confidence threshold through application
  settings.
- Keep confidence-gated LLM and rule fallbacks.
- Compare the candidate route with the current routing baseline.
- Emit route-decision metadata and an evaluation report.
- Add unit, integration, and workflow tests.

### Out of scope

- Changing the three intent labels.
- Modifying chat/RAG execution after a `chat` decision.
- Modifying internal/external provider behavior after a `search` decision.
- Modifying `ToolAgentLoop`, tool selection, MCP discovery, approval, or tool
  execution after a `tool` decision.
- Changing explicit request modes or providers.
- Frontend changes.

## Runtime architecture

Automatic requests retain the existing safe cascade:

```text
explicit mode or provider
  -> authoritative existing behavior

auto request
  -> high-precision deterministic route
  -> configured IntentPipeline
       -> confidence >= threshold: chat | search | tool
       -> confidence < threshold: defer
  -> available LLM classifier
  -> rule-based fallback
  -> existing route dispatcher
```

The trained model does not bypass authorization, select a particular MCP tool,
or execute an action. It selects only the broad execution family. Explicit modes
and providers remain authoritative.

## Components

### Training data

Replace the stale legacy-label artifact with examples using only `chat`,
`search`, and `tool`. The generator may use the local corpus for domain terms,
but the dataset must not consist only of phrases already decided by regex.

The dataset includes:

- direct and paraphrased examples for all three labels;
- difficult near-boundary pairs, such as explaining how to send an email versus
  actually sending one;
- short and long requests;
- requests containing misleading keywords;
- ambiguous and unsupported examples used to test abstention;
- multi-domain examples not copied from the local retrieval corpus.

Every record has a stable identifier, text, label, source, and optional tags such
as `ambiguous`, `hard_negative`, or `paraphrase`. Dataset validation rejects
unknown labels, empty text, duplicate identifiers, and conflicting duplicate
texts.

Splits are deterministic from a configured seed and grouped so paraphrases or
examples derived from the same template/source cannot cross split boundaries.
The test split is never used for threshold selection.

### Model training

The existing lightweight embedding/MLP implementation remains the initial model
family. The modeling pipeline fixes correctness issues that affect reliable
training and serving:

- pooling excludes padding tokens;
- vocabulary size is derived from or checked against the built vocabulary;
- random seeds control Python and PyTorch behavior where supported;
- training validates that all route labels have examples;
- saved checkpoints include their ordered labels, preprocessing contract,
  architecture configuration, dataset fingerprint, and format version.

The command accepts input/output paths and core hyperparameters, writes the
checkpoint atomically, and never overwrites the active serving artifact as a
side effect of training.

### Evaluation and threshold selection

Validation data selects the confidence threshold. Test data reports final
performance once for the selected candidate. The report includes:

- split sizes and label counts;
- accuracy and macro-F1;
- per-label precision, recall, and F1;
- confusion matrix;
- coverage and error rate at the selected threshold;
- high-confidence misroute count;
- LLM fallback rate;
- routing latency distribution;
- baseline-versus-candidate comparison.

`tool` precision is weighted more heavily than `tool` recall because incorrectly
routing an ordinary request toward an action workflow is more costly than
deferring a genuine tool request to the LLM classifier.

### Baseline comparison and promotion

The evaluation harness runs the same held-out requests through:

```text
baseline:  regex -> LLM classifier when available -> rule fallback
candidate: regex -> IntentPipeline -> same LLM/rule fallback
```

An artifact is marked promotable only when all configured gates pass:

- candidate macro-F1 is not below the baseline;
- `tool` precision meets the configured minimum;
- high-confidence misroutes do not exceed the configured maximum;
- model-resolved requests reduce LLM-classifier usage;
- model-resolved routing latency is lower than LLM classification latency;
- explicit and deterministic routes remain behaviorally unchanged.

The report records each gate and its measured value. Failure leaves the candidate
artifact available for inspection but does not activate it.

### Checkpoint loading and configuration

Loading rejects a checkpoint when:

- the format version is unsupported;
- its ordered labels differ from `INTENT_LABELS`;
- required preprocessing or architecture metadata is missing;
- vocabulary indices exceed the embedding table;
- model tensors do not match declared dimensions.

Intent model path and confidence threshold become typed application settings.
Environment variables use the existing implementation names:

- `AGENTIC_SEARCH_INTENT_MODEL_PATH`
- `AGENTIC_SEARCH_INTENT_MODEL_MIN_CONFIDENCE`

Documentation uses those exact names. With no artifact configured, behavior is
unchanged. A failed load is observable and safely disables the model route.

### Observability

Each automatic route records:

- deciding mechanism: `explicit_source`, `regex`, `model`, `classifier`, or
  `rule_based`;
- predicted intent and confidence when a model was evaluated;
- configured threshold;
- whether the model abstained;
- fallback mechanism and reason;
- routing latency.

No raw user request is added to new logs solely for intent telemetry.

## Error handling

- Invalid training data fails before training begins with record-level context.
- An incompatible checkpoint never falls back to reinterpreting class indices.
- A missing or unreadable checkpoint disables model routing and records a clear
  diagnostic; the request continues through the existing fallback.
- Prediction failure is isolated to the request and falls through to the LLM or
  rule router.
- Low confidence never forces `tool`; it abstains.
- Evaluation or promotion failure never mutates the configured production
  artifact.

## Verification

### Unit tests

- Dataset validation and grouped deterministic splitting.
- Three-label enforcement.
- Padding-masked pooling.
- Vocabulary/embedding bounds.
- Reproducible training under the supported deterministic settings.
- Checkpoint metadata and incompatible-label rejection.
- Confidence threshold and abstention behavior.
- Metric and promotion-gate calculations.

### Integration tests

- Train a tiny real checkpoint, load it through application settings, and route
  representative `chat`, `search`, and `tool` requests.
- Confirm a confident model decision reaches the existing matching dispatcher.
- Confirm a low-confidence decision reaches the existing LLM or rule fallback.
- Confirm explicit modes and deterministic regex routes bypass model decisions.
- Confirm invalid checkpoints fail closed without failing the user request.
- Confirm route traces expose model confidence and fallback reasons.

### Workflow test

Run the supported training/evaluation command on a small fixture dataset and
assert that it emits a checkpoint, split manifest, metrics report, promotion
decision, and nonzero exit status when mandatory promotion gates fail.

## Success criteria

- One supported command produces a reproducible, validated three-label model
  artifact and evaluation report.
- The web application can load that artifact using typed configuration.
- Confident predictions dispatch to the existing `chat`, `search`, and `tool`
  paths; uncertain predictions retain the existing fallback behavior.
- Incompatible artifacts cannot silently change class meaning.
- Held-out results show that the enabled candidate meets every promotion gate,
  including reduced LLM-classifier usage without reduced macro-F1.
- Existing focused router tests and the new modeling workflow tests pass.

## Risks and mitigations

- **Synthetic-data overfitting:** group splits by derivation source and include
  independently authored hard examples.
- **Misleading softmax confidence:** select the threshold on validation data and
  report coverage versus error instead of treating softmax as certainty.
- **Distribution drift:** retain abstention/fallback and make route outcomes
  observable for later evaluation datasets.
- **False tool routing:** use a stricter promotion requirement for tool precision.
- **Latency regression:** load once, measure routing latency, and compare it with
  the LLM-classifier baseline before promotion.
