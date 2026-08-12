# Intent Modeling Pipeline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a reproducible, validated `chat`/`search`/`tool` intent-modeling workflow whose artifacts can safely improve the existing automatic web router.

**Architecture:** Keep the existing route cascade and downstream dispatchers. Add focused data, evaluation, and training modules around `IntentPipeline`; version and validate checkpoints; expose typed serving configuration; and promote a candidate only when an offline baseline comparison passes explicit quality, safety, fallback, and latency gates.

**Tech Stack:** Python 3.10+, PyTorch, scikit-learn metrics, dataclasses, JSON/JSONL, pytest, FastAPI application settings.

## Global Constraints

- Preserve `INTENT_LABELS: list[str] = ["chat", "search", "tool"]` in that order.
- Do not modify chat/RAG, search-provider, ToolAgentLoop, MCP, explicit-mode, or frontend behavior.
- Explicit modes/providers and confident deterministic regex routes remain authoritative.
- Low confidence abstains; it must never force the `tool` route.
- Use `AGENTIC_SEARCH_INTENT_MODEL_PATH` and `AGENTIC_SEARCH_INTENT_MODEL_MIN_CONFIDENCE` exactly.
- Do not activate or overwrite a serving artifact as a side effect of training.
- Do not add a new runtime dependency.

## File Structure

- Create `src/model/intent_data.py`: typed examples, validation, dataset fingerprinting, and grouped deterministic splits.
- Modify `src/model/intent_classifier.py`: padding-masked pooling, seeded training, vocabulary bounds, and version-2 checkpoint validation.
- Create `src/model/intent_evaluation.py`: prediction records, classification/coverage metrics, threshold selection, baseline comparison, and promotion gates.
- Modify `src/model/intent_training.py`: compatible example generation plus the single offline train/evaluate CLI and atomic artifact/report writes.
- Modify `src/model/__init__.py` and `src/__init__.py`: public exports for stable modeling interfaces.
- Modify `src/internal/configs/app_configs.py` and `src/internal/configs/default_config.py`: typed model path and threshold.
- Modify `src/internal/servers/web/ml_intent.py`: load from typed settings and expose evaluated prediction details without importing Torch until configured.
- Modify `src/internal/servers/web/intent_routing.py`: record model decisions, abstentions, fallback reasons, and latency without changing route precedence.
- Replace `data/intent_examples.json`: current three-label examples and metadata.
- Modify `docs/configuration.md`, `docs/request-routing.md`, and `docs/training-and-evaluation.md`: exact operational workflow.

---

### Task 1: Validated Intent Dataset and Deterministic Splits

**Files:**
- Create: `src/model/intent_data.py`
- Create: `tests/unit/test_intent_data.py`
- Modify: `src/model/__init__.py`
- Modify: `src/__init__.py`

**Interfaces:**
- Produces: `IntentExample(id: str, text: str, label: str, source: str, tags: tuple[str, ...])`.
- Produces: `IntentDatasetSplit(train, validation, test, seed, fingerprint)`.
- Produces: `load_intent_examples(path: Path) -> list[IntentExample]`.
- Produces: `split_intent_examples(examples, *, seed=17, train_fraction=0.70, validation_fraction=0.15) -> IntentDatasetSplit`.
- Depends on: `src.model.intent_classifier.INTENT_LABELS`.

- [ ] **Step 1: Write failing validation and split tests**

```python
from pathlib import Path

import pytest

from src.model.intent_data import load_intent_examples, split_intent_examples


def test_load_rejects_unknown_label(tmp_path: Path):
    path = tmp_path / "examples.json"
    path.write_text('[{"id":"x","text":"buy it","label":"purchase","source":"manual"}]')
    with pytest.raises(ValueError, match="purchase"):
        load_intent_examples(path)


def test_load_rejects_conflicting_duplicate_text(tmp_path: Path):
    path = tmp_path / "examples.json"
    path.write_text(
        '[{"id":"a","text":"open the report","label":"search","source":"s1"},'
        '{"id":"b","text":"open the report","label":"tool","source":"s2"}]'
    )
    with pytest.raises(ValueError, match="conflicting"):
        load_intent_examples(path)


def test_grouped_split_is_reproducible_and_has_no_source_leakage(tmp_path: Path):
    rows = []
    for label in ("chat", "search", "tool"):
        for group in range(10):
            rows.append({"id": f"{label}-{group}-a", "text": f"{label} {group} a", "label": label, "source": f"{label}-{group}"})
            rows.append({"id": f"{label}-{group}-b", "text": f"{label} {group} b", "label": label, "source": f"{label}-{group}"})
    path = tmp_path / "examples.json"
    path.write_text(__import__("json").dumps(rows))
    examples = load_intent_examples(path)
    first = split_intent_examples(examples, seed=23)
    second = split_intent_examples(examples, seed=23)
    assert first == second
    sources = [
        {item.source for item in first.train},
        {item.source for item in first.validation},
        {item.source for item in first.test},
    ]
    assert not (sources[0] & sources[1] or sources[0] & sources[2] or sources[1] & sources[2])
```

- [ ] **Step 2: Run the tests and verify the module is missing**

Run: `python -m pytest tests/unit/test_intent_data.py -q`

Expected: FAIL during collection with `ModuleNotFoundError: No module named 'src.model.intent_data'`.

- [ ] **Step 3: Implement the data boundary**

Create frozen dataclasses, normalize duplicate detection with `text.casefold().strip()`, reject empty fields/duplicate IDs/unknown labels/conflicting text, require every label before splitting, group by `source`, use `random.Random(seed)`, stratify groups by label, and compute `fingerprint` as SHA-256 over canonical sorted JSON records.

```python
@dataclass(frozen=True)
class IntentExample:
    id: str
    text: str
    label: str
    source: str
    tags: tuple[str, ...] = ()


@dataclass(frozen=True)
class IntentDatasetSplit:
    train: tuple[IntentExample, ...]
    validation: tuple[IntentExample, ...]
    test: tuple[IntentExample, ...]
    seed: int
    fingerprint: str
```

Use group counts, not row counts, for the 70/15/15 allocation; ensure each nonempty split contains each label or raise a message instructing the caller to add more independent source groups.

- [ ] **Step 4: Export the stable interfaces and run focused tests**

Run: `python -m pytest tests/unit/test_intent_data.py tests/unit/test_intent_classifier.py -q`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/model/intent_data.py src/model/__init__.py src/__init__.py tests/unit/test_intent_data.py
git commit -m "feat(intent): validate and split intent datasets"
```

---

### Task 2: Correct Model Pooling, Reproducibility, and Checkpoint Contract

**Files:**
- Modify: `src/model/intent_classifier.py`
- Modify: `tests/unit/test_intent_classifier.py`

**Interfaces:**
- Consumes: `INTENT_LABELS` unchanged.
- Produces: `IntentPipeline.train(data, *, epochs, lr, min_freq, seed=17)`.
- Produces: checkpoint format version `2` with `intent_labels`, `preprocessing`, `dataset_fingerprint`, and architecture `config`.
- Produces: `IntentPipeline.save(path, *, dataset_fingerprint: str)`.

- [ ] **Step 1: Add failing masked-pooling and bounds tests**

Add tests that set known embedding/linear weights, compare the same short sequence alone and beside a longer padded sequence, and require equal logits. Add a training test with `vocab_size=2` and three retained tokens that expects `ValueError` containing `vocab_size`. Add `seed` reproducibility by training two tiny pipelines and comparing their predictions.

```python
def test_batch_padding_does_not_change_prediction_logits():
    torch = pytest.importorskip("torch")
    model = _IntentClassifier(vocab_size=8, embedding_dim=4, hidden_dim=4, num_classes=3)
    model._net.eval()
    single = model._net(model._pad_sequences([[1]]))
    batched = model._net(model._pad_sequences([[1], [1, 2, 3]]))[0:1]
    assert torch.allclose(single, batched, atol=1e-6)
```

- [ ] **Step 2: Run the new correctness tests**

Run: `python -m pytest tests/unit/test_intent_classifier.py -q`

Expected: FAIL because mean pooling includes padding, `seed` is unsupported, and vocabulary bounds are unchecked.

- [ ] **Step 3: Implement mask-aware pooling and seeded training**

Change `_Net.forward` to mask ID `0` and divide by the non-padding count:

```python
mask = ids.ne(0).unsqueeze(-1)
embedded = self.embedding(ids)
x = (embedded * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1)
```

Set Python and Torch seeds before model initialization/training, reject a built vocabulary whose maximum index is outside the embedding table, and keep CPU-safe behavior.

- [ ] **Step 4: Add failing checkpoint-compatibility tests**

Test that save emits version 2 and its ordered labels, that loading a checkpoint with reordered or legacy labels raises `ValueError` containing `intent_labels`, and that tensor/config dimension mismatch is rejected before `load_state_dict` leaks an opaque error.

- [ ] **Step 5: Implement and verify checkpoint version 2**

Store:

```python
{
    "version": 2,
    "intent_labels": list(INTENT_LABELS),
    "preprocessing": {"tokenizer": "document_index.tokenize_text", "padding_id": 0, "pooling": "masked_mean"},
    "dataset_fingerprint": dataset_fingerprint,
    "vocab": ...,
    "model_state": ...,
    "config": ...,
}
```

Reject version 1 with a clear retraining message rather than interpreting old class indices.

Run: `python -m pytest tests/unit/test_intent_classifier.py -q`

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add src/model/intent_classifier.py tests/unit/test_intent_classifier.py
git commit -m "fix(intent): make training and checkpoints deterministic"
```

---

### Task 3: Metrics, Threshold Selection, and Promotion Gates

**Files:**
- Create: `src/model/intent_evaluation.py`
- Create: `tests/unit/test_intent_evaluation.py`
- Modify: `src/model/__init__.py`
- Modify: `src/__init__.py`

**Interfaces:**
- Produces: `IntentPredictionRecord(example_id, expected, predicted, confidence, latency_ms, mechanism)`.
- Produces: `IntentEvaluationReport` with JSON-serializable `to_dict()`.
- Produces: `select_confidence_threshold(records, *, tool_precision_min, max_high_confidence_errors) -> float` using validation records only.
- Produces: `PromotionCriteria(min_tool_precision=0.95, max_high_confidence_errors=0, require_macro_f1_non_decreasing=True, require_llm_fallback_reduction=True, require_latency_improvement=True)`.
- Produces: `compare_for_promotion(candidate, baseline, criteria) -> PromotionDecision`.

- [ ] **Step 1: Write failing metric and threshold tests**

```python
def test_threshold_selection_prioritizes_tool_precision():
    records = [
        IntentPredictionRecord("1", "chat", "tool", 0.70, 1.0, "model"),
        IntentPredictionRecord("2", "tool", "tool", 0.91, 1.0, "model"),
        IntentPredictionRecord("3", "tool", "tool", 0.95, 1.0, "model"),
        IntentPredictionRecord("4", "search", "search", 0.80, 1.0, "model"),
    ]
    threshold = select_confidence_threshold(
        records, tool_precision_min=0.95, max_high_confidence_errors=0
    )
    assert threshold > 0.70


def test_promotion_fails_when_macro_f1_regresses():
    decision = compare_for_promotion(
        candidate=_report(macro_f1=0.80, tool_precision=1.0, fallback_rate=0.2, p50_latency_ms=2.0),
        baseline=_report(macro_f1=0.85, tool_precision=1.0, fallback_rate=0.5, p50_latency_ms=50.0),
        criteria=PromotionCriteria(),
    )
    assert decision.promotable is False
    assert "macro_f1_non_decreasing" in decision.failed_gates
```

- [ ] **Step 2: Run and observe missing-module failure**

Run: `python -m pytest tests/unit/test_intent_evaluation.py -q`

Expected: FAIL during collection because `intent_evaluation` does not exist.

- [ ] **Step 3: Implement evaluation using scikit-learn**

Use `precision_recall_fscore_support(..., labels=INTENT_LABELS, zero_division=0)`, `accuracy_score`, and `confusion_matrix`. Define coverage as the fraction with confidence at or above threshold, fallback rate as `1 - coverage`, and high-confidence errors as covered records where expected differs from predicted. Threshold candidates are sorted unique confidences plus `1.0`; choose the lowest threshold satisfying safety constraints, breaking ties by macro-F1 over covered records and then coverage.

- [ ] **Step 4: Implement explicit promotion-gate reporting**

Return every gate as `{name, passed, candidate, baseline_or_limit}` and derive `failed_gates` from it. Treat missing baseline latency/fallback fields as failed gates, not zero values.

- [ ] **Step 5: Run tests and export interfaces**

Run: `python -m pytest tests/unit/test_intent_evaluation.py -q`

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add src/model/intent_evaluation.py src/model/__init__.py src/__init__.py tests/unit/test_intent_evaluation.py
git commit -m "feat(intent): evaluate thresholds and promotion gates"
```

---

### Task 4: Three-Label Corpus and Training/Evaluation Command

**Files:**
- Modify: `src/model/intent_training.py`
- Modify: `tests/unit/test_intent_training.py`
- Replace: `data/intent_examples.json`
- Create: `tests/fixtures/intent/intent_examples.json`
- Create: `tests/fixtures/intent/baseline_predictions.json`

**Interfaces:**
- Consumes: `IntentExample`, `IntentDatasetSplit`, `IntentPipeline`, and evaluation interfaces from Tasks 1–3.
- Produces: `IntentTrainingConfig(examples_path, output_dir, baseline_path, seed=17, epochs=10, ...)`.
- Produces: `run_intent_training(config) -> IntentTrainingRun`.
- Produces CLI: `python -m src.model.intent_training train --examples ... --baseline ... --output-dir ...`.

- [ ] **Step 1: Write failing end-to-end workflow test**

Create `tests/fixtures/intent/intent_examples.json` with at least four independent source groups per label. Create `tests/fixtures/intent/baseline_predictions.json` as a JSON array with one record per test example using the exact shape `{ "example_id": str, "expected": str, "predicted": str, "confidence": float, "latency_ms": float, "mechanism": "classifier|rule_based" }`. Test the real command-facing function:

```python
def test_training_workflow_writes_artifact_manifest_and_report(tmp_path):
    pytest.importorskip("torch")
    run = run_intent_training(
        IntentTrainingConfig(
            examples_path=FIXTURES / "intent_examples.json",
            baseline_path=FIXTURES / "baseline_predictions.json",
            output_dir=tmp_path,
            epochs=1,
            embedding_dim=8,
            hidden_dim=16,
            seed=17,
        )
    )
    assert run.checkpoint_path.exists()
    assert (tmp_path / "split_manifest.json").exists()
    report = json.loads((tmp_path / "evaluation_report.json").read_text())
    assert report["labels"] == ["chat", "search", "tool"]
    assert "promotion" in report
```

- [ ] **Step 2: Run and verify failure**

Run: `python -m pytest tests/unit/test_intent_training.py -q`

Expected: FAIL because `IntentTrainingConfig` and `run_intent_training` do not exist.

- [ ] **Step 3: Expand generation templates beyond regex duplicates**

Keep current direct templates and add paired hard cases with stable IDs/tags, including:

```python
("Explain how to send an email securely", "chat", ("hard_negative",)),
("Send the security summary to the project owner", "tool", ("paraphrase",)),
("Discuss approaches to finding duplicate documents", "chat", ("hard_negative",)),
("Locate the duplicate-document policy", "search", ("paraphrase",)),
```

Generated `source` must group every template family/document combination so derived paraphrases remain in one split.

- [ ] **Step 4: Implement the offline training workflow**

Load/validate/split data, train only on `split.train`, select threshold only on validation predictions, evaluate once on test, load baseline records for the same test IDs, compare promotion gates, and atomically write:

```text
output_dir/
  intent_model.pt
  split_manifest.json
  evaluation_report.json
```

Write temporary sibling files and use `Path.replace()` only after serialization succeeds. The report records hyperparameters, dataset fingerprint, selected threshold, candidate metrics, baseline metrics, and promotion decision. Do not copy to the configured serving path.

- [ ] **Step 5: Add argparse entry point and failure exit code**

`main(argv: Sequence[str] | None = None) -> int` supports the `train` subcommand. Return `0` only for a promotable artifact and `2` when training/evaluation succeeds but promotion gates fail; malformed inputs raise through `argparse.error` or produce exit `1` with a concise diagnostic.

- [ ] **Step 6: Regenerate the tracked example file with current labels**

Run the module's `generate` subcommand against `data/corpus.jsonl` and the repository's current vocabulary artifact when present. If no current vocabulary artifact exists, make `--vocabulary` optional and derive terms solely from corpus titles/content, then run:

```bash
python -m src.model.intent_training generate \
  --corpus data/corpus.jsonl \
  --output data/intent_examples.json
```

Verify:

```bash
python - <<'PY'
import json
from collections import Counter
rows = json.load(open("data/intent_examples.json"))
print(Counter(row["label"] for row in rows))
assert {row["label"] for row in rows} == {"chat", "search", "tool"}
assert all(row.get("id") and row.get("source") for row in rows)
PY
```

- [ ] **Step 7: Run focused workflow tests**

Run: `python -m pytest tests/unit/test_intent_training.py tests/unit/test_intent_data.py tests/unit/test_intent_evaluation.py -q`

Expected: PASS.

- [ ] **Step 8: Commit**

```bash
git add src/model/intent_training.py tests/unit/test_intent_training.py tests/fixtures/intent data/intent_examples.json
git commit -m "feat(intent): add reproducible training workflow"
```

---

### Task 5: Typed Serving Configuration and Safe Lazy Loading

**Files:**
- Modify: `src/internal/configs/app_configs.py`
- Modify: `src/internal/configs/default_config.py`
- Modify: `src/internal/servers/web/ml_intent.py`
- Modify: `tests/unit/test_configs.py`
- Modify: `tests/unit/servers/web/test_ml_intent.py`

**Interfaces:**
- Produces: `AppSettings.intent_model_path: Path | None`.
- Produces: `AppSettings.intent_model_min_confidence: float` in `[0.0, 1.0]`.
- Produces: `load_intent_model(settings: AppSettings | None = None)`.
- Produces: `predict_route(query, *, settings=None) -> IntentModelDecision | None`, where the decision includes intent, confidence, latency, and threshold.

- [ ] **Step 1: Write failing typed-configuration tests**

```python
def test_load_app_settings_reads_intent_model_configuration():
    settings = load_app_settings({
        "AGENTIC_SEARCH_INTENT_MODEL_PATH": "/models/intent.pt",
        "AGENTIC_SEARCH_INTENT_MODEL_MIN_CONFIDENCE": "0.73",
    })
    assert settings.intent_model_path == Path("/models/intent.pt")
    assert settings.intent_model_min_confidence == 0.73


@pytest.mark.parametrize("value", ["-0.1", "1.1", "nan", "inf"])
def test_intent_threshold_must_be_finite_probability(value):
    with pytest.raises(ValueError, match="INTENT_MODEL_MIN_CONFIDENCE"):
        load_app_settings({"AGENTIC_SEARCH_INTENT_MODEL_MIN_CONFIDENCE": value})
```

- [ ] **Step 2: Run and verify configuration tests fail**

Run: `python -m pytest tests/unit/test_configs.py -q`

Expected: FAIL because `AppSettings` lacks the fields.

- [ ] **Step 3: Implement typed settings and defaults**

Parse the path with `Path(value)` only when nonempty. Parse the threshold with `get_env_float`, then reject non-finite or values outside `[0, 1]`. Add both exact environment keys to `DEFAULT_CONFIG`.

- [ ] **Step 4: Write failing lazy-loader and decision tests**

Add tests proving settings, not direct `os.environ` reads, choose the artifact and threshold; no configured path avoids importing/loading Torch; an incompatible checkpoint returns `None` and caches failure; and a stub prediction yields:

```python
IntentModelDecision(
    strategy=RouteStrategy.SEARCH,
    confidence=0.91,
    threshold=0.73,
    latency_ms=pytest.approx(...),
)
```

- [ ] **Step 5: Implement settings-backed lazy loading**

Cache by resolved artifact path rather than one process-global success/failure sentinel so deterministic tests and explicit application settings cannot reuse the wrong model. Preserve lazy import of `IntentPipeline` inside the configured load branch.

- [ ] **Step 6: Run focused tests**

Run: `python -m pytest tests/unit/test_configs.py tests/unit/servers/web/test_ml_intent.py -q`

Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add src/internal/configs/app_configs.py src/internal/configs/default_config.py src/internal/servers/web/ml_intent.py tests/unit/test_configs.py tests/unit/servers/web/test_ml_intent.py
git commit -m "feat(intent): configure safe model serving"
```

---

### Task 6: Route Observability and End-to-End Dispatch Verification

**Files:**
- Modify: `src/internal/servers/web/intent_routing.py`
- Modify: `src/internal/servers/web/app.py`
- Modify: `tests/unit/servers/web/test_agent_router.py`
- Modify: `tests/unit/servers/web/test_stage_emits_intent.py`
- Modify: `tests/unit/servers/web/test_web_experience_app.py`

**Interfaces:**
- Consumes: `IntentModelDecision` from Task 5.
- Produces: trace detail fields `predicted_intent`, `confidence`, `threshold`, `abstained`, `fallback_reason`, and `latency_ms`.
- Preserves: `route_query(query, *, llm, explicit_source) -> RouteStrategy` and the existing dispatch contract.

- [ ] **Step 1: Add failing confident/abstaining trace tests**

Assert a confident model records mechanism `model`, its intent/confidence/threshold/latency, and skips the LLM. Assert a low-confidence decision records an abstention detail and the final classifier/rule stage contains `fallback_reason="model_below_threshold"`. Assert no model configured records no fabricated prediction.

- [ ] **Step 2: Run router tests and verify failure**

Run: `python -m pytest tests/unit/servers/web/test_agent_router.py tests/unit/servers/web/test_stage_emits_intent.py -q`

Expected: FAIL because model decisions currently expose only a tuple and low-confidence evaluation is not observable.

- [ ] **Step 3: Thread typed application settings into model prediction**

Pass the already-resolved `AppSettings` from `create_web_app` through `_run_agent_impl`/`_run_auto_routed` to `route_query` as an optional keyword-only setting while retaining `None` for direct/unit callers. Do not reload environment settings per request.

- [ ] **Step 4: Record model evaluation and final decision without changing precedence**

Keep explicit source and regex early returns. For a model decision, record an `intent_model` evaluation stage; if covered, record `intent · model` and return. If abstaining, carry a local fallback detail into the existing LLM/rule `_record_intent` call. Do not expose prompt text or add raw query logging.

- [ ] **Step 5: Add real-checkpoint dispatcher integration test**

Train a tiny checkpoint in `tmp_path`, configure `AppSettings(intent_model_path=...)`, bypass regex with phrases such as `vendor renewal terms`, and monkeypatch the three existing downstream runners. Assert confident predictions invoke the matching existing runner; assert a deliberately high threshold invokes the existing LLM/rule fallback instead. Keep the test independent of external servers.

- [ ] **Step 6: Run web routing tests**

Run:

```bash
python -m pytest \
  tests/unit/servers/web/test_agent_router.py \
  tests/unit/servers/web/test_stage_emits_intent.py \
  tests/unit/servers/web/test_web_experience_app.py \
  tests/unit/servers/web/test_ml_intent.py -q
```

Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add src/internal/servers/web/intent_routing.py src/internal/servers/web/app.py tests/unit/servers/web/test_agent_router.py tests/unit/servers/web/test_stage_emits_intent.py tests/unit/servers/web/test_web_experience_app.py
git commit -m "feat(intent): observe and verify model routing"
```

---

### Task 7: Operator Documentation and Full Verification

**Files:**
- Modify: `docs/configuration.md`
- Modify: `docs/request-routing.md`
- Modify: `docs/training-and-evaluation.md`
- Modify: `.env.example` if present; otherwise document environment variables only in the existing configuration reference.

**Interfaces:**
- Documents the CLI and artifacts introduced in Task 4.
- Documents the exact settings introduced in Task 5.
- Documents promotion semantics and fallback behavior introduced in Tasks 3 and 6.

- [ ] **Step 1: Correct configuration names and document defaults**

Replace every generic `INTENT_MIN_CONFIDENCE` reference with `AGENTIC_SEARCH_INTENT_MODEL_MIN_CONFIDENCE`. Add `AGENTIC_SEARCH_INTENT_MODEL_PATH`, default disabled, and threshold default `0.6`.

- [ ] **Step 2: Document the reproducible command**

Include a complete example:

```bash
python -m src.model.intent_training train \
  --examples data/intent_examples.json \
  --baseline data/eval/intent_baseline_predictions.json \
  --output-dir models/intent-candidate \
  --seed 17
```

Explain the checkpoint, split manifest, evaluation report, exit codes `0`, `1`, and `2`, and that operators activate a passing artifact explicitly by setting `AGENTIC_SEARCH_INTENT_MODEL_PATH`.

- [ ] **Step 3: Document runtime precedence and observability**

State that explicit modes/providers and regex decisions precede the model; covered predictions skip the LLM classifier; abstentions fall through; downstream chat/search/tool execution is unchanged; and traces identify the deciding mechanism and fallback reason.

- [ ] **Step 4: Run placeholder, documentation, lint, and focused test gates**

Run:

```bash
rg -n "INTENT_MIN_CONFIDENCE|purchase|navigate|recommendation|\bqa\b" \
  src/model src/internal/servers/web/ml_intent.py docs/configuration.md docs/request-routing.md docs/training-and-evaluation.md data/intent_examples.json
```

Expected: no stale configuration name or legacy intent-label references, except an intentional migration error message in checkpoint validation.

Run:

```bash
python -m ruff check \
  src/model/intent_classifier.py src/model/intent_data.py \
  src/model/intent_evaluation.py src/model/intent_training.py \
  src/internal/configs/app_configs.py src/internal/servers/web/ml_intent.py \
  src/internal/servers/web/intent_routing.py
python -m ruff format --check \
  src/model/intent_classifier.py src/model/intent_data.py \
  src/model/intent_evaluation.py src/model/intent_training.py \
  src/internal/configs/app_configs.py src/internal/servers/web/ml_intent.py \
  src/internal/servers/web/intent_routing.py
python -m pytest \
  tests/unit/test_intent_classifier.py \
  tests/unit/test_intent_data.py \
  tests/unit/test_intent_evaluation.py \
  tests/unit/test_intent_training.py \
  tests/unit/test_configs.py \
  tests/unit/servers/web/test_ml_intent.py \
  tests/unit/servers/web/test_agent_router.py \
  tests/unit/servers/web/test_stage_emits_intent.py \
  tests/unit/servers/web/test_web_experience_app.py -q
```

Expected: all commands exit `0`.

- [ ] **Step 5: Run the repository unit/regression suite**

Run: `python -m pytest -q`

Expected: PASS. If an unrelated environment-dependent test fails, record its full node ID and error separately; do not weaken the intent tests.

- [ ] **Step 6: Commit**

```bash
git add docs/configuration.md docs/request-routing.md docs/training-and-evaluation.md .env.example
git commit -m "docs(intent): document training and promotion workflow"
```

- [ ] **Step 7: Final review checkpoint**

Review `git diff HEAD~7..HEAD` against every success criterion in `docs/superpowers/specs/2026-08-12-intent-modeling-pipeline-design.md`. Confirm no downstream chat/search/tool implementation changed beyond passing settings and observing the selected route.
