# Intent model: accuracy on realistic phrasing — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the three-label intent model classify realistic phrasing well enough to be worth promoting, and measure it on a hand-authored instrument the generator never produces.

**Architecture:** Four independent changes to the offline modeling pipeline, none of which touch serving: (1) unknown tokens get their own trained embedding instead of being deleted by pad-masking, which forces the checkpoint format to version 3; (2) the per-document template list becomes a frame set whose slot fillers carry ordinary English, so the vocabulary grows from authored phrasing with training signal behind it; (3) a new tracked `data/intent_eval_queries.json` is loaded and scored as a fixed instrument; (4) `realistic_accuracy` enters the evaluation report and the unachievable out-of-scope gate is demoted to a reported metric.

**Tech Stack:** Python 3, PyTorch (lazily imported), scikit-learn metrics, pytest.

## Global Constraints

- `INTENT_LABELS = ["chat", "search", "tool"]` is unchanged. Every generated example, evaluation query, and prediction uses only these.
- `Vocabulary` in `src/internal/document_index/text.py` is **not** modified — `src/internal/document_index/cli.py` shares it. Remapping happens inside `IntentPipeline`.
- No change to the runtime cascade, the clarification path, `intent_routing.py`, `ml_intent.py`, any dispatcher, or the frontend.
- Padding id stays `0`; unknown id is `1`; real vocabulary tokens start at `2` (`Vocabulary.build` sets `num_token = 2`).
- Checkpoint format version becomes `3`. Versions `1` and `2` are rejected with a retraining message and never reinterpreted.
- Generation is fully deterministic — no `random`, no `Date`/time, no set iteration order in output.
- Environment variable names in docs stay exactly `AGENTIC_SEARCH_INTENT_MODEL_PATH` and `AGENTIC_SEARCH_INTENT_MODEL_MIN_CONFIDENCE`.
- `data/` is gitignored; tracked data files are force-added (`git add -f`), matching `data/intent_out_of_scope.json`.
- Work happens on the existing branch `feat/intent-oos-separability`. One commit per task. Open a PR after the last task.
- Lint before each commit: `ruff check . --fix && ruff format .`

**Grouping decision (assumption, stated because the spec permits either reading):** generated examples get `source = "frame:<frame_id>"`. The spec's risk mitigation says "a frame's outputs cannot span splits", which only holds when the group key is the frame. Consequence: a document's terms now appear in more than one split, while a phrasing pattern never does. The templated split therefore stops measuring memorized phrasing, which is the point — `realistic_accuracy` is the instrument that matters.

---

### Task 1: Unknown tokens get their own embedding, checkpoint goes to version 3

**Files:**
- Modify: `src/model/intent_classifier.py:104-267` (`IntentPipeline`)
- Test: `tests/unit/test_intent_classifier.py`

**Interfaces:**
- Consumes: nothing from other tasks.
- Produces: module constants `PADDING_ID: int = 0` and `UNKNOWN_ID: int = 1`; private method `IntentPipeline._encode(self, tokens: Sequence[str]) -> list[int]`; checkpoints whose `version` is `3` and whose `preprocessing` dict contains `unknown_id: 1`.

- [ ] **Step 1: Write the failing tests**

Add to `tests/unit/test_intent_classifier.py`:

```python
def test_unknown_token_changes_the_pooled_vector():
    """An unread word must be visible to the model, not silently dropped."""
    torch = pytest.importorskip("torch")
    pipeline = IntentPipeline(vocab_size=16, embedding_dim=4, hidden_dim=8)
    pipeline.train(
        [(["hello"], "chat"), (["find"], "search"), (["run"], "tool")],
        epochs=1,
        min_freq=1,
        seed=17,
    )

    known = pipeline._encode(["find"])
    with_unknown = pipeline._encode(["find", "zzzznotatoken"])

    assert known == [pipeline._vocab.token2idx["find"]]
    assert with_unknown == [pipeline._vocab.token2idx["find"], 1]

    net = pipeline._model._net
    net.eval()
    with torch.no_grad():
        pooled_known = net.embedding(
            pipeline._model._pad_sequences([known])
        ).sum(dim=1)
        pooled_unknown = net.embedding(
            pipeline._model._pad_sequences([with_unknown])
        ).sum(dim=1)
    assert not torch.allclose(pooled_known, pooled_unknown)


def test_encode_maps_an_all_unknown_query_to_the_unknown_id():
    pytest.importorskip("torch")
    pipeline = IntentPipeline(vocab_size=16, embedding_dim=4, hidden_dim=8)
    pipeline.train(
        [(["hello"], "chat"), (["find"], "search"), (["run"], "tool")],
        epochs=1,
        min_freq=1,
        seed=17,
    )

    assert pipeline._encode(["zzzznotatoken"]) == [1]
    assert pipeline._encode([]) == [1]


def test_load_rejects_version_two_checkpoint_with_retraining_message(tmp_path):
    torch = pytest.importorskip("torch")
    path = tmp_path / "v2-intent.pt"
    torch.save({"version": 2}, path)

    with pytest.raises(ValueError, match="retrain"):
        IntentPipeline.load(str(path))
```

Edit the existing `test_save_writes_version_two_checkpoint_contract` (currently at `tests/unit/test_intent_classifier.py:247`): rename it to `test_save_writes_version_three_checkpoint_contract`, change `assert checkpoint["version"] == 2` to `== 3`, and change the preprocessing assertion to:

```python
    assert checkpoint["preprocessing"] == {
        "tokenizer": "document_index.tokenize_text",
        "padding_id": 0,
        "unknown_id": 1,
        "pooling": "masked_mean",
    }
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `pytest tests/unit/test_intent_classifier.py -v`
Expected: FAIL — `AttributeError: 'IntentPipeline' object has no attribute '_encode'`, and the version assertions fail with `3 != 2`.

- [ ] **Step 3: Implement the encoding change**

In `src/model/intent_classifier.py`, add the constants under `INTENT_LABELS`:

```python
INTENT_LABELS: list[str] = ["chat", "search", "tool"]

# Padding stays 0 so masked-mean pooling keeps ignoring it. Unknown words take
# 1, which `Vocabulary.build` never assigns (real tokens start at 2), so an
# unread word gets a trained embedding instead of vanishing into the mask.
PADDING_ID = 0
UNKNOWN_ID = 1
```

Add the method to `IntentPipeline` (place it just above `train`):

```python
    def _encode(self, tokens: Sequence[str]) -> list[int]:
        """Encode tokens, giving unknown words their own embedding index.

        ``Vocabulary.encode`` returns 0 for a token it does not recognise, and
        0 is the padding id that pooling masks out, so an unknown word would
        otherwise be deleted. Every 0 it returns is unambiguously unknown.
        """
        encoded = [
            UNKNOWN_ID if index == PADDING_ID else index
            for index in self._vocab.encode(list(tokens))
        ]
        return encoded or [UNKNOWN_ID]
```

Replace the two `encode` call sites:

- In `train` (currently line 148): `encoded = [self._encode(tokens) for tokens, _ in data]`
- In `predict` (currently line 156): `encoded = self._encode(list(tokens))`

- [ ] **Step 4: Implement the checkpoint version change**

In `save`, replace `"version": 2` with `"version": 3` and the preprocessing block with:

```python
            "preprocessing": {
                "tokenizer": "document_index.tokenize_text",
                "padding_id": PADDING_ID,
                "unknown_id": UNKNOWN_ID,
                "pooling": "masked_mean",
            },
```

In `load`, replace the version checks (currently lines 213-221) with:

```python
        version = checkpoint.get("version")
        if version in (1, 2):
            raise ValueError(
                f"Checkpoint version {version} was trained with an encoding that "
                "deleted unknown words; retrain the intent model before loading it."
            )
        if version != 3:
            raise ValueError(f"Unsupported checkpoint version: {version}")
```

and the expected preprocessing dict (currently lines 226-230) with:

```python
        expected_preprocessing = {
            "tokenizer": "document_index.tokenize_text",
            "padding_id": PADDING_ID,
            "unknown_id": UNKNOWN_ID,
            "pooling": "masked_mean",
        }
```

- [ ] **Step 5: Run the intent test files**

Run: `pytest tests/unit/test_intent_classifier.py tests/unit/test_intent_training.py -v`
Expected: PASS. `test_batch_padding_does_not_change_prediction_logits` must still pass — padding is untouched.

- [ ] **Step 6: Commit**

```bash
ruff check . --fix && ruff format .
git add src/model/intent_classifier.py tests/unit/test_intent_classifier.py
git commit -m "feat(intent): give unknown tokens their own embedding, checkpoint v3"
```

---

### Task 2: Frame-based generation

**Files:**
- Modify: `src/model/intent_training.py:202-274` (`build_examples_for_document`), `src/model/intent_training.py:277-303` (`generate_intent_examples`)
- Test: `tests/unit/test_intent_training.py`

**Interfaces:**
- Consumes: nothing from Task 1.
- Produces: `build_examples_for_document(document: dict[str, Any], vocabulary_tokens: list[str], *, document_index: int = 0) -> list[dict[str, Any]]`, emitting one example per frame with `id = f"corpus:{document_id}:{frame_id}"` and `source = f"frame:{frame_id}"`. Module constant `_FRAMES: tuple[tuple[str, str, str, tuple[str, ...]], ...]` of `(frame_id, template, label, tags)`.

- [ ] **Step 1: Write the failing tests**

Add to `tests/unit/test_intent_training.py`:

```python
def test_frames_group_by_frame_and_produce_unique_ids():
    documents = [
        {"id": "d1", "title": "vector search", "contents": "dense retrieval"},
        {"id": "d2", "title": "bm25 ranking", "contents": "sparse retrieval"},
    ]
    examples = [
        example
        for index, document in enumerate(documents)
        for example in intent_training.build_examples_for_document(
            document, [], document_index=index
        )
    ]

    ids = [example["id"] for example in examples]
    assert len(ids) == len(set(ids))
    assert all(example["source"].startswith("frame:") for example in examples)
    # A frame's outputs share one source, so a phrasing pattern cannot span splits.
    by_source: dict[str, set[str]] = {}
    for example in examples:
        by_source.setdefault(example["source"], set()).add(example["id"])
    assert all(len(members) == len(documents) for members in by_source.values())


def test_frames_introduce_function_words_absent_from_the_corpus():
    """The corpus contributes domain nouns; real queries need ordinary English."""
    document = {"id": "d1", "title": "vector search", "contents": "dense retrieval"}
    corpus_tokens = set(
        tokenize_text(f"{document['title']} {document['contents']}")
    )

    generated = " ".join(
        example["text"]
        for example in intent_training.build_examples_for_document(document, [])
    )
    generated_tokens = set(tokenize_text(generated))

    for function_word in ("where", "we", "need", "from", "can", "you", "before"):
        assert function_word in generated_tokens
        assert function_word not in corpus_tokens


def test_generated_examples_split_without_source_leakage(tmp_path):
    corpus = tmp_path / "corpus.jsonl"
    corpus.write_text(
        "\n".join(
            json.dumps({"id": f"d{index}", "title": f"topic {index}", "contents": "x"})
            for index in range(12)
        )
        + "\n",
        encoding="utf-8",
    )
    examples_path = tmp_path / "examples.json"
    intent_training.write_intent_examples(
        intent_training.generate_intent_examples(corpus_path=corpus), examples_path
    )

    split = split_intent_examples(load_intent_examples(examples_path), seed=17)

    train_sources = {example.source for example in split.train}
    assert not train_sources & {example.source for example in split.validation}
    assert not train_sources & {example.source for example in split.test}
```

Add the imports these need at the top of the file:

```python
from src.internal.document_index.text import tokenize_text
from src.model.intent_data import load_intent_examples, split_intent_examples
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `pytest tests/unit/test_intent_training.py -k "frames or leakage" -v`
Expected: FAIL — `TypeError: build_examples_for_document() got an unexpected keyword argument 'document_index'`, and the function-word assertions fail because the current templates contain none of them.

- [ ] **Step 3: Add the frame set and fillers**

In `src/model/intent_training.py`, add above `_HARD_CASES`:

```python
# Slot fillers supply ordinary English. They rotate by document so each function
# word appears many times across the dataset rather than once.
_ROLES = ("the platform team", "my manager", "the on-call engineer", "the legal team")
_TIMES = ("last quarter", "this morning", "the last release", "yesterday")
_ARTIFACTS = ("the design doc", "the runbook", "the invoice", "the summary")
_ACTIONS = ("share", "send", "post", "file")

# (frame_id, template, label, tags). Templates use {title}, {t1}, {t2}, {role},
# {time}, {artifact}, {action}. Every frame includes a document term so no two
# documents produce identical text.
_FRAMES: tuple[tuple[str, str, str, tuple[str, ...]], ...] = (
    ("s-find", "find {title}", "search", ("direct",)),
    ("s-lookup", "look up {t1}", "search", ("paraphrase",)),
    ("s-bare", "{title}", "search", ("short",)),
    ("s-retrieve", "retrieve the {t2} documentation", "search", ("paraphrase",)),
    (
        "s-where-from",
        "where is the {t1} {artifact} from {time}",
        "search",
        ("ambiguous", "question"),
    ),
    (
        "s-we-need",
        "we need the {t1} numbers from {time} on file",
        "search",
        ("ambiguous", "statement"),
    ),
    (
        "s-official",
        "the official {t1} configuration reference page",
        "search",
        ("ambiguous", "statement"),
    ),
    (
        "s-anyone-have",
        "does anyone have {artifact} for {t2}",
        "search",
        ("ambiguous", "question"),
    ),
    (
        "s-multi",
        "I need background on {t2} and the latest benchmark table",
        "search",
        ("multi_intent",),
    ),
    ("c-what-is", "what is {title} and how is it used?", "chat", ("direct",)),
    ("c-explain", "explain {t1} in {title}", "chat", ("paraphrase",)),
    ("c-compare", "compare {t1} and {t2}", "chat", ("comparison",)),
    ("c-summarize", "summarize {title}", "chat", ("direct",)),
    (
        "c-walk-through",
        "walk me through the tradeoffs of {t1} versus {t2}",
        "chat",
        ("ambiguous", "imperative"),
    ),
    (
        "c-confused",
        "I'm confused about when {t1} beats {t2}",
        "chat",
        ("ambiguous", "statement"),
    ),
    (
        "c-why-would",
        "why would we use {t1} here instead of {t2}",
        "chat",
        ("ambiguous", "question"),
    ),
    (
        "c-help-understand",
        "help me understand what {t1} is doing under the hood",
        "chat",
        ("ambiguous", "polite"),
    ),
    ("t-email", "send an email about {title}", "tool", ("direct",)),
    ("t-ticket", "create a ticket for {t1}", "tool", ("direct",)),
    ("t-meeting", "schedule a meeting about {title}", "tool", ("direct",)),
    ("t-pr", "open a pull request for {t2}", "tool", ("direct",)),
    (
        "t-invite",
        "the {title} rollout needs a calendar invite for {role}",
        "tool",
        ("ambiguous", "statement"),
    ),
    (
        "t-push",
        "please {action} the {t1} summary to the shared channel",
        "tool",
        ("ambiguous", "polite"),
    ),
    (
        "t-can-you",
        "can you {action} the {t1} summary to {role} before {time}",
        "tool",
        ("ambiguous", "question"),
    ),
    (
        "t-multi",
        "the {t1} contract needs a review summary and a note to {role}",
        "tool",
        ("multi_intent",),
    ),
)
```

- [ ] **Step 4: Rewrite `build_examples_for_document`**

Replace the body of `build_examples_for_document` (lines 202-274) with:

```python
def build_examples_for_document(
    document: dict[str, Any],
    vocabulary_tokens: list[str],
    *,
    document_index: int = 0,
) -> list[dict[str, Any]]:
    """Build one intent-labeled example per frame for a corpus document."""

    title = document.get("title", "retrieval topic")
    contents = document.get("contents", "")
    terms = build_domain_terms(document, vocabulary_tokens)
    slots = {
        "title": title,
        "t1": _pick_term(terms, 0, "retrieval"),
        "t2": _pick_term(terms, 1, "search"),
        "role": _ROLES[document_index % len(_ROLES)],
        "time": _TIMES[document_index % len(_TIMES)],
        "artifact": _ARTIFACTS[document_index % len(_ARTIFACTS)],
        "action": _ACTIONS[document_index % len(_ACTIONS)],
    }

    document_id = str(document.get("_intent_document_id", document.get("id", title)))
    examples: list[dict[str, Any]] = []
    for frame_id, template, label, tags in _FRAMES:
        examples.append(
            {
                "id": f"corpus:{document_id}:{frame_id}",
                "text": template.format(**slots),
                "label": label,
                # Grouping by frame keeps one phrasing pattern inside one split.
                "source": f"frame:{frame_id}",
                "tags": list(tags),
                "source_doc_id": document.get("id", document_id),
                "source_title": title,
                "keywords": terms[:4],
                "context_hint": contents[:120],
            }
        )
    return examples
```

Delete the now-unused `Counter[str]` import usage inside this function only — `Counter` is still used by `_split_manifest`, so the module import stays.

- [ ] **Step 5: Pass the document index through the generator**

In `generate_intent_examples` (line 289-291), replace the loop with:

```python
    examples: list[dict[str, Any]] = []
    for document_index, document in enumerate(documents):
        examples.extend(
            build_examples_for_document(
                document, vocabulary_tokens, document_index=document_index
            )
        )
```

- [ ] **Step 6: Run the tests**

Run: `pytest tests/unit/test_intent_training.py tests/unit/test_intent_classifier.py -v`
Expected: PASS. If `test_build_examples_emit_ambiguous_cases_for_every_label` or `test_build_examples_label_multi_intent_requests_by_route_precedence` reference removed template text, update their expected strings to the new frame text — the ambiguous and multi-intent cases are preserved as frames `s-multi`, `c-confused`, `t-invite`, `t-multi`, etc.

- [ ] **Step 7: Commit**

```bash
ruff check . --fix && ruff format .
git add src/model/intent_training.py tests/unit/test_intent_training.py
git commit -m "feat(intent): generate examples from phrasing frames, not fixed templates"
```

---

### Task 3: Realistic evaluation set and its loader

**Files:**
- Create: `data/intent_eval_queries.json` (force-added)
- Modify: `src/model/intent_data.py`
- Test: `tests/unit/test_intent_data.py`

**Interfaces:**
- Consumes: nothing from Tasks 1-2.
- Produces: `IntentEvalQuery(id: str, text: str, label: str)` frozen dataclass and `load_intent_eval_queries(path: Path) -> tuple[IntentEvalQuery, ...]`, both exported from `src.model.intent_data`. Task 4 consumes them.

- [ ] **Step 1: Write the failing tests**

Add to `tests/unit/test_intent_data.py`:

```python
def test_load_intent_eval_queries_reads_id_text_and_label(tmp_path: Path):
    path = tmp_path / "eval.json"
    path.write_text(
        '[{"id": "e1", "text": "where did we land on the index rebuild",'
        ' "label": "search"}]',
        encoding="utf-8",
    )

    queries = load_intent_eval_queries(path)

    assert queries == (
        IntentEvalQuery(
            id="e1", text="where did we land on the index rebuild", label="search"
        ),
    )


@pytest.mark.parametrize(
    "payload, message",
    [
        ('[{"id": "e1", "text": "hi", "label": "purchase"}]', "Unknown intent label"),
        ('[{"id": "e1", "text": " ", "label": "chat"}]', "empty 'text'"),
        (
            '[{"id": "e1", "text": "a", "label": "chat"},'
            ' {"id": "e1", "text": "b", "label": "chat"}]',
            "Duplicate",
        ),
        ("[]", "no records"),
    ],
)
def test_load_intent_eval_queries_rejects_invalid_records(
    tmp_path: Path, payload: str, message: str
):
    path = tmp_path / "eval.json"
    path.write_text(payload, encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        load_intent_eval_queries(path)
```

Update the import at the top of the file:

```python
from src.model.intent_data import (
    IntentEvalQuery,
    load_intent_eval_queries,
    load_intent_examples,
    split_intent_examples,
)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `pytest tests/unit/test_intent_data.py -v`
Expected: FAIL with `ImportError: cannot import name 'IntentEvalQuery'`.

- [ ] **Step 3: Implement the loader**

In `src/model/intent_data.py`, add the dataclass after `IntentDatasetSplit`:

```python
@dataclass(frozen=True)
class IntentEvalQuery:
    """One hand-authored request used only to measure realistic accuracy."""

    id: str
    text: str
    label: str
```

Generalize the existing `_required_text` helper so its messages name the record kind:

```python
def _required_text(
    record: Mapping[str, object],
    field: str,
    index: int,
    *,
    kind: str = "Intent example",
) -> str:
    value = record.get(field)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{kind} at index {index} has empty {field!r}")
    return value
```

Add the loader after `load_out_of_scope_probes`:

```python
def load_intent_eval_queries(path: Path) -> tuple[IntentEvalQuery, ...]:
    """Load the fixed realistic-evaluation set.

    This set is never trained on and never split: it is an instrument, so it is
    validated strictly and used exactly as authored.
    """
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"Invalid intent evaluation query JSON in {path}: {exc.msg}"
        ) from exc

    if not isinstance(payload, list):
        raise ValueError("Intent evaluation query JSON must contain a list of records")

    queries: list[IntentEvalQuery] = []
    ids: set[str] = set()
    for index, record in enumerate(payload):
        if not isinstance(record, Mapping):
            raise ValueError(
                f"Intent evaluation query at index {index} must be an object"
            )
        kind = "Intent evaluation query"
        query_id = _required_text(record, "id", index, kind=kind)
        text = _required_text(record, "text", index, kind=kind)
        label = _required_text(record, "label", index, kind=kind)
        if label not in INTENT_LABELS:
            raise ValueError(f"Unknown intent label: {label!r}")
        if query_id in ids:
            raise ValueError(f"Duplicate intent evaluation query id: {query_id!r}")
        ids.add(query_id)
        queries.append(IntentEvalQuery(id=query_id, text=text, label=label))

    if not queries:
        raise ValueError(f"Intent evaluation query file contains no records: {path}")
    return tuple(queries)
```

- [ ] **Step 4: Run the tests**

Run: `pytest tests/unit/test_intent_data.py -v`
Expected: PASS.

- [ ] **Step 5: Author the evaluation set**

Create `data/intent_eval_queries.json` with 30 queries — 10 per label — written **without looking at `_FRAMES`**, in the phrasing a colleague would actually type. None may be a frame output, and none may repeat an out-of-scope probe. Format:

```json
[
  {"id": "eval-search-01", "text": "where did we land on the index rebuild last week", "label": "search"},
  {"id": "eval-search-02", "text": "I can't find the rollout checklist anywhere", "label": "search"},
  {"id": "eval-chat-01", "text": "does reranking actually help if recall is already high", "label": "chat"},
  {"id": "eval-tool-01", "text": "ping the on-call engineer about the failing job", "label": "tool"}
]
```

Write all 30 in that shape. Vary register: questions, statements, imperatives, polite requests, short fragments, long sentences. Include boundary pairs (asking *how* to do something is `chat`; asking for it to *be done* is `tool`).

Verify it loads and its label counts are balanced:

```bash
python -c "
from pathlib import Path
from collections import Counter
from src.model.intent_data import load_intent_eval_queries
q = load_intent_eval_queries(Path('data/intent_eval_queries.json'))
print(len(q), Counter(x.label for x in q))
"
```
Expected: `30 Counter({'chat': 10, 'search': 10, 'tool': 10})`

- [ ] **Step 6: Commit**

```bash
ruff check . --fix && ruff format .
git add src/model/intent_data.py tests/unit/test_intent_data.py
git add -f data/intent_eval_queries.json
git commit -m "feat(intent): add the hand-authored realistic evaluation set and loader"
```

---

### Task 4: `realistic_accuracy` in the report, out-of-scope gate demoted

**Files:**
- Modify: `src/model/intent_evaluation.py` (add `realistic_accuracy_report`, drop the out-of-scope gate), `src/model/intent_training.py` (config, wiring, CLI)
- Test: `tests/unit/test_intent_evaluation.py`, `tests/unit/test_intent_training.py`

**Interfaces:**
- Consumes: `IntentEvalQuery` and `load_intent_eval_queries` from Task 3; `IntentExample` from `src.model.intent_data`; `_predict_examples` from `src.model.intent_training`.
- Produces: `realistic_accuracy_report(records: Iterable[IntentPredictionRecord], *, threshold: float) -> dict[str, Any]`; `IntentTrainingConfig.eval_queries_path: Path | None`; CLI flag `--eval-queries`; a `realistic_accuracy` key in `evaluation_report.json` (a dict, or `null` when no set is configured). `PromotionCriteria` no longer has `min_out_of_scope_abstention`; `IntentTrainingConfig` no longer has it; the CLI no longer accepts `--min-out-of-scope-abstention`.

- [ ] **Step 1: Write the failing tests**

Add to `tests/unit/test_intent_evaluation.py`:

```python
def test_realistic_accuracy_reports_argmax_accuracy_and_covered_accuracy():
    records = [
        IntentPredictionRecord("e1", "search", "search", 0.80, 1.0, "model"),
        IntentPredictionRecord("e2", "chat", "search", 0.40, 1.0, "model"),
        IntentPredictionRecord("e3", "tool", "tool", 0.90, 1.0, "model"),
    ]

    report = realistic_accuracy_report(records, threshold=0.75)

    assert report["total_queries"] == 3
    assert report["accuracy"] == pytest.approx(2 / 3)
    assert report["coverage"] == pytest.approx(2 / 3)
    assert report["covered_accuracy"] == pytest.approx(1.0)
    assert set(report["per_label_metrics"]) == {"chat", "search", "tool"}


def test_out_of_scope_abstention_is_reported_but_never_gates():
    decision = compare_for_promotion(
        candidate=_report(
            macro_f1=0.95,
            tool_precision=0.99,
            fallback_rate=0.1,
            p50_latency_ms=1.0,
            model_tool_precision=0.99,
        ),
        baseline=_report(
            macro_f1=0.90,
            tool_precision=0.97,
            fallback_rate=0.5,
            p50_latency_ms=40.0,
        ),
        criteria=PromotionCriteria(),
    )

    gates = {gate["name"] for gate in decision.gates}
    assert "out_of_scope_abstention_minimum" not in gates
    assert decision.promotable is True
```

Update the existing gate-set assertions at `tests/unit/test_intent_evaluation.py:152-175`: remove `"out_of_scope_abstention_minimum"` from the expected `set(gates)` and from the expected `decision.failed_gates`.

Add the import: `from src.model.intent_evaluation import realistic_accuracy_report` (extend the existing import block).

Add to `tests/unit/test_intent_training.py`:

```python
def test_training_report_records_realistic_accuracy(tmp_path):
    pytest.importorskip("torch")
    run = _run_fixture_training(tmp_path, with_eval_queries=True)

    report = json.loads(run.evaluation_report_path.read_text(encoding="utf-8"))
    assert report["realistic_accuracy"]["total_queries"] == 3
    assert 0.0 <= report["realistic_accuracy"]["accuracy"] <= 1.0
    assert set(report["realistic_accuracy"]) >= {
        "accuracy",
        "coverage",
        "covered_accuracy",
        "macro_f1",
        "per_label_metrics",
        "threshold",
        "total_queries",
    }


def test_training_report_records_null_realistic_accuracy_without_a_set(tmp_path):
    pytest.importorskip("torch")
    run = _run_fixture_training(tmp_path, with_eval_queries=False)

    report = json.loads(run.evaluation_report_path.read_text(encoding="utf-8"))
    assert report["realistic_accuracy"] is None


def test_training_rejects_an_evaluation_set_the_generator_produces(tmp_path):
    pytest.importorskip("torch")
    with pytest.raises(ValueError, match="cannot measure generalization"):
        _run_fixture_training(tmp_path, with_eval_queries=True, copy_training_text=True)
```

Implement `_run_fixture_training` in the same file, reusing the corpus/baseline fixture setup already used by `test_training_workflow_writes_artifact_manifest_and_report` (`tests/unit/test_intent_training.py:70`). It must: generate examples from a small tmp corpus, generate the baseline via `generate_baseline_predictions`, write an eval-query file of exactly 3 queries (one per label — hand-written, not frame output, unless `copy_training_text=True`, in which case every query's `text` is copied verbatim from a generated example), and call `run_intent_training` with `eval_queries_path` set or `None`.

- [ ] **Step 2: Run the tests to verify they fail**

Run: `pytest tests/unit/test_intent_evaluation.py tests/unit/test_intent_training.py -v`
Expected: FAIL — `ImportError: cannot import name 'realistic_accuracy_report'`, and `TypeError: IntentTrainingConfig.__init__() got an unexpected keyword argument 'eval_queries_path'`.

- [ ] **Step 3: Add `realistic_accuracy_report`**

In `src/model/intent_evaluation.py`, add after `evaluate_intent_predictions`:

```python
def realistic_accuracy_report(
    records: Iterable[IntentPredictionRecord], *, threshold: float
) -> dict[str, Any]:
    """Score hand-authored queries the generator never produced.

    Accuracy is over every query, using the model's argmax, so the number stays
    comparable with the hand-scored diagnosis baseline. Coverage and covered
    accuracy then show what survives the serving threshold.
    """
    records = _validated_records(records)
    _validate_probability(threshold, name="threshold")

    expected = [record.expected for record in records]
    predicted = [record.predicted for record in records]
    precision, recall, f1, _ = precision_recall_fscore_support(
        expected, predicted, labels=INTENT_LABELS, zero_division=0
    )
    covered = tuple(record for record in records if record.confidence >= threshold)
    return {
        "threshold": threshold,
        "total_queries": len(records),
        "accuracy": float(accuracy_score(expected, predicted)),
        "macro_f1": float(sum(f1) / len(INTENT_LABELS)),
        "per_label_metrics": {
            label: {
                "precision": float(precision[index]),
                "recall": float(recall[index]),
                "f1": float(f1[index]),
            }
            for index, label in enumerate(INTENT_LABELS)
        },
        "coverage": len(covered) / len(records),
        "covered_accuracy": (
            sum(record.expected == record.predicted for record in covered)
            / len(covered)
            if covered
            else None
        ),
    }
```

- [ ] **Step 4: Demote the out-of-scope gate**

In `src/model/intent_evaluation.py`:

- Delete the `min_out_of_scope_abstention: float = 1.0` field from `PromotionCriteria` and replace its docstring line with a note:

```python
@dataclass(frozen=True)
class PromotionCriteria:
    """Safety and operational requirements for serving a candidate model.

    Out-of-scope abstention is deliberately absent: it is reported on the
    evaluation report, but this model family cannot reach a useful abstention
    rate at any threshold that leaves coverage. Out-of-scope safety comes from
    the LLM-classifier fallback and the clarification path, not the model.
    """

    min_tool_precision: float = 0.95
    max_high_confidence_errors: int = 0
    require_macro_f1_non_decreasing: bool = True
    require_llm_fallback_reduction: bool = True
    require_latency_improvement: bool = True
```

- Delete the entire `_gate("out_of_scope_abstention_minimum", ...)` entry (currently lines 451-463) from the `gates` tuple in `compare_for_promotion`.
- Leave `IntentEvaluationReport.out_of_scope_abstention`, `out_of_scope_abstention_rate`, and the `select_confidence_threshold` / `calibration_report` parameters untouched — the metric stays measured and reported.

- [ ] **Step 5: Wire the evaluation set through training**

In `src/model/intent_training.py`:

- Extend the imports:

```python
from .intent_data import (
    IntentDatasetSplit,
    IntentEvalQuery,
    IntentExample,
    load_intent_eval_queries,
    load_intent_examples,
    load_out_of_scope_probes,
    split_intent_examples,
)
from .intent_evaluation import (
    IntentPredictionRecord,
    PromotionCriteria,
    PromotionDecision,
    authoritative_routes_match,
    calibration_report,
    compare_for_promotion,
    compose_candidate_cascade,
    evaluate_intent_predictions,
    out_of_scope_abstention_rate,
    realistic_accuracy_report,
    select_confidence_threshold,
)
```

- In `IntentTrainingConfig`: replace `min_out_of_scope_abstention: float = 1.0` with `eval_queries_path: Path | None = None` (keep it beside `out_of_scope_path`).

- Add the independence check near the other module-level helpers:

```python
def _validate_eval_queries_are_held_out(
    queries: Sequence[IntentEvalQuery], examples: Sequence[IntentExample]
) -> None:
    """Reject an evaluation set the generator already produces verbatim."""
    trained = {example.text.casefold().strip() for example in examples}
    if all(query.text.casefold().strip() in trained for query in queries):
        raise ValueError(
            "Evaluation queries all appear verbatim in the training examples, so "
            "the set cannot measure generalization."
        )
```

- In `run_intent_training`, right after `examples = load_intent_examples(...)`:

```python
    eval_queries = (
        load_intent_eval_queries(Path(config.eval_queries_path))
        if config.eval_queries_path is not None
        else ()
    )
    if eval_queries:
        _validate_eval_queries_are_held_out(eval_queries, examples)
```

- Change the `select_confidence_threshold` call to stop constraining on out-of-scope abstention (the gate is gone, so the constraint would silently keep blocking coverage):

```python
    selected_threshold = select_confidence_threshold(
        validation_records,
        tool_precision_min=config.min_tool_precision,
        max_high_confidence_errors=config.max_high_confidence_errors,
    )
```

- After `calibration = calibration_report(...)`, add:

```python
    realistic = (
        realistic_accuracy_report(
            _predict_examples(
                pipeline,
                [
                    IntentExample(
                        id=query.id,
                        text=query.text,
                        label=query.label,
                        source="eval",
                    )
                    for query in eval_queries
                ],
            ),
            threshold=selected_threshold,
        )
        if eval_queries
        else None
    )
```

- Add `"realistic_accuracy": realistic,` to the `report` dict.
- Remove `min_out_of_scope_abstention=...` from the `PromotionCriteria(...)` construction.
- In `_hyperparameters`, add `"eval_queries": str(config.eval_queries_path) if config.eval_queries_path else None`.

- [ ] **Step 6: Update the CLI**

In `_build_parser`, delete `train.add_argument("--min-out-of-scope-abstention", ...)` and add:

```python
    train.add_argument("--eval-queries", type=Path)
```

In `main`, delete `min_out_of_scope_abstention=args.min_out_of_scope_abstention,` from the `IntentTrainingConfig(...)` call and add `eval_queries_path=args.eval_queries,`.

- [ ] **Step 7: Run the full intent suite**

Run: `pytest tests/unit/test_intent_evaluation.py tests/unit/test_intent_training.py tests/unit/test_intent_data.py tests/unit/test_intent_classifier.py -v`
Expected: PASS. Then grep for stragglers: `grep -rn "min_out_of_scope_abstention" src tests docs` should only show `select_confidence_threshold` / `calibration_report` parameters and their own unit tests.

- [ ] **Step 8: Commit**

```bash
ruff check . --fix && ruff format .
git add src/model/intent_evaluation.py src/model/intent_training.py tests/unit/
git commit -m "feat(intent): report realistic accuracy, demote the out-of-scope gate"
```

---

### Task 5: Regenerate the dataset, pin the regression bar, update the docs

**Files:**
- Modify: `data/intent_examples.json` (regenerated, force-added), `docs/training-and-evaluation.md:38-65`
- Test: `tests/unit/test_intent_training.py`

**Interfaces:**
- Consumes: everything from Tasks 1-4.
- Produces: a committed frame-based `data/intent_examples.json`, a pinned `_REALISTIC_ACCURACY_FLOOR` constant in the test module, and operator documentation matching the new flags and report.

- [ ] **Step 1: Regenerate the committed dataset**

```bash
python -m src.model.intent_training generate \
  --corpus data/corpus.jsonl \
  --output data/intent_examples.json
python -c "
import json
from collections import Counter
rows = json.load(open('data/intent_examples.json'))
print(len(rows), Counter(r['label'] for r in rows), len({r['source'] for r in rows}))
"
```
Expected: 500 rows (25 frames × 20 documents), balanced-ish label counts, and ~27 sources (25 frames + 2 manual hard-case groups).

- [ ] **Step 2: Measure realistic accuracy on a real training run**

```bash
python - <<'PY'
from collections import Counter
from pathlib import Path

from src.internal.document_index.text import tokenize_text
from src.model.intent_classifier import IntentPipeline
from src.model.intent_data import (
    load_intent_eval_queries,
    load_intent_examples,
    split_intent_examples,
)
from src.model.intent_evaluation import realistic_accuracy_report, IntentPredictionRecord

split = split_intent_examples(
    load_intent_examples(Path("data/intent_examples.json")), seed=17
)
pipeline = IntentPipeline(vocab_size=5000, embedding_dim=128, hidden_dim=256)
pipeline.train(
    [(tokenize_text(e.text), e.label) for e in split.train],
    epochs=10, lr=1e-3, min_freq=1, seed=17,
)
print("vocabulary:", len(pipeline._vocab.token2idx))
queries = load_intent_eval_queries(Path("data/intent_eval_queries.json"))
records = []
for query in queries:
    prediction = pipeline.predict_text(query.text)
    records.append(
        IntentPredictionRecord(
            query.id, query.label, prediction.intent, prediction.confidence, 1.0, "model"
        )
    )
report = realistic_accuracy_report(records, threshold=0.5)
print("accuracy:", report["accuracy"])
print("per label:", report["per_label_metrics"])
PY
```

Record the printed vocabulary size and accuracy. The measured accuracy is the regression bar.

**Checkpoint — report before continuing:** if the accuracy is below `0.60` (the 3/5 diagnosis baseline), stop and report the number, the vocabulary size, and the per-label metrics rather than pinning a bar the design was meant to raise.

- [ ] **Step 3: Pin the bar with a test**

Add to `tests/unit/test_intent_training.py`, filling `_REALISTIC_ACCURACY_FLOOR` with the value measured in Step 2:

```python
# Pinned to the first frame-based run (see docs/superpowers/plans/
# 2026-08-12-intent-model-realistic-accuracy.md). Raise it when a run beats it;
# never lower it without recording why.
_REALISTIC_ACCURACY_FLOOR = 0.00  # <- replace with the measured value


def test_frame_trained_model_holds_the_realistic_accuracy_bar():
    """The templated split measures memorization; this set measures the model."""
    pytest.importorskip("torch")
    data_dir = Path(__file__).resolve().parents[2] / "data"

    split = split_intent_examples(
        load_intent_examples(data_dir / "intent_examples.json"), seed=17
    )
    pipeline = IntentPipeline(vocab_size=5000, embedding_dim=128, hidden_dim=256)
    pipeline.train(
        [(tokenize_text(example.text), example.label) for example in split.train],
        epochs=10,
        lr=1e-3,
        min_freq=1,
        seed=17,
    )

    records = []
    for query in load_intent_eval_queries(data_dir / "intent_eval_queries.json"):
        prediction = pipeline.predict_text(query.text)
        records.append(
            IntentPredictionRecord(
                example_id=query.id,
                expected=query.label,
                predicted=prediction.intent,
                confidence=prediction.confidence,
                latency_ms=1.0,
                mechanism="model",
            )
        )

    report = realistic_accuracy_report(records, threshold=0.5)
    assert report["accuracy"] >= _REALISTIC_ACCURACY_FLOOR
```

Add the imports it needs: `IntentPipeline` from `src.model.intent_classifier`, `load_intent_eval_queries` from `src.model.intent_data`, `realistic_accuracy_report` and `IntentPredictionRecord` from `src.model.intent_evaluation`.

- [ ] **Step 4: Run the test**

Run: `pytest tests/unit/test_intent_training.py::test_frame_trained_model_holds_the_realistic_accuracy_bar -v`
Expected: PASS in a few seconds.

If instead `split_intent_examples` raises "Unable to place every label in every split", or `run_intent_training` raises "Validation split has no model-eligible examples after regex routing", the seed-17 frame allocation is unlucky. Try `--seed 19`, then `23`; use the first seed that works, use it in this test, and record it in the Step 6 documentation edit.

- [ ] **Step 5: Run the whole suite**

Run: `pytest`
Expected: PASS. Anything red here is almost certainly a test asserting old template text, the version-2 checkpoint contract, or the removed gate — fix those assertions, not the implementation.

- [ ] **Step 6: Update the operator documentation**

In `docs/training-and-evaluation.md`, edit the intent-model section:

- In the `train` command block (line 41-47), add the new flag:

```bash
python -m src.model.intent_training train \
  --examples data/intent_examples.json \
  --baseline data/eval/intent_baseline_predictions.json \
  --out-of-scope data/intent_out_of_scope.json \
  --eval-queries data/intent_eval_queries.json \
  --output-dir models/intent-candidate \
  --seed 17
```

- Replace the `--out-of-scope` paragraph (line 49) with:

> `--out-of-scope` supplies unlabeled requests the router should decline entirely. They carry no label, because the three-label taxonomy cannot express "none of these": out-of-scope safety is measured as the fraction of probes whose confidence falls below the serving threshold. That rate is **reported, not gated**. This model family cannot reach a useful abstention rate at any threshold that leaves coverage — its confidence falls as evidence rises, so a threshold strict enough to decline chatter rejects genuine requests first. Out-of-scope safety comes from the LLM-classifier fallback and the clarification path, not from the model. Without probes the rate is reported as `null` — unmeasured, never assumed safe.

- Add a paragraph after it:

> `--eval-queries` supplies `data/intent_eval_queries.json`, a hand-authored set written independently of the generator. It is never trained on and never split. The report's `realistic_accuracy` block scores it: argmax accuracy and per-label precision/recall/F1 over every query, plus coverage and covered accuracy at the selected threshold. This is the number that says whether the model handles phrasing a person would actually type; the templated test split, whose coverage is `1.00` by construction, measures memorization of the generator. Training refuses a set whose queries all appear verbatim in the training examples. Without the flag, `realistic_accuracy` is recorded as `null`.

- In the artifact list (line 57), extend the `evaluation_report.json` bullet with `, realistic accuracy,` before `and every promotion-gate result`.

- [ ] **Step 7: Commit and open the PR**

```bash
ruff check . --fix && ruff format .
git add docs/training-and-evaluation.md tests/unit/test_intent_training.py
git add -f data/intent_examples.json
git commit -m "feat(intent): pin the realistic-accuracy bar and document the new workflow"
git push -u origin feat/intent-oos-separability
gh pr create --title "feat(intent): train on authored phrasing and measure realistic accuracy" --body "..."
```

The PR body must state the measured realistic accuracy, the vocabulary size before (126) and after, the seed used, and that the out-of-scope gate is now a reported metric. Link the spec (`docs/superpowers/specs/2026-08-12-intent-model-realistic-accuracy-design.md`) and this plan.

---

## Verification summary

| Spec requirement | Task |
|---|---|
| Frame-based generation, authored phrasing grows vocabulary | 2 |
| Frames produce unique stable ids, source grouping intact | 2 |
| Generated vocabulary contains function words absent from the corpus | 2 |
| Generated examples still use only the three labels | 2 (existing test retained) |
| Hand-authored realistic evaluation set, never trained/split | 3 |
| Loader rejects unknown labels, empty text, duplicate ids | 3 |
| Verbatim-overlap evaluation set is rejected | 4 |
| `realistic_accuracy` block in the report, `null` when absent | 4 |
| Unknown-token embedding distinct from padding | 1 |
| Padding still contributes nothing (batching invariance) | 1 (existing test retained) |
| Checkpoint version 3; version 2 rejected with retraining message | 1 |
| Out-of-scope abstention reported, in no gate | 4 |
| Tool-precision gate still blocks a failing candidate | 4 (existing test retained) |
| Regression bar pinned; templated split kept for continuity | 5 |
| Runtime cascade, clarification path, dispatchers unchanged | all — no file outside `src/model/`, `data/`, `docs/`, `tests/` is touched |
