# Intent Router Distillation — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add `src/model/intent_distillation.py` — label a real/broad query set with the current router (regex → LLM → rule-based) and train the `{chat,search,tool}` MLP on those labels, plus a query-corpus reader and a CLI.

**Architecture:** Teacher-driven labeling reuses the router's own decision functions (lazy-imported so the training module has no import-time web dependency). Data flow: queries → label → examples JSON → `train_intent_classifier` → `.pt`. Query sources: a file and the SQLite store's logged user messages.

**Tech Stack:** Python 3.12, PyTorch (via the existing `IntentPipeline`), pytest.

## Global Constraints

- Router helpers (`_regex_route`, `_rule_based_route`, `classify_route`) are imported **lazily inside `label_query`** — no module-top import of `src.internal.servers.web.*`.
- Teacher precedence: `_regex_route` (confident) → `classify_route(q, llm)` (only when an LLM is given) → `_rule_based_route`. A `classify_route` exception for a query falls to `_rule_based_route` (never aborts the batch).
- No test requires an LLM or loads a real external model; LLM-teacher paths use a fake LLM with a `.complete(...)` method. The CLI test runs offline.
- Existing `intent_training.py`, `intent_classifier.py`, `route_query`, and `ml_intent` are unchanged.
- Run `ruff check <files> --fix && ruff format <files>` before each commit (pre-commit hook; if it reformats and aborts, `git add -A` and re-run).
- Branch: `feat/intent-distillation` (spec already committed there).

---

### Task 1: Store reader — `get_user_query_texts`

**Files:**
- Modify: `src/internal/db/store.py` (add a method near the other chat readers)
- Test: `tests/unit/db/test_user_query_texts.py` (new)

**Interfaces:**
- Produces: `AgenticSearchStore.get_user_query_texts(limit: int | None = None) -> list[str]` — distinct non-empty user message contents, newest first, optional cap.

- [ ] **Step 1: Write the failing test**

Create `tests/unit/db/test_user_query_texts.py`:

```python
from src.internal.db.store import AgenticSearchStore


def _seed(store, session_id, pairs):
    store.create_chat_session(session_id=session_id, user_id="u1")
    for role, content in pairs:
        store.add_chat_message(session_id, role=role, content=content)


def test_returns_distinct_user_queries_newest_first():
    store = AgenticSearchStore(":memory:")
    _seed(
        store,
        "s1",
        [
            ("user", "what is FAISS"),
            ("assistant", "FAISS is a library"),
            ("user", "find the Q3 report"),
            ("user", "what is FAISS"),  # duplicate
        ],
    )
    got = store.get_user_query_texts()
    assert got == ["find the Q3 report", "what is FAISS"]  # distinct, newest first
    assert store.get_user_query_texts(limit=1) == ["find the Q3 report"]


def test_ignores_blank_and_non_user():
    store = AgenticSearchStore(":memory:")
    _seed(store, "s2", [("assistant", "hi"), ("user", "   "), ("user", "HNSW")])
    assert store.get_user_query_texts() == ["HNSW"]
```

(Confirm the exact `create_chat_session` / `add_chat_message` signatures against `src/internal/db/store.py` before running; adjust the seed helper to match if they differ.)

- [ ] **Step 2: Run to verify it fails**

Run: `python -m pytest tests/unit/db/test_user_query_texts.py -q`
Expected: FAIL — `AttributeError: ... 'get_user_query_texts'`.

- [ ] **Step 3: Add the method**

In `src/internal/db/store.py`, add (near `list_chat_messages`):

```python
    def get_user_query_texts(self, limit: int | None = None) -> list[str]:
        """Distinct non-empty user message texts, newest first (for distillation)."""
        sql = (
            "SELECT content FROM chat_messages "
            "WHERE role = 'user' AND TRIM(content) != '' "
            "GROUP BY content ORDER BY MAX(created_at) DESC, MAX(id) DESC"
        )
        params: tuple = ()
        if limit is not None:
            sql += " LIMIT ?"
            params = (limit,)
        rows = self._conn.execute(sql, params).fetchall()
        return [str(row["content"]) for row in rows]
```

- [ ] **Step 4: Run to verify it passes**

Run: `python -m pytest tests/unit/db/test_user_query_texts.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
ruff check src/internal/db/store.py tests/unit/db/test_user_query_texts.py --fix && ruff format src/internal/db/store.py tests/unit/db/test_user_query_texts.py
git add src/internal/db/store.py tests/unit/db/test_user_query_texts.py
git commit -m "feat(distill): store reader for distinct logged user queries"
```

---

### Task 2: Distillation library — `intent_distillation.py`

**Files:**
- Create: `src/model/intent_distillation.py`
- Test: `tests/unit/test_intent_distillation.py` (new)

**Interfaces:**
- Consumes: `train_intent_classifier` (`src/model/intent_training`); `IntentPipeline` (`src/model/intent_classifier`); router helpers `_regex_route`/`_rule_based_route`/`classify_route` (`src/internal/servers/web/intent_routing`, lazy).
- Produces:
  - `label_query(query, *, llm=None) -> tuple[str, str]`
  - `build_distillation_examples(queries, *, llm=None) -> list[dict]`
  - `DistillResult` (frozen dataclass: `pipeline`, `num_examples`, `label_counts`, `teacher_counts`)
  - `distill_and_train(queries, *, output_path, examples_path, llm=None, epochs=15, lr=1e-3, min_freq=1) -> DistillResult`
  - `load_queries_from_file(path) -> list[str]`

- [ ] **Step 1: Write the failing tests**

Create `tests/unit/test_intent_distillation.py`:

```python
import json
from pathlib import Path

import pytest

from src.model.intent_classifier import IntentPipeline
from src.model.intent_distillation import (
    build_distillation_examples,
    distill_and_train,
    label_query,
    load_queries_from_file,
)


class _FakeLLM:
    def __init__(self, reply):
        self.reply = reply
        self.calls = 0

    def complete(self, _messages, **_kwargs):
        self.calls += 1
        return self.reply


def test_label_query_regex_path_ignores_llm():
    llm = _FakeLLM("tool")
    label, teacher = label_query("find the Q3 report", llm=llm)
    assert (label, teacher) == ("search", "regex")
    assert llm.calls == 0


def test_label_query_llm_tail_when_regex_defers():
    # A query the anchored regex returns None for, with currency cue deferral.
    label, teacher = label_query("the current bitcoin price trend", llm=_FakeLLM("search"))
    assert teacher == "llm"
    assert label in {"chat", "search", "tool"}


def test_label_query_rule_based_when_no_llm():
    label, teacher = label_query("the current bitcoin price trend", llm=None)
    assert teacher == "rule_based"


def test_label_query_llm_error_falls_to_rule_based():
    class _Boom:
        def complete(self, *_a, **_k):
            raise RuntimeError("boom")

    label, teacher = label_query("the current bitcoin price trend", llm=_Boom())
    assert teacher == "rule_based"


def test_build_examples_shape_and_drops_blanks():
    ex = build_distillation_examples(["find X", "  ", "what is Y"])
    assert ex == [
        {"text": "find X", "label": "search"},
        {"text": "what is Y", "label": "chat"},
    ]


def test_distill_and_train_roundtrip(tmp_path):
    queries = (
        ["find " + t for t in ("faiss", "bm25", "hnsw")]
        + ["what is " + t for t in ("faiss", "bm25", "hnsw")]
        + ["create a ticket for " + t for t in ("faiss", "bm25", "hnsw")]
    )
    pt = tmp_path / "m.pt"
    ex = tmp_path / "ex.json"
    result = distill_and_train(
        queries, output_path=pt, examples_path=ex, epochs=40, min_freq=1
    )
    assert result.num_examples == len(queries)
    assert sum(result.teacher_counts.values()) == len(queries)
    assert set(result.label_counts) <= {"chat", "search", "tool"}
    reloaded = IntentPipeline.load(str(pt))
    assert reloaded.predict_text("find hnsw").intent in {"chat", "search", "tool"}


def test_distill_and_train_empty_raises(tmp_path):
    with pytest.raises(ValueError):
        distill_and_train([], output_path=tmp_path / "m.pt", examples_path=tmp_path / "e.json")


def test_load_queries_from_file_txt_and_json(tmp_path):
    txt = tmp_path / "q.txt"
    txt.write_text("find X\n\nwhat is Y\n")
    assert load_queries_from_file(txt) == ["find X", "what is Y"]
    js = tmp_path / "q.json"
    js.write_text(json.dumps(["a", {"text": "b"}, {"question": "c"}]))
    assert load_queries_from_file(js) == ["a", "b", "c"]
    with pytest.raises(FileNotFoundError):
        load_queries_from_file(tmp_path / "missing.txt")
```

- [ ] **Step 2: Run to verify they fail**

Run: `python -m pytest tests/unit/test_intent_distillation.py -q`
Expected: FAIL — `ModuleNotFoundError: ...intent_distillation`.

- [ ] **Step 3: Create the module**

Create `src/model/intent_distillation.py`:

```python
"""Distill the current intent router (regex + LLM) into the trained classifier.

Label a query set with the router's own decisions, then train the MLP on those
labels so it learns intent cues (verb/structure) instead of topic tokens.
"""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from src.model.intent_classifier import IntentPipeline
from src.model.intent_training import train_intent_classifier


def label_query(query: str, *, llm=None) -> tuple[str, str]:
    """Label one query with the router. Returns (label, teacher)."""
    from src.internal.servers.web.intent_routing import (
        _regex_route,
        _rule_based_route,
        classify_route,
    )

    regex = _regex_route(query)
    if regex is not None:
        return regex.value, "regex"
    if llm is not None:
        try:
            return classify_route(query, llm)[0].value, "llm"
        except Exception:
            pass
    return _rule_based_route(query).value, "rule_based"


def build_distillation_examples(queries, *, llm=None) -> list[dict]:
    """Map queries to [{text, label}], dropping blank queries."""
    examples: list[dict] = []
    for q in queries:
        text = (q or "").strip()
        if not text:
            continue
        label, _teacher = label_query(text, llm=llm)
        examples.append({"text": text, "label": label})
    return examples


@dataclass(frozen=True)
class DistillResult:
    pipeline: IntentPipeline
    num_examples: int
    label_counts: dict[str, int]
    teacher_counts: dict[str, int]


def distill_and_train(
    queries,
    *,
    output_path: Path,
    examples_path: Path,
    llm=None,
    epochs: int = 15,
    lr: float = 1e-3,
    min_freq: int = 1,
) -> DistillResult:
    """Label queries with the router, write examples, and train the classifier."""
    examples: list[dict] = []
    teacher_counts: Counter = Counter()
    label_counts: Counter = Counter()
    for q in queries:
        text = (q or "").strip()
        if not text:
            continue
        label, teacher = label_query(text, llm=llm)
        examples.append({"text": text, "label": label})
        teacher_counts[teacher] += 1
        label_counts[label] += 1
    if not examples:
        raise ValueError("No non-empty queries to distill.")

    examples_path = Path(examples_path)
    examples_path.parent.mkdir(parents=True, exist_ok=True)
    examples_path.write_text(json.dumps(examples, ensure_ascii=False, indent=2))

    result = train_intent_classifier(
        examples_path=examples_path,
        output_path=Path(output_path),
        epochs=epochs,
        lr=lr,
        min_freq=min_freq,
    )
    return DistillResult(
        pipeline=result.pipeline,
        num_examples=result.num_examples,
        label_counts=dict(label_counts),
        teacher_counts=dict(teacher_counts),
    )


def load_queries_from_file(path) -> list[str]:
    """Load queries from a .txt (one per line) or a .json list of str/dicts."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Query file not found: {path!r}")
    if path.suffix == ".json":
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ValueError(f"Query file is not valid JSON: {path!r}") from exc
        out: list[str] = []
        for item in raw:
            text = item if isinstance(item, str) else (item.get("text") or item.get("question") or "")
            if text and text.strip():
                out.append(text.strip())
        return out
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
```

- [ ] **Step 4: Run to verify they pass**

Run: `python -m pytest tests/unit/test_intent_distillation.py -q`
Expected: PASS. (If `test_label_query_llm_tail_when_regex_defers` shows `teacher != "llm"`, the chosen query is actually regex-decidable — swap it for one where `_regex_route` returns `None`; verify with `python -c "from src.internal.servers.web.intent_routing import _regex_route; print(_regex_route('the current bitcoin price trend'))"` → should print `None` because of the currency cross-cue.)

- [ ] **Step 5: Commit**

```bash
ruff check src/model/intent_distillation.py tests/unit/test_intent_distillation.py --fix && ruff format src/model/intent_distillation.py tests/unit/test_intent_distillation.py
git add src/model/intent_distillation.py tests/unit/test_intent_distillation.py
git commit -m "feat(distill): router-teacher labeling + distill_and_train library"
```

---

### Task 3: CLI — `python -m src.model.intent_distillation`

**Files:**
- Modify: `src/model/intent_distillation.py` (append `main()` + `if __name__ == "__main__"`)
- Test: `tests/unit/test_intent_distillation_cli.py` (new)

**Interfaces:**
- Consumes: `load_queries_from_file`, `distill_and_train`, `AgenticSearchStore.get_user_query_texts` (Task 1); `LLMConfig` + `OpenAICompatibleLLM` for the optional teacher.
- Produces: `main(argv: list[str] | None = None) -> int`.

- [ ] **Step 1: Write the failing test**

Create `tests/unit/test_intent_distillation_cli.py`:

```python
from pathlib import Path

from src.model.intent_classifier import IntentPipeline
from src.model.intent_distillation import main


def test_cli_offline_trains_from_file(tmp_path):
    q = tmp_path / "q.txt"
    q.write_text(
        "\n".join(
            ["find faiss", "find bm25", "what is faiss", "what is hnsw",
             "create a ticket for faiss", "create a ticket for bm25"]
        )
    )
    pt = tmp_path / "model.pt"
    rc = main(["--queries-file", str(q), "--output", str(pt), "--epochs", "30"])
    assert rc == 0
    assert pt.exists()
    assert IntentPipeline.load(str(pt)).predict_text("find hnsw").intent in {"chat", "search", "tool"}
```

- [ ] **Step 2: Run to verify it fails**

Run: `python -m pytest tests/unit/test_intent_distillation_cli.py -q`
Expected: FAIL — `ImportError: cannot import name 'main'`.

- [ ] **Step 3: Add the CLI**

Append to `src/model/intent_distillation.py`:

```python
def _build_teacher_llm(args):
    if not args.vllm_url:
        return None
    from src.internal.llm.interfaces import LLMConfig
    from src.internal.llm.providers import OpenAICompatibleLLM

    return OpenAICompatibleLLM(
        LLMConfig(
            model_provider=args.model_provider,
            model_name=args.model,
            api_key=args.api_key,
            api_base=args.vllm_url,
        )
    )


def _collect_queries(args) -> list[str]:
    queries: list[str] = []
    if args.queries_file:
        queries.extend(load_queries_from_file(args.queries_file))
    if args.from_db:
        from src.internal.db.store import AgenticSearchStore

        store = AgenticSearchStore(args.from_db)
        queries.extend(store.get_user_query_texts())
    seen: set[str] = set()
    deduped: list[str] = []
    for q in queries:
        if q not in seen:
            seen.add(q)
            deduped.append(q)
    return deduped


def main(argv: "list[str] | None" = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Distill the intent router into a trained classifier.")
    parser.add_argument("--queries-file", type=str, default=None, help="File of queries (.txt one-per-line or .json list).")
    parser.add_argument("--from-db", type=str, default=None, help="SQLite store path to pull distinct logged user queries from.")
    parser.add_argument("--output", type=str, required=True, help="Output .pt path.")
    parser.add_argument("--examples-out", type=str, default=None, help="Where to write the labeled examples JSON (defaults beside --output).")
    parser.add_argument("--vllm-url", type=str, default=None, help="OpenAI-compatible base URL for the LLM teacher (ambiguous tail). Omit for offline regex+rule-based.")
    parser.add_argument("--model", type=str, default=None, help="Teacher model name (with --vllm-url).")
    parser.add_argument("--model-provider", dest="model_provider", type=str, default="openai")
    parser.add_argument("--api-key", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args(argv)

    queries = _collect_queries(args)
    if not queries:
        parser.error("no queries collected; provide --queries-file and/or --from-db")

    output = Path(args.output)
    examples_out = Path(args.examples_out) if args.examples_out else output.with_suffix(".examples.json")
    llm = _build_teacher_llm(args)

    result = distill_and_train(
        queries, output_path=output, examples_path=examples_out, llm=llm, epochs=args.epochs, lr=args.lr
    )
    print(f"queries={len(queries)} num_examples={result.num_examples}")
    print(f"label_counts={result.label_counts}")
    print(f"teacher_counts={result.teacher_counts}")
    print(f"saved -> {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Run to verify it passes**

Run: `python -m pytest tests/unit/test_intent_distillation_cli.py -q`
Expected: PASS.

- [ ] **Step 5: Regression + import-hygiene check**

Run: `python -c "import src.model.intent_distillation"` (imports without pulling the web layer / torch at module top) and
`python -m pytest tests/unit/test_intent_distillation.py tests/unit/test_intent_distillation_cli.py tests/unit/db/test_user_query_texts.py -q`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
ruff check src/model/intent_distillation.py tests/unit/test_intent_distillation_cli.py --fix && ruff format src/model/intent_distillation.py tests/unit/test_intent_distillation_cli.py
git add src/model/intent_distillation.py tests/unit/test_intent_distillation_cli.py
git commit -m "feat(distill): CLI to label a query corpus and train a distilled model"
```

---

## Self-Review

**Spec coverage:** teacher-driven `label_query` (regex→llm→rule_based, lazy import) → Task 2; `build_distillation_examples`/`DistillResult`/`distill_and_train`/`load_queries_from_file` → Task 2; store query corpus → Task 1; CLI with file+DB sources and optional LLM teacher → Task 3. Error handling (empty→ValueError, per-query llm error→rule_based, missing file→FileNotFoundError) → Tasks 2 tests. All spec sections covered.

**Placeholder scan:** every step has concrete code, exact paths, commands, expected output. No TBD/TODO.

**Type consistency:** `label_query -> (str, str)` unpacked as `(label, teacher)` in `build_distillation_examples`/`distill_and_train`/CLI. `distill_and_train -> DistillResult` with `.num_examples/.label_counts/.teacher_counts` asserted in tests. `train_intent_classifier(examples_path, output_path, epochs, lr, min_freq)` matches the real signature (verified in `src/model/intent_training.py`); it returns `IntentTrainingResult(pipeline, num_examples, label_counts)`, and `distill_and_train` re-wraps with its own teacher_counts. `OpenAICompatibleLLM(LLMConfig(model_provider, model_name, api_key, api_base))` matches `app.py`'s construction. `get_user_query_texts(limit)` (Task 1) is consumed by the CLI's `_collect_queries`.

**Import hygiene:** `intent_distillation` module top imports only stdlib + `src.model.*`; the web-layer router helpers and torch-backed pipeline are pulled lazily (router inside `label_query`, LLM inside `_build_teacher_llm`, store inside `_collect_queries`). Task 3 Step 5 asserts `import src.model.intent_distillation` stays light.
