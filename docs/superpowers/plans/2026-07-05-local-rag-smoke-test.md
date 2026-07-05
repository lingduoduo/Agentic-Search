# Local RAG Smoke-Test Workflow Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an offline command that retrieves useful context from `data/corpus.jsonl` and produces compact RAG training records without requiring FlashRAG or Natural Questions assets.

**Architecture:** A focused example module will own a few corpus-backed QA examples and orchestration. It will reuse `TfidfRetriever` for retrieval plus `format_rag_reference` and `build_search_rag_record` for canonical training-record formatting, then preview records or serialize them through Hugging Face `Dataset`.

**Tech Stack:** Python 3.12, scikit-learn TF-IDF retrieval, Hugging Face `datasets`, PyArrow parquet, pytest.

## Global Constraints

- The default workflow must run without network access.
- Use the existing `data/corpus.jsonl`; do not modify or duplicate the corpus.
- Do not download datasets, build a production retrieval index, or generate Natural Questions caches.
- Keep `examples.prepare_search_rag_dataset` as the optional FlashRAG/NQ adapter.
- Reuse existing retrieval and RAG record-formatting code instead of duplicating it.

## File Structure

- Create `examples/prepare_local_rag_smoke_dataset.py`: bundled smoke examples, validation, retrieval adaptation, preview, parquet writing, and CLI entry point.
- Create `tests/unit/test_prepare_local_rag_smoke_dataset.py`: focused behavior, error, preview, and parquet tests for the example module.
- Modify `README.md`: lead with the offline smoke workflow and label full NQ preparation as optional with explicit prerequisites.

---

### Task 1: Corpus-Backed Retrieval and RAG Record Construction

**Files:**
- Create: `examples/prepare_local_rag_smoke_dataset.py`
- Create: `tests/unit/test_prepare_local_rag_smoke_dataset.py`

**Interfaces:**
- Consumes: `TfidfRetriever(corpus_path: str)` from `src.internal.servers.retrieval.demo`; `format_rag_reference(documents)` and `build_search_rag_record(example, *, context, split, index, data_source, ability)` from `src.training.data`.
- Produces: `SMOKE_EXAMPLES: tuple[dict[str, object], ...]`; `build_smoke_records(corpus_path: str | Path, *, topk: int = 3) -> list[dict[str, object]]`.

- [ ] **Step 1: Write failing tests for useful retrieved context and compact records**

Create `tests/unit/test_prepare_local_rag_smoke_dataset.py` with:

```python
from pathlib import Path

import pytest

from examples.prepare_local_rag_smoke_dataset import build_smoke_records


REPO_CORPUS = Path(__file__).parents[2] / "data" / "corpus.jsonl"


def test_build_smoke_records_retrieves_expected_context():
    records = build_smoke_records(REPO_CORPUS, topk=1)

    assert records
    first = records[0]
    assert set(first) == {
        "data_source",
        "prompt",
        "ability",
        "reward_model",
        "extra_info",
    }
    assert "FAISS" in first["prompt"][0]["content"]
    assert first["reward_model"]["ground_truth"]["target"] == ["FAISS"]
    assert first["extra_info"] == {"split": "smoke", "index": 0}


def test_build_smoke_records_rejects_non_positive_topk():
    with pytest.raises(ValueError, match="topk must be at least 1"):
        build_smoke_records(REPO_CORPUS, topk=0)
```

- [ ] **Step 2: Run the focused tests and verify they fail because the module is absent**

Run: `pytest tests/unit/test_prepare_local_rag_smoke_dataset.py -v`

Expected: collection fails with `ModuleNotFoundError: No module named 'examples.prepare_local_rag_smoke_dataset'`.

- [ ] **Step 3: Implement the bundled examples and retrieval-to-record adapter**

Create `examples/prepare_local_rag_smoke_dataset.py` with these core definitions (CLI work remains for Task 2):

```python
from __future__ import annotations

from pathlib import Path
from typing import Any

from src.internal.servers.retrieval.demo import TfidfRetriever
from src.training.data import build_search_rag_record, format_rag_reference


SMOKE_EXAMPLES: tuple[dict[str, object], ...] = (
    {
        "question": "What system enables efficient nearest-neighbor search over millions of high-dimensional vectors?",
        "golden_answers": ["FAISS"],
    },
    {
        "question": "What ranking function uses term frequency and inverse document frequency?",
        "golden_answers": ["BM25"],
    },
    {
        "question": "What does retrieval-augmented generation combine?",
        "golden_answers": ["a retriever and a generative language model"],
    },
    {
        "question": "Which Python web framework provides automatic OpenAPI documentation?",
        "golden_answers": ["FastAPI"],
    },
)


def _validate_corpus_path(corpus_path: str | Path) -> Path:
    path = Path(corpus_path)
    if not path.is_file():
        raise FileNotFoundError(f"Corpus file not found: {path}")
    if path.stat().st_size == 0:
        raise ValueError(f"Corpus file is empty: {path}")
    return path


def _context_documents(hits: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "id": hit["document"]["id"],
            "title": hit["document"].get("title", ""),
            "contents": hit["document"]["text"],
        }
        for hit in hits
    ]


def build_smoke_records(
    corpus_path: str | Path,
    *,
    topk: int = 3,
) -> list[dict[str, object]]:
    if topk < 1:
        raise ValueError("topk must be at least 1.")
    path = _validate_corpus_path(corpus_path)
    retriever = TfidfRetriever(str(path))
    questions = [str(example["question"]) for example in SMOKE_EXAMPLES]
    retrieval_rows = retriever.retrieve(questions, topk)

    records: list[dict[str, object]] = []
    for index, (example, hits) in enumerate(zip(SMOKE_EXAMPLES, retrieval_rows)):
        if not hits:
            raise ValueError(
                f"No retrieval results for smoke question: {example['question']}"
            )
        context = format_rag_reference(_context_documents(hits))
        records.append(
            build_search_rag_record(
                example,
                context=context,
                split="smoke",
                index=index,
                data_source="local-demo",
                ability="fact-reasoning",
            )
        )
    return records
```

- [ ] **Step 4: Run focused tests and verify they pass**

Run: `pytest tests/unit/test_prepare_local_rag_smoke_dataset.py -v`

Expected: 2 tests pass.

- [ ] **Step 5: Add failing tests for missing, empty, malformed, and no-match corpora**

Append:

```python
def test_build_smoke_records_rejects_missing_corpus(tmp_path):
    with pytest.raises(FileNotFoundError, match="Corpus file not found"):
        build_smoke_records(tmp_path / "missing.jsonl")


def test_build_smoke_records_rejects_empty_corpus(tmp_path):
    corpus = tmp_path / "empty.jsonl"
    corpus.write_text("", encoding="utf-8")

    with pytest.raises(ValueError, match="Corpus file is empty"):
        build_smoke_records(corpus)


def test_build_smoke_records_names_question_when_retrieval_is_empty(
    tmp_path, monkeypatch
):
    corpus = tmp_path / "corpus.jsonl"
    corpus.write_text(
        '{"id":"1","title":"Unrelated","contents":"xyzzy plugh"}\n',
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="No retrieval results for smoke question"):
        build_smoke_records(corpus)


def test_build_smoke_records_surfaces_malformed_jsonl(tmp_path):
    corpus = tmp_path / "broken.jsonl"
    corpus.write_text("not-json\n", encoding="utf-8")

    with pytest.raises(ValueError, match="Malformed corpus JSONL"):
        build_smoke_records(corpus)
```

- [ ] **Step 6: Run the new error tests and verify the malformed-corpus assertion fails**

Run: `pytest tests/unit/test_prepare_local_rag_smoke_dataset.py -v`

Expected: the malformed JSONL test fails because `json.JSONDecodeError` is not yet translated; all other tests pass.

- [ ] **Step 7: Translate malformed-corpus and empty-vocabulary errors at the corpus boundary**

Add `import json`, then replace direct retriever construction with:

```python
    try:
        retriever = TfidfRetriever(str(path))
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"Malformed corpus JSONL at {path}:{exc.lineno}: {exc.msg}"
        ) from exc
    except ValueError as exc:
        if "empty vocabulary" in str(exc).lower():
            raise ValueError(f"Corpus has no searchable text: {path}") from exc
        raise
```

- [ ] **Step 8: Run the complete focused test file**

Run: `pytest tests/unit/test_prepare_local_rag_smoke_dataset.py -v`

Expected: 6 tests pass.

- [ ] **Step 9: Commit the core workflow**

```bash
git add examples/prepare_local_rag_smoke_dataset.py tests/unit/test_prepare_local_rag_smoke_dataset.py
git commit -m "feat: build local RAG smoke records"
```

---

### Task 2: Preview, Parquet Output, and CLI

**Files:**
- Modify: `examples/prepare_local_rag_smoke_dataset.py`
- Modify: `tests/unit/test_prepare_local_rag_smoke_dataset.py`

**Interfaces:**
- Consumes: `build_smoke_records(corpus_path: str | Path, *, topk: int = 3) -> list[dict[str, object]]` from Task 1.
- Produces: `preview_records(records: list[dict[str, object]]) -> None`; `write_parquet(records: list[dict[str, object]], output_path: str | Path) -> Path`; `parse_args() -> argparse.Namespace`; `main() -> None`.

- [ ] **Step 1: Write failing preview and parquet tests**

Append these imports and tests:

```python
import json

from examples.prepare_local_rag_smoke_dataset import (
    preview_records,
    write_parquet,
)


def test_preview_records_prints_auditable_fields(capsys):
    records = build_smoke_records(REPO_CORPUS, topk=1)

    preview_records(records[:1])

    lines = capsys.readouterr().out.strip().splitlines()
    preview = json.loads(lines[-1])
    assert lines[0] == "Local RAG smoke-test preview"
    assert "FAISS" in preview["context_excerpt"]
    assert preview["reward_target"] == ["FAISS"]
    assert preview["extra_info"] == {"split": "smoke", "index": 0}


def test_write_parquet_writes_loadable_compact_records(tmp_path):
    datasets = pytest.importorskip("datasets")
    records = build_smoke_records(REPO_CORPUS, topk=1)
    output = tmp_path / "nested" / "smoke.parquet"

    result = write_parquet(records, output)
    loaded = datasets.Dataset.from_parquet(str(output))

    assert result == output
    assert output.is_file()
    assert len(loaded) == len(records)
    assert set(loaded.column_names) == set(records[0])
```

- [ ] **Step 2: Run tests and verify missing interfaces fail collection**

Run: `pytest tests/unit/test_prepare_local_rag_smoke_dataset.py -v`

Expected: collection fails because `preview_records` and `write_parquet` are not defined.

- [ ] **Step 3: Implement preview and parquet writing**

Add:

```python
def preview_records(records: list[dict[str, object]]) -> None:
    print("Local RAG smoke-test preview")
    for record in records:
        prompt = record["prompt"][0]["content"]
        context = prompt.split("Context:\n", 1)[-1]
        preview = {
            "question": prompt.split("Question:", 1)[-1].split("Context:", 1)[0].strip(),
            "reward_target": record["reward_model"]["ground_truth"]["target"],
            "context_excerpt": context[:300],
            "extra_info": record["extra_info"],
        }
        print(json.dumps(preview, ensure_ascii=False))


def write_parquet(
    records: list[dict[str, object]], output_path: str | Path
) -> Path:
    try:
        import datasets
    except Exception as exc:
        raise RuntimeError(
            "Failed to import Hugging Face datasets; install project requirements."
        ) from exc
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    datasets.Dataset.from_list(records).to_parquet(path)
    return path
```

- [ ] **Step 4: Run preview and parquet tests**

Run: `pytest tests/unit/test_prepare_local_rag_smoke_dataset.py -v`

Expected: 8 tests pass.

- [ ] **Step 5: Write a failing CLI-defaults test**

Append:

```python
from examples.prepare_local_rag_smoke_dataset import parse_args


def test_parse_args_defaults_to_demo_corpus_and_preview(monkeypatch):
    monkeypatch.setattr("sys.argv", ["prepare_local_rag_smoke_dataset"])

    args = parse_args()

    assert args.corpus_path == "data/corpus.jsonl"
    assert args.output_path == "data/local_rag_smoke.parquet"
    assert args.topk == 3
    assert args.preview is False
```

- [ ] **Step 6: Run the CLI test and verify it fails because `parse_args` is absent**

Run: `pytest tests/unit/test_prepare_local_rag_smoke_dataset.py::test_parse_args_defaults_to_demo_corpus_and_preview -v`

Expected: collection fails because `parse_args` is not defined.

- [ ] **Step 7: Implement argument parsing and the command entry point**

Add `import argparse` and:

```python
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a small offline RAG parquet dataset from the demo corpus."
    )
    parser.add_argument("--corpus_path", default="data/corpus.jsonl")
    parser.add_argument("--output_path", default="data/local_rag_smoke.parquet")
    parser.add_argument("--topk", type=int, default=3)
    parser.add_argument(
        "--preview",
        action="store_true",
        help="Print converted records and skip parquet writing.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    records = build_smoke_records(args.corpus_path, topk=args.topk)
    if args.preview:
        preview_records(records)
        return
    output_path = write_parquet(records, args.output_path)
    print(f"Wrote {len(records)} local RAG smoke records to {output_path}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 8: Run the complete focused test file**

Run: `pytest tests/unit/test_prepare_local_rag_smoke_dataset.py -v`

Expected: 9 tests pass.

- [ ] **Step 9: Manually verify preview and parquet output**

Run:

```bash
python3 -m examples.prepare_local_rag_smoke_dataset --topk 1 --preview
python3 -m examples.prepare_local_rag_smoke_dataset --topk 1 --output_path /tmp/local_rag_smoke.parquet
python3 -c 'import datasets; d=datasets.Dataset.from_parquet("/tmp/local_rag_smoke.parquet"); print(len(d), d.column_names)'
```

Expected: preview prints four records with relevant context; write reports 4 rows; inspection prints `4` and the five compact schema columns.

- [ ] **Step 10: Commit the CLI workflow**

```bash
git add examples/prepare_local_rag_smoke_dataset.py tests/unit/test_prepare_local_rag_smoke_dataset.py
git commit -m "feat: expose local RAG smoke-test CLI"
```

---

### Task 3: Documentation and Regression Verification

**Files:**
- Modify: `README.md:421-441`

**Interfaces:**
- Consumes: `python3 -m examples.prepare_local_rag_smoke_dataset [--preview] [--topk N] [--corpus_path PATH] [--output_path PATH]` from Task 2.
- Produces: a README workflow that distinguishes offline smoke preparation from optional full FlashRAG/NQ preparation.

- [ ] **Step 1: Replace the dataset-preparation introduction with the offline smoke workflow**

Insert before the Search-QA commands:

```markdown
# Offline local RAG smoke test (4 examples, existing 30-document demo corpus)
python3 -m examples.prepare_local_rag_smoke_dataset --topk 1 --preview

# Write compact RAG parquet after inspecting the preview
python3 -m examples.prepare_local_rag_smoke_dataset \
  --topk 1 --output_path data/local_rag_smoke.parquet
```

Explain directly below that the command requires no retrieval server, network access, FlashRAG dataset, or retrieval caches.

- [ ] **Step 2: Label Search-QA and full NQ RAG as optional large-dataset workflows**

Keep the current Search-QA commands, then revise the RAG comment and add prerequisites:

```markdown
# Optional: full NQ RAG parquet preparation
# Requires an external Wikipedia corpus plus retrieval-cache JSON files keyed
# by NQ question. prepare_search_qa_dataset does not create these inputs.
python3 -m examples.prepare_search_rag_dataset \
  --dataset_name RUC-NLPIR/FlashRAG_datasets --dataset_config nq \
  --corpus_path data/wiki-18.jsonl \
  --train_retrieval_cache data/nq_train_retrieval_cache.json \
  --test_retrieval_cache data/nq_test_retrieval_cache.json \
  --topk 3 --local_dir data/nq_rag
```

- [ ] **Step 3: Verify every documented command is exposed by its CLI**

Run:

```bash
python3 -m examples.prepare_local_rag_smoke_dataset --help
python3 -m examples.prepare_search_qa_dataset --help
python3 -m examples.prepare_search_rag_dataset --help
```

Expected: all commands exit 0 and show every documented option.

- [ ] **Step 4: Run focused and boundary regression tests**

Run:

```bash
pytest tests/unit/test_prepare_local_rag_smoke_dataset.py tests/unit/test_data.py tests/unit/servers/retrieval/test_demo_retrieval.py -v
```

Expected: all selected tests pass.

- [ ] **Step 5: Run formatting and diff checks**

Run:

```bash
ruff check examples/prepare_local_rag_smoke_dataset.py tests/unit/test_prepare_local_rag_smoke_dataset.py
ruff format --check examples/prepare_local_rag_smoke_dataset.py tests/unit/test_prepare_local_rag_smoke_dataset.py
git diff --check
```

Expected: all commands exit 0 with no formatting or whitespace errors.

- [ ] **Step 6: Commit documentation**

```bash
git add README.md
git commit -m "docs: distinguish local and full NQ RAG preparation"
```

- [ ] **Step 7: Confirm the final commit set and clean worktree**

Run:

```bash
git log --oneline -4
git status --short
```

Expected: the design commit plus three implementation commits are visible, and `git status --short` prints nothing.
