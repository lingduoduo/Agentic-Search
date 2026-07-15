# Generated Context Pack

# Local Rag Smoke Test

## Sources

- [Specification: 2026-07-05-local-rag-smoke-test-design.md](../specs/2026-07-05-local-rag-smoke-test-design.md)
- [Plan: 2026-07-05-local-rag-smoke-test.md](../plans/2026-07-05-local-rag-smoke-test.md)

## Specification Context

### Goal

Provide a small, offline workflow that proves the repository can retrieve context and produce compact RAG training parquet files using the existing `data/corpus.jsonl`. Keep the current FlashRAG/Natural Questions preparation workflow available, but label it as an optional large-dataset path with external prerequisites.

### Scope

The workflow will add `examples.prepare_local_rag_smoke_dataset`. It will use a small set of bundled question-and-answer examples whose answers are present in the existing demo corpus, retrieve passages with the repository's lightweight TF-IDF retriever, convert each example with the existing RAG training-record helpers, and either preview or write the resulting records.

It will not download datasets, build a production retrieval index, generate Natural Questions caches, modify the existing corpus, or replace `examples.prepare_search_rag_dataset`.

### Architecture and Data Flow

The new example module will contain a small immutable collection of QA examples covering documents in `data/corpus.jsonl`. Each example will contain a question and one or more accepted answers.

At runtime the command will:

1. Validate and load the requested corpus path.
2. Construct the existing `TfidfRetriever` directly; no HTTP server is required.
3. Retrieve the requested top-k documents for every bundled question.
4. Fail if a question has no retrieved documents rather than producing an empty-context training record.
5. Convert retrieval results into the context format expected by `build_search_rag_record`.
6. Print auditable records in preview mode or write a compact parquet dataset in output mode.

The command will default to `data/corpus.jsonl`, a small top-k value, and an output path under `data/`. CLI arguments will allow callers to override the corpus, top-k, output path, and preview behavior.

### Component Boundaries

- `TfidfRetriever` remains responsible only for ranking corpus documents.
- The new example module owns the bundled QA examples, result adaptation, validation, preview rendering, and parquet orchestration.
- Existing functions in `src.training.data` remain responsible for canonical RAG prompt and reward-record formatting.
- `examples.prepare_search_rag_dataset` remains the optional adapter for FlashRAG/NQ plus externally generated retrieval caches.

The smoke workflow will import and reuse these components instead of duplicating retrieval or training-record logic.

### Testing and Verification

Unit tests will verify:

- retrieval results are adapted into valid RAG records;
- a known corpus-backed question includes the expected document context;
- missing retrieval results fail clearly;
- preview output exposes the key audit fields;
- argument validation rejects invalid top-k values.

Integration verification will run the preview command and write a temporary parquet output, then inspect its row count and schema. The existing focused data and demo-retriever tests will also run to guard the reused boundaries.

## Implementation Plan Context

### Global Constraints

- The default workflow must run without network access.
- Use the existing `data/corpus.jsonl`; do not modify or duplicate the corpus.
- Do not download datasets, build a production retrieval index, or generate Natural Questions caches.
- Keep `examples.prepare_search_rag_dataset` as the optional FlashRAG/NQ adapter.
- Reuse existing retrieval and RAG record-formatting code instead of duplicating it.

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

_[Section compacted.]_

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

_[Section compacted.]_

### Task 3: Documentation and Regression Verification

**Files:**
- Modify: `README.md:421-441`

**Interfaces:**
- Consumes: `python3 -m examples.prepare_local_rag_smoke_dataset [--preview] [--topk N] [--corpus_path PATH] [--output_path PATH]` from Task 2.
- Produces: a README workflow that distinguishes offline smoke preparation from optional full FlashRAG/NQ preparation.

- [ ] **Step 1: Replace the dataset-preparation introduction with the offline smoke workflow**

Insert before the Search-QA commands:

```markdown

### Offline local RAG smoke test (4 examples, existing 30-document demo corpus)

python3 -m examples.prepare_local_rag_smoke_dataset --topk 1 --preview

## Context Boundary

This pack summarizes its linked sources. Consult those documents for complete details; no implementation status is inferred here.
