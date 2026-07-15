# Generated Context Pack

# Query Transformations

## Sources

- [Specification: 2026-06-16-query-transformations-design.md](../specs/2026-06-16-query-transformations-design.md)
- [Plan: 2026-06-16-query-transformations.md](../plans/2026-06-16-query-transformations.md)

## Specification Context

### Out of Scope

- Replacing `QueryEnhancer` or `AgenticRAGLoop` (they continue to work as-is)
- HTTP endpoint for query transformation
- Streaming transformed results
- Training or fine-tuning query transformation models
- Query routing to different HTTP retrieval servers (only backend-strategy routing via filters)

---

### 2. Architecture

```
RetrievalService.search(query, top_k, filters, ...)
        │
        ▼  [if pipeline is set]
QueryTransformPipeline.transform(query, filters)
  ├── QueryRewriter     → step_back variant        [reuses existing step_back()]
  ├── QueryDecomposer   → sub-query list            [reuses existing decompose()]
  ├── HyDEExpander      → hypothetical passage      [reuses existing hyde()]
  ├── KeywordExpander   → BM25 keyword variants     [reuses existing expand_keywords()]
  └── QueryConstructor  → extracted filters (NEW)   [LLM: NL → structured dict]
        │
        ▼
TransformedQueryBundle
  { original, sub_queries, hyde_text, step_back, keywords, merged_filters }
        │
        ▼  parallel retrieval per variant (ThreadPoolExecutor)
  _search_one(original), _search_one(sub_q1), _search_one(hyde_text), ...
        │
        ▼
RAG-Fusion: rrf_fuse([result_set_per_variant])     [existing rrf_fuse, new wiring]
        │
        ▼
MMR reranking → optional cross-encoder reranker    [unchanged]
        │
        ▼
retrieval_mode: "hybrid+rag_fusion" | "hybrid+rag_fusion+reranked"
```

When `pipeline` is `None` (default), `RetrievalService.search()` behaves exactly as today — single query, no fusion step, no extra latency.

---

### 7. Testing Strategy

All tests are unit tests — no model downloads, no HTTP calls.

### `test_query_transform.py`

- Pipeline with all flags off → `TransformedQueryBundle` with only `original`, all lists empty
- `retrieval_variants()` deduplicates case-insensitively and respects `max_variants`
- `retrieval_variants()` always includes `original` even when all transformers return empty
- Each `QT_*` flag independently enables its transformer (monkeypatch `QueryEnhancer`)
- `from_env()` returns `None` when no `QT_*` vars set
- `from_env()` returns pipeline when at least one `QT_*` var is `true`

### `test_query_constructor.py`

- Extracts `date_year` from "FAISS papers from 2023"
- Extracts `source` from "arxiv articles about dense retrieval"
- Extracts multiple fields from "Hinton papers on arxiv before 2024"
- Malformed LLM JSON → returns `(original_query, {})`, no exception
- LLM raises exception → returns `(original_query, {})`, no exception
- Caller filters override extracted filters on conflict

### `test_service.py` additions

- `test_pipeline_called_when_injected` — `pipeline.transform` called with query and filters
- `test_rag_fusion_runs_retrieval_per_variant` — `_search_one` called N times for N variants
- `test_mode_has_rag_fusion_suffix` — mode ends with `+rag_fusion` when variants > 1
- `test_no_pipeline_single_query_path_unchanged` — no `+rag_fusion`, single retrieval call

---

## Implementation Plan Context

### Task 1: `TransformedQueryBundle` + `QueryTransformConfig`

**Files:**
- Create: `src/context/query_transform.py`
- Create: `tests/unit/test_query_transform.py`

- [ ] **Step 1: Write the failing tests**

```python

### tests/unit/test_query_transform.py

from __future__ import annotations

from src.context.query_transform import QueryTransformConfig, TransformedQueryBundle


def test_bundle_no_variants_returns_original():
    """When all expansions are empty, retrieval_variants returns [original]."""
    bundle = TransformedQueryBundle(original="what is FAISS?")
    assert bundle.retrieval_variants() == ["what is FAISS?"]


def test_retrieval_variants_deduplicates_case_insensitively():
    bundle = TransformedQueryBundle(
        original="what is FAISS?",
        sub_queries=["FAISS vector search", "faiss vector search", "FAISS indexing"],
    )
    variants = bundle.retrieval_variants()
    assert variants.count("FAISS vector search") == 1
    assert "faiss vector search" not in variants or "FAISS vector search" not in variants
    # exactly one of the two casings, not both
    combined = [v.lower() for v in variants]
    assert combined.count("faiss vector search") == 1


def test_retrieval_variants_respects_max_variants():
    bundle = TransformedQueryBundle(
        original="q",
        sub_queries=[f"sub{i}" for i in range(10)],
    )
    assert len(bundle.retrieval_variants(max_variants=3)) == 3


def test_retrieval_variants_always_includes_original():
    """original must appear even when max_variants is tight."""
    bundle = TransformedQueryBundle(
        original="original query",
        sub_queries=["q1", "q2", "q3"],
    )
    variants = bundle.retrieval_variants(max_variants=2)
    assert len(variants) == 2
    assert "original query" in variants


def test_config_defaults_all_false():

_[Section compacted.]_

### Task 2: `QueryConstructor`

**Files:**
- Create: `src/internal/retrieval/query_constructor.py`
- Create: `tests/unit/retrieval/test_query_constructor.py`

- [ ] **Step 1: Write the failing tests**

```python

### tests/unit/retrieval/test_query_constructor.py

from __future__ import annotations

import json
from unittest.mock import MagicMock

from src.internal.retrieval.query_constructor import QueryConstructor


def _llm(response: str) -> MagicMock:
    m = MagicMock()
    m.complete.return_value = response
    return m


def test_extracts_date_year():
    payload = json.dumps({"query": "FAISS papers", "filters": {"date_year": 2023}})
    constructor = QueryConstructor(_llm(payload))
    cleaned, filters = constructor.extract_filters("FAISS papers from 2023")
    assert filters.get("date_year") == 2023
    assert cleaned == "FAISS papers"


def test_extracts_source():
    payload = json.dumps({"query": "dense retrieval", "filters": {"source": "arxiv"}})
    constructor = QueryConstructor(_llm(payload))
    _, filters = constructor.extract_filters("arxiv articles about dense retrieval")
    assert filters.get("source") == "arxiv"


def test_extracts_multiple_fields():
    payload = json.dumps({
        "query": "Hinton papers",
        "filters": {"author": "Hinton", "source": "arxiv", "date_before": "2024-01-01"},
    })
    constructor = QueryConstructor(_llm(payload))
    _, filters = constructor.extract_filters("Hinton papers on arxiv before 2024")
    assert filters["author"] == "Hinton"
    assert filters["source"] == "arxiv"
    assert filters["date_before"] == "2024-01-01"


def test_fallback_on_malformed_json():
    """Invalid JSON from LLM → (original_query, {}) with no exception."""
    constructor = QueryConstructor(_llm("not valid json {{"))

_[Section compacted.]_

### Task 3: `QueryTransformPipeline`

**Files:**
- Modify: `src/context/query_transform.py` (append class)
- Modify: `tests/unit/test_query_transform.py` (append tests)

- [ ] **Step 1: Write the failing tests — append to existing test file**

```python

### Append to tests/unit/test_query_transform.py

import os
from unittest.mock import MagicMock, patch


def _llm(responses: list[str]) -> MagicMock:
    """LLM mock returning responses in order for each complete() call."""
    m = MagicMock()
    m.complete.side_effect = responses
    return m


def test_pipeline_all_flags_off_returns_original_only():
    from src.context.query_transform import QueryTransformPipeline, QueryTransformConfig
    llm = MagicMock()
    pipeline = QueryTransformPipeline(QueryTransformConfig(), llm)
    bundle = pipeline.transform("what is FAISS?")
    assert bundle.original == "what is FAISS?"
    assert bundle.sub_queries == []
    assert bundle.hyde_text is None
    assert bundle.step_back is None
    assert bundle.keywords == []
    assert bundle.merged_filters == {}
    assert bundle.retrieval_variants() == ["what is FAISS?"]


def test_pipeline_decompose_flag_calls_enhancer():
    from src.context.query_transform import QueryTransformPipeline, QueryTransformConfig
    llm = _llm(["sub-q1\nsub-q2"])
    pipeline = QueryTransformPipeline(QueryTransformConfig(decompose=True), llm)
    bundle = pipeline.transform("compare FAISS and ScaNN")
    assert "sub-q1" in bundle.sub_queries
    assert "sub-q2" in bundle.sub_queries


def test_pipeline_hyde_flag_populates_hyde_text():
    from src.context.query_transform import QueryTransformPipeline, QueryTransformConfig
    llm = _llm(["FAISS is a fast library."])
    pipeline = QueryTransformPipeline(QueryTransformConfig(hyde=True), llm)
    bundle = pipeline.transform("what is FAISS?")
    assert bundle.hyde_text == "FAISS is a fast library."

_[Section compacted.]_

### Task 4: Extract `_search_one()` from `RetrievalService`

This is a pure refactor — existing tests must still pass after this change. No new tests needed.

**Files:**
- Modify: `src/internal/retrieval/service.py`

- [ ] **Step 1: Verify existing tests pass before touching anything**

```bash
pytest tests/unit/retrieval/test_service.py -v
```

Expected: all current tests pass (note the count for comparison after)

- [ ] **Step 2: Refactor `service.py` — extract `_search_one()`**

Replace the body of `search()` in `src/internal/retrieval/service.py` so it delegates to a new `_search_one()`. The full updated `service.py` content from `class RetrievalService:` onward:

```python
class RetrievalService:
    def __init__(
        self, backend: RetrievalBackend, reranker: "Reranker | None" = None
    ) -> None:
        self._backend = backend
        self._reranker = reranker

    @classmethod
    def from_env(cls) -> "RetrievalService":
        """Construct service from environment variables."""
        from src.internal.retrieval.reranker import Reranker

        return cls(_build_backend(), reranker=Reranker.from_env())

    def _search_one(
        self,
        query: str,
        over_fetch: int,
        filters: dict | None,
    ) -> tuple[list[RetrievalResult], str]:
        """Run sparse+dense retrieval for one query variant. Returns (results, base_mode)."""
        sparse_results: list[RetrievalResult] = []
        dense_results: list[RetrievalResult] = []
        sparse_ok = dense_ok = False

        with ThreadPoolExecutor(max_workers=2) as executor:
            sparse_future = executor.submit(

_[Section compacted.]_

### Task 5: Wire Pipeline + RAG-Fusion into `RetrievalService`

**Files:**
- Modify: `src/internal/retrieval/service.py`
- Modify: `tests/unit/retrieval/test_service.py`

- [ ] **Step 1: Write the failing tests — append to `tests/unit/retrieval/test_service.py`**

```python

## Context Boundary

This pack summarizes its linked sources. Consult those documents for complete details; no implementation status is inferred here.
