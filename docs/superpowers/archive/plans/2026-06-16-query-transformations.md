# Query Transformations Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a composable `QueryTransformPipeline` that generates multiple query variants (decomposition, HyDE, step-back, keyword expansion, filter extraction) and wires RAG-Fusion into `RetrievalService.search()` for multi-variant parallel retrieval.

**Architecture:** `QueryTransformPipeline.transform()` runs all enabled transformations (each independently fallback-safe) and returns a `TransformedQueryBundle`; `RetrievalService.search()` retrieves per variant in parallel with a `ThreadPoolExecutor`, then fuses all result sets via the existing `rrf_fuse()`. When no `QT_*` env vars are set, `from_env()` returns `None` and `search()` behaves exactly as today (zero overhead).

**Tech Stack:** Python 3.12 · `dataclasses` · `concurrent.futures.ThreadPoolExecutor` · existing `QueryEnhancer` (`src/context/query_enhancer.py`) · existing `expand_keywords` (`src/internal/servers/secondary_llm_flows/query_expansion.py`) · existing `rrf_fuse` + `mmr_rerank` (`src/internal/retrieval/fusion.py`) · `OpenAICompatibleLLM` (`src/internal/llm/providers.py`) · `pytest` + `unittest.mock`

---

## File Map

| Action | Path | Responsibility |
|--------|------|----------------|
| **Create** | `src/context/query_transform.py` | `QueryTransformConfig`, `TransformedQueryBundle`, `QueryTransformPipeline` |
| **Create** | `src/internal/retrieval/query_constructor.py` | `QueryConstructor` — LLM-based NL → structured filter extraction |
| **Modify** | `src/internal/retrieval/service.py` | Add `_build_llm()`, `_search_one()`, `pipeline` param; RAG-Fusion path in `search()` |
| **Create** | `tests/unit/test_query_transform.py` | Tests for `TransformedQueryBundle`, `QueryTransformPipeline` |
| **Create** | `tests/unit/retrieval/test_query_constructor.py` | Tests for filter extraction, fallback, merge semantics |
| **Modify** | `tests/unit/retrieval/test_service.py` | 4 new tests: pipeline injection, variant count, `+rag_fusion` mode, no-pipeline path |
| **Modify** | `.env.example` | Add `QT_*` env vars section |

---

## Task 1: `TransformedQueryBundle` + `QueryTransformConfig`

**Files:**
- Create: `src/context/query_transform.py`
- Create: `tests/unit/test_query_transform.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/unit/test_query_transform.py
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
    config = QueryTransformConfig()
    assert config.decompose is False
    assert config.hyde is False
    assert config.step_back is False
    assert config.keywords is False
    assert config.construct_filters is False
    assert config.max_variants == 5
```

- [ ] **Step 2: Run tests to confirm they fail**

```bash
cd /Users/linghuang/Git/Agentic-Search
pytest tests/unit/test_query_transform.py -v 2>&1 | head -20
```

Expected: `ModuleNotFoundError: No module named 'src.context.query_transform'`

- [ ] **Step 3: Write the implementation**

```python
# src/context/query_transform.py
"""QueryTransformConfig, TransformedQueryBundle, QueryTransformPipeline."""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class QueryTransformConfig:
    decompose: bool = False
    hyde: bool = False
    step_back: bool = False
    keywords: bool = False
    construct_filters: bool = False
    max_variants: int = 5


@dataclass(frozen=True)
class TransformedQueryBundle:
    original: str
    sub_queries: list[str] = field(default_factory=list)
    hyde_text: str | None = None
    step_back: str | None = None
    keywords: list[str] = field(default_factory=list)
    merged_filters: dict = field(default_factory=dict)

    def retrieval_variants(self, max_variants: int = 5) -> list[str]:
        """Return deduplicated query variants, always including original.

        Order: sub_queries → hyde_text → step_back → keywords → original (always present).
        Truncated to max_variants total.
        """
        seen: set[str] = set()
        candidates: list[str] = []

        def _add(text: str | None) -> None:
            if text and text.lower() not in seen:
                seen.add(text.lower())
                candidates.append(text)

        for q in self.sub_queries:
            _add(q)
        _add(self.hyde_text)
        _add(self.step_back)
        for kw in self.keywords:
            _add(kw)

        original_already_in = self.original.lower() in seen
        if original_already_in:
            result = candidates[:max_variants]
        else:
            result = candidates[: max_variants - 1]
            result.append(self.original)

        return result if result else [self.original]
```

- [ ] **Step 4: Run tests to confirm they pass**

```bash
pytest tests/unit/test_query_transform.py -v
```

Expected: `5 passed`

- [ ] **Step 5: Commit**

```bash
git add src/context/query_transform.py tests/unit/test_query_transform.py
git commit -m "feat: add TransformedQueryBundle and QueryTransformConfig"
```

---

## Task 2: `QueryConstructor`

**Files:**
- Create: `src/internal/retrieval/query_constructor.py`
- Create: `tests/unit/retrieval/test_query_constructor.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/unit/retrieval/test_query_constructor.py
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
    cleaned, filters = constructor.extract_filters("FAISS papers from 2023")
    assert cleaned == "FAISS papers from 2023"
    assert filters == {}


def test_fallback_on_llm_error():
    """LLM exception → (original_query, {}) with no exception."""
    llm = MagicMock()
    llm.complete.side_effect = RuntimeError("LLM down")
    constructor = QueryConstructor(llm)
    cleaned, filters = constructor.extract_filters("any query")
    assert cleaned == "any query"
    assert filters == {}


def test_unknown_filter_fields_dropped():
    payload = json.dumps({
        "query": "papers",
        "filters": {"source": "arxiv", "unknown_field": "value", "another_unknown": 42},
    })
    constructor = QueryConstructor(_llm(payload))
    _, filters = constructor.extract_filters("papers")
    assert "unknown_field" not in filters
    assert "another_unknown" not in filters
    assert filters["source"] == "arxiv"
```

- [ ] **Step 2: Run tests to confirm they fail**

```bash
pytest tests/unit/retrieval/test_query_constructor.py -v 2>&1 | head -10
```

Expected: `ModuleNotFoundError: No module named 'src.internal.retrieval.query_constructor'`

- [ ] **Step 3: Write the implementation**

```python
# src/internal/retrieval/query_constructor.py
"""LLM-based extraction of structured metadata filters from natural-language queries."""

from __future__ import annotations

import json
import logging

from src.context.models import ChatMessage, LLMClient

logger = logging.getLogger(__name__)

_KNOWN_FILTER_FIELDS = frozenset(
    {"source", "date_year", "date_after", "date_before", "author", "doc_type"}
)

_EXTRACT_PROMPT = """Extract metadata filters from the user's query. Return JSON with exactly two keys:
- "query": the cleaned query with metadata phrases removed
- "filters": an object with any of these fields (omit fields not present):
  - "source": string (e.g. "arxiv", "confluence", "sharepoint")
  - "date_year": integer (e.g. 2023)
  - "date_after": string in "YYYY-MM-DD" format
  - "date_before": string in "YYYY-MM-DD" format
  - "author": string
  - "doc_type": string (e.g. "papers", "tickets", "pages")

Examples:
Query: "FAISS papers from 2023 on arxiv"
Output: {{"query": "FAISS papers", "filters": {{"date_year": 2023, "source": "arxiv"}}}}

Query: "what is attention mechanism"
Output: {{"query": "what is attention mechanism", "filters": {{}}}}

Query: {query}
Output:""".strip()


def _llm_text(response: object) -> str:
    if isinstance(response, str):
        return response
    if hasattr(response, "text"):
        return response.text
    if hasattr(response, "content"):
        return response.content
    return str(response)


class QueryConstructor:
    """Extract structured metadata filters from a natural-language query via LLM."""

    def __init__(self, llm: LLMClient) -> None:
        self._llm = llm

    def extract_filters(self, query: str) -> tuple[str, dict]:
        """Return (cleaned_query, filters).

        Falls back to (query, {}) on any LLM error or JSON parse failure — never raises.
        """
        try:
            raw = _llm_text(
                self._llm.complete(
                    [ChatMessage(role="user", content=_EXTRACT_PROMPT.format(query=query))]
                )
            ).strip()
            # Strip markdown code fences if present
            if raw.startswith("```"):
                parts = raw.split("```")
                raw = parts[1].lstrip("json").strip() if len(parts) > 1 else raw
            parsed = json.loads(raw)
            cleaned_query = str(parsed.get("query", query))
            raw_filters: dict = parsed.get("filters") or {}
            filters: dict = {
                k: v
                for k, v in raw_filters.items()
                if k in _KNOWN_FILTER_FIELDS and v is not None
            }
            return cleaned_query, filters
        except Exception:
            return query, {}
```

- [ ] **Step 4: Run tests to confirm they pass**

```bash
pytest tests/unit/retrieval/test_query_constructor.py -v
```

Expected: `6 passed`

- [ ] **Step 5: Commit**

```bash
git add src/internal/retrieval/query_constructor.py tests/unit/retrieval/test_query_constructor.py
git commit -m "feat: add QueryConstructor for NL to structured filter extraction"
```

---

## Task 3: `QueryTransformPipeline`

**Files:**
- Modify: `src/context/query_transform.py` (append class)
- Modify: `tests/unit/test_query_transform.py` (append tests)

- [ ] **Step 1: Write the failing tests — append to existing test file**

```python
# Append to tests/unit/test_query_transform.py

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


def test_pipeline_step_back_flag_populates_step_back():
    from src.context.query_transform import QueryTransformPipeline, QueryTransformConfig
    llm = _llm(["What are vector similarity search algorithms?"])
    pipeline = QueryTransformPipeline(QueryTransformConfig(step_back=True), llm)
    bundle = pipeline.transform("how does FAISS GPU indexing work?")
    assert bundle.step_back == "What are vector similarity search algorithms?"


def test_pipeline_keywords_flag_calls_expand_keywords():
    import json
    from src.context.query_transform import QueryTransformPipeline, QueryTransformConfig

    llm = MagicMock()
    with patch(
        "src.context.query_transform.expand_keywords",
        return_value=["FAISS", "ANN index"],
    ) as mock_expand:
        pipeline = QueryTransformPipeline(QueryTransformConfig(keywords=True), llm)
        bundle = pipeline.transform("what is FAISS?")

    mock_expand.assert_called_once_with("what is FAISS?", llm)
    assert "FAISS" in bundle.keywords


def test_pipeline_construct_filters_merges_with_caller_filters():
    import json
    from src.context.query_transform import QueryTransformPipeline, QueryTransformConfig

    extracted_payload = json.dumps(
        {"query": "FAISS papers", "filters": {"date_year": 2023, "source": "arxiv"}}
    )
    llm = _llm([extracted_payload])
    caller_filters = {"source": "confluence"}  # caller wins on conflict
    pipeline = QueryTransformPipeline(QueryTransformConfig(construct_filters=True), llm)
    bundle = pipeline.transform("FAISS papers from 2023 on arxiv", filters=caller_filters)
    # caller's source wins over extracted "arxiv"
    assert bundle.merged_filters["source"] == "confluence"
    assert bundle.merged_filters["date_year"] == 2023


def test_from_env_returns_none_when_no_qt_vars_set(monkeypatch):
    from src.context.query_transform import QueryTransformPipeline
    for var in ("QT_DECOMPOSE", "QT_HYDE", "QT_STEP_BACK", "QT_KEYWORDS", "QT_CONSTRUCT_FILTERS"):
        monkeypatch.delenv(var, raising=False)
    result = QueryTransformPipeline.from_env(MagicMock())
    assert result is None


def test_from_env_returns_pipeline_when_one_qt_var_set(monkeypatch):
    from src.context.query_transform import QueryTransformPipeline
    monkeypatch.setenv("QT_DECOMPOSE", "true")
    for var in ("QT_HYDE", "QT_STEP_BACK", "QT_KEYWORDS", "QT_CONSTRUCT_FILTERS"):
        monkeypatch.delenv(var, raising=False)
    pipeline = QueryTransformPipeline.from_env(MagicMock())
    assert pipeline is not None
    assert pipeline._config.decompose is True
    assert pipeline._config.hyde is False
```

- [ ] **Step 2: Run tests to confirm they fail**

```bash
pytest tests/unit/test_query_transform.py -v 2>&1 | tail -15
```

Expected: failures on the new tests (ImportError for `QueryTransformPipeline`)

- [ ] **Step 3: Write the implementation — append to `src/context/query_transform.py`**

Add these imports at the top of the existing file (after `import os`):

```python
# Add to existing imports in src/context/query_transform.py
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.internal.retrieval.query_constructor import QueryConstructor
```

Then append at the bottom of `src/context/query_transform.py`:

```python
class QueryTransformPipeline:
    """Orchestrates query transformation techniques behind one interface.

    Each transformer is independently fallback-safe: on LLM failure, that
    transformer returns its empty/None default and the pipeline continues.
    """

    def __init__(self, config: QueryTransformConfig, llm: object) -> None:
        from src.context.query_enhancer import QueryEnhancer

        self._config = config
        self._llm = llm
        self._enhancer = QueryEnhancer(llm)  # type: ignore[arg-type]
        self._constructor: QueryConstructor | None = None
        if config.construct_filters:
            from src.internal.retrieval.query_constructor import QueryConstructor as QC

            self._constructor = QC(llm)  # type: ignore[arg-type]

    def transform(
        self,
        query: str,
        filters: dict | None = None,
    ) -> TransformedQueryBundle:
        """Run enabled transformations and return a bundle of all query variants."""
        sub_queries: list[str] = []
        hyde_text: str | None = None
        step_back_q: str | None = None
        keywords: list[str] = []
        extracted_filters: dict = {}

        if self._config.decompose:
            sub_queries = self._enhancer.decompose(query)
        if self._config.hyde:
            hyde_text = self._enhancer.hyde(query)
        if self._config.step_back:
            step_back_q = self._enhancer.step_back(query)
        if self._config.keywords:
            from src.internal.servers.secondary_llm_flows.query_expansion import (
                expand_keywords,
            )

            keywords = expand_keywords(query, self._llm)  # type: ignore[arg-type]
        if self._config.construct_filters and self._constructor is not None:
            _, extracted_filters = self._constructor.extract_filters(query)

        # Caller-supplied filters win on key conflict.
        merged_filters: dict = {**extracted_filters, **(filters or {})}

        return TransformedQueryBundle(
            original=query,
            sub_queries=sub_queries,
            hyde_text=hyde_text,
            step_back=step_back_q,
            keywords=keywords,
            merged_filters=merged_filters,
        )

    @classmethod
    def from_env(cls, llm: object) -> "QueryTransformPipeline | None":
        """Return None if no QT_* env vars are enabled (zero overhead for callers)."""

        def _bool(name: str) -> bool:
            return os.environ.get(name, "").lower() in ("1", "true", "yes")

        config = QueryTransformConfig(
            decompose=_bool("QT_DECOMPOSE"),
            hyde=_bool("QT_HYDE"),
            step_back=_bool("QT_STEP_BACK"),
            keywords=_bool("QT_KEYWORDS"),
            construct_filters=_bool("QT_CONSTRUCT_FILTERS"),
            max_variants=int(os.environ.get("QT_MAX_VARIANTS", "5")),
        )

        if not any(
            [
                config.decompose,
                config.hyde,
                config.step_back,
                config.keywords,
                config.construct_filters,
            ]
        ):
            return None

        return cls(config, llm)
```

- [ ] **Step 4: Run all query_transform tests**

```bash
pytest tests/unit/test_query_transform.py -v
```

Expected: `13 passed`

- [ ] **Step 5: Commit**

```bash
git add src/context/query_transform.py tests/unit/test_query_transform.py
git commit -m "feat: add QueryTransformPipeline with all transformation flags"
```

---

## Task 4: Extract `_search_one()` from `RetrievalService`

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
                self._backend.search_sparse, query, top_k=over_fetch, filters=filters
            )
            dense_future = executor.submit(
                self._backend.search_dense, query, top_k=over_fetch, filters=filters
            )

        try:
            sparse_results = sparse_future.result()
            sparse_ok = True
        except Exception as exc:
            logger.warning("Sparse retrieval leg failed: %s", exc)

        try:
            dense_results = dense_future.result()
            dense_ok = True
        except NotImplementedError:
            pass
        except Exception as exc:
            logger.warning("Dense retrieval leg failed: %s", exc)

        if not sparse_ok and not dense_ok:
            raise RuntimeError("Both retrieval legs failed")

        if not dense_ok:
            return sparse_results, "sparse_only"
        elif not sparse_ok:
            return dense_results, "dense_only"
        else:
            return rrf_fuse([sparse_results, dense_results]), "hybrid"

    def search(
        self,
        query: str,
        top_k: int = 5,
        filters: dict | None = None,
    ) -> tuple[list[RetrievalResult], str]:
        """Run sparse and dense legs, fuse with RRF+MMR, fall back gracefully.

        filters: optional key/value pairs applied by each backend before returning results.
        Returns (results, retrieval_mode) where mode is 'hybrid' | 'sparse_only' | 'dense_only'.
        """
        over_fetch = top_k * int(os.environ.get("OVER_FETCH_MULTIPLIER", "2"))
        raw, base_mode = self._search_one(query, over_fetch, filters)

        if base_mode == "hybrid":
            fused = mmr_rerank(raw, top_k=top_k)
        else:
            fused = raw[:top_k]

        if self._reranker:
            fused = self._reranker.rerank(query, fused, top_k)
            base_mode = f"{base_mode}+reranked"

        return fused, base_mode

    def graph_search(
        self,
        query: str,
        top_k: int = 10,
        initial_k: int = 5,
        max_entity_queries: int = 3,
    ) -> list[RetrievalResult]:
        """Graph-augmented retrieval: seed search → entity expansion → RRF fusion."""
        from .graph_rag import graph_rag_search

        return graph_rag_search(
            query,
            service=self,
            top_k=top_k,
            initial_k=initial_k,
            max_entity_queries=max_entity_queries,
        )
```

- [ ] **Step 3: Run existing tests — must all still pass**

```bash
pytest tests/unit/retrieval/test_service.py -v
```

Expected: same count as Step 1, all pass

- [ ] **Step 4: Commit**

```bash
git add src/internal/retrieval/service.py
git commit -m "refactor: extract _search_one() from RetrievalService.search()"
```

---

## Task 5: Wire Pipeline + RAG-Fusion into `RetrievalService`

**Files:**
- Modify: `src/internal/retrieval/service.py`
- Modify: `tests/unit/retrieval/test_service.py`

- [ ] **Step 1: Write the failing tests — append to `tests/unit/retrieval/test_service.py`**

```python
# Append to tests/unit/retrieval/test_service.py

from unittest.mock import patch


def _pipeline_mock(variants: list[str], merged_filters: dict | None = None) -> MagicMock:
    """Helper: mock QueryTransformPipeline returning given variants."""
    pipeline = MagicMock()
    bundle = MagicMock()
    bundle.retrieval_variants.return_value = variants
    bundle.merged_filters = merged_filters or {}
    pipeline.transform.return_value = bundle
    pipeline._config.max_variants = 5
    return pipeline


def test_pipeline_transform_called_with_query_and_filters():
    backend = _sparse_only_backend([_make_result("d1")])
    pipeline = _pipeline_mock(["variant q"])

    service = RetrievalService(backend, pipeline=pipeline)
    service.search("original query", top_k=1, filters={"source": "wiki"})

    pipeline.transform.assert_called_once_with("original query", {"source": "wiki"})


def test_rag_fusion_retrieves_once_per_variant():
    backend = _sparse_only_backend([_make_result("d1")])
    pipeline = _pipeline_mock(["q1", "q2", "q3"])

    service = RetrievalService(backend, pipeline=pipeline)
    service.search("q", top_k=1)

    # _search_one runs once per variant; each calls search_sparse once
    assert backend.search_sparse.call_count == 3


def test_mode_has_rag_fusion_suffix_when_pipeline_set():
    backend = _sparse_only_backend([_make_result("d1")])
    pipeline = _pipeline_mock(["q1", "q2"])

    service = RetrievalService(backend, pipeline=pipeline)
    _, mode = service.search("q", top_k=1)

    assert "+rag_fusion" in mode


def test_no_pipeline_path_unchanged():
    """Without pipeline, mode must not contain +rag_fusion and only one retrieval fires."""
    backend = _sparse_only_backend([_make_result("d1")])
    service = RetrievalService(backend)

    _, mode = service.search("q", top_k=1)

    assert "+rag_fusion" not in mode
    backend.search_sparse.assert_called_once()
```

- [ ] **Step 2: Run tests to confirm they fail**

```bash
pytest tests/unit/retrieval/test_service.py::test_pipeline_transform_called_with_query_and_filters -v
```

Expected: `FAILED` with `TypeError: __init__() got an unexpected keyword argument 'pipeline'`

- [ ] **Step 3: Update `service.py` — add `_build_llm()`, `pipeline` param, RAG-Fusion path**

At the top of `service.py`, add to `TYPE_CHECKING` block:

```python
if TYPE_CHECKING:
    from src.internal.retrieval.reranker import Reranker
    from src.context.query_transform import QueryTransformPipeline
```

Add `_build_llm()` helper function after `_build_backend()`:

```python
def _build_llm() -> object:
    """Build an LLM client from GEN_AI_* environment variables."""
    from src.internal.llm.interfaces import LLMConfig
    from src.internal.llm.providers import OpenAICompatibleLLM

    return OpenAICompatibleLLM(
        LLMConfig(
            model_provider=os.environ.get("GEN_AI_MODEL_PROVIDER", "openai"),
            model_name=os.environ.get("GEN_AI_MODEL_VERSION", "gpt-4o-mini"),
            api_key=os.environ.get("GEN_AI_API_KEY") or os.environ.get("OPENAI_API_KEY"),
            api_base=os.environ.get("GEN_AI_API_BASE"),
            max_input_tokens=int(os.environ.get("GEN_AI_MAX_INPUT_TOKENS", "8192")),
        )
    )
```

Replace `class RetrievalService:` with the full updated class:

```python
class RetrievalService:
    def __init__(
        self,
        backend: RetrievalBackend,
        reranker: "Reranker | None" = None,
        pipeline: "QueryTransformPipeline | None" = None,
    ) -> None:
        self._backend = backend
        self._reranker = reranker
        self._pipeline = pipeline

    @classmethod
    def from_env(cls) -> "RetrievalService":
        """Construct service from environment variables."""
        from src.internal.retrieval.reranker import Reranker

        pipeline = None
        _qt_flags = (
            "QT_DECOMPOSE",
            "QT_HYDE",
            "QT_STEP_BACK",
            "QT_KEYWORDS",
            "QT_CONSTRUCT_FILTERS",
        )
        if any(os.environ.get(v, "").lower() in ("1", "true", "yes") for v in _qt_flags):
            from src.context.query_transform import QueryTransformPipeline

            pipeline = QueryTransformPipeline.from_env(_build_llm())

        return cls(_build_backend(), reranker=Reranker.from_env(), pipeline=pipeline)

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
                self._backend.search_sparse, query, top_k=over_fetch, filters=filters
            )
            dense_future = executor.submit(
                self._backend.search_dense, query, top_k=over_fetch, filters=filters
            )

        try:
            sparse_results = sparse_future.result()
            sparse_ok = True
        except Exception as exc:
            logger.warning("Sparse retrieval leg failed: %s", exc)

        try:
            dense_results = dense_future.result()
            dense_ok = True
        except NotImplementedError:
            pass
        except Exception as exc:
            logger.warning("Dense retrieval leg failed: %s", exc)

        if not sparse_ok and not dense_ok:
            raise RuntimeError("Both retrieval legs failed")

        if not dense_ok:
            return sparse_results, "sparse_only"
        elif not sparse_ok:
            return dense_results, "dense_only"
        else:
            return rrf_fuse([sparse_results, dense_results]), "hybrid"

    def search(
        self,
        query: str,
        top_k: int = 5,
        filters: dict | None = None,
    ) -> tuple[list[RetrievalResult], str]:
        """Run retrieval, fuse with RRF+MMR, fall back gracefully.

        When pipeline is set: generates query variants, retrieves per variant in
        parallel, fuses all result sets via RRF → mode gains '+rag_fusion' suffix.
        Without pipeline: single query, identical behaviour to previous releases.
        """
        over_fetch = top_k * int(os.environ.get("OVER_FETCH_MULTIPLIER", "2"))

        if self._pipeline:
            bundle = self._pipeline.transform(query, filters)
            variants = bundle.retrieval_variants(self._pipeline._config.max_variants)
            active_filters: dict | None = bundle.merged_filters or None
        else:
            variants = [query]
            active_filters = filters

        max_workers = min(len(variants), 4)
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [
                executor.submit(self._search_one, v, over_fetch, active_filters)
                for v in variants
            ]
        result_sets_with_modes = [f.result() for f in futures]
        all_result_sets = [rs for rs, _ in result_sets_with_modes]
        base_mode = result_sets_with_modes[0][1] if result_sets_with_modes else "sparse_only"

        if len(all_result_sets) > 1:
            fused = rrf_fuse(all_result_sets)
            fused = mmr_rerank(fused, top_k=top_k)
            mode = f"{base_mode}+rag_fusion"
        else:
            raw = all_result_sets[0] if all_result_sets else []
            if base_mode == "hybrid":
                fused = mmr_rerank(raw, top_k=top_k)
            else:
                fused = raw[:top_k]
            mode = base_mode

        if self._reranker:
            fused = self._reranker.rerank(query, fused, top_k)
            mode = f"{mode}+reranked"

        return fused, mode

    def graph_search(
        self,
        query: str,
        top_k: int = 10,
        initial_k: int = 5,
        max_entity_queries: int = 3,
    ) -> list[RetrievalResult]:
        """Graph-augmented retrieval: seed search → entity expansion → RRF fusion."""
        from .graph_rag import graph_rag_search

        return graph_rag_search(
            query,
            service=self,
            top_k=top_k,
            initial_k=initial_k,
            max_entity_queries=max_entity_queries,
        )
```

- [ ] **Step 4: Run all service tests**

```bash
pytest tests/unit/retrieval/test_service.py -v
```

Expected: all original tests + 4 new tests pass

- [ ] **Step 5: Run the full unit suite**

```bash
pytest tests/unit/ -q
```

Expected: all tests pass (no regressions)

- [ ] **Step 6: Commit**

```bash
git add src/internal/retrieval/service.py tests/unit/retrieval/test_service.py
git commit -m "feat: wire QueryTransformPipeline into RetrievalService with RAG-Fusion"
```

---

## Task 6: Update `.env.example`

**Files:**
- Modify: `.env.example`

- [ ] **Step 1: Add the `QT_*` section to `.env.example`**

Append the following block before the final comment in `.env.example` (after the `RETRIEVAL_SERVER_PORT` line):

```bash
# Query Transformations — all default to false (disabled)
# Enable any combination to activate multi-query RAG-Fusion retrieval.
# QT_MAX_VARIANTS caps the total number of retrieval variants (default 5).
# When any QT_* flag is true, GEN_AI_* env vars must also be set.
QT_DECOMPOSE=false
QT_HYDE=false
QT_STEP_BACK=false
QT_KEYWORDS=false
QT_CONSTRUCT_FILTERS=false
QT_MAX_VARIANTS=5
```

- [ ] **Step 2: Verify the file looks correct**

```bash
grep -A8 "QT_DECOMPOSE" .env.example
```

Expected: shows the 7 QT_* lines

- [ ] **Step 3: Run a final full suite check**

```bash
pytest tests/unit/ -q
```

Expected: all tests pass

- [ ] **Step 4: Commit**

```bash
git add .env.example
git commit -m "docs: add QT_* env vars to .env.example"
```

---

## Self-Review

**Spec coverage check:**

| Spec requirement | Task |
|-----------------|------|
| `QueryTransformConfig` frozen dataclass | Task 1 |
| `TransformedQueryBundle` with `retrieval_variants()` | Task 1 |
| Case-insensitive dedup, `original` always present | Task 1 |
| `QueryConstructor` NL → structured filters | Task 2 |
| Fallback `(query, {})` on any error | Task 2 |
| Unknown filter fields dropped | Task 2 |
| `QueryTransformPipeline` composing all transformers | Task 3 |
| Each transformer independently fallback-safe | Task 3 (uses existing `QueryEnhancer` fallbacks) |
| `from_env()` returns `None` when no `QT_*` set | Task 3 |
| Caller filters win on conflict | Task 3 |
| `_search_one()` extraction | Task 4 |
| `pipeline` param in `RetrievalService.__init__` | Task 5 |
| Parallel retrieval per variant | Task 5 |
| RAG-Fusion via `rrf_fuse` across all result sets | Task 5 |
| `+rag_fusion` mode suffix | Task 5 |
| `from_env()` builds `_build_llm()` lazily | Task 5 |
| Single-variant path unchanged (no `+rag_fusion`) | Task 5 |
| All existing callers unaffected | Task 5 (default `pipeline=None`) |
| `QT_*` env vars documented | Task 6 |

**Placeholder check:** None found.

**Type consistency check:**
- `TransformedQueryBundle.step_back: str | None` — used as `bundle.step_back` throughout ✓
- `TransformedQueryBundle.merged_filters: dict` — used as `bundle.merged_filters` in service ✓
- `QueryTransformPipeline._config` — accessed as `self._pipeline._config.max_variants` in service ✓
- `_search_one()` returns `tuple[list[RetrievalResult], str]` — unpacked correctly in both single and multi-variant paths ✓
