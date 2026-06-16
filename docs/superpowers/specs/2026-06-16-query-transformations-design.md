# Query Transformations PRD — Design Spec

**Date:** 2026-06-16
**Status:** Draft

---

## 1. Goals & Success Criteria

### Problem

The repo already contains individual query transformation primitives — `QueryEnhancer` (HyDE, decompose, step-back), `expand_keywords`, `rrf_fuse`, `semantic_query_rephrase` — but they are scattered across three packages and not composable. `RetrievalService.search()` always retrieves for exactly one query. There is no RAG-Fusion wiring (multi-query → parallel retrieval → RRF across variants), and no structured filter extraction (NL → metadata filters).

### Success Criteria

- A single `QueryTransformPipeline` composes all transformation techniques behind one interface
- `RetrievalService.search()` accepts an optional `pipeline` param; when set, retrieval runs for every variant in parallel and results are fused via RRF
- NL queries containing metadata cues (`"arxiv papers from 2023"`) have filters automatically extracted and merged with retrieval backend filters
- `retrieval_mode` gains `+rag_fusion` suffix when multi-variant retrieval ran
- All existing callers (`AgenticRAGLoop`, `SearchAgentLoop`, `answer_with_retrieval`) are unaffected — zero breakage
- All `QT_*` env vars default to `false`; disabling all of them produces identical behaviour to today

### Out of Scope

- Replacing `QueryEnhancer` or `AgenticRAGLoop` (they continue to work as-is)
- HTTP endpoint for query transformation
- Streaming transformed results
- Training or fine-tuning query transformation models
- Query routing to different HTTP retrieval servers (only backend-strategy routing via filters)

---

## 2. Architecture

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

## 3. Components

### 3.1 `TransformedQueryBundle`

**File:** `src/context/query_transform.py`

```python
@dataclass(frozen=True)
class TransformedQueryBundle:
    original: str
    sub_queries: list[str]      # from QueryDecomposer
    hyde_text: str | None       # from HyDEExpander
    step_back: str | None       # from QueryRewriter
    keywords: list[str]         # from KeywordExpander
    merged_filters: dict        # caller_filters merged with QueryConstructor output; {} when no filters

    def retrieval_variants(self, max_variants: int = 5) -> list[str]:
        """Return deduplicated text queries to retrieve against.

        Order: sub_queries → hyde_text → step_back → keywords → original (always last fallback).
        Truncated to max_variants. original is always included.
        """
```

Deduplication is case-insensitive and order-preserving. `original` is always present even if all other expansions are empty.

---

### 3.2 `QueryTransformConfig`

**File:** `src/context/query_transform.py`

```python
@dataclass(frozen=True)
class QueryTransformConfig:
    decompose: bool = False         # sub-question decomposition
    hyde: bool = False              # hypothetical document embedding
    step_back: bool = False         # abstract step-back rewriting
    keywords: bool = False          # BM25 keyword expansion
    construct_filters: bool = False # NL → metadata filter extraction
    max_variants: int = 5           # hard cap; prevents runaway parallel retrieval
```

All features default to `False`. `max_variants` caps `TransformedQueryBundle.retrieval_variants()`.

---

### 3.3 `QueryTransformPipeline`

**File:** `src/context/query_transform.py`

```python
class QueryTransformPipeline:
    def __init__(self, config: QueryTransformConfig, llm: LLMClient) -> None:
        self._config = config
        self._enhancer = QueryEnhancer(llm)           # existing class, reused
        self._constructor = QueryConstructor(llm) if config.construct_filters else None

    def transform(
        self,
        query: str,
        filters: dict | None = None,
    ) -> TransformedQueryBundle:
        """Run enabled transformers and return a bundle of all query variants."""

    @classmethod
    def from_env(cls, llm: LLMClient) -> QueryTransformPipeline | None:
        """Return None if no QT_* env vars are set (zero overhead for callers)."""
```

Each transformer is called independently. If an LLM call fails (network error, timeout), that transformer returns its empty/`None` fallback — the pipeline continues with whatever variants succeeded, always including `original`.

---

### 3.4 `QueryConstructor` (New)

**File:** `src/internal/retrieval/query_constructor.py`

Extracts structured metadata filters from a natural-language query via a one-shot LLM prompt. Returns both a cleaned query (metadata phrases removed) and a filter dict.

**Supported filter fields:**

| Field | Type | Example trigger |
|---|---|---|
| `source` | `str` | "on arxiv", "from confluence", "in sharepoint" |
| `date_year` | `int` | "from 2023", "in 2024" |
| `date_after` | `str` | "after January 2023" → `"2023-01-01"` |
| `date_before` | `str` | "before 2024" → `"2024-01-01"` |
| `author` | `str` | "by Hinton", "authored by LeCun" |
| `doc_type` | `str` | "papers", "tickets", "pages" |

**Interface:**

```python
class QueryConstructor:
    def __init__(self, llm: LLMClient) -> None: ...

    def extract_filters(self, query: str) -> tuple[str, dict]:
        """Return (cleaned_query, filters).

        Example:
            "FAISS papers from 2023 on arxiv"
            → ("FAISS papers", {"date_year": 2023, "source": "arxiv"})

        Falls back to (query, {}) on LLM error or JSON parse failure — never raises.
        """
```

**Prompt strategy:** Single-turn structured extraction. LLM is asked to return JSON with `"query"` and `"filters"` keys. Empty filters → `{}` (not `null`). Unknown fields are dropped.

**Filter merge semantics:** `merged = {**caller_filters, **extracted}`. Caller-supplied filters win on conflict — the caller always has authoritative metadata context.

---

## 4. `RetrievalService` Integration

### 4.1 Constructor changes

```python
class RetrievalService:
    def __init__(
        self,
        backend: RetrievalBackend,
        reranker: Reranker | None = None,
        pipeline: QueryTransformPipeline | None = None,  # new
    ) -> None:
        self._backend = backend
        self._reranker = reranker
        self._pipeline = pipeline
```

### 4.2 `search()` changes

```python
def search(
    self,
    query: str,
    top_k: int = 5,
    filters: dict | None = None,
) -> tuple[list[RetrievalResult], str]:
    # 1. Transform
    if self._pipeline:
        bundle = self._pipeline.transform(query, filters)
        variants = bundle.retrieval_variants(self._pipeline.config.max_variants)
        filters  = bundle.merged_filters
    else:
        variants = [query]

    # 2. Parallel retrieval per variant
    over_fetch = top_k * int(os.environ.get("OVER_FETCH_MULTIPLIER", "2"))
    with ThreadPoolExecutor(max_workers=min(len(variants), 4)) as executor:
        futures = [
            executor.submit(self._search_one, v, over_fetch, filters)
            for v in variants
        ]
    all_result_sets = [f.result() for f in futures]

    # 3. RAG-Fusion
    if len(all_result_sets) > 1:
        fused = rrf_fuse(all_result_sets)
        mode = f"{base_mode}+rag_fusion"
    else:
        fused = all_result_sets[0]
        mode = base_mode

    # 4. MMR + reranker (unchanged)
    fused = mmr_rerank(fused, top_k=top_k)
    if self._reranker:
        fused = self._reranker.rerank(query, fused, top_k)
        mode = f"{mode}+reranked"

    return fused[:top_k], mode
```

`_search_one(query, over_fetch, filters)` is a private method extracted from the current sparse+dense+RRF logic, so it can be called per variant without duplicating the fallback logic.

### 4.3 `from_env()` extension

```python
@classmethod
def from_env(cls) -> RetrievalService:
    from src.context.query_transform import QueryTransformPipeline
    llm = _build_llm()
    return cls(
        _build_backend(),
        reranker=Reranker.from_env(),
        pipeline=QueryTransformPipeline.from_env(llm),
    )
```

`_build_llm()` follows the existing `GEN_AI_*` env var pattern used elsewhere in the serving stack.

---

## 5. Environment Variables

| Variable | Default | Effect |
|---|---|---|
| `QT_DECOMPOSE` | `false` | Enable sub-query decomposition |
| `QT_HYDE` | `false` | Enable HyDE expansion |
| `QT_STEP_BACK` | `false` | Enable step-back rewriting |
| `QT_KEYWORDS` | `false` | Enable BM25 keyword variants |
| `QT_CONSTRUCT_FILTERS` | `false` | Enable NL → filter extraction |
| `QT_MAX_VARIANTS` | `5` | Cap on total retrieval variants |

When none of these are set, `QueryTransformPipeline.from_env()` returns `None` and `RetrievalService` behaves identically to its current state.

---

## 6. File Map

| Action | Path | Responsibility |
|---|---|---|
| **Create** | `src/context/query_transform.py` | `QueryTransformConfig`, `TransformedQueryBundle`, `QueryTransformPipeline` |
| **Create** | `src/internal/retrieval/query_constructor.py` | `QueryConstructor` — LLM-based NL → filter extraction |
| **Modify** | `src/internal/retrieval/service.py` | Add `pipeline` param; extract `_search_one()`; parallel variant retrieval; RAG-Fusion mode |
| **Create** | `tests/unit/context/test_query_transform.py` | Pipeline config, feature flags, `retrieval_variants()` dedup/cap, `from_env()` |
| **Create** | `tests/unit/retrieval/test_query_constructor.py` | Filter extraction, fallback on bad JSON, filter merge semantics |
| **Modify** | `tests/unit/retrieval/test_service.py` | 4 new tests: pipeline injection, variant retrieval count, `+rag_fusion` mode, no-pipeline path unchanged |

**Not changed:** `src/context/query_enhancer.py`, `src/agents/agentic_rag.py`, `src/agents/search.py`, `src/internal/retrieval/fusion.py`, any existing test.

---

## 7. Testing Strategy

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

## 8. Error Handling

| Scenario | Behavior |
|---|---|
| LLM unavailable during `transform()` | Each transformer catches and returns `None`/`[]`; `original` always included |
| `QueryConstructor` returns invalid JSON | Falls back to `(original_query, {})` — no exception, no log noise |
| `max_variants` exceeded | `retrieval_variants()` truncates; logs count at DEBUG level |
| All variant retrievals return empty | `rrf_fuse([[], []])` → `[]`; caller gets empty list, mode still set |
| `QT_CONSTRUCT_FILTERS=true` but `GEN_AI_*` not set | `_build_llm()` raises at startup (fail-fast, same as today) |
| Single variant (no transformation) | Skips `rrf_fuse`, no `+rag_fusion` suffix, zero extra overhead |
