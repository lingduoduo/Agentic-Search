# Retrieval PRD — Milestone 3: OpenSearch + Weaviate Backends + CI Eval Gate

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add `OpenSearchBackend` (BM25 + kNN) and `WeaviateBackend` (BM25 + nearVector) so `RETRIEVAL_BACKEND` can be set to `local|opensearch|weaviate`. Add a BEIR eval script and a CI job that fails PRs when eval metrics drop. Remove the three legacy server files.

**Architecture:** Each new backend wraps a vendor client via a monkeypatchable factory function; no live server is needed for unit tests. `service.py._build_backend()` is extended with two new branches. The CI gate reads `data/eval/baseline_metrics.json` (written by a human after the first green run) and fails if Recall@10 drops > 2pp or NDCG@10 drops > 1pp.

**Tech Stack:** Python 3.12, `opensearch-py`, `weaviate-client>=4.9`, GitHub Actions.

**Spec:** `docs/superpowers/specs/2026-06-15-retrieval-prd-design.md` Milestone 3.

**Gate:** NDCG@10 ≥ 0.45 on nfcorpus, fiqa, scifact (requires live indexes). CI green on 3 consecutive PRs.

---

## File Map

| Action | Path | Responsibility |
|---|---|---|
| Create | `src/internal/retrieval/backends/opensearch.py` | BM25 `match` + kNN on OpenSearch |
| Create | `src/internal/retrieval/backends/weaviate.py` | BM25 + `nearVector` on Weaviate |
| Modify | `src/internal/retrieval/service.py` | Wire opensearch/weaviate in `_build_backend()` |
| Create | `src/internal/retrieval/beir_eval.py` | BEIR dataset download + eval runner |
| Create | `.github/workflows/eval-gate.yml` | CI job: pytest + optional eval gate |
| Modify | `src/__init__.py` | Remove lazy refs to deleted modules |
| Delete | `src/internal/servers/retrieval/retrieval_server.py` | Legacy — replaced by server.py |
| Delete | `src/internal/servers/retrieval/retrieval_rerank.py` | Legacy — replaced by eval_router.py |
| Delete | `src/internal/servers/retrieval/hybrid_rerank.py` | Legacy — replaced by eval_router.py |
| Create | `tests/unit/retrieval/test_opensearch_backend.py` | Unit tests for OpenSearchBackend |
| Create | `tests/unit/retrieval/test_weaviate_backend.py` | Unit tests for WeaviateBackend |

---

### Task 1: `backends/opensearch.py`

**Env vars:** `OPENSEARCH_URL` (e.g. `http://localhost:9200`), `OPENSEARCH_INDEX`, `OPENSEARCH_CONTENT_FIELD` (default `content`), `OPENSEARCH_VECTOR_FIELD` (default `content_vector`), `OPENSEARCH_DOC_ID_FIELD` (default `document_id`), `OPENSEARCH_TITLE_FIELD` (default `title`), `OPENSEARCH_URL_FIELD` (default `source_links`).

**Dense search:** Needs a query embedder to convert query text to a vector. Pass `embedder: Callable[[str], list[float]] | None`; if None, `search_dense` raises `NotImplementedError`.

**Hit mapping:** `_source["document_id"]` → `doc_id`, `_source["title"]` → `title`, `_source["content"]` → `text`, `_source.get("source_links")` → `url`, `_score` → `score`.

**Files:**
- Create: `src/internal/retrieval/backends/opensearch.py`
- Create: `tests/unit/retrieval/test_opensearch_backend.py`

### Task 2: `backends/weaviate.py`

**Env vars:** `WEAVIATE_URL` (e.g. `http://localhost:8080`), `WEAVIATE_COLLECTION` (e.g. `Document`).

**Weaviate v4 API:** Uses `weaviate.connect_to_local()` / `connect_to_custom()`. BM25 via `collection.query.bm25(query, limit)`. nearVector via `collection.query.near_vector(near_vector=vec, limit)`. Both return `weaviate.types.WeaviateReturn` with `.objects` list.

**Hit mapping:** `obj.uuid` → `doc_id`, `obj.properties.get("title", "")` → `title`, `obj.properties.get("content", "")` → `text`, `obj.properties.get("source_links")` → `url`, `obj.metadata.score` → `score` (may be None — fall back to `obj.metadata.distance` negated or 0.0).

**Dense search:** Same embedder pattern as OpenSearch backend.

**Files:**
- Create: `src/internal/retrieval/backends/weaviate.py`
- Create: `tests/unit/retrieval/test_weaviate_backend.py`

### Task 3: Wire backends in `service.py`

Add `opensearch` and `weaviate` branches to `_build_backend()`.

**Files:**
- Modify: `src/internal/retrieval/service.py`

### Task 4: BEIR eval script

Thin script that downloads BEIR corpora (nfcorpus, fiqa, scifact) via `ir-datasets` or direct BEIR download, runs `run_eval()` per dataset, writes results JSON, and optionally compares against a baseline.

**Files:**
- Create: `src/internal/retrieval/beir_eval.py`

### Task 5: Metadata filtering in OpenSearch and Weaviate backends

**PRD reference:** Section 7 — `"filters": { "source": "confluence" }` in `POST /search`. M1 Task 9 added post-hoc Python filtering to `LocalBackend`. OpenSearch and Weaviate support server-side filtering — more efficient and must be wired here.

**Files:**
- Modify: `src/internal/retrieval/backends/opensearch.py`
- Modify: `src/internal/retrieval/backends/weaviate.py`
- Modify: `tests/unit/retrieval/test_opensearch_backend.py` (append)
- Modify: `tests/unit/retrieval/test_weaviate_backend.py` (append)

**OpenSearch filtering** — wrap `match` query in `bool+filter` when `filters` is provided:

```python
def _build_sparse_body(
    query: str, content_field: str, filters: dict | None
) -> dict:
    match_clause = {"match": {content_field: query}}
    if not filters:
        return {"query": match_clause}
    return {
        "query": {
            "bool": {
                "must": [match_clause],
                "filter": [{"term": {k: v}} for k, v in filters.items()],
            }
        }
    }
```

For kNN, add a `"filter"` key inside the knn clause:

```python
def _build_dense_body(
    vec: list[float], vector_field: str, top_k: int, filters: dict | None
) -> dict:
    knn_clause: dict = {"vector": vec, "k": top_k}
    if filters:
        knn_clause["filter"] = {
            "bool": {"filter": [{"term": {k: v}} for k, v in filters.items()]}
        }
    return {"query": {"knn": {vector_field: knn_clause}}}
```

Update `search_sparse()` and `search_dense()` signatures to `(self, query, top_k, filters=None)`.

**Failing tests to append to `test_opensearch_backend.py`:**

```python
def test_sparse_query_wrapped_in_bool_when_filters_set(monkeypatch):
    import src.internal.retrieval.backends.opensearch as os_mod
    fake_client = _make_fake_client([])
    monkeypatch.setattr(os_mod, "_make_client", lambda url: fake_client)
    backend = OpenSearchBackend.from_env()
    backend.search_sparse("q", top_k=5, filters={"source": "confluence"})
    body = fake_client.search.call_args[1]["body"]
    assert body["query"]["bool"]["filter"] == [{"term": {"source": "confluence"}}]


def test_sparse_query_is_plain_match_without_filters(monkeypatch):
    import src.internal.retrieval.backends.opensearch as os_mod
    fake_client = _make_fake_client([])
    monkeypatch.setattr(os_mod, "_make_client", lambda url: fake_client)
    backend = OpenSearchBackend.from_env()
    backend.search_sparse("q", top_k=5)
    body = fake_client.search.call_args[1]["body"]
    assert "match" in body["query"]
    assert "bool" not in body["query"]
```

**Weaviate filtering** — use `Filter.by_property()` from `weaviate.classes.query`:

```python
from weaviate.classes.query import Filter as WvFilter

def _weaviate_filter(filters: dict | None):
    if not filters:
        return None
    clauses = [WvFilter.by_property(k).equal(v) for k, v in filters.items()]
    return clauses[0] if len(clauses) == 1 else WvFilter.all_of(clauses)

# In search_sparse:
results = self._collection.query.bm25(query=query, limit=top_k, filters=_weaviate_filter(filters))

# In search_dense:
results = self._collection.query.near_vector(near_vector=vec, limit=top_k, filters=_weaviate_filter(filters))
```

**Failing tests to append to `test_weaviate_backend.py`:**

```python
def test_bm25_passes_filter_object_when_filters_set(monkeypatch):
    import src.internal.retrieval.backends.weaviate as wv_mod
    fake_collection = _make_fake_collection([])
    monkeypatch.setattr(wv_mod, "_make_collection", lambda url, name: fake_collection)
    backend = WeaviateBackend.from_env()
    backend.search_sparse("q", top_k=5, filters={"source": "confluence"})
    kw = fake_collection.query.bm25.call_args[1]
    assert kw.get("filters") is not None


def test_bm25_no_filter_kwarg_when_filters_none(monkeypatch):
    import src.internal.retrieval.backends.weaviate as wv_mod
    fake_collection = _make_fake_collection([])
    monkeypatch.setattr(wv_mod, "_make_collection", lambda url, name: fake_collection)
    backend = WeaviateBackend.from_env()
    backend.search_sparse("q", top_k=5)
    kw = fake_collection.query.bm25.call_args[1]
    assert kw.get("filters") is None
```

- [ ] **Step 1: Implement OpenSearch filtering in `opensearch.py`** (extract `_build_sparse_body` + `_build_dense_body` helpers; update both search methods)
- [ ] **Step 2: Run and verify OpenSearch filter tests pass**

```bash
pytest tests/unit/retrieval/test_opensearch_backend.py -v
```

Expected: all pass (including 2 new filter tests)

- [ ] **Step 3: Implement Weaviate filtering in `weaviate.py`** (add `_weaviate_filter` helper; update `search_sparse` and `search_dense`)
- [ ] **Step 4: Run and verify Weaviate filter tests pass**

```bash
pytest tests/unit/retrieval/test_weaviate_backend.py -v
```

Expected: all pass (including 2 new filter tests)

- [ ] **Step 5: Commit**

```bash
git add src/internal/retrieval/backends/opensearch.py \
        src/internal/retrieval/backends/weaviate.py \
        tests/unit/retrieval/test_opensearch_backend.py \
        tests/unit/retrieval/test_weaviate_backend.py
git commit -m "feat(retrieval): server-side metadata filtering in OpenSearch (bool+filter) and Weaviate (Filter.by_property)"
```

---

### Task 6: CI eval gate + legacy removal

Create `.github/workflows/eval-gate.yml`, remove three legacy server files, clean `src/__init__.py`.

**Files:**
- Create: `.github/workflows/eval-gate.yml`
- Modify: `src/__init__.py`
- Delete: three legacy server files
