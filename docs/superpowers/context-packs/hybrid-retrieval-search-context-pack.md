# Generated Context Pack

# Hybrid Retrieval Search

## Sources

- [Specification: 2026-06-21-hybrid-retrieval-search-design.md](../specs/2026-06-21-hybrid-retrieval-search-design.md)
- [Plan: 2026-06-21-hybrid-retrieval-search.md](../plans/2026-06-21-hybrid-retrieval-search.md)

## Specification Context

### Goal

Serve **RRF-fused dense + sparse** results from the internal retrieval provider, reusing the
existing hybrid library, without adding a Java dependency and without changing the web/agent
layers. Dense uses a local e5 embedding model on Apple Silicon (MPS).

### Scope

- New: `src/internal/servers/retrieval/hybrid.py` (`/retrieve` server)
- Reuse: `combine_retrieval_results` (RRF, `src/internal/document_index/hybrid_retriever.py`),
  `TfidfRetriever` (`src/internal/servers/retrieval/demo.py`), and the demo `/retrieve`
  request/response shape.
- New dense leg: `sentence-transformers` (`intfloat/e5-base-v2`) — already a dependency.
- Tests under `tests/unit/`

Out of scope: Pyserini/Java BM25; `DenseRetriever`/FAISS-on-disk plumbing (the demo corpus
is tiny and `faiss-cpu` is unreliable on macOS Apple Silicon — see Decision below);
cross-encoder reranking (web backend already supports `rerank_url` separately); changing the
web backend, agent loops, or frontend.

### Decision: dense leg implementation

Use `sentence-transformers` directly with an **in-memory** embedding matrix and numpy
dot-product search — not `DenseRetriever`/FAISS-on-disk. Rationale: the demo corpus is ~30
docs (brute-force dot product is instant), `DenseRetriever` only exposes e5 *query* encoding
(passage encoding lives behind private helpers), and `import faiss` currently fails on this
macOS Apple Silicon setup. The hybrid *fusion* (`combine_retrieval_results`) and the sparse
leg (`TfidfRetriever`) are still reused unchanged.

### Components

- **`hybrid.py`** — a `/retrieve` FastAPI server with the **same request/response contract
  as `demo.py`**, so `SearchClient` and the whole web stack are unchanged.
- **Dense half (new, sentence-transformers):** a small `DenseEmbeddingRetriever` wrapping
  `SentenceTransformer("intfloat/e5-base-v2")` on MPS. Corpus passages are embedded once at
  startup into an in-memory L2-normalized matrix; per query, encode `"query: <q>"` and take
  the top matches by dot product. Output shape matches `TfidfRetriever`:
  `list[list[{"document": {...}, "score": float}]]`.
- **Sparse half (reuse):** `TfidfRetriever` from `demo.py` (sklearn, no Java).
- **Fusion (reuse):** `combine_retrieval_results([dense, sparse], rrf_k=60)`.

### Testing

- **Fusion:** a doc ranked high by *both* legs outranks a doc ranked high by only one
  (through `combine_retrieval_results`).
- **Contract parity:** hybrid `/retrieve` returns the same shape as `demo.py` (single-query
  dict, batch list-of-lists, `return_scores` honored).
- **Degradation:** dense forced to fail / `--no-dense` → server serves TF-IDF-only, 200.
- **Dense contributes:** with a stub embedder (no e5 download in CI), a query returns a
  dense-only hit the sparse leg misses.

The real e5/MPS path is excluded from CI (stub the embedder); exercised via manual E2E.

## Implementation Plan Context

### Global Constraints

- Never commit to `main`; work on branch `feat/hybrid-retrieval-search`. Run `git branch --show-current` before every commit.
- No Java/Pyserini and no FAISS dependency (faiss-cpu is unreliable on macOS Apple Silicon).
- The `/retrieve` request/response contract MUST match `demo.py` exactly (so `SearchClient` is unchanged): request is `RetrieveRequest{queries?, query?, topk=5, return_scores=False}`; response is `{"results": <row>}` for a single `query`, `{"results": <list-of-rows>}` for batch `queries`; each row item is `{"document": {id,title,text,url}, "score": float}`, or just the `document` when `return_scores` is False.
- Dense embeddings are L2-normalized so dot product == cosine; e5 prefixes: passages `"passage: <text>"`, queries `"query: <text>"`.
- Each leg over-fetches `2·topk` candidates before RRF, then the fused list is truncated to `topk`.
- Reuse `combine_retrieval_results` (`src/internal/document_index/hybrid_retriever.py`) and `TfidfRetriever` / `RetrieveRequest` (`src/internal/servers/retrieval/demo.py`); do not reimplement them.
- Test command: `PYTHONPATH=src:. python -m pytest <path> -q`. CI must not download e5 — inject a stub encoder.

---

### Task 1: `DenseEmbeddingRetriever` (in-memory e5 dense leg)

**Files:**
- Create: `src/internal/servers/retrieval/hybrid.py` (dense retriever portion)
- Test: `tests/unit/servers/retrieval/test_hybrid_retrieval.py`

**Interfaces:**
- Produces:
  - `DenseEmbeddingRetriever(docs: list[dict], *, encoder: Callable[[list[str]], np.ndarray])` with method `retrieve(queries: list[str], topk: int) -> list[list[dict]]` returning rows of `{"document": {id,title,text,url}, "score": float}` (same shape as `TfidfRetriever.retrieve`).
  - `build_e5_encoder(model_name: str = "intfloat/e5-base-v2", device: str = "mps") -> Callable[[list[str]], np.ndarray]` — lazy-imports sentence-transformers, returns a callable that encodes texts to an L2-normalized float32 matrix.

- [ ] **Step 1: Write the failing test**

Create `tests/unit/servers/retrieval/test_hybrid_retrieval.py`:

```python
import numpy as np
import pytest

from src.internal.servers.retrieval.hybrid import DenseEmbeddingRetriever


def _stub_encoder(vectors_by_text):
    """Return a deterministic encoder mapping known texts to unit vectors."""

    def encode(texts):
        rows = []
        for t in texts:
            vec = np.array(vectors_by_text[t], dtype=np.float32)
            vec = vec / (np.linalg.norm(vec) or 1.0)
            rows.append(vec)
        return np.stack(rows)

    return encode


def test_dense_retriever_ranks_by_dot_product():
    docs = [
        {"id": "a", "title": "Cats", "text": "feline animals"},
        {"id": "b", "title": "Dogs", "text": "canine animals"},
    ]
    vecs = {
        "passage: Cats feline animals": [1.0, 0.0],

_[Section compacted.]_

### Task 2: Hybrid `/retrieve` server (RRF fusion + degradation + CLI)

**Files:**
- Modify: `src/internal/servers/retrieval/hybrid.py` (add `create_app`, `parse_args`, `main`)
- Modify: `.claude/CLAUDE.md` (startup docs — note the hybrid server option)
- Test: `tests/unit/servers/retrieval/test_hybrid_retrieval.py`

**Interfaces:**
- Consumes: `DenseEmbeddingRetriever` (Task 1), `TfidfRetriever` / `RetrieveRequest` (demo), `combine_retrieval_results`.
- Produces: `create_app(*, dense: object | None, sparse: object) -> FastAPI` exposing `POST /retrieve`; `main()` CLI entrypoint with `--corpus_path`, `--no-dense`, `--device`, host/port.

- [ ] **Step 1: Write the failing tests**

Append to `tests/unit/servers/retrieval/test_hybrid_retrieval.py`:

```python
from fastapi.testclient import TestClient

from src.internal.servers.retrieval.hybrid import create_app


class _FakeRetriever:
    """Returns preconfigured rows per call, shaped like TfidfRetriever.retrieve."""

    def __init__(self, rows_by_query):
        self._rows_by_query = rows_by_query

    def retrieve(self, queries, topk):
        out = []
        for q in queries:
            out.append(self._rows_by_query.get(q, [])[:topk])
        return out


def _doc(doc_id, score):
    return {"document": {"id": doc_id, "title": doc_id, "text": "", "url": None}, "score": score}


def test_hybrid_retrieve_fuses_dense_and_sparse():
    # 'shared' appears in both legs → should outrank single-leg docs after RRF.
    dense = _FakeRetriever({"q": [_doc("shared", 0.9), _doc("dense_only", 0.8)]})
    sparse = _FakeRetriever({"q": [_doc("shared", 0.5), _doc("sparse_only", 0.4)]})

_[Section compacted.]_

### Task 3: End-to-end verification

**Files:** none (manual verification)

- [ ] **Step 1: Verify TF-IDF-only mode boots and matches the demo contract**

```bash
PYTHONPATH=src:. python -m src.internal.servers.retrieval.hybrid --corpus_path data/corpus.jsonl --no-dense --port 8011 &
sleep 3
curl -s -X POST http://localhost:8011/retrieve -H "Content-Type: application/json" -d '{"query":"FAISS","topk":3}' | python3 -c "import sys,json; print(json.load(sys.stdin)['results'][0]['title'])"
```
Expected: a FAISS-related title (e.g. "Dense Retrieval with FAISS"), confirming `/retrieve` contract parity.

- [ ] **Step 2: Verify hybrid (dense enabled) returns fused results**

```bash
PYTHONPATH=src:. python -m src.internal.servers.retrieval.hybrid --corpus_path data/corpus.jsonl --port 8012 &
sleep 20   # first run downloads e5-base-v2
curl -s -X POST http://localhost:8012/retrieve -H "Content-Type: application/json" -d '{"query":"compare vector index types","topk":5,"return_scores":true}' | python3 -c "import sys,json; d=json.load(sys.stdin); print(len(d['results']), 'results'); print([r['document']['id'] for r in d['results']])"
```
Expected: 5 fused results, returns 200. (Dense leg active; results reflect RRF of e5 + TF-IDF.)

- [ ] **Step 3: Point the web backend at the hybrid server and confirm the UI path works**

With the web backend on :7860 and the hybrid server on :8001 (the backend's default retrieval URL), run a search-intent query and confirm internal results return without error cards:

```bash

_[Section compacted.]_

## Context Boundary

This pack summarizes its linked sources. Consult those documents for complete details; no implementation status is inferred here.
