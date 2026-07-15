# Hybrid Retrieval Search Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A Java-free `/retrieve` server that returns RRF-fused dense (e5, sentence-transformers, in-memory) + sparse (TF-IDF) results, drop-in for the existing `demo.py` so the web "Local Retrieval" provider gets hybrid results with no web/agent changes.

**Architecture:** New `hybrid.py` retrieval server. Dense leg is a small `DenseEmbeddingRetriever` wrapping `SentenceTransformer("intfloat/e5-base-v2")` on MPS, embedding the corpus into an in-memory L2-normalized matrix and scoring by dot product. Sparse leg reuses `demo.py`'s `TfidfRetriever`. The two ranked sets are fused per query with the existing `combine_retrieval_results` (RRF). If the dense leg fails to initialize, the server degrades to TF-IDF-only.

**Tech Stack:** Python 3.12, FastAPI, sentence-transformers, numpy, scikit-learn (TF-IDF). No Java, no FAISS.

## Global Constraints

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
        "passage: Dogs canine animals": [0.0, 1.0],
        "query: tell me about cats": [0.9, 0.1],
    }
    dense = DenseEmbeddingRetriever(docs, encoder=_stub_encoder(vecs))
    rows = dense.retrieve(["tell me about cats"], topk=2)
    assert [item["document"]["id"] for item in rows[0]] == ["a", "b"]
    assert rows[0][0]["score"] > rows[0][1]["score"]


def test_dense_retriever_empty_corpus_returns_empty_rows():
    dense = DenseEmbeddingRetriever([], encoder=lambda texts: np.empty((0, 0), dtype=np.float32))
    assert dense.retrieve(["anything"], topk=3) == [[]]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `PYTHONPATH=src:. python -m pytest tests/unit/servers/retrieval/test_hybrid_retrieval.py -q`
Expected: FAIL — `ModuleNotFoundError`/`ImportError` (hybrid.py / `DenseEmbeddingRetriever` does not exist).

- [ ] **Step 3: Implement the dense retriever**

Create `src/internal/servers/retrieval/hybrid.py`:

```python
"""Hybrid retrieval server — RRF-fused dense (e5, in-memory) + sparse (TF-IDF).

Java-free and FAISS-free. Exposes the same /retrieve API as demo.py so the web
backend's "Local Retrieval" provider gets hybrid results with no changes.
"""

from __future__ import annotations

import argparse
import logging
from collections.abc import Callable

import numpy as np

from src.internal.document_index.hybrid_retriever import combine_retrieval_results
from src.internal.servers.app import (
    add_host_port_args,
    create_base_app,
    load_environment,
    run_uvicorn_app,
)
from src.internal.servers.retrieval.demo import (
    DEFAULT_TOPK,
    RetrieveRequest,
    TfidfRetriever,
    _load_corpus,
)

logger = logging.getLogger(__name__)

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8001
DEFAULT_MODEL = "intfloat/e5-base-v2"


def _passage_text(doc: dict) -> str:
    body = doc.get("contents", doc.get("text", ""))
    return f"passage: {doc.get('title', '')} {body}".strip()


def _as_document(doc: dict, index: int) -> dict:
    return {
        "id": doc.get("id", str(index)),
        "title": doc.get("title", ""),
        "text": doc.get("contents", doc.get("text", "")),
        "url": doc.get("url"),
    }


class DenseEmbeddingRetriever:
    """Dense retrieval over an in-memory e5 embedding matrix (no FAISS).

    Embeds corpus passages once at construction; retrieve() encodes each query
    as "query: <q>" and ranks documents by dot product (embeddings are
    L2-normalized, so dot product == cosine similarity).
    """

    def __init__(
        self, docs: list[dict], *, encoder: Callable[[list[str]], np.ndarray]
    ) -> None:
        self._docs = docs
        self._encoder = encoder
        if docs:
            self._matrix = encoder([_passage_text(d) for d in docs])
        else:
            self._matrix = np.empty((0, 0), dtype=np.float32)

    def retrieve(self, queries: list[str], topk: int) -> list[list[dict]]:
        if self._matrix.size == 0:
            return [[] for _ in queries]
        query_vecs = self._encoder([f"query: {q}" for q in queries])
        sims = query_vecs @ self._matrix.T
        results: list[list[dict]] = []
        for row in sims:
            ranked = sorted(enumerate(row), key=lambda x: x[1], reverse=True)[:topk]
            results.append(
                [
                    {"document": _as_document(self._docs[i], i), "score": float(score)}
                    for i, score in ranked
                ]
            )
        return results


def build_e5_encoder(
    model_name: str = DEFAULT_MODEL, device: str = "mps"
) -> Callable[[list[str]], np.ndarray]:
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_name, device=device)

    def encode(texts: list[str]) -> np.ndarray:
        return model.encode(
            texts, normalize_embeddings=True, convert_to_numpy=True
        ).astype(np.float32)

    return encode
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `PYTHONPATH=src:. python -m pytest tests/unit/servers/retrieval/test_hybrid_retrieval.py -q`
Expected: PASS (2 tests).

- [ ] **Step 5: Commit**

```bash
git add src/internal/servers/retrieval/hybrid.py tests/unit/servers/retrieval/test_hybrid_retrieval.py
git commit -m "feat(retrieval): in-memory e5 DenseEmbeddingRetriever for hybrid search"
```

---

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
    client = TestClient(create_app(dense=dense, sparse=sparse))
    resp = client.post("/retrieve", json={"query": "q", "topk": 3, "return_scores": True})
    assert resp.status_code == 200
    ids = [item["document"]["id"] for item in resp.json()["results"]]
    assert ids[0] == "shared"
    assert set(ids) == {"shared", "dense_only", "sparse_only"}


def test_hybrid_retrieve_contract_single_and_batch():
    dense = _FakeRetriever({"q1": [_doc("a", 0.9)], "q2": [_doc("b", 0.9)]})
    sparse = _FakeRetriever({"q1": [_doc("a", 0.5)], "q2": [_doc("b", 0.5)]})
    client = TestClient(create_app(dense=dense, sparse=sparse))
    # single query → results is a flat list of documents (return_scores False)
    single = client.post("/retrieve", json={"query": "q1", "topk": 2}).json()
    assert single["results"][0]["id"] == "a"
    # batch → results is a list of rows
    batch = client.post("/retrieve", json={"queries": ["q1", "q2"], "topk": 2}).json()
    assert [row[0]["id"] for row in batch["results"]] == ["a", "b"]


def test_hybrid_retrieve_degrades_to_sparse_when_dense_none():
    sparse = _FakeRetriever({"q": [_doc("s1", 0.5), _doc("s2", 0.4)]})
    client = TestClient(create_app(dense=None, sparse=sparse))
    resp = client.post("/retrieve", json={"query": "q", "topk": 2, "return_scores": True})
    assert resp.status_code == 200
    assert [item["document"]["id"] for item in resp.json()["results"]] == ["s1", "s2"]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `PYTHONPATH=src:. python -m pytest tests/unit/servers/retrieval/test_hybrid_retrieval.py -q`
Expected: FAIL — `create_app` does not exist / `ImportError`.

- [ ] **Step 3: Implement `create_app`, `parse_args`, `main`**

Append to `src/internal/servers/retrieval/hybrid.py`:

```python
def _fuse_rows(
    dense_rows: list[list[dict]], sparse_rows: list[list[dict]], topk: int
) -> list[list[dict]]:
    fused: list[list[dict]] = []
    for dense_row, sparse_row in zip(dense_rows, sparse_rows):
        result_sets = [sparse_row] if not dense_row else [dense_row, sparse_row]
        fused.append(combine_retrieval_results(result_sets)[:topk])
    return fused


def create_app(*, dense: object | None, sparse: object):
    app = create_base_app("Hybrid Retrieval Server")

    @app.post("/retrieve")
    def retrieve_endpoint(body: RetrieveRequest):
        queries = body.resolved_queries()
        if not queries:
            return {"results": [] if body.query is not None else []}
        fetch_k = body.topk * 2
        sparse_rows = sparse.retrieve(queries, topk=fetch_k)
        if dense is not None:
            dense_rows = dense.retrieve(queries, topk=fetch_k)
        else:
            dense_rows = [[] for _ in queries]
        rows = _fuse_rows(dense_rows, sparse_rows, body.topk)
        if not body.return_scores:
            rows = [[item["document"] for item in row] for row in rows]
        if body.query is not None:
            return {"results": rows[0] if rows else []}
        return {"results": rows}

    return app


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Hybrid (dense + sparse) retrieval server")
    parser.add_argument(
        "--corpus_path", type=str, required=True, help="Path to corpus.jsonl"
    )
    parser.add_argument("--topk", type=int, default=DEFAULT_TOPK)
    parser.add_argument(
        "--device", type=str, default="mps", help="Device for e5 (mps/cpu/cuda)"
    )
    parser.add_argument(
        "--no-dense", action="store_true", help="Disable the dense leg (TF-IDF only)"
    )
    add_host_port_args(
        parser,
        "HYBRID_RETRIEVAL_HOST",
        "HYBRID_RETRIEVAL_PORT",
        default_host=DEFAULT_HOST,
        default_port=DEFAULT_PORT,
    )
    return parser.parse_args()


def _build_dense(corpus_path: str, device: str) -> DenseEmbeddingRetriever | None:
    try:
        docs = _load_corpus(corpus_path)
        encoder = build_e5_encoder(device=device)
        return DenseEmbeddingRetriever(docs, encoder=encoder)
    except Exception as exc:  # missing deps, model download, MPS unavailable
        logger.warning("Dense leg unavailable, falling back to TF-IDF only: %s", exc)
        return None


def main() -> None:
    load_environment()
    args = parse_args()
    sparse = TfidfRetriever(args.corpus_path)
    dense = None if args.no_dense else _build_dense(args.corpus_path, args.device)
    app = create_app(dense=dense, sparse=sparse)
    run_uvicorn_app(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `PYTHONPATH=src:. python -m pytest tests/unit/servers/retrieval/test_hybrid_retrieval.py -q`
Expected: PASS (5 tests total).

- [ ] **Step 5: Update startup docs**

In `.claude/CLAUDE.md`, under "Running the 3-process local stack", replace the Terminal 1 demo line's comment to note the hybrid option. Add directly below the existing demo command:

```bash
# Terminal 1 (hybrid: RRF-fused dense e5 + sparse TF-IDF, drop-in for demo)
python3 -m src.internal.servers.retrieval.hybrid --corpus_path data/corpus.jsonl
# add --no-dense to force TF-IDF only (skips the e5 model download)
```

- [ ] **Step 6: Lint + commit**

```bash
ruff check src/internal/servers/retrieval/hybrid.py tests/unit/servers/retrieval/test_hybrid_retrieval.py --fix && ruff format src/internal/servers/retrieval/hybrid.py tests/unit/servers/retrieval/test_hybrid_retrieval.py
git add src/internal/servers/retrieval/hybrid.py tests/unit/servers/retrieval/test_hybrid_retrieval.py .claude/CLAUDE.md
git commit -m "feat(retrieval): hybrid /retrieve server with RRF fusion + TF-IDF degradation"
```

---

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
curl -s -X POST http://localhost:7860/api/agent -H "Content-Type: application/json" -d '{"query":"FAISS index types","top_k":5}' | python3 -c "import sys,json; d=json.load(sys.stdin); print('intent=',d['intent']); print([x['title'][:40] for x in d['documents']])"
```
Expected: real internal documents (no "Search error" card).

---

## Self-Review

**Spec coverage:**
- Java-free hybrid `/retrieve` reusing RRF + TF-IDF → Task 1 (dense) + Task 2 (server/fusion). ✅
- Dense = sentence-transformers e5 in-memory, no FAISS → Task 1 `DenseEmbeddingRetriever` + `build_e5_encoder`. ✅
- Contract parity with `demo.py` → Task 2 `create_app` + `test_hybrid_retrieve_contract_single_and_batch`. ✅
- `2·topk` over-fetch then truncate → Task 2 `fetch_k = body.topk * 2`, `combine_retrieval_results(...)[:topk]`. ✅
- Graceful degradation to TF-IDF-only + `--no-dense` → Task 2 `_build_dense` try/except, `dense=None` path, `test_hybrid_retrieve_degrades_to_sparse_when_dense_none`. ✅
- Run on :8001 in place of demo, web stack unchanged → Task 2 defaults + Task 3 Step 3. ✅
- Tests don't download e5 in CI → Task 1/2 inject stub/fake retrievers; real model only in Task 3 manual E2E. ✅
- Ops docs → Task 2 Step 5 (CLAUDE.md). ✅

**Placeholder scan:** No TBD/TODO; every code step has full code.

**Type consistency:** `DenseEmbeddingRetriever(docs, *, encoder)` + `.retrieve(queries, topk)` defined in Task 1 and consumed in Task 2's `main`/`create_app`. `create_app(*, dense, sparse)` defined and consumed within Task 2. Row shape `{"document": {...}, "score": float}` is consistent across dense, sparse (demo), and `combine_retrieval_results`. ✅
