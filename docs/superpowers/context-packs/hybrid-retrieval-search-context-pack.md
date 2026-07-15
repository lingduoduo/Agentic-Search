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

…

### Decision: dense leg implementation

Use `sentence-transformers` directly with an **in-memory** embedding matrix and numpy
dot-product search — not `DenseRetriever`/FAISS-on-disk. Rationale: the demo corpus is ~30
docs (brute-force dot product is instant), `DenseRetriever` only exposes e5 *query* encoding
(passage encoding lives behind private helpers), and `import faiss` currently fails on this
macOS Apple Silicon setup. The hybrid *fusion* (`combine_retrieval_results`) and the sparse
leg (`TfidfRetriever`) are still reused unchanged.

## Implementation Plan Context

### Task 1: `DenseEmbeddingRetriever` (in-memory e5 dense leg)

**Files:**
- Create: `src/internal/servers/retrieval/hybrid.py` (dense retriever portion)
- Test: `tests/unit/servers/retrieval/test_hybrid_retrieval.py`

**Interfaces:**
- Produces:
  - `DenseEmbeddingRetriever(docs: list[dict], *, encoder: Callable[[list[str]], np.ndarray])` with method `retrieve(queries: list[str], topk: int) -> list[list[dict]]` returning rows of `{"document": {id,title,text,url}, "score": float}` (same shape as `TfidfRetriever.retrieve`).
  - `build_e5_encoder(model_name: str = "intfloat/e5-base-v2", device: str = "mps") -> Callable[[list[str]], np.ndarray]` — lazy-imports sentence-transformers, returns a callable that encodes texts to an L2-normalized float32 matrix.

…

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

…

### Task 3: End-to-end verification

**Files:** none (manual verification)

- [ ] **Step 1: Verify TF-IDF-only mode boots and matches the demo contract**

Expected: a FAISS-related title (e.g. "Dense Retrieval with FAISS"), confirming `/retrieve` contract parity.

- [ ] **Step 2: Verify hybrid (dense enabled) returns fused results**

Expected: 5 fused results, returns 200. (Dense leg active; results reflect RRF of e5 + TF-IDF.)

- [ ] **Step 3: Point the web backend at the hybrid server and confirm the UI path works**

With the web backend on :7860 and the hybrid server on :8001 (the backend's default retrieval URL), run a search-intent query and confirm internal results return without error cards:

…

## Context Boundary

This pack summarizes its linked sources. Consult those documents for complete details; no implementation status is inferred here.
