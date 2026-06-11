# Task Plan: Agentic RAG — Best-in-Class Search & Answer Quality

**Goal:** Implement a best-in-class Agentic RAG pipeline over a hybrid index (dense + sparse + rerank) with iterative AI-agent retrieval, query enhancement, and grounded answer synthesis.

**Success criteria:**
- New retrieval server combining hybrid search + neural reranking in a single endpoint
- Query enhancement (decomposition + HyDE) to maximise recall
- Iterative agentic retrieval loop: generate follow-up queries until evidence is sufficient
- Grounded answer synthesis with citation
- All new components have unit tests; full suite stays green

---

## What Already Exists

| Component | Location | Status |
|-----------|----------|--------|
| `DenseRetriever` (FAISS + E5/BGE) | `src/retrieval/dense_retriever.py` | ✅ ready |
| `SparseRetriever` (BM25) | `src/retrieval/sparse_retriever.py` | ✅ ready |
| `HybridRetriever` (RRF fusion) | `src/retrieval/hybrid_retriever.py` | ✅ ready |
| `SentenceTransformerReranker` | `src/retrieval/reranker.py` | ✅ ready |
| `SearchAgentLoop` (XML trace) | `src/agents/search.py` | ✅ ready |
| `answer_with_retrieval` pipeline | `src/context/pipeline.py` | ✅ ready |
| `expand_keywords` (BM25 expansion) | `src/internal/secondary_llm_flows/query_expansion.py` | ✅ ready |
| Retrieval + rerank server | `src/internal/servers/retrieval/retrieval_rerank.py` | ✅ single-backend only |

**Gap:** No server combines hybrid (dense+sparse) with reranking. No query decomposition or HyDE. No iterative sufficiency check in the RAG loop.

---

## Architecture

```
User query
    │
    ▼
Phase 2: QueryEnhancer
  ├── decompose(query) → sub-queries
  └── hyde(query)      → hypothetical doc embedding
    │
    ▼
Phase 1: HybridRerankServer  [POST /retrieve]
  ├── DenseRetriever  (sub-queries + HyDE vector)
  ├── SparseRetriever (original + expanded keywords)
  ├── RRF fusion
  └── CrossEncoder rerank → top-k docs
    │
    ▼
Phase 3: AgenticRAGLoop
  ├── Assess evidence sufficiency
  ├── Generate follow-up queries if insufficient
  └── Loop (max N rounds)
    │
    ▼
Phase 4: Grounded answer synthesis + citation
    │
    ▼
Phase 5: Wire into /api/agent + README
```

---

## Phase 1: Hybrid + Rerank Retrieval Server

**Files:**
- Create: `src/internal/servers/retrieval/hybrid_rerank.py`
- Create: `tests/unit/retrieval/test_hybrid_rerank_server.py`

**What it does:**
- `HybridRerankEngine` wraps `HybridRetriever` + `SentenceTransformerReranker`
- Follows same `batch_search` → `create_search_app` pattern as `serp.py`, `google.py`, `browser.py`
- CLI: `python3 -m src.internal.servers.retrieval.hybrid_rerank --dense_model intfloat/e5-base-v2 --index_path indexes/e5_Flat.index --corpus_path data/corpus.jsonl`

**Status:** [ ] pending

---

## Phase 2: Query Enhancer

**Files:**
- Create: `src/context/query_enhancer.py`
- Create: `tests/unit/test_query_enhancer.py`

**What it does:**
- `QueryEnhancer` with two strategies:
  - `decompose(query, llm)` → list of sub-queries (for multi-hop questions)
  - `hyde(query, llm)` → hypothetical answer text (for HyDE dense retrieval)
- `QueryBundle` dataclass: `original`, `sub_queries`, `hyde_text`
- Both LLM calls are optional/gracefully degraded (return original query on failure)

**Status:** [ ] pending

---

## Phase 3: Agentic RAG Loop

**Files:**
- Create: `src/agents/agentic_rag.py`
- Create: `tests/unit/test_agentic_rag.py`

**What it does:**
- `AgenticRAGLoop` — iterative retrieval loop (max `max_rounds` iterations):
  1. Enhance query (decompose + HyDE)
  2. Retrieve via hybrid+rerank server
  3. Assess evidence sufficiency with LLM: "Is the retrieved context sufficient to answer?"
  4. If insufficient + rounds remaining: generate follow-up queries → go to step 2
  5. Synthesize grounded answer with citations
- `AgenticRAGConfig`: `max_rounds`, `sufficiency_threshold`, `topk`, `retrieval_url`
- `AgenticRAGResult`: `answer`, `citations`, `rounds_used`, `all_contexts`

**Status:** [ ] pending

---

## Phase 4: Wire into Web API

**Files:**
- Modify: `src/internal/servers/web/app.py` or `src/internal/servers/query_and_chat/`
- Add `mode=agentic_rag` to `POST /api/agent`

**Status:** [ ] pending

---

## Phase 5: README Update

**Files:**
- Modify: `README.md`

**Status:** [ ] pending

---

## Errors Encountered

| Error | Attempt | Resolution |
|-------|---------|------------|
| — | — | — |
