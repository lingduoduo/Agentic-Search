# Findings

## Existing Retrieval Infrastructure

### HybridRetriever (`src/retrieval/hybrid_retriever.py`)
- Takes `DenseRetrieverConfig` + optional `SparseRetrieverConfig` + `alpha` weight
- `batch_search(queries)` returns `list[list[dict]]` with optional scores
- Uses RRF (`combine_retrieval_results`) for fusion
- **Key**: already works, just no server wrapping it with reranking

### SentenceTransformerReranker (`src/retrieval/reranker.py`)
- `rerank(query, docs)` → sorted docs
- Used in `retrieval_rerank.py` server (but that only wraps single-backend retrieval)

### Retrieval Server Pattern (`src/backend/servers/retrieval/app.py`)
- `create_search_app(title, engine)` — engine must implement `batch_search(queries)`
- `format_document(title, content, url)` → `{"document": {"title", "contents", "url"}}`
- `add_host_port_args`, `load_environment`, `run_uvicorn_app` — all reusable

### Context Pipeline (`src/context/pipeline.py`)
- `answer_with_retrieval(question, *, search_url, llm, top_k, ...)` — top-level async fn
- `retrieve_context(question, *, search_url, top_k, filters)` → `SearchContextBundle`
- `generate_answer(request, *, llm)` → `AnswerGenerationResult`
- `build_chat_prompt(question, context, history, config)` → `ChatPrompt`

### Query Expansion (`src/backend/secondary_llm_flows/query_expansion.py`)
- `expand_keywords(user_query, llm)` → list of expanded BM25 keyword queries
- LLM-backed, gracefully returns `[]` on failure
- Uses `KEYWORD_EXPANSION_PROMPT` from `src/backend/prompts/query_expansion.py`

### SearchAgentLoop (`src/agents/search.py`)
- XML protocol: `<think>`, `<search>`, `<information>`, `<answer>`
- Multi-turn with dedup, budget (max_search_rounds), fetch support
- Parser handles `[task_id] query` prefixes for multi-query decomposition

## Key Design Decisions for Agentic RAG

### Why HyDE?
HyDE (Hypothetical Document Embeddings) generates a fake "ideal answer" and uses its embedding as the query vector. This dramatically improves dense retrieval for questions where the query text differs a lot from document vocabulary.

### Why query decomposition?
Multi-hop questions ("What did CEO X say about Y and how did it affect Z?") require separate retrievals. Decomposition lets each sub-query find its own evidence before synthesis.

### Sufficiency assessment
Rather than fixed-round retrieval, the agent asks the LLM: "Given these documents, can I fully answer the question?" This is the key agentic element — the loop terminates early when evidence is good enough, or runs more rounds when it isn't.

### Why separate HybridRerankServer?
The existing `retrieval_rerank.py` only wraps a single retrieval method. A hybrid+rerank server enables the full quality pipeline behind a single `/retrieve` endpoint — drop-in replacement, no client changes needed.

## Interfaces to Build Against

### `HybridRetriever.__init__`
```python
HybridRetriever(
    dense_config: DenseRetrieverConfig,
    sparse_config: SparseRetrieverConfig | None = None,
    alpha: float = 0.5,
)
```

### `SentenceTransformerReranker`
```python
reranker = get_reranker(RerankerConfig(model_name="cross-encoder/ms-marco-MiniLM-L-6-v2"))
docs = reranker.rerank(query, docs)
```

### LLMClient protocol (`src/context/models.py`)
```python
class LLMClient(Protocol):
    def complete(self, messages: list[ChatMessage]) -> LLMResponse | str: ...
```
