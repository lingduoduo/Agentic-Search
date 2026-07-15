# Retrieval

[← Back to README](../README.md)

This guide covers retrieval services, ranking modes, reranking, and query optimization.

## Grounded answer safety

The shared `src.context` answer pipeline treats retrieval results and approved
read-only tool results as the complete evidence boundary for factual claims.
Retrieved documents are normalized as stable `D*` evidence and successful tool
results as stable `T*` evidence. The model's internal structured draft must cite
known evidence IDs for every claim; malformed drafts, unknown IDs, and claims
without sufficient lexical support fail verification. Evidence can still be
incorrect at its source—the verifier establishes support by the supplied
evidence, not independent truth.

Guarded generation allows at most one corrective retry with the original
evidence and the verifier's findings. After that retry, unsupported claims are
removed. If no supported claim remains, or usable evidence is absent, the result
is exactly `I don't know based on the available evidence.` A supported answer is
never replaced with that canonical abstention: partially supported drafts render
only their supported claims. The no-LLM extractive path also abstains instead of
selecting an unrelated first sentence.

Confidence is deterministic rather than model-reported. It combines the verified
claim fraction, evidence coverage, and an optional evidence-sufficiency signal,
is clamped to `[0.0, 1.0]`, and is `0.0` for abstention. A fully verified answer
can therefore have confidence below `1.0` when its evidence is weak. The safety
guard is enabled by default; direct Python callers can explicitly pass
`GroundedGenerationConfig(enabled=False)` to retain legacy unconstrained
generation during compatibility migration.

### Approved tool evidence

Tool evidence is opt-in. A caller supplies a registry and selector, and only
uniquely named tools explicitly classified `read_only` are offered to the
selector or invoked. Unknown, duplicate, side-effecting, and unspecified tools
are rejected. Defaults bound execution to two calls with a five-second timeout;
selection and invocation failures or timeouts degrade to retrieval-only evidence.
Tool outputs must be JSON-serializable and are normalized as data, never treated
as instructions.

`ToolRequest` arguments are intended to be JSON-like values built from standard
containers. Mappings are copied and exposed read-only, while nested mappings,
lists/tuples, and sets/frozensets are recursively converted to immutable standard
container snapshots. This does not promise deep immutability for arbitrary
user-defined objects stored inside those containers; registries and selectors
should exchange JSON-like arguments only.

Synchronous selector calls and bounded iteration run through
`asyncio.to_thread` so they do not block the event loop. `asyncio.wait_for`
limits how long the pipeline awaits them, but timing out does not stop the
underlying worker thread, which may continue running. Selectors must therefore
be trusted, independently bounded, and nonblocking; the timeout is a pipeline
latency/failure boundary, not cancellation of synchronous work.

### Result and operational metadata

Existing result fields—`answer`, `citations`, `context`, `prompt`, and
`grounding_report`—remain available. The shared result adds defaulted safety
metadata: `confidence`, `verification_status` (`verified`, `partial`, or
`abstained`), `abstained`, summarized `tool_evidence`, and `retry_count`. The MCP
chat adapter preserves its established keys and adds confidence, verification,
abstention, and tool-source summaries.

Tracing records counts and categories, tool names and statuses, retry count,
verification status, confidence, and abstention. It deliberately excludes
evidence bodies, raw tool output, full prompts, and tool arguments. Tool failures
are operational signals rather than fatal answer errors, because generation can
continue from retrieval evidence.

## Retrieval setup

`src.internal.document_index` is the single indexing entry point — filtering, chunking, embedding, retry-isolated writes, and failure reporting. Query-time retrievers and the retrieval HTTP client live in `src.context`. Reranker utilities live in `src.internal.servers.retrieval`.

Index construction is upstream and asynchronous: connectors and ingestion jobs prepare documents, then the document-index pipeline writes the searchable sparse/dense indexes. Query requests consume those existing indexes; they do not re-ingest or retrain on documents.

**Retrieval servers** (`src/internal/servers/retrieval/`):

| Module | Description |
|--------|-------------|
| `demo.py` | TF-IDF over corpus.jsonl — no Java, no FAISS |
| `hybrid.py` | RRF-fused dense (E5) + sparse TF-IDF; Java-free, FAISS-free — recommended for `AgenticRAGLoop` |
| `server.py` | Full `RetrievalService` (BM25 / dense / hybrid, env-configured via `RETRIEVAL_BACKEND`) with per-mode + admin endpoints |
| `rerank.py` | Standalone cross-encoder reranker (no retrieval) |

**Web search servers** (`src/internal/servers/web_search/`):

| Module | Description |
|--------|-------------|
| `google.py` | Google Custom Search proxy |
| `serp.py` | SerpAPI proxy |
| `browser.py` | playwright-cli browser automation; no API key, ~5–10s/query |

**Start a retrieval server:**

```bash
# Demo — TF-IDF, no Java/FAISS
python3 -m src.internal.servers.retrieval.demo --corpus_path data/corpus.jsonl

# Hybrid — RRF-fused dense E5 + sparse TF-IDF (add --no-dense for TF-IDF only)
python3 -m src.internal.servers.retrieval.hybrid --corpus_path data/corpus.jsonl
```

**Build indexes:**

```bash
python3 -m src.internal.document_index.index_builder \
  --retrieval_method e5 --model_path intfloat/e5-base-v2 \
  --corpus_path data/corpus.jsonl --faiss_type Flat --save_dir data/indexes/

python3 -m src.internal.document_index.index_builder \
  --retrieval_method bm25 --corpus_path data/corpus.jsonl --save_dir data/indexes/
```

**Web search servers:**

```bash
python3 -m src.internal.servers.web_search.serp \
  --search_url "https://serpapi.com/search" --topk 3 --serp_api_key "$SERP_API_KEY"

python3 -m src.internal.servers.web_search.google \
  --api_key "$GOOGLE_API_KEY" --topk 5 --cse_id "$GOOGLE_CSE_ID" --snippet_only
```

**Health check:**

```bash
curl -i -sS http://127.0.0.1:8001/health
curl -i -sS -X POST http://127.0.0.1:8001/retrieve \
  -H "Content-Type: application/json" -d '{"query":"What is FAISS?","topk":5}'
```

For complete request and response payloads, see the [HTTP API reference](api-reference.md).

## Retrieval in auto-routed API requests

The web API has an evidence-first search path above the retrieval services. For an unfiltered `/api/agent` request that routes to `search` with `source_provider=auto`, the backend:

1. queries internal retrieval;
2. accepts an exact-title, fuzzy-plus-semantic, or semantic match that clears the direct sufficiency gate;
3. otherwise tries SerpAPI with the original query;
4. otherwise calls the configured browser-search HTTP service;
5. returns a deterministic no-results or sources-unreachable response when no evidence exists.

This sequence is different from explicit `mode=hybrid_search`, whose helper can query internal retrieval in parallel with a cascading web leg and then merge/rerank results. It is also different from the internal retrieval router described below.

Authenticated requests carry document-access filters and use the filter-aware pipeline rather than the unfiltered direct-first shortcut. Internal retrieval receives the ACL filters; external web providers do not receive internal document ACL objects. See [API request routing](request-routing.md) for exact modes, metadata, and fallbacks.

The filter-aware path uses the same internal stage sequence throughout the web backend:

1. bounded session history resolves continuation-style queries into a retrieval query while retaining the original user question;
2. the selected existing provider returns a normalized candidate set and receives ACL filters when it is internal retrieval;
3. one ranking stage deduplicates candidates, optionally invokes the existing reranker, and applies MMR/truncation;
4. inference synthesizes from ranked evidence, or the pipeline returns deterministic status/results when evidence or synthesis is unavailable;
5. shared response finalization persists citations, documents, and stage metadata.

These stages are internal adapters. Existing `/retrieve`, `/search`, and `/rerank` endpoints remain available with their current payloads, and no new retrieval API was added. Backend RRF inside `RetrievalService` remains distinct from web-layer candidate ranking: RRF fuses backend result lists; the web ranking stage normalizes, deduplicates, optionally reranks, and diversifies the resulting evidence.

## Neural reranking

`RetrievalService` optionally reranks hybrid-fused results via a layered wrapper chain. Set `RERANKER_PROVIDER` to enable; all wrappers are opt-in via env vars and compose on top of the unchanged `Reranker` leaf.

**Wrapper chain** (outermost → innermost):
```
TwoStageReranker → CachedReranker → AsyncReranker → Reranker (leaf)
```

**Enable local BGE reranking:**
```bash
RERANKER_PROVIDER=local RERANKER_MODEL=BAAI/bge-reranker-v2-m3 \
  PYTHONPATH=src:. uvicorn src.internal.servers.web.app:app --host 127.0.0.1 --port 7860
```

**Enable Cohere reranking:**
```bash
RERANKER_PROVIDER=cohere RERANKER_MODEL=rerank-english-v3.0 COHERE_API_KEY=... \
  PYTHONPATH=src:. uvicorn src.internal.servers.web.app:app --host 127.0.0.1 --port 7860
```

**Enable async + Redis cache wrapper:**
```bash
RERANKER_PROVIDER=local RERANKER_ASYNC=true \
  RERANKER_TIMEOUT_MS=500 RERANKER_CACHE_REDIS_URL=redis://localhost:6379 \
  PYTHONPATH=src:. uvicorn src.internal.servers.web.app:app --host 127.0.0.1 --port 7860
```

**Enable two-stage pipeline** (fast pre-filter → heavy scorer):
```bash
RERANKER_PROVIDER=local RERANKER_TWO_STAGE=true \
  RERANKER_FAST_MODEL=BAAI/bge-reranker-base \
  RERANKER_PRE_FILTER_TOP_N=50 RERANKER_OVER_FETCH_MULTIPLIER=2.0 \
  PYTHONPATH=src:. uvicorn src.internal.servers.web.app:app --host 127.0.0.1 --port 7860
```

**ONNX runtime** (lower latency than PyTorch, requires `pip install optimum[onnxruntime]`):
```bash
RERANKER_PROVIDER=local RERANKER_USE_ONNX=true RERANKER_MODEL=BAAI/bge-reranker-base \
  PYTHONPATH=src:. uvicorn src.internal.servers.web.app:app --host 127.0.0.1 --port 7860
```

**Evaluate reranker quality and latency:**
```bash
# Baseline vs reranked NDCG/MRR + per-query latency
python -m src.internal.retrieval.eval_runner \
  --dataset data/eval/qa_pairs.jsonl --top_k 10 \
  --reranker local --reranker_model BAAI/bge-reranker-v2-m3 \
  --compare-baseline --slo-ms 200

# Output JSON:
# { "retrieval":  {"ndcg@10": 0.48, "mrr": 0.63},
#   "reranked":   {"ndcg@10": 0.55, "mrr": 0.71, "map@10": 0.52},
#   "latency_ms": {"mean": 312, "p50": 290, "p99": 680, "n": 50},
#   "reranker_improvement_ratio": 0.145 }
```

**Benchmark model configurations offline:**
```bash
python -m src.internal.retrieval.reranker_benchmark \
  --qa-pairs data/eval/qa_pairs.jsonl \
  --models BAAI/bge-reranker-base BAAI/bge-reranker-v2-m3 \
  --batch-sizes 8 16 32 \
  --max-tokens 256 512 \
  --output results/reranker_bench.jsonl
# Prints ranked table sorted by NDCG@10
```

## Retrieval optimization

All optimization components are opt-in; unset env vars = unchanged M1–M4 behavior.

**Tune BM25 parameters against your QA pairs:**
```bash
curl -s -X POST http://localhost:8001/internal/optimize/bm25-tune \
  -H "Content-Type: application/json" \
  -d '{"qa_pairs_path": "data/eval/qa_pairs.jsonl", "k1_range": [0.6, 0.9, 1.2], "b_range": [0.5, 0.75]}' \
  -H "Authorization: Bearer $TOKEN"
# → {"k1": 0.9, "b": 0.6, "score": 0.86}
```

**Learn fusion weights (sparse vs dense RRF weights):**
```bash
curl -s -X POST http://localhost:8001/internal/optimize/fusion-weights \
  -H "Content-Type: application/json" \
  -d '{"qa_pairs_path": "data/eval/qa_pairs.jsonl"}' \
  -H "Authorization: Bearer $TOKEN"
# → {"w_sparse": 0.38, "w_dense": 0.62}
```

**Tune HNSW ef_search for a recall target:**
```bash
curl -s -X POST http://localhost:8001/internal/optimize/hnsw-tune \
  -H "Content-Type: application/json" \
  -d '{"target_recall": 0.82}' \
  -H "Authorization: Bearer $TOKEN"
# → {"ef_search": 96, "measured_recall": 0.831}
```

**Retrieval stats (cache hit rate, latency, throughput):**
```bash
curl -s http://localhost:7860/api/admin/retrieval/stats \
  -H "Authorization: Bearer $TOKEN"
# → {"result_cache_hit_rate": 0.42, "p99_latency_ms": 112, "throughput_qps": 87, ...}
```

**Hot-reload tunable parameters without restart:**
```bash
curl -s -X PATCH http://localhost:7860/api/admin/retrieval/config \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $TOKEN" \
  -d '{"rrf_k": 80, "mmr_lambda": 0.4, "nprobe": 96, "result_cache_ttl": 600}'
# → {"applied": ["rrf_k", "mmr_lambda", "nprobe", "result_cache_ttl"]}
```

**Enable query expansion and result caching:**
```bash
QUERY_EXPANSION_ENABLED=true SPELL_CORRECTION_ENABLED=true EXPANSION_MAX_TERMS=3 \
  BM25_VARIANT=bm25plus \
  RESULT_CACHE_REDIS_URL=redis://localhost:6379 RESULT_CACHE_TTL=300 \
  ADAPTIVE_MMR=true \
  PYTHONPATH=src:. uvicorn src.internal.servers.web.app:app --host 127.0.0.1 --port 7860
```

**Build an IVF-PQ FAISS index** (cuts memory from ~30 GB to ≤ 4 GB at 10 M docs):
```python
from src.internal.retrieval.index_optimizer import FAISSIndexBuilder
import numpy as np

builder = FAISSIndexBuilder()
index = builder.build_ivfpq(embeddings, nlist=4096, m=96, nbits=8, nprobe=64)
# Save alongside existing index; load via FAISS_INDEX_TYPE=ivfpq
```

## Query transformation optimization

A layered-wrapper optimization stack over `QueryTransformPipeline`, parallel to Neural Reranking. Every layer is opt-in; with all `QT_*` unset, `RetrievalService` runs the single-query path unchanged (`build_query_transform_pipeline_from_env` returns `None`).

**Wrapper chain** (outermost → innermost):
```
RoutedQueryTransformPipeline → CachedQueryTransformPipeline → AsyncQueryTransformPipeline → QueryTransformPipeline (leaf)
```

**Enable parallel transforms + Redis bundle cache:**
```bash
QT_DECOMPOSE=true QT_HYDE=true QT_STEP_BACK=true \
  QT_ASYNC=true QT_TRANSFORM_TIMEOUT_MS=400 \
  QT_CACHE_REDIS_URL=redis://localhost:6379 QT_CACHE_TTL_SECONDS=600 \
  PYTHONPATH=src:. uvicorn src.internal.servers.web.app:app --host 127.0.0.1 --port 7860
```

**Enable Multi-Query + weighted RAG-Fusion:**
```bash
QT_MULTI_QUERY=true QT_MULTI_QUERY_N=3 QT_FUSION_WEIGHTED=true \
  PYTHONPATH=src:. uvicorn src.internal.servers.web.app:app --host 127.0.0.1 --port 7860
```

**Enable per-query learned routing** (heuristic until an artifact exists):
```bash
QT_ROUTER=true QT_ROUTER_MODEL_PATH=data/query_router.joblib \
  PYTHONPATH=src:. uvicorn src.internal.servers.web.app:app --host 127.0.0.1 --port 7860
```
`QT_ROUTER` and `QT_MULTI_QUERY` each activate the pipeline on their own — no other `QT_*` flag is required.

Query transformation is **backend-only** — there is no dedicated HTTP endpoint and no query-transform-specific UI. The pipeline runs inside `RetrievalService.from_env()`, so it applies to **both** the retrieval server's `/search` and the web backend's `/api/agent`. Its observable effect is the `+rag_fusion` suffix on `retrieval_mode`.

**Test it on the retrieval server** (`POST /search` — `retrieval_mode` reflects the transform):
```bash
# Start the retrieval server with QT flags enabled, then:
curl -s -X POST http://localhost:8001/search \
  -H "Content-Type: application/json" \
  -d '{"query": "Compare dense and sparse retrieval", "top_k": 5}' \
  | python -c "import sys, json; print(json.load(sys.stdin)['retrieval_mode'])"
# → hybrid+rag_fusion
```

**Test it on the web backend** (`POST /api/agent`):
```bash
curl -s -X POST http://localhost:7860/api/agent \
  -H "Content-Type: application/json" \
  -d '{"query": "Compare dense and sparse retrieval", "mode": "chat_loop", "top_k": 5}' \
  | python -m json.tool | grep -i retrieval_mode
# → "retrieval_mode": "hybrid+rag_fusion"   (or "hybrid+rag_fusion+reranked" with a reranker)
```

**Extract metadata filters from natural language** (numeric operators behind `QT_CONSTRUCT_OPERATORS`):
```bash
QT_CONSTRUCT_FILTERS=true QT_CONSTRUCT_OPERATORS=true \
  PYTHONPATH=src:. uvicorn src.internal.servers.web.app:app --host 127.0.0.1 --port 7860
# "arxiv papers after 2023 rated above 4" → filters {date_after: "2023-...", rating_gte: 4}
curl -s -X POST http://localhost:7860/api/agent \
  -H "Content-Type: application/json" \
  -d '{"query": "arxiv papers after 2023 rated above 4 on retrieval", "mode": "chat_loop", "top_k": 5}'
```

**Train the learned router offline:**
```bash
python -m src.training.train_query_router --out data/query_router.joblib
# → wrote data/query_router.joblib
# Predicts 7 transform labels: decompose, hyde, step_back, keywords, construct_filters, multi_query, rewrite
```

**Gate transform latency in CI:**
```bash
python -m src.internal.retrieval.eval_runner \
  --dataset data/eval/qa_pairs.jsonl --top_k 10 --qt-slo-ms 300
# Records per-query "qt_latency_ms"; exits non-zero when P99 transform latency > 300ms
```

**Benchmark technique combinations offline** (Python API; the `--dataset` CLI ships a stub `retrieve_fn` to wire to your retriever):
```python
from src.context.query_transform import QueryTransformConfig
from src.internal.retrieval.query_transform_benchmark import run_query_transform_benchmark

dataset = [("what is FAISS", {"doc-1"}), ("compare BM25 and dense", {"doc-2"})]

def retrieve(query, config):
    # build a pipeline from `config`, run RetrievalService.search, return ranked doc_ids
    ...

rows = run_query_transform_benchmark(dataset, retrieve, [
    QueryTransformConfig(),
    QueryTransformConfig(multi_query=True),
    QueryTransformConfig(decompose=True, hyde=True),
], k=10)
# → [{"config_signature": "...", "recall": 0.91, "ndcg": 0.78, "mean_latency_ms": 142.0}, ...]
```

## Routing and query construction

The RAG **Routing → Query Construction** stage (`src/internal/routing/`). It decides **where** a query should go (domain → source → retriever) and **how** to express it for the chosen backend. Distinct from [Intent Routing](architecture.md#intent-routing) (web-level `search`/`chat`/`tool`) and from `QueryRouter` (which picks *transforms*): this layer picks the *retriever/construction target* per query.

**Backend-only and default-off.** With no `ROUTING_*` env set, `build_router_from_env()` returns `None`, `RetrievalService.search` skips the routing branch entirely, and behavior is byte-identical to today — zero overhead, no frontend change. There is no dedicated HTTP endpoint or UI; routing runs inside `RetrievalService.from_env()`.

**Pipeline:**
```
query → Router.route() → RouteDecision(domain, sources, retriever, construction_target)
      → QueryConstructor.construct() → ConstructedQuery(target, payload, text)
```

**Router strategies** (heuristic default; LLM strategies fall back to it on any failure):

| Strategy | Env | How it routes |
|----------|-----|---------------|
| Heuristic | (default) | Rule-based cue matching → SQL / GRAPH / API / default HYBRID. No LLM; the path the accuracy gate runs against |
| Logical | `ROUTING_LOGICAL=true` | LLM structured-classification into a registered route by name |
| Semantic | `ROUTING_SEMANTIC=true` | Embedding cosine between the query and each route's description |

Routes come from a config-driven registry (`ROUTING_REGISTRY_PATH` → JSON of `{name, description, sources, retriever}`; a built-in default mirrors the local corpus). `RetrieverTarget` ∈ `sparse · dense · hybrid · metadata · sql · graph · api`.

**Six query constructors** (`construction/`, one `construct(query, route) -> ConstructedQuery` interface):

| Constructor | Target | Backing | Output |
|-------------|--------|---------|--------|
| Metadata Filter | `metadata` | wraps `QueryConstructor` | NL → `{filters}` + cleaned query |
| Vector Search | `dense` | params | `{top_k, namespace, filters}` |
| Hybrid Retrieval | `hybrid` | reuses `adaptive_mmr_lambda` | `{rrf_k, w_sparse, w_dense, mmr_lambda}` |
| SQL Generation | `sql` | net-new (no exec) | schema-aware Text-to-SQL, SELECT-only + table allowlist + multi-statement reject |
| Knowledge Graph | `graph` | net-new (no exec) | read-only Cypher (`MATCH…RETURN`), word-boundary write-clause rejection |
| API Request | `api` | net-new (no exec) | `{endpoint, params}` filtered to an `ApiSpec` allowlist |

The three net-new constructors **build and validate but never execute** a query — there is no live SQL/KG/API backend, so `RetrievalService` short-circuits the `sql`/`graph`/`api` targets to `([], "routed:<target>")`. When a real backend is wired later, only the executor changes. Every `route()`/`construct()` degrades to a safe empty/None payload rather than raising.

**Enable per-query routing:**
```bash
ROUTING_ENABLED=true \
  PYTHONPATH=src:. uvicorn src.internal.servers.web.app:app --host 127.0.0.1 --port 7860
# Optional LLM strategies + a custom route registry:
ROUTING_ENABLED=true ROUTING_LOGICAL=true ROUTING_SEMANTIC=true \
  ROUTING_REGISTRY_PATH=data/routes.json  uvicorn ...
```

**Score routing accuracy** (heuristic router; no LLM needed):
```bash
python -m src.internal.retrieval.eval_runner \
  --routing-eval --dataset data/eval/routing_labels.jsonl
# → {"routing_accuracy": 1.0, "num_queries": 12}
```
