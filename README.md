# Agentic-Search

`Agentic-Search` is a small FastAPI codebase for search-backed retrieval services. It currently includes:

- a Google Custom Search server
- a SerpAPI-backed search server
- an index builder for dense and BM25 retrieval
- a dense retriever for querying FAISS indexes
- rerank and retrieval+rerank servers for local ranking pipelines
- an `llm_agent` package for multi-turn generation and agentic search workflows

Both services support:

- batched search requests
- a simple healthcheck endpoint for local development

The Google server also supports optional page fetching to extract paragraph context from result links.

## Project Structure

```text
src/
  __init__.py
  agent_loop/
    __init__.py
    agent_loop.py          # AgentLoopBase, AgentLoopConfig, AgentLoopOutput
    context.py             # SearchResult, SearchContext, AgentContext
    search_agent_loop.py   # SearchAgentLoop (multi-turn, registered as "search_agent")
    search_client.py       # SearchClient — async aiohttp client for /retrieve endpoints
    single_turn_agent_loop.py
    tool.py                # Tool, FunctionTool — tool abstraction and JSON schema
    tool_agent_loop.py     # ToolAgentLoop (multi-turn tool use, registered as "tool_agent")
    tool_parser.py         # ToolParser — Hermes / Llama3 / JSON tool-call parsers
  llm_agent/
    __init__.py
    generation.py
    tensor_helper.py
  search/
    __init__.py
    search_app.py
    google_search_server.py
    index_builder.py
    rerank.py
    retrieval.py
    retrieval_rerank_server.py
    retrieval_server.py
    serp_search_server.py
    vocabulary.py
tests/
  conftest.py
  unit/
    test_agent_loop.py
    test_index_builder.py
    test_llm_agent_generation.py
    test_llm_agent_tensor_helper.py
    test_rerank.py
    test_search_app.py
    test_vocabulary.py
  regression/
    test_regression.py
  load/
    test_load.py
```

## Requirements

- Python 3.10+

API keys (required only for the corresponding server):

- Google Custom Search: a JSON API key and a Programmable Search Engine ID (`cx`)
- SerpAPI: a SerpAPI key

Install all dependencies:

```bash
pip install -r requirements.txt
```

Or manually:

```bash
pip install fastapi uvicorn google-api-python-client requests aiohttp beautifulsoup4 chardet numpy torch transformers datasets tqdm faiss-cpu sentence-transformers python-dotenv pyserini
```

## Environment Variables

Both search servers load a `.env` file automatically at startup via `python-dotenv`. Copy `.env.example` to `.env` and fill in your keys — no `export` commands needed.

```bash
cp .env.example .env
```

`.env` (never commit this file — it is already in `.gitignore`):

```
GOOGLE_API_KEY=your_google_api_key
GOOGLE_CSE_ID=your_custom_search_engine_id
SERP_API_KEY=your_serpapi_key
```

All variables can also be passed as CLI flags or set as shell environment variables — whichever takes precedence in your workflow.

Full list of supported variables (with defaults) is in [`.env.example`](.env.example).

### Java (BM25 only)

BM25 indexing uses [pyserini](https://github.com/castorini/pyserini), which wraps Apache Lucene and requires a **Java 11+ JDK**. On Apple Silicon the JDK must be the **arm64** build — the x86_64 Corretto build will not work.

Install an arm64 JDK via Homebrew:

```bash
brew install openjdk
```

Then add `JAVA_HOME` to your `.env`:

```
JAVA_HOME=/opt/homebrew/opt/openjdk/libexec/openjdk.jdk/Contents/Home
```

The index builder loads `.env` automatically at startup, so no shell `export` is needed. Dense indexing and retrieval do not require Java.

## Google Search Configuration

With keys in `.env`, no flags are needed:

```bash
python3 -m src.search.google_search_server
```

Optional flags:

- `--topk N`: maximum number of documents returned per query (default: 3)
- `--snippet_only`: return Google snippets only, without fetching result pages
- `--host` / `--port`: bind address (defaults: `0.0.0.0:8000`)

> **Note:** do not pass `--api_key "$GOOGLE_API_KEY"` from the shell — if the variable is not already exported the shell expands it to an empty string before Python starts, which overrides the `.env` value. Let the server read the key from `.env` directly.

If you need to override a key at the command line, pass the literal value:

```bash
python3 -m src.search.google_search_server --api_key "AIza..." --cse_id "305b..."
```

## SerpAPI Configuration

With keys in `.env`, no flags are needed:

```bash
python3 -m src.search.serp_search_server
```

Optional flags mirror the Google server (`--topk`, `--host`, `--port`) plus `--serp_engine` and `--search_url`.

If you need to override a key at the command line, pass the literal value:

```bash
python3 -m src.search.serp_search_server --serp_api_key "your_key_here"
```

## Running a Server

Once running, either server listens on `http://localhost:8000` by default.

## Building an Index

Build the dense FAISS index first — the retrieval command requires a pre-built index and will not build one on the fly.

Dense index (downloads the encoder model on first run, then uses the local cache):

```bash
python3 -m src.search.index_builder \
  --retrieval_method bge \
  --model_path BAAI/bge-base-en-v1.5 \
  --corpus_path data/corpus.jsonl \
  --save_dir indexes/
```

BM25 index (requires Java — see [Java (BM25 only)](#java-bm25-only) above):

```bash
python3 -m src.search.index_builder \
  --retrieval_method bm25 \
  --corpus_path data/corpus.jsonl \
  --save_dir indexes/
```

Notes:

- dense indexing needs `model_path` unless you supply `--embedding_path`
- GPU is used automatically when available for embedding generation
- `--faiss_gpu` requires GPU-enabled FAISS support
- both commands write `indexes/vocabulary_corpus.json` alongside the index by default
- `--bm25_threads N` controls how many Lucene indexing threads are used (default: all available CPUs)

### Vocabulary Metadata

The index builder writes a combined `vocabulary_corpus.json` artifact that stores both corpus-level vocabulary statistics and per-document token metadata in one file.

Top-level fields include:

- `corpus_path`
- `retrieval_method`
- `keyword_limit`
- `vocab_max_length`
- `vocabulary`
- `corpus`

The `vocabulary` section contains:

- `num_token`
- `token2idx`
- `token2cnt`
- `idx2token`

Each item in `corpus` contains:

- `doc_id`
- `id`
- `title`
- `contents`
- `tokens`
- `keywords`
- `token_count`

Disable this sidecar artifact with:

```bash
python3 -m src.search.index_builder \
  --retrieval_method bm25 \
  --corpus_path data/corpus.jsonl \
  --save_dir indexes/ \
  --no_save_vocabulary
```

## Querying a Dense Index

```bash
python3 -m src.search.retrieval \
  --model_path BAAI/bge-base-en-v1.5 \
  --index_path indexes/bge_Flat.index \
  --corpus_path data/corpus.jsonl \
  --retrieval_method bge \
  --queries "What is agentic search?" "How does FAISS work?" \
  --topk 5
```

The command prints JSON results containing the matched corpus entries and similarity scores.

> **Model loading:** the encoder is loaded from the local HuggingFace cache on every run. The first run downloads the model (~438 MB for `bge-base-en-v1.5`); subsequent runs start in a few seconds with no network traffic.

## Running a Dense Retrieval Server

Example:

```bash
python3 -m src.search.retrieval_server \
  --model_path BAAI/bge-base-en-v1.5 \
  --index_path indexes/bge_Flat.index \
  --corpus_path data/corpus.jsonl \
  --retrieval_method bge \
  --topk 5
```

Then query it with:

```bash
curl -X POST http://localhost:8000/retrieve \
  -H "Content-Type: application/json" \
  -d '{
    "queries": ["What is agentic search?"],
    "topk": 3,
    "return_scores": true
  }'
```

## Running a Rerank Server

The rerank server listens on port **6980** by default.

```bash
python3 -m src.search.rerank_server \
  --rerank_model_name_or_path cross-encoder/ms-marco-MiniLM-L12-v2 \
  --rerank_topk 3
```

Query it with:

```bash
curl -X POST http://localhost:6980/rerank \
  -H "Content-Type: application/json" \
  -d '{
    "queries": ["What is agentic search?"],
    "documents": [[{"document": {"contents": "\"Example\"\nSome text."}}]],
    "rerank_topk": 3,
    "return_scores": true
  }'
```

When `return_scores` is `false` (the default), each document is returned as a plain string instead of a scored object.

## Running a Retrieval + Rerank Server

Example:

```bash
python3 -m src.search.retrieval_rerank_server \
  --retriever_model BAAI/bge-base-en-v1.5 \
  --index_path indexes/bge_Flat.index \
  --corpus_path data/corpus.jsonl \
  --retrieval_method bge \
  --retrieval_topk 10 \
  --rerank_topk 3
```

## Agentic Search Loop

The `src.agent_loop` package provides a multi-turn agent loop that issues `<search>` queries, calls a retrieval server, and injects results back into the conversation as `<information>` blocks.

### Registered loops

| Name | Class | Description |
|------|-------|-------------|
| `"single_turn_agent"` | `SingleTurnAgentLoop` | One generation step, no search |
| `"search_agent"` | `SearchAgentLoop` | Multi-turn with `<search>`/`<answer>` protocol |
| `"tool_agent"` | `ToolAgentLoop` | Multi-turn with parallel tool execution |

### Usage

```python
from src.agent_loop import SearchAgentLoop, SearchAgentLoopConfig

loop = SearchAgentLoop(
    tokenizer=tokenizer,
    server_manager=server_manager,
    search_config=SearchAgentLoopConfig(
        search_url="http://localhost:8000/retrieve",
        topk=5,
        max_turns=5,
    ),
)
output = await loop.run(
    messages=[{"role": "user", "content": "What is FAISS?"}],
    sampling_params={"temperature": 0.7},
)
# output.context.turns      — list of SearchContext (query + results per turn)
# output.context.num_searches  — total searches issued
```

### Tool agent usage

```python
from src.agent_loop import ToolAgentLoop, ToolAgentLoopConfig
from src.agent_loop.tool import FunctionTool

@FunctionTool.from_fn(
    description="Search the web",
    parameters={"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]},
)
async def web_search(query: str) -> str:
    ...

loop = ToolAgentLoop(
    tokenizer=tokenizer,
    server_manager=server_manager,
    tools=[web_search],
    config=ToolAgentLoopConfig(
        tool_parser_format="hermes",  # or "llama3" / "json"
        max_assistant_turns=10,
        max_parallel_calls=4,
    ),
)
output = await loop.run(
    messages=[{"role": "user", "content": "What is FAISS?"}],
    sampling_params={"temperature": 0.7},
)
# output.response_mask — 1 for model tokens, 0 for injected tool-response tokens
```

Supported `tool_parser_format` values:

| Format | Model family |
|--------|-------------|
| `"hermes"` | NousResearch Hermes 2.5 / 3 |
| `"llama3"` | Meta Llama 3.1 / 3.2 |
| `"json"` | Generic fallback (best-effort) |

### Context objects

- `SearchResult(contents, score)` — one passage from the server
- `SearchContext(query, results)` — one search round; `.to_information_block()` formats results for injection
- `AgentContext(turns)` — full run history; attached to `AgentLoopOutput.context`

### Lookup by name

```python
from src.agent_loop import get_registered_agent_loop, list_registered_agent_loops

print(list_registered_agent_loops())          # ["search_agent", "single_turn_agent"]
cls = get_registered_agent_loop("search_agent")
```

## LLM Agent Utilities

The `src.llm_agent` package provides reusable helpers for multi-turn LLM generation loops that interleave model responses with search observations.

Main modules:

- `src.llm_agent.generation`: `LLMGenerationManager` — multi-turn loop with action parsing, batched search dispatch, and search simulation helpers. Supported `search_mode` values: `google`, `wiki`, `local`, `simulate_sft`, `simulate_prompt`
- `src.llm_agent.tensor_helper`: padding, trimming, attention-mask, and position-id utilities for batched tensor handling

Import example:

```python
from src.llm_agent.generation import GenerationConfig, LLMGenerationManager
from src.llm_agent.tensor_helper import TensorConfig, TensorHelper
```

The `local` search mode calls the repo's own `retrieval_server` — set `retrieval_url` in `GenerationConfig`:

```python
config = GenerationConfig(
    ...,
    search_mode="local",
    retrieval_url="http://localhost:8000/retrieve",
)
```

Notes:

- optional runtime dependencies (`openai`, `serpapi`) are loaded lazily
- `SERP_API_KEY` is read from the environment for the `google` search mode

## Shared API

### `GET /health`

Healthcheck endpoint:

```bash
curl http://localhost:8000/health
```

Example response:

```json
{"status":"ok"}
```

### `POST /retrieve`

Submit one or more queries to either service:

```bash
curl -X POST http://localhost:8000/retrieve \
  -H "Content-Type: application/json" \
  -d '{
    "queries": [
      "OpenAI latest API docs",
      "FastAPI tutorial"
    ]
  }'
```

Example response shape:

```json
{
  "result": [
    [
      {
        "document": {
          "contents": "\"Example title\"\nRelevant snippet or extracted paragraph."
        }
      }
    ],
    [
      {
        "document": {
          "contents": "\"Another title\"\nRelevant snippet or extracted paragraph."
        }
      }
    ]
  ]
}
```

## How Google Search Works

For each query, the service:

1. sanitizes the search string
2. calls Google Custom Search
3. collects top result links
4. optionally fetches HTML pages concurrently
5. extracts paragraph text and returns short retrieval contexts

If `--snippet_only` is enabled, the service skips page fetching and returns cleaned snippets directly.

## How SerpAPI Search Works

For each query, the service:

1. calls SerpAPI
2. extracts the answer box when present
3. adds organic results
4. fills remaining slots with related questions
5. returns up to `topk` formatted documents

## Notes

- Google Custom Search usage is subject to Google API quotas and billing rules.
- SerpAPI usage is subject to SerpAPI quotas and billing rules.
- Some result pages may block scraping or return little usable paragraph text.
- Empty or fully invalid queries return empty result lists.

## Testing

Install test dependencies:

```bash
pip install pytest httpx
```

### Run all tests

```bash
python3 -m pytest
```

### Unit tests

Pure-logic tests that require no running server or ML models:

```bash
python3 -m pytest tests/unit/ -v
```

Coverage:

| File | What is tested |
|------|---------------|
| `test_vocabulary.py` | `Vocabulary`, `normalize_text`, `tokenize_text`, `build_vocabulary_from_sequences`, `extract_keywords` |
| `test_index_builder.py` | `IndexBuilderConfig.validate`, `prepare_texts`, `resolve_pooling_method`, `pooling` (skipped when torch unavailable) |
| `test_llm_agent_generation.py` | action parsing, search payload alignment, inactive-example handling, unknown search-mode fallback |
| `test_llm_agent_tensor_helper.py` | padding conversion and example-level batch re-expansion |
| `test_rerank.py` | `passage_to_string`, `string_to_document`, `RerankerConfig.validate`, `SentenceTransformerReranker.rerank` |
| `test_search_app.py` | `format_document`, `/health` endpoint, `/retrieve` endpoint |

### Regression tests

Snapshot tests that pin exact outputs for known inputs, catching unintended behaviour changes:

```bash
python3 -m pytest tests/regression/ -v
```

### Load tests

Concurrent-request tests that verify the FastAPI endpoints handle parallelism correctly and meet basic latency/throughput bounds:

```bash
python3 -m pytest tests/load/ -v -s -m load
```

The `-s` flag lets latency percentiles (p50/p95/p99) print to stdout.

### Development syntax check

```bash
python3 -m py_compile \
  src/__init__.py \
  src/agent_loop/__init__.py \
  src/agent_loop/agent_loop.py \
  src/agent_loop/context.py \
  src/agent_loop/search_agent_loop.py \
  src/agent_loop/search_client.py \
  src/agent_loop/single_turn_agent_loop.py \
  src/llm_agent/__init__.py \
  src/llm_agent/generation.py \
  src/llm_agent/tensor_helper.py \
  src/search/__init__.py \
  src/search/search_app.py \
  src/search/google_search_server.py \
  src/search/index_builder.py \
  src/search/rerank.py \
  src/search/rerank_server.py \
  src/search/retrieval.py \
  src/search/retrieval_rerank_server.py \
  src/search/retrieval_server.py \
  src/search/serp_search_server.py \
  src/search/vocabulary.py
```
