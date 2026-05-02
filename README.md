# Agentic-Search

`Agentic-Search` is a small FastAPI codebase for search-backed retrieval services. It currently includes:

- a Google Custom Search server
- a SerpAPI-backed search server
- an index builder for dense and BM25 retrieval
- a dense retriever for querying FAISS indexes
- rerank and retrieval+rerank servers for local ranking pipelines

Both services support:

- batched search requests
- a simple healthcheck endpoint for local development

The Google server also supports optional page fetching to extract paragraph context from result links.

## Project Structure

```text
src/
  __init__.py
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
```

## Requirements

- Python 3.10+

API keys (required only for the corresponding server):

- Google Custom Search: a JSON API key and a Programmable Search Engine ID (`cx`)
- SerpAPI: a SerpAPI key

Python dependencies used by the server:

- `fastapi`
- `uvicorn`
- `google-api-python-client`
- `requests`
- `aiohttp`
- `beautifulsoup4`
- `chardet`
- `numpy`
- `torch`
- `transformers`
- `datasets`
- `tqdm`
- `faiss-cpu` or `faiss-gpu`
- `sentence-transformers`

Example install:

```bash
pip install fastapi uvicorn google-api-python-client requests aiohttp beautifulsoup4 chardet numpy torch transformers datasets tqdm faiss-cpu sentence-transformers
```

## Google Search Configuration

The Google server accepts configuration either from CLI flags or environment variables.

Environment variables:

```bash
export GOOGLE_API_KEY="your_google_api_key"
export GOOGLE_CSE_ID="your_custom_search_engine_id"
export GOOGLE_SEARCH_HOST="0.0.0.0"
export GOOGLE_SEARCH_PORT="8000"
```

CLI flags:

```bash
python3 -m src.search.google_search_server \
  --api_key "$GOOGLE_API_KEY" \
  --cse_id "$GOOGLE_CSE_ID" \
  --topk 3 \
  --host 0.0.0.0 \
  --port 8000
```

Optional flags:

- `--snippet_only`: return Google snippets only, without fetching result pages
- `--topk`: maximum number of documents returned per query

Start the Google API:

```bash
python3 -m src.search.google_search_server
```

## SerpAPI Configuration

Environment variables:

```bash
export SERP_API_KEY="your_serpapi_key"
export SERP_ENGINE="google"
export SERP_SEARCH_URL="https://serpapi.com/search"
export SERP_SEARCH_HOST="0.0.0.0"
export SERP_SEARCH_PORT="8000"
```

CLI flags:

```bash
python3 -m src.search.serp_search_server \
  --serp_api_key "$SERP_API_KEY" \
  --serp_engine google \
  --search_url https://serpapi.com/search \
  --topk 3 \
  --host 0.0.0.0 \
  --port 8000
```

Start the SerpAPI service:

```bash
python3 -m src.search.serp_search_server
```

## Running a Server

Once running, either server listens on `http://localhost:8000` by default.

## Building an Index

Dense index example:

```bash
python3 -m src.search.index_builder \
  --retrieval_method bge \
  --model_path BAAI/bge-base-en-v1.5 \
  --corpus_path data/corpus.jsonl \
  --save_dir indexes/
```

BM25 example:

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

## Querying a Dense Index

Example:

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
python3 -m src.search.rerank \
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

## Development Check

Syntax can be checked with:

```bash
python3 -m py_compile \
  src/__init__.py \
  src/search/__init__.py \
  src/search/search_app.py \
  src/search/google_search_server.py \
  src/search/index_builder.py \
  src/search/rerank.py \
  src/search/rerank_server.py \
  src/search/retrieval.py \
  src/search/retrieval_rerank_server.py \
  src/search/retrieval_server.py \
  src/search/serp_search_server.py
```
