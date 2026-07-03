---
name: agentic-search
description: Run searches, retrieval, and agent loops in the Agentic Search repo. Use when the user wants to query a local index, run a search agent trace, start a retrieval server, or exercise the training/reward pipeline.
---

# Agentic Search — Agent Tool

This repo provides local dense/sparse retrieval, multi-turn search-agent loops, web search tooling, and RL training helpers. All entry points are Python modules; no external CLI is required.

## Entry Points

### Run the full agentic search loop

```bash
# Multi-turn deep-research loop against a running retrieval server
python3 -m examples.run_agentic_search \
    --mode search \
    --question "Compare dense vs sparse retrieval" \
    --model meta-llama/Llama-3.1-8B-Instruct \
    --vllm_url http://localhost:8080 \
    --search_url http://localhost:8000/retrieve

# Single-turn generation (no search)
python3 -m examples.run_agentic_search \
    --mode single \
    --question "What is FAISS?" \
    --model Qwen/Qwen2.5-1.5B-Instruct \
    --local --device cpu --max_tokens 256

# Tool-calling loop (structured tool calls)
python3 -m examples.run_agentic_search \
    --mode tool \
    --question "What papers use BM25?" \
    --model meta-llama/Llama-3.1-8B-Instruct \
    --local --device cpu
```

| Flag | Description |
|------|-------------|
| `--mode` | `single` / `search` / `tool` |
| `--question` | Query string |
| `--model` | HuggingFace model ID |
| `--vllm_url` | OpenAI-compatible completions endpoint (vLLM, Ollama, LiteLLM) |
| `--search_url` | Running `/retrieve` endpoint URL |
| `--local` | Load model in-process (no server required) |
| `--device` | `cpu` / `cuda` (with `--local`) |
| `--max_tokens` | Max tokens to generate |

### Start a retrieval server

```bash
# Demo — TF-IDF over a local corpus (no Java, no FAISS)
python3 -m src.internal.servers.retrieval.demo --corpus_path data/corpus.jsonl

# Hybrid — RRF-fused dense E5 + sparse TF-IDF (add --no-dense for TF-IDF only)
python3 -m src.internal.servers.retrieval.hybrid --corpus_path data/corpus.jsonl

# Standalone cross-encoder reranker
python3 -m src.internal.servers.retrieval.rerank
```

### Run a deterministic search trace (no model required)

```bash
# Full XML trace: plan → search → fetch → answer
python3 -m examples.run_search_trace_workflow

# Same trace + build an SFT training example
python3 -m examples.run_search_trace_workflow --sft
```

### Build training datasets

```bash
# Search-QA parquet (NQ/FlashRAG)
python3 -m examples.prepare_search_qa_dataset \
    --dataset_name RUC-NLPIR/FlashRAG_datasets \
    --dataset_config nq \
    --local_dir data/nq_search

# RAG parquet
python3 -m examples.prepare_search_rag_dataset \
    --dataset_name RUC-NLPIR/FlashRAG_datasets \
    --local_dir data/nq_rag

# GRPO/reward smoke test
python3 -m examples.run_grpo_training_pipeline
```

## Key Python APIs

### SearchClient — async HTTP client

```python
from src.context.retrieval.client import SearchClient, SearchClientConfig

client = SearchClient(SearchClientConfig(url="http://localhost:8000/retrieve", topk=5))
results = await client.retrieve_one("What is BM25?")
# results: list[SearchResult]  — each has .text, .url, .score
```

### HybridRetriever — in-process dense + sparse fusion

```python
from src.internal.document_index.hybrid_retriever import HybridRetriever, HybridRetrieverConfig

retriever = HybridRetriever(HybridRetrieverConfig(
    dense_index_path="indexes/dense",
    sparse_index_path="indexes/sparse",
    alpha=0.5,   # 0 = sparse only, 1 = dense only
    topk=10,
))
results = retriever.retrieve("machine learning for IR")
```

### SearchAgentLoop — multi-turn research loop

```python
from src.agents.search import SearchAgentLoop
from src.agents.base import AgentLoopConfig

loop = SearchAgentLoop(AgentLoopConfig(
    model=server_manager,
    search_client=client,
    max_search_rounds=3,
))
output = await loop.run("What is retrieval-augmented generation?")
print(output.answer)
```

## Web Search (optional)

Set environment variables to enable external search:

```bash
export GOOGLE_API_KEY=...
export GOOGLE_CSE_ID=...     # Google Custom Search
export SERP_API_KEY=...      # SerpAPI (alternative)
```

Start the web search server:

```bash
python3 -m src.internal.servers.web_search.google   # Google Custom Search
python3 -m src.internal.servers.web_search.serp     # SerpAPI
```

## When to Use Each Mode

Use `--mode search` (SearchAgentLoop) when:
- The question requires iterative planning and multi-step retrieval
- You want `<think>`, `<search>`, `<information>`, `<fetch>`, `<answer>` XML traces
- Building SFT training data from agent traces

Use `--mode tool` (ToolAgentLoop) when:
- The model should emit structured JSON tool calls
- You are integrating with an OpenAPI tool registry (`ApiToolRegistry`)

Use `--mode single` when:
- No search is needed — just generation from a model
- Baselining model quality without retrieval

Use the retrieval server + `SearchClient` directly when:
- You need retrieval in your own Python code
- You want programmatic access without an agent loop

Do NOT use this skill when:
- The user is asking about general Python or ML knowledge (use your own knowledge)
- The user wants to query an external knowledge base unrelated to this repo
