# Agentic Search

Agentic Search is a compact playground for retrieval-backed agents and
search-policy training. It includes local dense and sparse retrieval, web search
tooling, multi-turn XML search traces, SFT example building, and PPO/GRPO-style
reward helpers.

## What Is Here

| Area | Main modules |
|------|--------------|
| Agent loops | `src/agents/` |
| Retrieval and search servers | `src/retrieval/` |
| Tool schemas and search tools | `src/tools/` |
| Model generation and intent routing | `src/model/` |
| SFT, rewards, PPO, and GRPO helpers | `src/training/` |
| Runnable examples | `examples/` |

`src/search/` is a compatibility layer for older imports. New code should use
`src/retrieval/` and `src/retrieval/servers/` directly.

## Features

- Local dense retrieval with FAISS-compatible indexes.
- Local sparse retrieval with BM25/Pyserini.
- Optional web search through Google Custom Search, Bing, Brave, and SerpAPI.
- Multi-turn `SearchAgentLoop` traces with `<think>`, `<search>`,
  `<information>`, `<fetch>`, and `<answer>` actions.
- One-shot generation, one-shot RAG, full search-agent, and generic tool-agent
  loops.
- Reward shaping and group-relative advantage helpers for PPO, GRPO, and
  REINFORCE-style experiments.
- Intent classifier utilities for query routing and model routing.

## Install

Requires Python 3.10+.

```bash
pip install -r requirements.txt
```

For BM25, Java must be available. On Apple Silicon, FAISS is usually more stable
when installed through conda:

```bash
conda install -c conda-forge faiss-cpu
```

Optional environment variables:

```bash
GOOGLE_API_KEY=...
GOOGLE_CSE_ID=...
BING_SEARCH_API_KEY=...
BRAVE_SEARCH_API_KEY=...
SERP_API_KEY=...
JAVA_HOME=/path/to/java
```

## Quick Start

Run a deterministic search-agent trace with fake model/search backends:

```bash
python3 -m examples.run_search_trace_workflow
```

Build the matching full-trace SFT example:

```bash
python3 -m examples.run_search_trace_workflow --sft
```

Run the reward and GRPO helper smoke test:

```bash
python3 -m examples.run_grpo_training_pipeline
```

Prepare NQ/FlashRAG-style question-answer pairs for search-agent training:

```bash
python3 -m examples.prepare_search_qa_dataset \
  --dataset_name RUC-NLPIR/FlashRAG_datasets \
  --dataset_config nq \
  --local_dir data/nq_search
```

Inspect converted question/answer pairs before writing parquet:

```bash
python3 -m examples.prepare_search_qa_dataset \
  --dataset_name RUC-NLPIR/FlashRAG_datasets \
  --dataset_config nq \
  --splits test \
  --max_examples 20 \
  --preview \
  --preview_rows 5
```

If this command fails with a `pyarrow` extension error, refresh dependencies
with `python -m pip install -r requirements.txt`; this repo pins a
`datasets`-compatible `pyarrow` range.

Create a tiny local parquet slice for a dry run:

```bash
python3 -m examples.prepare_search_qa_dataset \
  --dataset_name RUC-NLPIR/FlashRAG_datasets \
  --dataset_config nq \
  --splits test \
  --max_examples 100 \
  --local_dir data/nq_search_debug
```

Prepare RAG-style NQ records from cached retrieval results:

```bash
python3 -m examples.prepare_search_rag_dataset \
  --dataset_name RUC-NLPIR/FlashRAG_datasets \
  --dataset_config nq \
  --corpus_path data/wiki-18.jsonl \
  --train_retrieval_cache data/nq_train_retrieval_cache.json \
  --test_retrieval_cache data/nq_test_retrieval_cache.json \
  --topk 3 \
  --local_dir data/nq_rag
```

Preview the RAG prompt/context shape before writing parquet:

```bash
python3 -m examples.prepare_search_rag_dataset \
  --dataset_name RUC-NLPIR/FlashRAG_datasets \
  --dataset_config nq \
  --corpus_path data/wiki-18.jsonl \
  --train_retrieval_cache data/nq_train_retrieval_cache.json \
  --test_retrieval_cache data/nq_test_retrieval_cache.json \
  --splits test \
  --topk 3 \
  --max_examples 20 \
  --preview \
  --preview_rows 5
```

## Run An Agent

`examples/run_agentic_search.py` is the main CLI.

```bash
python3 -m examples.run_agentic_search \
  --mode single \
  --question "What is FAISS?" \
  --model Qwen/Qwen2.5-1.5B-Instruct \
  --local --device cpu \
  --max_tokens 64 --temperature 0
```

Modes:

| Mode | Loop | Use it for |
|------|------|------------|
| `single` | `PlainGenerationLoop` | Simple local generation smoke tests |
| `search` | `SearchAgentLoop` | Multi-turn search traces for RAG, SFT, and RL |
| `tool` | `ToolAgentLoop` | Generic function/tool-calling experiments |

For server-backed inference:

```bash
python3 -m examples.run_agentic_search \
  --mode search \
  --question "Compare dense and sparse retrieval" \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --vllm_url http://localhost:8080 \
  --search_url http://localhost:8000/retrieve
```

## Local Retrieval

Start a dense retrieval server:

```bash
python3 -m src.retrieval.servers.retrieval \
  --model_path intfloat/e5-base-v2 \
  --index_path indexes/e5_Flat.index \
  --corpus_path data/corpus.jsonl \
  --retrieval_method e5 \
  --device cpu \
  --workers 1 \
  --topk 5
```

Start a sparse BM25 server:

```bash
python3 -m src.retrieval.servers.retrieval \
  --index_path indexes/bm25 \
  --corpus_path data/corpus.jsonl \
  --retrieval_method bm25 \
  --workers 1
```

Check the server:

```bash
curl -i -sS http://127.0.0.1:8000/health

curl -i -sS -X POST http://127.0.0.1:8000/retrieve \
  -H "Content-Type: application/json" \
  -d '{"query":"What is FAISS?","top_k":5}'
```

The retrieval endpoint accepts both single-query and batch-query request shapes.

## Build Indexes

Dense FAISS index:

```bash
python3 -m src.retrieval.index_builder \
  --retrieval_method e5 \
  --model_path intfloat/e5-base-v2 \
  --corpus_path data/corpus.jsonl \
  --faiss_type Flat \
  --save_dir indexes/
```

BM25 index:

```bash
python3 -m src.retrieval.index_builder \
  --retrieval_method bm25 \
  --corpus_path data/corpus.jsonl \
  --save_dir indexes/
```

## Web Search

`src.tools.search` routes calls to `retrieval`, `google`, `bing`, `brave`, or
`serpapi`. Missing API keys return structured tool errors.

Standalone web-search servers are available under `src.retrieval.servers`:

```bash
python3 -m src.retrieval.servers.serp \
  --search_url "https://serpapi.com/search" \
  --topk 3 \
  --serp_api_key "$SERP_API_KEY"

python3 -m src.retrieval.servers.google \
  --api_key "$GOOGLE_API_KEY" \
  --topk 5 \
  --cse_id "$GOOGLE_CSE_ID" \
  --snippet_only
```

## Retrieval Plus Rerank

```bash
python3 -m src.retrieval.servers.retrieval_rerank \
  --retriever_model intfloat/e5-base-v2 \
  --index_path indexes/e5_Flat.index \
  --corpus_path data/corpus.jsonl \
  --retrieval_method e5 \
  --retrieval_topk 10 \
  --rerank_topk 3
```

For BM25 plus reranking, use `--retrieval_method bm25` and omit
`--retriever_model`.

## Training Flow

The training pipeline is intentionally modular:

1. Generate trajectories with `SearchAgentLoop` or `LLMGenerationManager`.
2. Score trajectories with `SearchRewardFunction`.
3. Compute group-relative advantages with `src.training.grpo`.
4. Save JSONL batches or compute token log probabilities.
5. Optimize with PPO/GRPO/REINFORCE helpers in `src.model.generation` and
   `src.training.ppo`.

Useful entry points:

| Task | Command or module |
|------|-------------------|
| Deterministic trace | `python3 -m examples.run_search_trace_workflow` |
| SFT record from trace | `python3 -m examples.run_search_trace_workflow --sft` |
| QA parquet preparation | `python3 -m examples.prepare_search_qa_dataset` |
| Reward/GRPO smoke test | `python3 -m examples.run_grpo_training_pipeline` |
| Reward function | `src/training/reward.py` |
| GRPO helpers | `src/training/grpo.py` |
| PPO helpers | `src/training/ppo/` |
| Generation and policy loss | `src/model/generation.py` |

## XML Search Protocol

The search-agent trace uses a compact ReAct-style protocol:

```xml
<think>decide whether to answer or search</think>
<search>precise query</search>
<information>retrieval results injected by the environment</information>
<answer>final grounded answer</answer>
```

`<information>` is environment output and should be masked out of policy/SFT
action loss.

## Model Routing

`--model_routing intent` uses the intent classifier before model loading:

```bash
python3 -m examples.run_agentic_search \
  --mode search \
  --question "Recommend a dense retrieval setup for a small budget" \
  --model Qwen/Qwen2.5-1.5B-Instruct \
  --model_routing intent \
  --intent_model models/intent_classifier.pt \
  --fast_model Qwen/Qwen2.5-0.5B-Instruct \
  --balanced_model Qwen/Qwen2.5-1.5B-Instruct \
  --reasoning_model Qwen/Qwen2.5-7B-Instruct \
  --local --device cpu
```

Default routes:

| Intent | Route |
|--------|-------|
| `qa`, `navigate` | `fast_model` |
| `recommendation` | `balanced_model` |
| `purchase` | `reasoning_model` |

If confidence is too low or a route model is missing, the CLI falls back to
`--model`.

## Tests

```bash
python3 -m pytest -v
python3 -m pytest tests/unit/ -v
python3 -m pytest tests/unit/test_readme_examples.py tests/unit/test_run_agentic_search.py -v
python3 -m pytest tests/unit/test_reward.py tests/unit/test_grpo.py tests/unit/test_llm_agent_generation.py -v
```

## Notes

- Dense retrieval defaults to CPU to avoid competing with trainer GPU memory.
- Set dense retrieval to `--device cuda` only on a dedicated retrieval node.
- Empty or invalid queries return empty result lists.
- Some web pages block scraping or return little usable text.
- Google Custom Search, Bing, Brave, and SerpAPI are subject to their own quota
  and billing rules.
- BM25 serving requires Java because Pyserini uses Lucene.
