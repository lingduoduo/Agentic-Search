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

Common public classes and helpers are exported from top-level `src`. Retrieval
implementation details live in `src/retrieval/`; FastAPI services live in
`src/servers/`.

## Features

- Local dense retrieval with FAISS-compatible indexes (E5, BGE, and custom embedders).
- Local sparse retrieval with BM25/Pyserini.
- Optional web search through Google Custom Search and SerpAPI.
- Retrieval + cross-encoder reranking pipeline.
- Multi-turn `SearchAgentLoop` traces with `<think>`, `<search>`,
  `<information>`, `<fetch>`, and `<answer>` actions.
- `SingleTurnAgentLoop` for parse-and-dispatch single-action flows.
- One-shot generation, one-shot RAG, full search-agent, and generic tool-agent
  loops.
- Hermes, Llama-3, and JSON tool-call parsers.
- OpenAPI-based `ApiToolRegistry` for dynamic tool loading.
- Composite reward shaping (`SearchRewardFunction`) with format, search-use,
  answer-length, and exact-match components.
- Group-relative advantage helpers for PPO, GRPO, and REINFORCE-style
  experiments.
- PPO core algorithms: clipped policy loss, value loss, entropy, KL penalty,
  adaptive and fixed KL controllers.
- Intent classifier (`IntentPipeline`) for query routing and model routing.
- Intent-driven model routing: pick fast/balanced/reasoning model by intent.
- Training data builders for search-QA and RAG parquet datasets.
- SFT example builder from search traces.

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
SERP_API_KEY=...
JAVA_HOME=/path/to/java
```

## Web Search Provider Admin API

The repo also includes a lightweight FastAPI admin surface for configuring web
search providers without adding a database dependency:

```python
from src.server.web_search import create_app

app = create_app()
```

It exposes `/health` plus `/admin/web-search/search-providers` and
`/admin/web-search/content-providers` routes for listing, upserting, activating,
deactivating, deleting, and validation-testing provider settings. Validation
tests are local by default; pass `"live": true` to make a real provider request.

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

## Indexing Helpers

The repo-native indexing pipeline lives in `src/retrieval/index_builder.py`.
For a server-style facade, use `src.servers.indexing`:

```python
from src.connectors import Document
from src.servers.indexing import index_document_batch

result = index_document_batch(
    [Document(id="doc-1", title="Example", contents="hello world")],
    save_dir="indexes/example",
)
```

The facade includes `Chunker`, `DefaultIndexingEmbedder`, `ChunkBatchStore`,
`embed_and_stream`, document prefiltering, mini-chunk support, and vector-write
retry helpers.

## Search Context

`src.context` contains small, repo-native helpers for retrieval-grounded chat:

```python
from src import SearchResult, build_context_bundle, build_answer_prompt

context = build_context_bundle(
    "What is FAISS?",
    [SearchResult(title="FAISS", contents='"FAISS"\nA vector search library.')],
)
prompt = build_answer_prompt("What is FAISS?", context)
```

It includes normalized context documents, citation extraction, retrieval prompt
builders, answer prompt builders, LLM protocol types, and an
`answer_with_retrieval` pipeline for connecting a `/retrieve` service to answer
generation.

## Local Retrieval

Start a dense retrieval server:

```bash
python3 -m src.servers.retrieval.retrieval \
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
python3 -m src.servers.retrieval.retrieval \
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

`src.tools.search` routes calls to `retrieval`, `google`, or `serpapi`. Missing API keys return structured tool errors.

Standalone web-search servers are available under `src.servers.retrieval`:

```bash
python3 -m src.servers.retrieval.serp \
  --search_url "https://serpapi.com/search" \
  --topk 3 \
  --serp_api_key "$SERP_API_KEY"

python3 -m src.servers.retrieval.google \
  --api_key "$GOOGLE_API_KEY" \
  --topk 5 \
  --cse_id "$GOOGLE_CSE_ID" \
  --snippet_only
```

## Retrieval Plus Rerank

```bash
python3 -m src.servers.retrieval.retrieval_rerank \
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

## Module Reference

### Agent Loops (`src/agents/`)

| Class / function | Description |
|-----------------|-------------|
| `AgentLoopBase` | Abstract base all loops inherit from; exposes `run()` |
| `AgentLoopConfig` | Shared config dataclass (max tokens, temperature, etc.) |
| `AgentLoopOutput` | Return value of `run()`: steps, final answer, metrics |
| `RolloutStep` | Single turn record: role, text, token ids, action mask |
| `PlainGenerationLoop` | One-shot generation, no search |
| `SearchAgentLoop` | Multi-turn XML trace loop (`<think>/<search>/<answer>`) |
| `SearchAgentLoopConfig` | Adds topk, max search limit, search URL, fetch URL |
| `SingleTurnAgentLoop` | Parse first action from generation, dispatch tool call |
| `ToolAgentLoop` | Generic tool-calling loop with `ToolParser` |
| `AgentState` | Full state: `TaskNode` graph, `Plan`, `RouteDecision`, metrics |
| `register` / `get_registered_agent_loop` | Decorator-based loop registry |
| `build_search_agent_instruction` | Build the system-prompt instruction string |

### Retrieval (`src/retrieval/`)

| Class / function | Description |
|-----------------|-------------|
| `DenseRetriever` | FAISS-backed dense retrieval; supports E5, BGE, custom encoders |
| `DenseRetrieverConfig` | Model path, index path, device, topk, batch size |
| `SparseRetriever` | Pyserini BM25 retriever |
| `SparseRetrieverConfig` | Index path, topk, language |
| `SentenceTransformerReranker` | Cross-encoder reranker via `sentence-transformers` |
| `RerankerConfig` | Model name, device, batch size |
| `get_reranker` | Factory: returns a configured `SentenceTransformerReranker` |
| `SearchClient` | Async HTTP client for the retrieval server (`/retrieve`) |
| `SearchClientConfig` | Base URL, timeout, retry settings |
| `SearchResult` | Single retrieved document: id, title, contents, score |
| `SearchContext` | Ordered list of `SearchResult`s for one query |
| `AgentContext` | Per-turn context accumulator across search rounds |
| `Vocabulary` | Freq-filtered token vocabulary with `build` / `encode` |
| `TextProcessor` | Tokenization, stopword filtering, field extraction |
| `normalize_text` / `tokenize_text` | Fast text normalisation and word tokenisation |
| `normalize_document` / `tokenize_document` | Document-level normalisation |
| `extract_keywords` | TF-style keyword extraction from document fields |
| `build_vocabulary_from_sequences` | Build a `Vocabulary` from a token sequence list |

#### Retrieval Servers (`src/retrieval/servers/`)

| Module | Server |
|--------|--------|
| `retrieval` | Dense (E5/BGE) or sparse (BM25) retrieval; `/retrieve`, `/health` |
| `retrieval_rerank` | Retrieval + cross-encoder rerank in one server |
| `rerank` | Standalone rerank endpoint |
| `google` | Google Custom Search proxy server |
| `serp` | SerpAPI proxy server |

### Tools (`src/tools/`)

| Class / function | Description |
|-----------------|-------------|
| `Tool` / `FunctionTool` | Abstract tool base; `FunctionTool` wraps a plain callable |
| `ToolSchema` | JSON-schema definition attached to a tool |
| `SearchPage` | Search result page: url, title, snippet, contents |
| `build_search_tool` | Build a `FunctionTool` that calls retrieval, Google, or SerpAPI |
| `format_search_pages` | Render a list of `SearchPage`s to text for the model |
| `HermesToolParser` | Parse Hermes-format `<tool_call>` XML |
| `Llama3ToolParser` | Parse Llama-3 `<\|python_tag\|>` tool calls |
| `JSONToolParser` | Parse bare JSON tool call blobs |
| `FunctionCall` | Parsed tool call: name + arguments dict |
| `ApiToolRegistry` | Load tools from an OpenAPI 3.x schema string |
| `ApiRequestTool` | Auto-generated tool that executes one OpenAPI operation |
| `parse_openapi_schema` | Parse an OpenAPI 3.x YAML/JSON string into `OpenAPISchema` |

### Connectors (`src/connectors/`)

| Class / function | Description |
|-----------------|-------------|
| `Document` / `SlimDocument` | Native document containers emitted by connectors |
| `BaseConnector` / `LoadConnector` | Connector interfaces for shared behavior and full-state document loading |
| `PollConnector` | Incremental sync by time window |
| `CheckpointedConnector` | Incremental sync that returns a persisted checkpoint |
| `CheckpointedConnectorWithPermSync` | Checkpointed sync with document permission metadata |
| `SlimConnector` | Pull only document ids for pruning or expired document deletion |
| `SlimConnectorWithPermSync` | Pull document ids plus permission metadata |
| `OAuthConnector` | OAuth authorization-code connector contract |
| `InMemoryConnector` | Load documents from Python objects or dictionaries |
| `LocalFileConnector` | Load UTF-8 text files from paths, directories, or glob patterns |
| `SearchConnector` | Load search results as documents through retrieval, Google, or SerpAPI |
| `StaticCredentialsProvider` | Simple in-memory provider for connector credentials |

### Model (`src/model/`)

#### Generation (`src/model/generation.py`)

| Class / function | Description |
|-----------------|-------------|
| `LLMGenerationManager` | Orchestrates batched GRPO rollouts: sample → score → pack |
| `GenerationConfig` | vLLM sampling params + retrieval URLs + safety config |
| `EndpointRetriever` | HTTP retrieval via `/retrieve` endpoint |
| `SimulateRetriever` | Deterministic fake retriever for tests |
| `GoogleRetriever` | Google Custom Search retriever |
| `EndpointFetcher` | Fetch a URL via the retrieval server's fetch endpoint |
| `ask_llm` | Low-level single-prompt vLLM call |
| `search_simulate` | Run a full search-agent trace with any retriever |
| `score_group_rollout` | Score a group of rollouts with a reward function |
| `assign_group_relative_advantages` | Compute GRPO advantages for a scored group |
| `apply_rollout_safety_penalties` | Penalise length, repetition, and format violations |
| `apply_safety_penalties_to_scored_rollouts` | Batch version of the above |
| `trajectory_log_prob_pack` | Pack token logprobs into training tensors |
| `format_search_trajectory_log` | Render a `SearchTrajectoryLog` to human-readable text |
| `format_trajectory_batch` | Render a batch of trajectories |
| `save_training_batch_jsonl` | Write a scored rollout batch to JSONL |
| `RolloutTrajectory` | Full trajectory: prompt, steps, final answer, reward |
| `SearchTrajectoryLog` | Per-query search turn log with documents and scores |
| `ActorRolloutStep` | One generation step: tokens, logprobs, action mask |
| `ReActStep` | ReAct observation-action pair |
| `GroupedRolloutBatch` | G rollouts for one prompt (GRPO group) |
| `ScoredGroupedRollout` | `GroupedRolloutBatch` with rewards and advantages |
| `GRPORolloutSafetyConfig` | Thresholds for safety penalties |

#### Intent Classifier (`src/model/intent_classifier.py`)

| Class / function | Description |
|-----------------|-------------|
| `IntentPipeline` | Trainable feedforward classifier; `train`, `predict`, `save`, `load` |
| `IntentPrediction` | `(intent, confidence)` result dataclass |
| `INTENT_LABELS` | `["purchase", "navigate", "qa", "recommendation"]` |
| `resolve_search_settings` | Map a prediction to adjusted topk / evidence / internal-knowledge flags |
| `load_training_data` | Load a JSON examples file into `(token_list, label)` pairs |

#### Intent Training (`src/model/intent_training.py`)

| Function | Description |
|----------|-------------|
| `train_intent_classifier` | Train an `IntentPipeline` from an examples file and save it |
| `generate_intent_examples` | Generate intent-labelled examples from a JSONL corpus |
| `write_intent_examples` | Write examples list to a pretty JSON file |
| `load_corpus` | Load a JSONL corpus into a list of document dicts |
| `load_vocabulary_tokens` | Load top-N tokens from a vocabulary metadata file |

#### Tensor Helper (`src/model/tensor_helper.py`)

| Class | Description |
|-------|-------------|
| `TensorHelper` | Pad, pack, and mask token sequences for PPO/GRPO training |
| `TensorConfig` | Padding token id, max sequence length, device |

### Training (`src/training/`)

#### Reward (`src/training/reward.py`)

| Class / function | Description |
|-----------------|-------------|
| `SearchRewardFunction` | Composite reward: format + search-use + length + exact-match |
| `SearchRewardConfig` | Weights for each reward component |
| `simple_sparse_correctness_reward` | Fast token-overlap correctness reward |
| `normalize_answer_text` | Lowercase, strip articles and punctuation |

#### Evaluation (`src/training/evaluation.py`)

| Class | Description |
|-------|-------------|
| `SearchResultEvaluator` | Evaluate retrieval quality across a batch of queries |
| `SearchEvaluationConfig` | Relevance threshold, topk, exact-match mode |
| `QueryEvaluation` | Per-query evaluation: precision, recall, hit |
| `SearchRoundEvaluation` | Per-search-round aggregate metrics |

#### GRPO (`src/training/grpo.py`)

| Class / function | Description |
|-----------------|-------------|
| `score_prompt_group` | Score G rollouts for one prompt, return `ScoredGRPORollout` |
| `score_prompt_batch` | Score a full batch of prompt groups |
| `compute_grpo_outcome_advantage` | Normalise rewards to group-relative advantages |
| `build_grpo_sampling_params` | Build vLLM sampling params for G samples per prompt |
| `GRPORolloutSample` | One rollout sample: tokens, logprobs, reward |
| `ScoredGRPORollout` | Rollout with advantage assigned |
| `PromptGroupSamplingConfig` | Group size, temperature, top-p |

#### SFT (`src/training/sft.py`)

| Class / function | Description |
|-----------------|-------------|
| `SFTExample` | Prompt + completion pair with action mask |
| `build_search_sft_example` | Build an `SFTExample` from a search-agent trace |

#### Data (`src/training/data.py`)

| Class / function | Description |
|-----------------|-------------|
| `PromptOnlyDataset` | PyTorch `Dataset` over tokenised prompt records |
| `PromptBatch` | Collated batch of padded prompt tensors |
| `build_prompt_dataloader` | Build a `DataLoader` from a parquet dataset |
| `build_search_qa_record` | Build a search-QA prompt/answer training record |
| `build_search_rag_record` | Build a RAG prompt/answer training record |
| `build_search_qa_prompt` | Format a question into a search-agent prompt string |
| `build_search_qa_messages` | Format as an OpenAI-style messages list |
| `build_search_rag_prompt` | Inject retrieved context into a RAG prompt |
| `format_rag_reference` | Format a retrieval result list as a reference block |
| `make_search_qa_map_fn` | HuggingFace `datasets` map function for QA records |
| `make_search_rag_map_fn` | HuggingFace `datasets` map function for RAG records |
| `normalize_question_text` | Normalise raw question fields |
| `normalize_answer_aliases` | Normalise answer alias lists |
| `collate_prompt_batch` | DataLoader collate function for `PromptSample`s |
| `prompt_batch_to_search_batch` | Convert a `PromptBatch` to a `SearchBatch` |

#### PPO (`src/training/ppo/`)

| Class / function | Description |
|-----------------|-------------|
| `compute_ppo_policy_loss_core` | Clipped PPO surrogate loss |
| `compute_reinforce_policy_loss` | REINFORCE policy gradient loss |
| `compute_trajectory_policy_loss` | Full trajectory loss with KL and entropy terms |
| `compute_grpo_outcome_advantage` | GRPO outcome-level advantage normalisation |
| `compute_rewards` | Apply KL penalty to per-token reward signal |
| `compute_value_loss` | Clipped value function loss |
| `kl_penalty` | Per-token KL divergence penalty (k1–k3 estimators) |
| `entropy_from_logits` | Per-token entropy from raw logits |
| `masked_mean` / `masked_whiten` | Masked tensor statistics for variable-length sequences |
| `clip_by_value` | Symmetric value clipping |
| `AdaptiveKLController` | PID-style KL coefficient adapter |
| `FixedKLController` | Constant KL coefficient |
| `PPORewardManager` | Assign per-sample rewards during PPO rollout collection |
| `LocalGRPOController` | In-process GRPO rollout loop with reward assignment |
| `PPOPolicyLossConfig` | Clip epsilon, entropy coefficient, KL coefficient |

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
- Google Custom Search and SerpAPI are subject to their own quota and billing rules.
- BM25 serving requires Java because Pyserini uses Lucene.
