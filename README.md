# Agentic-Search

A FastAPI codebase for search-backed retrieval services and multi-turn agentic research loops.

- Google Custom Search and SerpAPI search servers
- Dense (FAISS) and sparse (BM25) retrieval with optional reranking
- `SearchAgentLoop`: plan → adaptive search decision → parallel queries → evidence evaluation → fetch → cited answer
- Config-driven text preprocessing for structured documents and `rec_texts` payloads

## Project Structure

```text
src/
  run_agentic_search.py       # CLI + importable entry point for all agent loop flows
  train_intent_classifier.py  # Offline: train and save the intent classifier (.pt)
  generate_intent_examples.py # Offline: generate intent training examples from corpus
  agent_loop/
    agent_loop.py          # AgentLoopBase, AgentLoopConfig, AgentLoopOutput
    context.py             # SearchResult, SearchContext, AgentContext
    evaluation.py          # SearchResultEvaluator, SearchEvaluationConfig
    intent_classifier.py    # IntentPipeline: train / save / load + resolve_search_settings
    search_agent_loop.py   # SearchAgentLoop (registered as "search_agent")
    search_client.py       # async aiohttp client for /retrieve and /fetch endpoints
    single_turn_agent_loop.py
    tool.py                # Tool, FunctionTool — tool abstraction and JSON schema
    tool_agent_loop.py     # ToolAgentLoop (registered as "tool_agent")
    tool_parser.py         # Hermes / Llama3 / JSON tool-call parsers
  search/
    search_app.py
    google_search_server.py
    index_builder.py
    rerank.py
    retrieval.py
    retrieval_rerank_server.py
    retrieval_server.py
    serp_search_server.py
    text_processor.py     # config-driven cleanup / segmentation for structured text
    vocabulary.py
tests/
  unit/
  regression/
  load/
```

## Requirements

- Python 3.10+
- API keys (only needed for the corresponding server): `GOOGLE_API_KEY`, `GOOGLE_CSE_ID`, `SERP_API_KEY`
- Java 11+ (arm64 on Apple Silicon) for BM25 indexing via pyserini

```bash
pip install -r requirements.txt
```

## Environment Variables

Copy `.env.example` to `.env` — servers load it automatically at startup.

```
GOOGLE_API_KEY=...
GOOGLE_CSE_ID=...
SERP_API_KEY=...
JAVA_HOME=/opt/homebrew/opt/openjdk/libexec/openjdk.jdk/Contents/Home  # BM25 only
```

## Running the Agent

`src/run_agentic_search.py` is the unified entry point. It works as both a CLI script and an importable module.

Choose one inference backend:

- `vLLM / OpenAI-compatible server`: fastest path for repeated runs and larger models
- `local Hugging Face model`: simplest offline path, no separate inference server

**Full retrieval-backed pipeline (first-time setup)**

```
1. Build an index        →  src.search.index_builder        (one-time, offline)
2. Start retrieval server→  src.search.retrieval_server     (or retrieval_rerank_server)
3. Train intent model    →  src.train_intent_classifier     (one-time, offline, optional)
4. Run the agent         →  src.run_agentic_search          (every query)
```

For quick experiments without a corpus, skip steps 1–3 and start with `--mode single`.

### vLLM / server-backed inference

Use this when you already have a serving stack running at `--vllm_url`. This is the recommended path for faster interactive runs.

```bash
# Deep-research loop (vLLM server + retrieval server)
python3 -m src.run_agentic_search \
    --mode search \
    --question "Compare dense vs sparse retrieval" \
    --model meta-llama/Llama-3.1-8B-Instruct \
    --vllm_url http://localhost:8080 \
    --search_url http://localhost:8000/retrieve

# Single-turn answer through the same server (no retrieval)
python3 -m src.run_agentic_search \
    --mode single \
    --question "What is FAISS?" \
    --model meta-llama/Llama-3.1-8B-Instruct \
    --vllm_url http://localhost:8080 \
    --max_tokens 256 \
    --temperature 0

# Tool-calling loop through vLLM
python3 -m src.run_agentic_search \
    --mode tool \
    --question "What is the capital of France?" \
    --model meta-llama/Llama-3.1-8B-Instruct \
    --vllm_url http://localhost:8080 \
    --tool_format hermes
```

Typical setup:

1. Start `vllm serve ... --port 8080`
2. Start a search backend at `http://localhost:8000/retrieve` if using `--mode search`
3. Run `python3 -m src.run_agentic_search ...` without `--local`

### Local inference

Use this when you want an all-in-one offline workflow and are okay with slower generation than a dedicated serving stack.

```bash
# Single-turn (no search), local model on CPU
python3 -m src.run_agentic_search \
  --mode single \
  --question "What is FAISS?" \
  --model Qwen/Qwen2.5-1.5B-Instruct \
  --local \
  --device cpu \
  --max_tokens 256 \
  --temperature 0 \
  --generation_timeout_seconds 120 \
  --generation_heartbeat_seconds 5

# Search loop with local generation + local retrieval server
python3 -m src.run_agentic_search \
  --mode search \
  --question "Compare dense vs sparse retrieval" \
  --model Qwen/Qwen2.5-1.5B-Instruct \
  --local \
  --device cpu \
  --search_url http://localhost:8000/retrieve

# MPS (Apple Silicon GPU) — requires --allow_unsafe_mps and PyTorch ≥2.5 + transformers ≥4.46
python3 -m src.run_agentic_search \
  --mode single \
  --question "What is FAISS?" \
  --model Qwen/Qwen2.5-1.5B-Instruct \
  --local \
  --device mps \
  --allow_unsafe_mps \
  --max_tokens 256 \
  --temperature 0 \
  --generation_timeout_seconds 120 \
  --generation_heartbeat_seconds 5
```

Typical setup:

1. Skip `--vllm_url` and add `--local`
2. Pick `--device cpu`, `--device cuda`, or `--device mps --allow_unsafe_mps`
3. Start a retrieval server separately if using `--mode search`

### Key flags

| Flag | Default | Purpose |
|------|---------|---------|
| `--mode` | `search` | `single` / `search` / `tool` |
| `--local` | off | Load model in-process (no vLLM) |
| `--device` | `auto` | `cpu` / `cuda` / `mps` (local mode only); MPS requires `--allow_unsafe_mps` |
| `--allow_unsafe_mps` | off | Unlock MPS device; disabled by default due to segfault risk with some causal LMs |
| `--dtype` | auto | Override model dtype (`bfloat16` / `float16` / `float32`); auto selects bfloat16 on Apple Silicon |
| `--vllm_url` | `http://localhost:8080` | Base URL for the OpenAI-compatible server when not using `--local` |
| `--search_url` | `http://localhost:8000/retrieve` | Retrieval endpoint used by `search` mode |
| `--max_tokens` | `512` | Maximum new tokens to generate; set too low and answers will be truncated |
| `--generation_timeout_seconds` | `120` | Wall-clock deadline for local generation; local mode only |
| `--no_evidence_gate` | off | Allow `<answer>` before evidence is sufficient |
| `--require_search` | off | Force search even when model has internal knowledge |
| `--max_search_limit` | 0 (= max_turns) | Cap on search rounds |
| `--intent_model` | none | Load a pre-trained intent classifier (`.pt`); preferred for production |
| `--intent_examples` | none | Train an intent classifier from a JSON examples file at startup (slow path, for development) |
| `--intent_min_confidence` | `0.6` | Minimum confidence required before intent routing overrides defaults |
| `--tool_format` | `hermes` | Tool-call parser for `tool` mode |

When intent routing is active (via `--intent_model` or `--intent_examples`), high-confidence `purchase`, `navigate`, and `recommendation` intents automatically force evidence gathering and disable direct internal-knowledge answers; `qa` leaves the current settings unchanged.

### Intent classifier: training and inference

Train once offline, then reuse across all agent runs:

```bash
# Step 1 — generate labelled examples from a local corpus (optional; edit the JSON directly if preferred)
python3 -m src.generate_intent_examples \
    --corpus data/corpus.jsonl \
    --vocabulary data/vocabulary_corpus.json \
    --output data/intent_examples.json

# Step 2 — train and save
python3 -m src.train_intent_classifier \
    --examples data/intent_examples.json \
    --output models/intent_classifier.pt

# Step 3 — load at runtime (no retraining overhead)
python3 -m src.run_agentic_search \
    --intent_model models/intent_classifier.pt \
    --mode search --question "Buy me a noise-cancelling headphone" \
    --model Qwen/Qwen2.5-1.5B-Instruct \
    --vllm_url http://localhost:8080 \
    --search_url http://localhost:8000/retrieve
```

`--intent_examples` trains the same classifier on startup and is convenient for quick iteration, but adds startup latency on every run. Use `--intent_model` once the classifier is stable.

### Local model notes

Use `--local` to load a HuggingFace model in-process instead of connecting to a vLLM server. Useful for offline development or when a server is not available.

```bash
python3 -m src.run_agentic_search \
    --mode single \
    --question "What is FAISS?" \
    --model Qwen/Qwen2.5-1.5B-Instruct \
    --local \
    --device cpu \
    --max_tokens 256 \
    --temperature 0 \
    --generation_timeout_seconds 120 \
    --generation_heartbeat_seconds 5
```

**Device and dtype selection**

| `--device` | `--dtype` auto-selected | Notes |
|-----------|------------------------|-------|
| `cpu` on Apple Silicon (arm64) | `bfloat16` | 2-3x faster than float32; requires PyTorch ≥2.0 |
| `cuda` | `float16` | Standard GPU half-precision |
| `mps` | `float16` | Requires `--allow_unsafe_mps`, PyTorch ≥2.5, transformers ≥4.46 |
| `cpu` on x86 | `float32` | Safe default; slowest |

Override with `--dtype bfloat16` / `float16` / `float32`. Do not use `mps` with old toolchain versions — the runtime validates the stack and raises an error before loading the model.

**Generation timeout**

`--generation_timeout_seconds` (default `120`) sets a wall-clock deadline enforced by a `StoppingCriteria` that fires at the first token check after the deadline. Unlike `max_time=`, it interrupts generation even during long prefill phases. `--generation_heartbeat_seconds` controls how often the criteria is polled.

Set the timeout to comfortably cover prefill + generation: on Apple Silicon CPU with bfloat16, `--max_tokens 256` typically completes in under 60s, but `120` gives safe headroom. A timeout that fires mid-generation will truncate the answer just as `--max_tokens` does.

### vLLM tokenizer and gated models

When using `--vllm_url`, the CLI still loads the tokenizer locally to build prompt IDs. If the tokenizer Hub fetch fails because the model is gated (e.g. `meta-llama/Llama-3.1-8B-Instruct`), it automatically retries from the local cache — so if vLLM already downloaded the model, no login is required.

If the tokenizer is not cached at all, authenticate first and retry with the Hub model ID:

```bash
hf auth login   # or: huggingface-cli login
python3 -m src.run_agentic_search \
    --mode tool --question "What is the capital of France?" \
    --model meta-llama/Llama-3.1-8B-Instruct \
    --vllm_url http://localhost:8080 --tool_format hermes
```

For `--local` mode, pass `--allow_remote_model_downloads` to permit downloading weights at runtime (disabled by default).

### Programmatic use

```python
import asyncio
from transformers import AutoTokenizer
from src.run_agentic_search import VLLMServerManager, run_search_agent

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")
server_manager = VLLMServerManager(
    tokenizer=tokenizer,
    base_url="http://localhost:8080",
    model="meta-llama/Llama-3.1-8B-Instruct",
)

asyncio.run(run_search_agent(
    tokenizer=tokenizer,
    server_manager=server_manager,
    question="Compare dense vs sparse retrieval.",
    sampling_params={"temperature": 0.7},
    search_url="http://localhost:8000/retrieve",
    topk=5,
    max_turns=8,
))
```

`run_search_agent` prints `output.context.queries` and `output.metrics`, then the formatted answer and search trace.

| Class | When to use |
|-------|-------------|
| `VLLMServerManager` | Any OpenAI-compatible server (vLLM, Ollama, LiteLLM) |
| `LocalServerManager` | Offline — loads HuggingFace model in-process |

## Search Servers

### Google Custom Search

```bash
python3 -m src.search.google_search_server
```

Optional flags: `--topk N`, `--snippet_only`, `--host`, `--port`.

> Pass API keys via `.env`, not via shell variable expansion (`"$GOOGLE_API_KEY"` expands to empty if not exported).

### SerpAPI

```bash
python3 -m src.search.serp_search_server
```

Same flags as Google plus `--serp_engine` and `--search_url`. Both servers listen on `http://localhost:8000` by default.

## Building an Index

Dense index:

```bash
python3 -m src.search.index_builder \
  --retrieval_method bge \
  --model_path BAAI/bge-base-en-v1.5 \
  --corpus_path data/corpus.jsonl \
  --save_dir indexes/
```

BM25 index (requires Java):

```bash
python3 -m src.search.index_builder \
  --retrieval_method bm25 \
  --corpus_path data/corpus.jsonl \
  --save_dir indexes/
```

Notes: GPU is used automatically when available; `--bm25_threads N` sets Lucene thread count (default: all CPUs); `--no_save_vocabulary` skips the `vocabulary_corpus.json` sidecar.

## Dense Retrieval Server

```bash
python3 -m src.search.retrieval_server \
  --model_path BAAI/bge-base-en-v1.5 \
  --index_path indexes/bge_Flat.index \
  --corpus_path data/corpus.jsonl \
  --retrieval_method bge
```

```bash
curl -X POST http://localhost:8000/retrieve \
  -H "Content-Type: application/json" \
  -d '{"queries": ["What is agentic search?"], "topk": 3}'
```

## Rerank Server

```bash
python3 -m src.search.rerank_server \
  --rerank_model_name_or_path cross-encoder/ms-marco-MiniLM-L12-v2 \
  --rerank_topk 3
# listens on port 6980 by default
```

## Retrieval + Rerank Server

```bash
python3 -m src.search.retrieval_rerank_server \
  --retriever_model BAAI/bge-base-en-v1.5 \
  --index_path indexes/bge_Flat.index \
  --corpus_path data/corpus.jsonl \
  --retrieval_method bge \
  --retrieval_topk 10 --rerank_topk 3
```

## Agentic Search Loop

### Registered loops

| Name | Class | Description |
|------|-------|-------------|
| `"single_turn_agent"` | `SingleTurnAgentLoop` | One generation step, no search |
| `"search_agent"` | `SearchAgentLoop` | Multi-turn research with planning, adaptive search, fetch, and evidence gating |
| `"tool_agent"` | `ToolAgentLoop` | Multi-turn with parallel tool execution |

### What `SearchAgentLoop` actually does

The current loop is not a simple "search once, then answer" flow. It behaves more like a small research controller:

1. The model can write a `<plan>`.
2. The model must decide whether to answer from internal knowledge or search using `<search_decision>`.
3. For multi-hop questions, it can register named tracks with `<subquestions>`.
4. It can issue one query with `<search>` or many in parallel with `<searches>`.
5. The loop evaluates each search round and injects both evidence and a search-quality verdict.
6. If snippets are weak, the model can refine queries or fetch full pages with `<fetch>`.
7. `<answer>` is accepted only when the loop allows it:
   - immediately, if internal knowledge answers are allowed and the model chose `<search_decision>answer</search_decision>`
   - after search, only when the latest evidence is sufficient overall and for every active subquestion

Important runtime behavior:

- Search rounds are capped by `max_search_limit`.
- Repeated queries are skipped and called out explicitly.
- Search and page fetches are cached for the duration of one run.
- Multiple actions in one model response are supported and processed in order.

### XML protocol

`SearchAgentLoop` is driven by XML tags the model emits and loop-generated feedback tags:

| Tag | Direction | Purpose |
|-----|-----------|---------|
| `<plan>` | model → loop | Record a short research plan |
| `<search_decision>answer\|search</search_decision>` | model → loop | Declare whether to answer directly or retrieve evidence |
| `<subquestions>` | model → loop | Register named research tracks such as `T1`, `T2` |
| `<search>query</search>` | model → loop | Send one query |
| `<searches>` | model → loop | Send multiple queries in parallel, one per line |
| `<fetch>url1, url2</fetch>` | model → loop | Fetch full-page content for URLs returned by search |
| `<answer>` | model → loop | Final answer candidate |
| `<plan_feedback>` | loop → model | Acknowledges the plan and tells the model to continue |
| `<decision_feedback>` | loop → model | Prompts for or acknowledges the current search decision |
| `<subquestions_feedback>` | loop → model | Confirms registered subquestions |
| `<information>` | loop → model | Search results with citation labels such as `[R1Q2D1]` |
| `<search_evaluation>` | loop → model | Sufficiency verdict and per-query feedback for the latest round |
| `<full_page>` | loop → model | Full fetched page content |
| `<search_feedback>` | loop → model | Explains repeated-query skips or search-limit enforcement |
| `<answer_feedback>` | loop → model | Rejects premature answers and explains what is still missing |

Inside `<searches>`, queries can optionally be task-scoped with prefixes like `[T1] FAISS benchmark` so the loop can track evidence per subquestion.

Multiple tags in the same response are processed in one turn. Evidence labels follow `R{round}Q{query}D{doc}`. Full-page fetches are surfaced separately from search rounds.

### Typical flow

```xml
<plan>Compare dense and sparse retrieval.</plan>
<search_decision>search</search_decision>
<subquestions>
T1: dense retrieval with FAISS
T2: sparse retrieval with BM25
</subquestions>
<searches>
[T1] dense retrieval FAISS overview
[T2] BM25 sparse retrieval Lucene
</searches>
```

The loop then injects:

```xml
<search_evaluation>
INSUFFICIENT
...
</search_evaluation>
<information>
Round 1
...
</information>
```

If one track is still weak, the model can refine just that track:

```xml
<searches>
[T1] FAISS vs BM25 benchmark 2024
</searches>
```

If snippets are not enough, it can fetch pages:

```xml
<fetch>https://example.com/faiss-benchmark</fetch>
```

And only then produce a cited answer:

```xml
<answer>
Dense retrieval [R1Q1D1] outperforms BM25 [R1Q2D1] on semantic queries,
but BM25 remains competitive for keyword-heavy tasks [R2Q1D1][R3P1].
</answer>
```

### Direct use (without `run_agentic_search.py`)

```python
from src.agent_loop import SearchAgentLoop, SearchAgentLoopConfig, SearchEvaluationConfig

loop = SearchAgentLoop(
    tokenizer=tokenizer,
    server_manager=server_manager,
    search_config=SearchAgentLoopConfig(
        search_url="http://localhost:8000/retrieve",
        topk=5,
        max_turns=8,
        max_search_limit=6,
        allow_internal_knowledge_answer=True,
        evaluation_config=SearchEvaluationConfig(
            min_results_per_query=1,
            min_total_results=2,
            min_content_length=10,
        ),
    ),
)
output = await loop.run(
    messages=[{"role": "user", "content": "Compare dense vs sparse retrieval."}],
    sampling_params={"temperature": 0.7},
)
print(output.context.queries)   # flat list of issued queries
print(output.context.tasks)     # registered subquestions, if any
print(output.metrics)           # timing, cache, gating, and search counters
```

### Tool agent

Use `--mode tool` from the CLI, or:

```python
from src.agent_loop import ToolAgentLoop, ToolAgentLoopConfig, FunctionTool

@FunctionTool.from_fn(description="Search", parameters={...})
async def search(query: str) -> str: ...

loop = ToolAgentLoop(tokenizer=tokenizer, server_manager=server_manager,
                    tools=[search],
                    config=ToolAgentLoopConfig(tool_parser_format="hermes"))
```

Supported `tool_parser_format` values:

| Format | Model family |
|--------|-------------|
| `"hermes"` | NousResearch Hermes 2.5 / 3 |
| `"llama3"` | Meta Llama 3.1 / 3.2 |
| `"json"` | Generic fallback (best-effort) |

### Context objects

- `SearchResult(contents, score, title, url)`
- `SearchContext(query, results, task_id, task_description)` — `.to_information_block(citation_prefix=...)` formats for injection
- `AgentContext` — attached to `AgentLoopOutput.context`:
  - `.rounds` — `list[list[SearchContext]]`, one list per search round
  - `.turns` — flat list of every `SearchContext`
  - `.tasks` — `dict[str, str]` of task id → description (from `<subquestions>`)
  - `.queries` — flat list of query strings in issue order
  - `.num_rounds`, `.num_searches`

### Metrics (`output.metrics`)

| Key | Meaning |
|-----|---------|
| `search_rounds` | Rounds that hit the retrieval server |
| `search_queries` | Individual queries sent (after dedup) |
| `search_cache_hits` | Queries served from the per-run cache |
| `page_cache_hits` | Fetched pages served from the per-run cache |
| `fetched_pages` | Pages retrieved via `<fetch>` |
| `answer_rejections` | Times `<answer>` was blocked by the evidence gate |
| `direct_answers` | Times model answered from internal knowledge |
| `decision_prompts` | Times a `<search_decision>` turn was injected |
| `repeated_search_queries` | Queries skipped as duplicates |
| `search_limit_hits` | Turns where the round cap was enforced |
| `active_subquestions` | Number of registered subquestion tasks |

## API Reference

### `GET /health`

```json
{"status": "ok"}
```

### `POST /retrieve`

```json
{"queries": ["query 1", "query 2"], "topk": 3}
```

Response:

```json
{
  "result": [
    [{"document": {"contents": "\"Title\"\nBody text."}}],
    [{"document": {"contents": "\"Title 2\"\nBody text."}}]
  ]
}
```

### `POST /fetch` (Google server only)

```json
{"urls": ["https://example.com/page"]}
```

Response shape mirrors `/retrieve`.

## Testing

```bash
pip install pytest httpx
python3 -m pytest              # all tests
python3 -m pytest tests/unit/  # unit tests only (no server or model required)
python3 -m pytest tests/load/ -v -s -m load  # latency/throughput tests
```

Unit test coverage:

| File | What is tested |
|------|---------------|
| `test_agent_loop.py` | `AgentLoopBase`; `SingleTurnAgentLoop`; `SearchAgentLoop` — plan, parallel search, multi-round refinement, subquestions, `<fetch>`, search+fetch in one turn, evaluation feedback, answer gating, adaptive search-decision, direct internal-knowledge answer, repeated-query dedup, search-round limit, cache and metrics; `SearchResultEvaluator`; `SearchClientConfig.get_fetch_url` |
| `test_intent_classifier.py` | `Vocabulary` sequence training; `IntentPipeline` untrained guard; `resolve_search_settings` purchase / low-confidence / qa / recommendation policies; `INTENT_LABELS` snapshot; save/load round-trip; save-before-train guard |
| `test_run_agentic_search.py` | `_build_prompt_ids_sync` chat-template fallback; `_validate_local_generation_config` encoder-only rejection; `_friendly_model_load_error` gated/missing/cache-miss messages; `_resolve_local_device`; `_has_accelerate`; `_parse_major_minor`; `_validate_local_runtime_device` MPS guard; `_validate_local_runtime_stack` old-stack rejection and CPU/new-stack allowance; `LocalServerManager._generate_sync` greedy-decode attention mask and wall-clock stopping criteria |
| `test_vocabulary.py` | `Vocabulary`, tokenization, keyword extraction |
| `test_index_builder.py` | `IndexBuilderConfig.validate`, `prepare_texts`, `resolve_pooling_method`, `pooling` |
| `test_llm_agent_generation.py` | action parsing, search payload, inactive examples, unknown search mode |
| `test_llm_agent_tensor_helper.py` | padding conversion, batch re-expansion |
| `test_rerank.py` | passage formatting, `RerankerConfig.validate`, `SentenceTransformerReranker.rerank` |
| `test_search_app.py` | `format_document`, `/health`, `/retrieve` endpoints |

## Notes

- Google Custom Search and SerpAPI usage are subject to their respective quota and billing rules.
- Some result pages may block scraping or return little usable text.
- Empty or invalid queries return empty result lists.
