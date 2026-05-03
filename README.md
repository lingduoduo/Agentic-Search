# Agentic-Search

A FastAPI codebase for search-backed retrieval services and multi-turn agentic research loops.

- Google Custom Search and SerpAPI search servers
- Dense (FAISS) and sparse (BM25) retrieval with optional reranking
- `SearchAgentLoop`: plan → adaptive search decision → parallel queries → evidence evaluation → fetch → cited answer
- Config-driven text preprocessing for structured documents and `rec_texts` payloads

## Project Structure

```text
src/
  run_agentic_search.py  # CLI + importable entry point for all agent loop flows
  agent_loop/
    agent_loop.py          # AgentLoopBase, AgentLoopConfig, AgentLoopOutput
    context.py             # SearchResult, SearchContext, AgentContext
    evaluation.py          # SearchResultEvaluator, SearchEvaluationConfig
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

### CLI

```bash
# Deep-research loop (vLLM server + retrieval server)
python3 -m src.run_agentic_search \
    --mode search \
    --question "Compare dense vs sparse retrieval" \
    --model meta-llama/Llama-3.1-8B-Instruct \
    --vllm_url http://localhost:8080 \
    --search_url http://localhost:8000/retrieve

# Single-turn (no search), local model
python3 -m src.run_agentic_search \
    --mode single --question "What is FAISS?" \
    --model BAAI/bge-base-en-v1.5 --local

# Tool-calling loop
python3 -m src.run_agentic_search \
    --mode tool --question "What is the capital of France?" \
    --model meta-llama/Llama-3.1-8B-Instruct \
    --vllm_url http://localhost:8080 --tool_format hermes
```

Key flags:

| Flag | Default | Purpose |
|------|---------|---------|
| `--mode` | `search` | `single` / `search` / `tool` |
| `--local` | off | Load model in-process (no vLLM) |
| `--no_evidence_gate` | off | Allow `<answer>` before evidence is sufficient |
| `--require_search` | off | Force search even when model has internal knowledge |
| `--max_search_limit` | 0 (= max_turns) | Cap on search rounds |
| `--intent_examples` | none | Train a lightweight intent classifier and route search policy from it |
| `--intent_min_confidence` | `0.6` | Minimum confidence required before intent routing overrides defaults |
| `--tool_format` | `hermes` | Tool-call parser for `tool` mode |

If `--intent_examples` is provided, the CLI trains a small intent classifier on startup and can automatically bias search behavior. High-confidence `purchase`, `navigate`, and `recommendation` intents force evidence gathering and disable direct internal-knowledge answers; `qa` leaves the current settings unchanged.

A ready-to-use sample file can be generated from the local corpus and vocabulary with:

```bash
python3 -m src.generate_intent_examples --output data/intent_examples.sample.json
```

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
| `"search_agent"` | `SearchAgentLoop` | Deep research: plan → search → evaluate → answer |
| `"tool_agent"` | `ToolAgentLoop` | Multi-turn with parallel tool execution |

### XML protocol

`SearchAgentLoop` is driven by XML tags the model emits:

| Tag | Direction | Purpose |
|-----|-----------|---------|
| `<plan>` | model → loop | Research plan; loop acknowledges and continues |
| `<search_decision>answer\|search</search_decision>` | model → loop | Skip search if internal knowledge is sufficient |
| `<subquestions>` | model → loop | Register named research tracks (`T1`, `T2`, …) |
| `<search>query</search>` | model → loop | Single query |
| `<searches>` | model → loop | One query per line — all fired in parallel |
| `<fetch>url1, url2</fetch>` | model → loop | Fetch full-page content for specific URLs |
| `<information>` | loop → model | Cited evidence injected after each search round |
| `<search_evaluation>` | loop → model | `SUFFICIENT` / `INSUFFICIENT` verdict alongside evidence |
| `<full_page>` | loop → model | Page content injected after `<fetch>` |
| `<answer>` | model → loop | Final cited answer; loop stops |

Multiple tags in the same response are processed in one turn. Evidence labels follow `R{round}Q{query}D{doc}`. Repeated queries are deduplicated; rounds past `max_search_limit` get a nudge instead of silently dropping.

**Example turns**

```xml
<!-- Turn 1: plan + decision + subquestions + parallel searches in one shot -->
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

<!-- Turn 2: refine a weak track -->
<searches>
[T1] FAISS vs BM25 benchmark 2024
</searches>

<!-- Turn 3: fetch a promising page -->
<fetch>https://example.com/faiss-benchmark</fetch>

<!-- Turn 4: answer with citations -->
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
print(output.context.queries)   # all queries issued during the run
print(output.metrics)           # timing and search-quality counters
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
