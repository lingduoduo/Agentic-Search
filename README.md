# Agentic-Search-GRPO

A unified codebase for search-backed retrieval systems, multi-turn agentic
research workflows, full-trace supervised fine-tuning (SFT), and end-to-end
reinforcement learning (RL) training with PPO/GRPO-style optimization.

## Core Features

- Support for local sparse retrievers such as BM25.
- Support for local dense retrievers with both flat indexing and ANN indexing,
  including FAISS index factory configurations.
- Integration with external web search providers such as Google Custom Search,
  Bing Search, and SerpAPI.
- Support for off-the-shelf intent classifiers and neural rerankers.
- Support for multiple RL algorithms, including PPO, GRPO, and REINFORCE.
- Support for multiple LLM families, including Llama 3 and Qwen 2.5.

## Retrieval & Search Infrastructure

- Google Custom Search, Bing, and SerpAPI-backed search providers.
- Hybrid retrieval pipelines combining sparse retrieval (BM25), dense retrieval
  (FAISS), and optional neural reranking.

## Agentic Search Framework

`SearchAgentLoop` supports:

- Planning.
- Adaptive search decisions.
- Sub-question decomposition.
- Parallel query execution.
- Evidence evaluation.
- Document fetching.
- Citation-grounded answer generation.

## Training & Optimization

`SearchRewardFunction` provides strategy-aware reward shaping and GRPO
within-group advantage normalization.

Full action-trace SFT support lets you train on complete trajectories such as:

```text
<plan> -> <search> -> <evidence> -> <answer>
```

instead of training only on final responses.

The end-to-end PPO/GRPO RL training pipeline covers rollout generation, reward
computation, advantage estimation, log-probability extraction, clipped policy
optimization, and optimizer updates.

## Systems & Scalability

Asynchronous rollout execution supports rollout-level concurrency:

```text
N_prompts x G parallel tasks
```

This overlaps HTTP search latency and retrieval I/O, improving throughput for
large-scale RL training and evaluation workloads.

This repo is organized around a progression from basic retrieval to trainable
agentic search:

| Stage | What it does | Main code path |
|-------|--------------|----------------|
| Simple search | Retrieve documents from a search or vector endpoint | `src/retrieval/`, `src/tools/search.py`, `/retrieve` |
| RAG | Retrieve evidence at inference time, inject it into the prompt, then answer | `SingleTurnAgentLoop` |
| Agentic Search | Let the model decide when to search, what to search, when evidence is sufficient, and when to answer | `SearchAgentLoop` |
| Agentic Search + RL | Train better search behavior from full trajectories, rewards, and PPO/GRPO/REINFORCE losses | `src/training/reward.py`, `src/training/sft.py`, `src/training/grpo.py`, `src/training/ppo/core_algos.py`, `src/model/generation.py` |

### RAG and fine-tuning solve different problems here:

| Dimension | RAG | Fine-tuning |
|-----------|-----|-------------|
| Primary goal | Add external knowledge at inference time | Change model behavior or policy |
| Best for | Fresh, large, private, or citation-sensitive knowledge | Stable formats, workflows, tool-use habits, search policy |
| Update path | Re-index or update the corpus | Regenerate traces, train, evaluate, redeploy |
| Failure mode | Bad retrieval, missing evidence, weak grounding | Overfitting, stale learned behavior, format drift |
| Cost profile | More runtime latency from retrieval and longer prompts | More training cost, cheaper repeated inference if behavior improves |
| Repo path | `src/retrieval/`, `src/tools/`, `SingleTurnAgentLoop`, `SearchAgentLoop` observations | `src/training/sft.py`, `SearchRewardFunction`, `src/training/grpo.py`, `src/model/generation.py` |

Use RAG when the model needs facts it should not memorize. Use fine-tuning when
the model already has enough capability but needs to behave consistently: when
to search, how to write queries, how to avoid repeated searches, how to cite
evidence, and when to stop.

In short: RAG teaches the system where to get knowledge; fine-tuning teaches
the model how to behave while using that knowledge.

### SFT vs RLHF / GRPO

For fast iteration in this repo, SFT is usually the first move. It is cheaper,
more deterministic, and easier to debug because it trains directly on known-good
search traces. Use it to teach the model the workflow shape: XML actions,
query style, evidence use, citation format, and stopping behavior.

RLHF-style optimization, represented here by reward functions plus PPO/GRPO/REINFORCE,
is slower to iterate but more useful when the target behavior is hard to write
as a single gold trace. Use it when there are tradeoffs: search more or answer
now, cite enough but avoid bloated answers, explore subquestions without
repeating queries, or optimize final answer quality under a search budget.

| Dimension | SFT | RLHF / GRPO |
|-----------|-----|-------------|
| Iteration speed | Faster first loop | Slower, needs rollouts and reward evaluation |
| Training signal | Demonstration traces | Rewards, preferences, or outcome/process scores |
| Best for | Teaching the protocol and a known-good workflow | Improving decisions with tradeoffs and delayed outcomes |
| Data shape | Full action traces from `build_search_sft_example()` | Groups of rollouts scored by `SearchRewardFunction` |
| Debuggability | High: inspect the target trace directly | Medium: inspect rewards, advantages, and trajectory metrics |
| Breakthrough point | Make a strong base model reliably follow the agent scaffold | Make the agent choose better trajectories than supervised examples cover |
| Repo path | `examples.run_search_trace_workflow --sft`, `src/training/sft.py` | `examples.run_grpo_training_pipeline`, `src/training/grpo.py`, `src/model/generation.py` |

As base models become stronger, the bottleneck shifts:

- SFT's leverage is no longer teaching basic language ability. Its leverage is
  turning a capable model into a reliable product-shaped agent that follows the
  repo's protocol and tool workflow consistently.
- RLHF / GRPO's leverage is no longer fixing simple formatting mistakes. Its
  leverage is optimizing decisions where supervised labels are incomplete:
  when to search, how much evidence is enough, which trajectory is better, and
  how to trade quality against latency and cost.

The practical path is: use SFT to establish the behavior scaffold quickly, then
use PPO/GRPO rewards to improve the policy choices that only show up after the
agent interacts with retrieval.

### Lowering Inference Cost

In multi-task or multi-agent systems, the main tradeoff is not simply small
model vs large model. It is deciding which work deserves expensive reasoning,
which work can be routed to cheaper paths, and when the system has enough
evidence to stop.

This repo exposes several cost-control layers:

| Lever | Lower-cost choice | Accuracy-oriented choice |
|-------|-------------------|--------------------------|
| Loop | `PlainGenerationLoop` or `SingleTurnAgentLoop` | `SearchAgentLoop` |
| Model | `--model_routing intent` with `fast_model` / `balanced_model` / `reasoning_model` | Always use the strongest model |
| Retrieval | smaller `topk`, fewer search rounds | larger `topk`, more rounds, fetch pages |
| Agent budget | lower `--max_turns` and `--max_search_limit` | allow more turns before forcing an answer |
| Evidence policy | answer when confidence is enough | keep `require_sufficient_evidence_before_answer` enabled |
| Training reward | penalize search cost and repeated queries | reward citation support and evidence sufficiency |

The intended pattern is hybrid routing:

- Use small or fast models for classification, routing, simple QA, and
  navigation-style tasks.
- Use retrieval before escalating to a larger model when the missing piece is
  external knowledge rather than deeper reasoning.
- Use larger reasoning models for high-ambiguity, high-stakes, multi-hop, or
  synthesis-heavy tasks.
- Use `SearchAgentLoop` only when the task benefits from multi-turn evidence
  gathering; otherwise prefer the simpler loop.
- Train with rewards that make cost visible, such as search penalties, repeated
  query penalties, and budget exhaustion metrics.

For multi-agent systems, optimize the system-level policy rather than each
agent in isolation. A cheap router can decide whether to answer directly,
retrieve once, invoke the full search agent, or escalate to a stronger model.
Accuracy improves when expensive agents are reserved for cases where they
change the outcome, not when every task uses the heaviest path by default.

## Project Structure

```text
src/
  agent_loop/
    ...                      # Backward-compatible imports for agents, tools, training, and retrieval
  agents/
    base.py                  # AgentLoopBase, AgentLoopConfig, AgentLoopOutput
    plain.py                 # PlainGenerationLoop (registered as "plain_generation")
    search.py                # SearchAgentLoop (registered as "search_agent")
    single_turn.py           # One-shot retrieval-assisted RAG loop
    tool_calling.py          # ToolAgentLoop (registered as "tool_agent")
  llm_agent/
    ...                      # Backward-compatible imports for model generation helpers
  model/
    generation.py            # LLMGenerationManager — rollout, log probs, policy loss, async
    intent_classifier.py     # IntentPipeline: train / save / load + resolve_search_settings
    intent_training.py       # Generate intent examples + train/save classifier utilities
    tensor_helper.py         # TensorConfig, TensorHelper — padding and batch helpers
  retrieval/
    client.py                # async aiohttp client with session reuse for /retrieve and /fetch
    context.py               # SearchResult, SearchContext, AgentContext
    dense_retriever.py       # DenseRetriever (CPU default to avoid VRAM contention)
    index_builder.py
    rerank.py
    sparse_retriever.py      # SparseRetriever for local BM25 / Pyserini indexes
    servers/
      app.py                 # Shared FastAPI app factory
      google.py              # Google Custom Search /retrieve server
      rerank.py              # Rerank FastAPI server
      retrieval.py           # Local dense or sparse retrieval FastAPI server
      retrieval_rerank.py    # Combined retrieval + rerank FastAPI server
      serp.py                # SerpAPI /retrieve server
    text_processor.py        # config-driven cleanup / segmentation for structured text
    vocabulary.py
  search/
    ...                      # Backward-compatible imports for retrieval modules and servers
  tools/
    api.py                   # OpenAPI schema -> ToolAgentLoop API tools
    base.py                  # Tool, FunctionTool — tool abstraction and JSON schema
    parsers.py               # Hermes / Llama3 / JSON tool-call parsers
    search.py                # Search provider router + FunctionTool builder
  trainer/
    ...                      # Backward-compatible imports for PPO training helpers
  training/
    data.py                  # Prompt datasets, dataloaders, and batch conversion helpers
    evaluation.py            # SearchResultEvaluator, SearchEvaluationConfig
    grpo.py                  # Prompt-group rollout sampling + within-group scoring helpers
    ppo/
      controller.py
      core_algos.py
      plot_rollouts.py
    reward.py                # SearchRewardFunction, SearchRewardConfig — reward + GRPO advantages
    sft.py                   # Build supervised examples from full search-agent trajectories
tests/
  unit/
  regression/
  load/
examples/
  run_agentic_search.py          # CLI + importable entry point for all agent loop flows
  run_search_agent_loop.py       # CLI + importable SearchAgentLoop wiring example
  run_search_trace_workflow.py   # Deterministic trace demo + SFT example builder
  run_grpo_training_pipeline.py  # Model-free reward / GRPO helper smoke test
  run_intent_training.py         # Generate examples + train classifier CLI wrapper
```

## Requirements

- Python 3.10+
- API keys, only for the providers you use: `GOOGLE_API_KEY`, `GOOGLE_CSE_ID`, `BING_SEARCH_API_KEY`, `BRAVE_SEARCH_API_KEY`, `SERP_API_KEY`
- Java 11+ (arm64 on Apple Silicon) for BM25 indexing and querying via Pyserini

```bash
pip install -r requirements.txt
```

## Environment Variables

Copy `.env.example` to `.env` — servers load it automatically at startup.

```
GOOGLE_API_KEY=...
GOOGLE_CSE_ID=...
BING_SEARCH_API_KEY=...
BRAVE_SEARCH_API_KEY=...
SERP_API_KEY=...
JAVA_HOME=/opt/homebrew/opt/openjdk/libexec/openjdk.jdk/Contents/Home  # BM25 only
```

---

## PPO / GRPO / REINFORCE Training Pipeline

The full RL training loop is implemented across `src/model/generation.py`,
`src/agents/`, `src/retrieval/`, `src/tools/`, and `src/training/`. The
legacy `src/llm_agent/`, `src/agent_loop/`, and `src/search/` packages remain
as compatibility import layers, but new code and docs should prefer the
current module layout.

The pipeline follows these 10 steps:

```
1. run_llm_loop()          → trajectory (RolloutTrajectory + SearchTrajectoryLog)
2. SearchTool              → FastAPI /retrieve (EndpointRetriever)
3. G=4 rollouts/prompt     → run_prompt_rollout_group / async_run_prompt_rollout_group
4. reward_fn               → SearchRewardFunction scores each rollout
5. compute_grpo_advantage  → assign_group_relative_advantages
6. save_training_batch_jsonl → JSONL training data
7. compute_log_probs       → trajectory_log_prob_pack (prompt-aligned)
8. Policy loss             → compute_trajectory_policy_loss / compute_policy_loss / compute_reinforce_policy_loss
9. run_grpo_training_step  → end-to-end on small model + small data
10. async                  → async_run_grpo_training_step (N×G concurrent rollouts)
```

For a quick, model-free smoke test of the reward and GRPO helper flow, run:

```bash
python3 -m examples.run_grpo_training_pipeline
```

The script builds deterministic `AgentLoopOutput` objects, scores them with
`SearchRewardFunction`, computes group-relative advantages, and, when PyTorch
is installed, also exercises trajectory packing plus PPO/GRPO policy loss.

### Step 1 — Trajectory output from `run_llm_loop`

`LLMGenerationManager.run_llm_loop()` returns a final batch plus per-trajectory
turn counts. The batch stores structured trajectory logs, rollout objects, and
prompt-aligned token arrays for later reward and loss computation.

Reference code: `src/model/generation.py`

Tests: `tests/unit/test_llm_agent_generation.py`

### Step 2 — SearchTool → FastAPI `/retrieve`

The retrieval server runs separately from the trainer to avoid VRAM contention.
`DenseRetrieverConfig` defaults to `device="cpu"`.

```bash
# Start the retrieval server (CPU, 1 worker)
python3 -m src.retrieval.servers.retrieval \
  --model_path intfloat/e5-base-v2 \
  --index_path indexes/e5_Flat.index \
  --corpus_path data/corpus.jsonl \
  --retrieval_method e5 \
  --device cpu \
  --workers 1 \
  --topk 5

# Test it
curl -X POST http://localhost:8000/retrieve \
  -H "Content-Type: application/json" \
  -d '{"query": "2024 Nobel Prize Physics", "top_k": 3}'
```

Inside the manager, set `search_mode="local"` and `retrieval_url="http://localhost:8000/retrieve"`.

### Step 3 — G=4 rollouts per prompt

Use `run_prompt_rollout_group()` for reproducible sequential rollouts, or
`async_run_prompt_rollout_group()` to overlap HTTP search I/O across
`N_prompts × G` rollout tasks. Each rollout keeps the same `group_id` and a
distinct `rollout_index`.

Reference tests: `tests/unit/test_llm_agent_generation.py`

### Step 4 — `reward_fn` scores each rollout

`SearchRewardFunction` supports sparse final-answer reward, first-pass search
penalty shaping, and second-pass shaping with citation/search-quality signals.
Use batch judge helpers when an LLM judge would otherwise require one API call
per rollout.

Runnable reference: `python3 -m examples.run_grpo_training_pipeline`

Source: `src/training/reward.py`

### Step 5 — `compute_grpo_advantage`

Group-relative advantages can be raw mean-centered or std-normalized. The
default training path uses std-normalized advantages for stability; sparse
outcome-only GRPO is still available through `GRPOAdvantageConfig`.

Source: `src/training/grpo.py` and `src/model/generation.py`

### Step 6 — Save JSONL training data

Each JSONL record contains: `group_id`, `rollout_index`, `reward`, `advantage`,
`reward_components`, `trajectory` (full `SearchTrajectoryLog.to_dict()`),
`tokens`, `response_mask`, `old_log_probs` (all prompt+response aligned).

Writer: `save_training_batch_jsonl()` in `src/model/generation.py`

### Step 7 — Offline `compute_log_probs`

Compute `old_log_probs` with the frozen rollout policy, `new_log_probs` with
the current policy, and optionally `ref_log_probs` for KL anchoring.

Alignment rule: `old_log_probs` is prepended with `len(prompt_token_ids)` zeros
so it aligns with `tokens` and `attention_mask` without slicing in the loss loop.
`response_mask` is `0` for prompt and environment `<information>` tokens.

### Step 8 — Policy loss

Formula:

```
ratio_t       = exp(new_log_probs_t − old_log_probs_t)
L_clip        = −mean(min(ratio * A, clip(ratio, 1−ε, 1+ε) * A) * mask)
kl_penalty    = β × KL(π_ref ∥ π_θ) × mask   # or KL(π_old ∥ π_θ) if no ref
entropy_bonus = −coef × mean(log π_θ × mask)
total_loss    = L_clip + kl_penalty − entropy_bonus
```

`response_mask` is `1` only for model-generated action tokens (`<search>`, `<plan>`,
`<fetch>`, `<answer>`); `0` for prompt tokens and environment `<information>` observations.
For simpler unclipped policy-gradient updates, `compute_reinforce_policy_loss()`
uses the same aligned token lists with reward-minus-baseline advantages.

Source: `compute_trajectory_policy_loss()`, `compute_reinforce_policy_loss()`,
and `PPOPolicyLossConfig`

### Step 9 — End-to-end on small model + small data

What `run_grpo_training_step` does internally:

1. For each prompt: `run_prompt_rollout_group()` → G rollouts
2. `score_group_rollout()` — reward + GRPO advantage per rollout
3. `apply_safety_penalties_to_scored_rollouts()` — penalize unsafe tool use
4. `collate_scored_rollouts_for_training()` — merge into one `SearchBatch`
5. `compute_log_prob()` for `old_log_probs`, `new_log_probs`, `ref_log_probs`
6. `compute_policy_loss()` — clipped surrogate + KL + entropy
7. `loss.backward(); optimizer.step()`

### Step 10 — Async / distributed rollout

Execution model: `N_prompts × G` rollout tasks run concurrently in a `ThreadPoolExecutor`.
While rollout A waits for HTTP search results, rollouts B/C/D continue generating.
Effective search wall-time drops from `G × search_latency` to roughly `max(per-rollout latency)`.

Each task calls `manager._run_one_rollout(prompt_batch, rollout_idx, variant, group_id, ...)`
on a `copy.copy(manager)`, so `self.config` mutations (temperature, seed) never race across threads.

Future path to Ray workers:

```
Rollout workers  →  async_run_prompt_rollout_group (already concurrent)
Learner          →  compute_policy_loss + optimizer.step (one central update)
```

### Safety constraints

The `allowed_actions` parser inspects `trajectory.steps` for valid-XML-but-disallowed tags
(e.g. `<fetch>` or `<plan>` when only `["search", "answer"]` are permitted), applying
`invalid_action_penalty` per disallowed tag. This is separate from `invalid_action_count`
in metrics (which only catches malformed XML).

Safety helpers add `invalid_action_penalty`, `repeated_query_penalty`,
`excess_search_penalty`, and `disallowed_action_penalty` into
`reward_components`, then can rederive group advantages after penalties.

---

## Running the Agent

`examples/run_agentic_search.py` is the unified entry point.

### vLLM / server-backed inference

```bash
python3 -m examples.run_agentic_search \
    --mode search \
    --question "Compare dense vs sparse retrieval" \
    --model meta-llama/Llama-3.1-8B-Instruct \
    --vllm_url http://localhost:8080 \
    --search_url http://localhost:8000/retrieve
```

### Local inference

`single`, `search`, and `tool` are CLI modes, not HTTP endpoints. First use
`curl` to check the retrieval service, then run the loop you want to test.

Use the modes this way:

| Mode | Best use | Notes |
|------|----------|-------|
| `single` | Simplest smoke test for local model generation | Uses `PlainGenerationLoop`; no retrieval, tools, or XML actions |
| `search` | Project-native search-agent / RL trajectory format | Uses `SearchAgentLoop` with XML actions such as `<think>`, `<search>`, `<information>`, and `<answer>` |
| `tool` | Generic function/tool calling | Optional path for tool-schema parsing; not the primary search-agent flow |

Loop differences:

| Loop | Retrieval | Turns | Output shape |
|------|-----------|-------|--------------|
| `PlainGenerationLoop` | Never | One model generation | Plain assistant text |
| `SingleTurnAgentLoop` | Optional one-shot RAG | One generation, or search then one more generation | Optional `<search>` / `<information>` / `<answer>` trace |
| `SearchAgentLoop` | Multi-turn search workflow | Repeated generate → search/fetch → observe | Full XML action trajectory for RL/SFT |
| `ToolAgentLoop` | Through registered tools | Repeated generate → tool calls → observe | Generic function-call trajectory with tool messages |

Briefly:

- `PlainGenerationLoop` is the baseline local-generation path. It sends the
  prompt to the model once and prints the raw answer, so it is the fastest way
  to check model loading and decoding.
- `SingleTurnAgentLoop` is the small RAG path. It can retrieve once, inject the
  retrieved evidence, and ask the model for one final answer.
- `SearchAgentLoop` is the full agentic-search path. It lets the model plan,
  search or fetch over multiple turns, inspect observations, and produce a
  trajectory suitable for RL/SFT experiments.
- `ToolAgentLoop` is the generic function-calling path. It uses `Tool`,
  `FunctionTool`, and `ToolParser` to expose JSON-schema-described tools,
  execute parsed tool calls, and feed tool results back into the conversation.

#### 1. Start local retrieval

In terminal 1, start the retrieval server and leave it running:

```bash
conda install -c conda-forge faiss-cpu
```

```bash
python3 -m src.retrieval.servers.retrieval \
  --model_path BAAI/bge-base-en-v1.5 \
  --index_path indexes/bge_Flat.index \
  --corpus_path data/corpus.jsonl \
  --retrieval_method bge \
  --device cpu \
  --workers 1 \
  --topk 5 \
  --host 0.0.0.0 --port 8000
```

#### 2. Test retrieval with curl

In terminal 2, test the running server:

Health check:

```bash
curl -i -sS http://127.0.0.1:8000/health
```

Single-query retrieval:

```bash
curl -i -sS -X POST http://127.0.0.1:8000/retrieve \
  -H "Content-Type: application/json" \
  -d '{"query":"What is FAISS?","top_k":5}'
```

Batch retrieval, matching the agent-loop request shape:

```bash
curl -i -sS -X POST http://127.0.0.1:8000/retrieve \
  -H "Content-Type: application/json" \
  -d '{"queries":["dense retrieval","BM25 sparse retrieval"],"topk":3}'
```

Retrieval with scores:

```bash
curl -i -sS -X POST http://127.0.0.1:8000/retrieve \
  -H "Content-Type: application/json" \
  -d '{"query":"hybrid retrieval","top_k":5,"return_scores":true}'
```

#### 3. Run the smoke tests

Start with `PlainGenerationLoop` to confirm the local model loads and can
generate text. This does not need the retrieval server.

```bash
python3 -m examples.run_agentic_search \
  --mode single \
  --question "What is FAISS?" \
  --model Qwen/Qwen2.5-1.5B-Instruct \
  --local --device cpu \
  --max_tokens 64 --temperature 0 \
  --generation_timeout_seconds 30
```

Then run `SearchAgentLoop` when you want the project-native retrieval-assisted
trajectory used by the RL/SFT code. Keep the retrieval server running first.

```bash
python3 -m examples.run_agentic_search \
  --mode search \
  --question "What is FAISS?" \
  --model Qwen/Qwen2.5-1.5B-Instruct \
  --local --device cpu \
  --search_url http://127.0.0.1:8000/retrieve \
  --topk 2 \
  --max_turns 2 --max_search_limit 1 \
  --max_tokens 128 --temperature 0 \
  --generation_timeout_seconds 45
```

`ToolAgentLoop` is the supported generic function/tool-calling path. Use it
when you want to test `Tool` / `FunctionTool` schemas and `ToolParser` formats
(`json`, `hermes`, or `llama3`) rather than the XML search-agent trajectory.

### Key flags

| Flag | Default | Purpose |
|------|---------|---------|
| `--mode` | `search` | `single` / `search` / `tool` |
| `--local` | off | Load model in-process (no vLLM) |
| `--model_routing` | `off` | `off` / `intent`; choose a generation model before loading it |
| `--fast_model` | unset | Low-latency model for simple QA / navigation when model routing is enabled |
| `--balanced_model` | unset | Medium model for synthesis / recommendation when model routing is enabled |
| `--reasoning_model` | unset | Larger model for complex or high-stakes intents when model routing is enabled |
| `--device` | `auto` | `cpu` / `cuda` / `mps` (local only) |
| `--allow_unsafe_mps` | off | Unlock MPS; disabled by default (segfault risk on some models) |
| `--vllm_url` | `http://localhost:8080` | OpenAI-compatible server base URL |
| `--search_url` | `http://localhost:8000/retrieve` | Retrieval endpoint |
| `--max_turns` | `8` | Max agent turns |
| `--max_search_limit` | 0 (= max_turns) | Cap on search rounds |
| `--max_tokens` | `512` | Max new tokens per generation step |
| `--generation_timeout_seconds` | `120` | Local generation wall-clock timeout; use `0` to disable |
| `--no_evidence_gate` | off | Allow `<answer>` before evidence is sufficient |

### Model routing

`--model_routing intent` reuses the intent classifier as a lightweight model
router. The router chooses the generation model before loading the tokenizer or
server manager, so the agent loops themselves do not need to change.

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

Default intent-to-model tiers:

| Intent | Route | Typical use |
|--------|-------|-------------|
| `qa`, `navigate` | `fast_model` | Simple factual answers or navigation |
| `recommendation` | `balanced_model` | Synthesis and comparison |
| `purchase` | `reasoning_model` | Higher-stakes recommendation / buying decisions |

If confidence is below `--model_routing_min_confidence`, or if a route-specific
model is not provided, the CLI falls back to `--model`.

---

## Reward Function & RL Training

### Reward presets

| Preset | Formula | When to use |
|--------|---------|-------------|
| `sparse_final_only()` | `correctness` | Phase 0 — baseline |
| `simple_sparse_with_search_penalty()` | `correctness − 0.02 × num_searches` | Phase 1 — recommended first-pass |
| `second_pass()` | Phase 1 + citation support + unsupported-claim penalty + dup-query | Phase 2 — after Phase 1 converges |

Runnable reference:

```bash
python3 -m examples.run_grpo_training_pipeline
```

### GRPO advantage presets

Use `GRPOAdvantageConfig.outcome_only()` for raw mean-centering, or
`GRPOAdvantageConfig.std_normalized()` for the default lower-variance
normalization.

### Reward components

| Component | Config key | Description |
|-----------|-----------|-------------|
| Answer correctness | `correctness_weight` | `judge_fn(final_answer, ground_truth)` |
| Citation support | `citation_support_weight` | Fraction of retrieved results cited via `[R{r}Q{q}D{d}]` |
| Subquestion coverage | `subquestion_coverage_weight` | Fraction of declared subquestions with sufficient evidence |
| Search quality | `search_quality_weight` | Evidence sufficiency × query quality |
| Per-search penalty | `per_search_penalty` | Per search round (Phase 1) |
| Duplicate-query penalty | `duplicate_query_penalty` | Per repeated query |
| Budget penalty | `budget_penalty` | Fired once when `rounds_used / max_search_rounds ≥ threshold` |
| Unsupported-claim penalty | `unsupported_claim_penalty` | Agent searched + got results + cited nothing |
| Fetch usefulness reward | `fetch_usefulness_reward` | Fetched pages cited in final answer |

---

## Infrastructure

### Local Retrieval Server

The same `/retrieve` service can serve dense FAISS indexes or sparse BM25 indexes.
Dense retrieval defaults to `device="cpu"` so it does not compete with the trainer for GPU memory.

```bash
# Dense FAISS retrieval
python3 -m src.retrieval.servers.retrieval \
  --model_path intfloat/e5-base-v2 \
  --index_path indexes/e5_Flat.index \
  --corpus_path data/corpus.jsonl \
  --retrieval_method e5 \
  --device cpu \
  --workers 1

# Sparse BM25 retrieval
python3 -m src.retrieval.servers.retrieval \
  --index_path indexes/bm25 \
  --corpus_path data/corpus.jsonl \
  --retrieval_method bm25 \
  --workers 1

# Trainer-friendly single-query
curl -X POST http://localhost:8000/retrieve \
  -H "Content-Type: application/json" \
  -d '{"query": "Nobel Prize Physics 2024", "top_k": 5}'

# Legacy batch query
curl -X POST http://localhost:8000/retrieve \
  -H "Content-Type: application/json" \
  -d '{"queries": ["query 1", "query 2"], "topk": 3}'
```

Both request shapes are supported; the server normalises responses automatically.

### Building an Index

```bash
# Dense FAISS (E5 / BGE). Use --faiss_type Flat for exact search.
python3 -m src.retrieval.index_builder \
  --retrieval_method e5 \
  --model_path intfloat/e5-base-v2 \
  --corpus_path data/corpus.jsonl \
  --faiss_type Flat \
  --save_dir indexes/

# CPU ANN indexing with HNSW64. This is faster than Flat search on large
# corpora, but approximate, so recall can be lower when top_k is small.
python3 -m src.retrieval.index_builder \
  --retrieval_method e5 \
  --model_path intfloat/e5-base-v2 \
  --corpus_path data/corpus.jsonl \
  --faiss_type HNSW64 \
  --hnsw_ef_construction 200 \
  --save_dir indexes/

python3 -m src.retrieval.servers.retrieval \
  --retrieval_method e5 \
  --model_path intfloat/e5-base-v2 \
  --index_path indexes/e5_HNSW64.index \
  --corpus_path data/corpus.jsonl \
  --hnsw_ef_search 128

# BM25 (requires Java)
python3 -m src.retrieval.index_builder \
  --retrieval_method bm25 \
  --corpus_path data/corpus.jsonl \
  --save_dir indexes/
```

### Web Search Providers

`src.tools.search` exposes a provider router for `retrieval`, `google`, `bing`,
and `serpapi`. Missing API keys return structured tool errors instead
of crashing the caller.

The repo also includes standalone online search servers for SerpAPI and Google
Custom Search. SerpAPI is the recommended backend for large RL training runs
because it can route to multiple engines such as Google, Bing, and Baidu, while
Google Custom Search has a hard monthly quota.

```bash
# SerpAPI online search server
export SERP_SEARCH_URL="https://serpapi.com/search"
export SERP_API_KEY="..."  # https://serpapi.com/

python3 -m src.search.serp_search_server \
  --search_url "$SERP_SEARCH_URL" \
  --topk 3 \
  --serp_api_key "$SERP_API_KEY"

# Google Custom Search server
export GOOGLE_API_KEY="..."  # https://developers.google.com/custom-search/v1/overview
export GOOGLE_CSE_ID="..."   # Google Custom Search Engine ID

python3 -m src.search.google_search_server \
  --api_key "$GOOGLE_API_KEY" \
  --topk 5 \
  --cse_id "$GOOGLE_CSE_ID" \
  --snippet_only
```

```python
from src.tools.search import search_for_tool_string

text = await search_for_tool_string(
    "agentic search with retrieval",
    provider="bing",
    page_size=5,
)
```

### Retrieval + Rerank Server

```bash
python3 -m src.retrieval.servers.retrieval_rerank \
  --retriever_model intfloat/e5-base-v2 \
  --index_path indexes/e5_Flat.index \
  --corpus_path data/corpus.jsonl \
  --retrieval_method e5 \
  --retrieval_topk 10 \
  --retrieval_query_batch_size 128 \
  --rerank_topk 3
```

For local BM25 retrieval with the same reranking layer, use
`--retrieval_method bm25` and omit `--retriever_model`. The service retrieves
each request in one batched backend call, then reranks all candidate
query-document pairs with the configured cross-encoder batch size. Responses
include rerank scores by default:

```json
{"result": [[{"document": {"contents": "\"Title\"\nBody"}, "score": 0.91}]]}
```

---

## Testing

```bash
pip install -r requirements.txt pytest httpx

# All tests
python3 -m pytest -v

# Unit tests only (no server, no model weights required)
python3 -m pytest tests/unit/ -v

# Agent loops and tool calling
python3 -m pytest tests/unit/test_agent_loop.py \
  tests/unit/test_api_tools.py tests/unit/test_search_tools.py -v

# Retrieval servers, clients, indexing, rerank, and text processing
python3 -m pytest tests/unit/test_search_app.py tests/unit/test_retrieval_server.py \
  tests/unit/test_search_client.py tests/unit/test_index_builder.py \
  tests/unit/test_rerank.py tests/unit/test_vocabulary.py -v

# Runnable examples and CLI helpers
python3 -m pytest tests/unit/test_readme_examples.py \
  tests/unit/test_run_agentic_search.py tests/unit/test_intent_classifier.py -v

# Training, SFT, rewards, and PPO/GRPO/REINFORCE helpers
python3 -m pytest tests/unit/test_data.py tests/unit/test_sft.py \
  tests/unit/test_reward.py tests/unit/test_grpo.py \
  tests/unit/test_llm_agent_generation.py tests/unit/test_llm_agent_tensor_helper.py -v
```

### PPO / GRPO / REINFORCE pipeline tests

```bash
# Trajectory output and log-prob alignment
python3 -m pytest tests/unit/test_llm_agent_generation.py \
  -k "trajectory_log_prob or compute_log_prob or log_probs_alignment" -v

# Policy loss (clipped surrogate, KL penalty, entropy bonus, REINFORCE)
python3 -m pytest tests/unit/test_llm_agent_generation.py \
  -k "compute_trajectory_policy_loss or compute_policy_loss" -v

python3 -m pytest tests/unit/test_grpo.py \
  -k "reinforce" -v

# GRPO advantage computation
python3 -m pytest tests/unit/test_llm_agent_generation.py \
  -k "assign_group_relative_advantages" -v

python3 -m pytest tests/unit/test_grpo.py \
  -k "compute_batch_advantages or outcome_advantage" -v

# Group rollout scoring (score_group_rollout, format_group_rollout)
python3 -m pytest tests/unit/test_llm_agent_generation.py \
  -k "score_group_rollout or format_group_rollout or format_scored_group" -v

# End-to-end training step
python3 -m pytest tests/unit/test_llm_agent_generation.py \
  -k "run_grpo_training_step" -v

# Async rollout (rollout-level N×G concurrency)
python3 -m pytest tests/unit/test_llm_agent_generation.py \
  -k "async_run_prompt_rollout_group or async_run_grpo_training_step" -v

# Safety constraints (allowed_actions parser, penalty breakdown)
python3 -m pytest tests/unit/test_llm_agent_generation.py \
  -k "safety_penalties or safety_config" -v

# JSONL export
python3 -m pytest tests/unit/test_llm_agent_generation.py \
  -k "save_training_batch_jsonl" -v

# Reward function (sparse, shaped, batch judge)
python3 -m pytest tests/unit/test_reward.py -v

# Run all PPO/GRPO tests at once
python3 -m pytest tests/unit/test_llm_agent_generation.py \
  tests/unit/test_reward.py tests/unit/test_grpo.py -v
```

### Test coverage by area

| File | What is tested |
|------|---------------|
| `test_llm_agent_generation.py` | Trajectory logging; `RolloutTrajectory` fields; `trajectory_log_prob_pack` alignment; `compute_log_prob` store/overwrite guard; `compute_trajectory_policy_loss` clipped surrogate, KL penalty, length-mismatch guard; `compute_policy_loss` entropy bonus, action-type weights, policy-family breakdown; `assign_group_relative_advantages` mean-center + std-norm modes; `score_group_rollout`; `format_group_rollout` diversity display; `collate_scored_rollouts_for_training` padding; `run_grpo_training_step` end-to-end; `async_run_prompt_rollout_group` N×G fan-out, per-rollout seeds, result ordering; `async_run_grpo_training_step`; `GRPORolloutSafetyConfig` defaults; `apply_rollout_safety_penalties` (invalid action, repeated query, excess search); `apply_safety_penalties_to_scored_rollouts` (allowed-actions parser, penalty breakdown, advantage recompute); `save_training_batch_jsonl` (write/append/overwrite, trajectory field, components); `_run_one_rollout` extraction; action parsing; search payload; tensor helper |
| `test_reward.py` | `normalize_answer_text`; `simple_sparse_correctness_reward`; `compute_batch_sparse_token_rewards`; `SearchRewardConfig` presets (`sparse_final_only`, `simple_sparse_with_search_penalty`, `second_pass`); `SearchRewardFunction.compute` full components; unsupported-claim penalty; `assign_grpo_outcome_token_advantages`; batch judge dispatch |
| `test_grpo.py` | `build_grpo_sampling_params` temperature diversification; `score_prompt_group` reward + advantage; `score_prompt_batch` batching; `compute_batch_advantages` within-group normalisation, cross-group independence, single-sample groups |
| `test_agent_loop.py` | `SearchAgentLoop` multi-turn; plan, parallel search, subquestions, fetch, gating; repeated-query dedup; search-round limit; cache and metrics |
| `test_api_tools.py` | OpenAPI schema parsing; API provider registry; derived tool schemas; API invocation request routing |
| `test_search_tools.py` | Search provider routing; tool-string formatting; detail fetch; `build_search_tool` |
| `test_data.py` | Prompt dataset normalization, dataloader padding, prompt batch conversion |
| `test_retrieval_server.py` | Single-query + batch-query request shapes; `--device` / `--workers` CLI flags |
| `test_sft.py` | Full action-trace SFT example construction |
| `test_search_client.py` | Session reuse; `results` / `result` response shape normalisation |
| `test_intent_classifier.py` | `IntentPipeline` train / save / load; `resolve_search_settings` |
| `test_run_agentic_search.py` | Local model config validation; MPS guard; `LocalServerManager` |
| `test_readme_examples.py` | Runnable README examples with fake model/search backends |
| `test_llm_agent_tensor_helper.py` | Padding conversion; batch re-expansion |
| `test_rerank.py` | `SentenceTransformerReranker.rerank` |
| `test_index_builder.py` | `IndexBuilderConfig.validate`, pooling methods |
| `test_vocabulary.py` | Tokenization, keyword extraction |
| `test_search_app.py` | `/health`, `/retrieve` endpoints |

---

## Agentic Search Loop

### Registered loops

| Name | Class | Description |
|------|-------|-------------|
| `"plain_generation"` | `PlainGenerationLoop` | Plain one-shot model generation; backs CLI `--mode single` |
| `"single_turn_agent"` | `SingleTurnAgentLoop` | One-shot retrieval-assisted RAG flow |
| `"search_agent"` | `SearchAgentLoop` | Project-native XML search-agent / RL trajectory format |
| `"tool_agent"` | `ToolAgentLoop` | Generic function/tool calling with parallel tool execution |

### XML protocol

The training-facing trace follows a compact ReAct shape:

```xml
<think>decide whether to answer or search; plan the next useful action</think>
<search>precise query when external evidence is needed</search>
<information>environment-injected evidence only; the model must not write this</information>
<answer>final response grounded in evidence</answer>
```

`<information>` is environment output and is masked out of policy/SFT action loss.

| Tag | Direction | Purpose |
|-----|-----------|---------|
| `<think>` | model → loop | Reasoning step — decide answer vs search |
| `<search_decision>answer\|search</search_decision>` | model → loop | Declare retrieval intent |
| `<subquestions>` | model → loop | Register named research tracks |
| `<search>query</search>` | model → loop | Single query |
| `<searches>` | model → loop | Multiple parallel queries |
| `<fetch>url1, url2</fetch>` | model → loop | Fetch full pages |
| `<answer>` | model → loop | Final answer |
| `<information>` | loop → model | Search results with citation labels `[R{r}Q{q}D{d}]` |
| `<search_evaluation>` | loop → model | Sufficiency verdict + per-query feedback |

### Direct use

For importable `SearchAgentLoop` wiring, use the maintained example module
instead of pasting setup code into the shell:

```bash
python3 -m examples.run_search_agent_loop
```

The module exposes `run_search_agent_loop_example(...)` for tests or notebooks
and also runs as a deterministic CLI smoke test with fake model/search backends.
It is covered by `tests/unit/test_readme_examples.py` so README-facing code
stays runnable.

For a deterministic screenshot-style trace with repeated
`<think>` → `<search>` → `<information>` steps, run:

```bash
python3 -m examples.run_search_trace_workflow
```

That example uses scripted fake model/search backends to show the workflow
shape without requiring a local LLM or retrieval server.

### Full-trace SFT

Build an SFT training example from the same deterministic search trace:

```bash
python3 -m examples.run_search_trace_workflow --sft
```

### Output fields (`AgentLoopOutput`)

| Field | Type | Description |
|-------|------|-------------|
| `prompt_ids` | `list[int]` | Tokenised prompt for the final turn |
| `response_ids` | `list[int]` | All generated token IDs across turns |
| `response_mask` | `list[int]` | `1` for every response token |
| `num_turns` | `int` | Generation steps taken |
| `final_answer` | `str \| None` | Content of the last accepted `<answer>` tag |
| `metrics` | `dict[str, float]` | `search_rounds`, `repeated_search_queries`, `invalid_action_count`, `rounds_used`, … |
| `group_id` | `str \| None` | Prompt-group ID for GRPO grouped rollouts |
| `rollout_index` | `int \| None` | Rollout index within a prompt group |
| `context` | `AgentContext` | Full search state — all rounds, tasks, results |
| `action_trace` | `str \| None` | Concatenated assistant XML trace |

### Metrics (`output.metrics`)

| Key | Meaning |
|-----|---------|
| `search_rounds` | Rounds that hit the retrieval server |
| `search_queries` | Individual queries dispatched |
| `repeated_search_queries` | Queries skipped (repeated) |
| `invalid_action_count` | Malformed XML / disallowed tags |
| `rounds_used` | Alias of `search_rounds` for reward computation |
| `repeated_query_ratio` | `repeated / (dispatched + repeated)` |
| `budget_used_ratio` | `rounds_used / max_search_rounds` |
| `search_quality_score` | Average per-round evaluator-approved fraction |
| `answer_when_evidence_insufficient` | `1.0` if answered before evidence was sufficient |
| `search_budget_exhausted_without_answer` | `1.0` if hit budget limit without answering |

---

## Notes

- Google Custom Search, Bing, and SerpAPI usage are subject to their respective quota and billing rules.
- Some result pages may block scraping or return little usable text.
- Empty or invalid queries return empty result lists.
- `DenseRetriever` defaults to `device="cpu"` to avoid competing with the trainer GPU — set `--device cuda` only on a dedicated retrieval node.
- `SparseRetriever` uses Pyserini/Lucene BM25 indexes, so Java must be available when serving BM25 locally.
