# Agentic-Search-GRPO

A FastAPI codebase for search-backed retrieval services, multi-turn agentic research loops, full-trace SFT targets, and end-to-end GRPO/PPO RL training.

- Google Custom Search and SerpAPI search servers
- Dense (FAISS) and sparse (BM25) retrieval with optional reranking
- `SearchAgentLoop`: plan → adaptive search decision → subquestions → parallel queries → evidence evaluation → fetch → cited answer
- `SearchRewardFunction`: strategy-aware reward signal + GRPO within-group advantage normalisation
- Full action-trace SFT support: train on `<plan> ... <answer>`, not only the final answer
- **End-to-end GRPO/PPO training loop**: rollout → reward → advantage → log probs → clipped loss → optimizer step
- Async rollout with rollout-level concurrency (`N_prompts × G` parallel tasks) to overlap HTTP search I/O

## Project Structure

```text
src/
  run_agentic_search.py          # CLI + importable entry point for all agent loop flows
  train_intent_classifier.py     # Offline: train and save the intent classifier (.pt)
  generate_intent_examples.py    # Offline: generate intent training examples from corpus
  agent_loop/
    agent_loop.py            # AgentLoopBase, AgentLoopConfig, AgentLoopOutput
    context.py               # SearchResult, SearchContext, AgentContext
    evaluation.py            # SearchResultEvaluator, SearchEvaluationConfig
    grpo.py                  # Prompt-group rollout sampling + within-group scoring helpers
    intent_classifier.py     # IntentPipeline: train / save / load + resolve_search_settings
    reward.py                # SearchRewardFunction, SearchRewardConfig — reward + GRPO advantages
    search_agent_loop.py     # SearchAgentLoop (registered as "search_agent")
    search_client.py         # async aiohttp client with session reuse for /retrieve and /fetch
    single_turn_agent_loop.py
    sft.py                   # Build supervised examples from full search-agent trajectories
    tool.py                  # Tool, FunctionTool — tool abstraction and JSON schema
    tool_agent_loop.py       # ToolAgentLoop (registered as "tool_agent")
    tool_parser.py           # Hermes / Llama3 / JSON tool-call parsers
  llm_agent/
    generation.py            # LLMGenerationManager — rollout, log probs, GRPO loss, async
    tensor_helper.py         # TensorConfig, TensorHelper — padding and batch helpers
  search/
    search_app.py
    google_search_server.py
    index_builder.py
    rerank.py
    retrieval.py             # DenseRetriever (CPU default to avoid VRAM contention)
    retrieval_rerank_server.py
    retrieval_server.py      # FastAPI /retrieve — single-query + batch modes
    serp_search_server.py
    text_processor.py        # config-driven cleanup / segmentation for structured text
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

---

## PPO / GRPO Training Pipeline

The full RL training loop is implemented in `src/llm_agent/generation.py` and `src/agent_loop/`. The pipeline follows these 10 steps:

```
1. run_llm_loop()          → trajectory (RolloutTrajectory + SearchTrajectoryLog)
2. SearchTool              → FastAPI /retrieve (EndpointRetriever)
3. G=4 rollouts/prompt     → run_prompt_rollout_group / async_run_prompt_rollout_group
4. reward_fn               → SearchRewardFunction scores each rollout
5. compute_grpo_advantage  → assign_group_relative_advantages
6. save_training_batch_jsonl → JSONL training data
7. compute_log_probs       → trajectory_log_prob_pack (prompt-aligned)
8. GRPO loss               → compute_trajectory_policy_loss / compute_policy_loss
9. run_grpo_training_step  → end-to-end on small model + small data
10. async                  → async_run_grpo_training_step (N×G concurrent rollouts)
```

### Step 1 — Trajectory output from `run_llm_loop`

```python
from src.llm_agent import LLMGenerationManager, GenerationConfig

manager = LLMGenerationManager(
    tokenizer=tokenizer,
    config=GenerationConfig(max_turns=3, max_search_rounds=3),
    generation_backend=actor_backend,
)

final_batch, trajectory_turns = manager.run_llm_loop(
    gen_batch=gen_batch,
    search_mode="local",   # "local" | "wiki" | "google" | "simulate"
)

# Per-trajectory structured log
log = final_batch.non_tensor_batch["trajectory_logs"][0]
print(log)                 # compact trace: "Step 1: search ... Step 2: answer ..."
print(log.to_dict())       # dict for JSONL / wandb logging

# Aligned token arrays (prompt + response length, same size)
from src.llm_agent import trajectory_log_prob_pack
traj = final_batch.non_tensor_batch["trajectories"][0]
pack = trajectory_log_prob_pack(traj)
# pack["tokens"], pack["attention_mask"], pack["response_mask"], pack["old_log_probs"]
```

### Step 2 — SearchTool → FastAPI `/retrieve`

The retrieval server runs separately from the trainer to avoid VRAM contention.
`DenseRetrieverConfig` defaults to `device="cpu"`.

```bash
# Start the retrieval server (CPU, 1 worker)
python3 -m src.search.retrieval_server \
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

```python
# Sequential (reproducible, RNG save/restore between rollouts)
grouped = manager.run_prompt_rollout_group(
    prompt_batch,
    search_mode="local",
    sampling_params={"temperature": 0.8, "top_p": 0.95},
    num_rollouts=4,
    base_seed=42,
)
# grouped[i].group_id  — shared across 4 rollouts
# grouped[i].rollout_index  — 0, 1, 2, 3
# grouped[i].final_output.trajectory_logs[0]  — per-rollout trace

# Async (N_prompts × G tasks concurrent — overlaps HTTP search I/O)
import asyncio
from src.llm_agent import async_run_prompt_rollout_group

all_groups = asyncio.run(
    async_run_prompt_rollout_group(
        manager,
        [single_batch_0, single_batch_1, single_batch_2],
        search_mode="local",
        sampling_params={"temperature": 0.8},
        num_rollouts=4,
        base_seed=0,
        # 12 concurrent tasks: each (prompt_i, rollout_j) is one thread-pool task
        # seed for (prompt i, rollout j) = base_seed + i*num_rollouts + j
    )
)
# all_groups[i] == list[GroupedRolloutBatch] for prompt i, sorted by rollout_index
```

### Step 4 — `reward_fn` scores each rollout

```python
from src.agent_loop import SearchRewardConfig, SearchRewardFunction
from src.agent_loop.reward import simple_sparse_correctness_reward

# Phase 0 — strict sparse (only final answer correctness)
reward_fn = SearchRewardFunction(SearchRewardConfig.sparse_final_only())

# Phase 1 — recommended first-pass (correctness - 0.02 * num_searches)
reward_fn = SearchRewardFunction(
    SearchRewardConfig.simple_sparse_with_search_penalty(per_search_penalty=-0.02)
)

# Phase 2 — full shaping (citation support, unsupported-claim penalty, dup query)
reward_fn = SearchRewardFunction(SearchRewardConfig.second_pass())

# Score one rollout
reward = reward_fn.compute(output, ground_truth=gt, judge_fn=simple_sparse_correctness_reward)

# Batch judge (fewer LLM API calls for LLM-based scoring)
from src.agent_loop.reward import compute_batch_sparse_token_rewards

token_rewards = compute_batch_sparse_token_rewards(
    outputs, ground_truths, judge_fn=simple_sparse_correctness_reward
)

# Full penalty breakdown per rollout
components = reward_fn.reward_components(output, ground_truth=gt, judge_fn=simple_sparse_correctness_reward)
# {"correctness": 1.0, "search_penalty": -0.04, "total": 0.96, ...}
```

### Step 5 — `compute_grpo_advantage`

```python
from src.llm_agent import assign_group_relative_advantages

# Mean-centering only (DeepSeek-R1 style)
scored = assign_group_relative_advantages(
    grouped,
    rewards=[1.0, 0.7, 0.0, 0.0],
    normalize=False,
)
# [0.575, 0.275, -0.425, -0.425]

# Std-normalized (default, more stable)
scored = assign_group_relative_advantages(
    grouped,
    rewards=[1.0, 0.7, 0.0, 0.0],
    normalize=True,  # (reward - mean) / (std + 1e-8)
)

# Full pipeline: score + advantage in one call
from src.llm_agent import score_group_rollout
scored = score_group_rollout(
    grouped,
    ground_truth="Hopfield and Hinton",
    judge_fn=simple_sparse_correctness_reward,
    reward_fn=reward_fn,
)
```

### Step 6 — Save JSONL training data

```python
from src.llm_agent import save_training_batch_jsonl

# Write / overwrite
n = save_training_batch_jsonl(scored_rollouts, "data/train.jsonl")

# Append across training steps
n = save_training_batch_jsonl(scored_rollouts, "data/train.jsonl", append=True)
```

Each JSONL record contains: `group_id`, `rollout_index`, `reward`, `advantage`,
`reward_components`, `trajectory` (full `SearchTrajectoryLog.to_dict()`),
`tokens`, `response_mask`, `old_log_probs` (all prompt+response aligned).

### Step 7 — Offline `compute_log_probs`

```python
# Compute old_log_probs (frozen at rollout time)
manager.compute_log_prob(training_batch, backend=rollout_backend, store_key="old_log_probs")

# Compute new_log_probs (current policy being trained)
manager.compute_log_prob(training_batch, backend=train_backend, store_key="new_log_probs")

# Optional: reference policy for KL penalty
manager.compute_log_prob(training_batch, backend=ref_backend, store_key="ref_log_probs")

# Extract aligned arrays from one trajectory
from src.llm_agent import trajectory_log_prob_pack

pack = trajectory_log_prob_pack(traj)
# All 4 lists are the same length: len(prompt_token_ids) + len(response_token_ids)
# old_log_probs: [0.0, ..., 0.0,  -0.5, -0.3, -1.2, ...]
#                 ^^^^prompt^^^^   ^^^^^^^^^response^^^^^^^^^
# response_mask: [0, ..., 0,       1,    1,    1,   0, ...]  (0 for <information>)
```

**Alignment rule**: `old_log_probs` is prepended with `len(prompt_token_ids)` zeros so
it aligns with `tokens` and `attention_mask` without any slicing in the loss loop.

### Step 8 — GRPO loss

```python
from src.llm_agent import PPOPolicyLossConfig, compute_trajectory_policy_loss

# Trajectory-level loss (no GPU batch, no tokenizer — pure Python/Torch)
result = compute_trajectory_policy_loss(
    new_log_probs=pack["new_log_probs"],
    old_log_probs=pack["old_log_probs"],
    advantages=pack["advantages"],
    response_mask=pack["response_mask"],
    ref_log_probs=pack.get("ref_log_probs"),   # optional KL anchor
    clip_epsilon=0.2,
    kl_beta=0.01,
)
# result["grpo_policy_loss"], ["kl_penalty"], ["total_loss"], ["clip_fraction"], ["mean_ratio"]

# Batch-level loss with action-type weights and entropy bonus
loss = manager.compute_policy_loss(
    training_batch,
    config=PPOPolicyLossConfig(
        clip_epsilon=0.2,
        kl_coefficient=0.01,
        entropy_coefficient=0.001,        # prevents mode collapse
        action_type_weights={"search": 1.5, "answer": 1.0},
    ),
)
loss.backward()
optimizer.step()
```

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

### Step 9 — End-to-end on small model + small data

```python
result = manager.run_grpo_training_step(
    prompt_batch,
    search_mode="local",
    sampling_params={"temperature": 0.8, "top_p": 0.95},
    judge_fn=simple_sparse_correctness_reward,
    num_rollouts=4,
    reward_fn=SearchRewardFunction(
        SearchRewardConfig.simple_sparse_with_search_penalty()
    ),
    old_backend=rollout_policy,    # frozen at rollout time
    new_backend=actor_policy,      # being trained
    ref_backend=reference_policy,  # optional KL anchor
    loss_config=PPOPolicyLossConfig(clip_epsilon=0.2, kl_coefficient=0.01),
    safety_config=GRPORolloutSafetyConfig(
        allowed_actions=("search", "answer"),
        max_search_rounds=3,
        invalid_action_penalty=-0.2,
        repeated_query_penalty=-0.1,
    ),
    optimizer=optimizer,
)

print(result.mean_reward)       # average reward across all rollouts this step
print(result.mean_advantage)    # average advantage
print(result.loss)              # scalar Torch tensor (already backpropped)
print(result.optimizer_stepped) # True if optimizer.step() was called
```

What `run_grpo_training_step` does internally:

1. For each prompt: `run_prompt_rollout_group()` → G rollouts
2. `score_group_rollout()` — reward + GRPO advantage per rollout
3. `apply_safety_penalties_to_scored_rollouts()` — penalize unsafe tool use
4. `collate_scored_rollouts_for_training()` — merge into one `SearchBatch`
5. `compute_log_prob()` for `old_log_probs`, `new_log_probs`, `ref_log_probs`
6. `compute_policy_loss()` — clipped surrogate + KL + entropy
7. `loss.backward(); optimizer.step()`

### Step 10 — Async / distributed rollout

```python
import asyncio
from src.llm_agent import async_run_grpo_training_step

result = asyncio.run(
    async_run_grpo_training_step(
        manager,
        prompt_batch,
        search_mode="local",
        sampling_params={"temperature": 0.8},
        judge_fn=simple_sparse_correctness_reward,
        num_rollouts=4,
        reward_fn=reward_fn,
        optimizer=optimizer,
        max_workers=16,   # up to N_prompts × num_rollouts concurrent threads
    )
)
```

**Execution model** — `N_prompts × G` rollout tasks run concurrently in a `ThreadPoolExecutor`.
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

```python
from src.llm_agent import GRPORolloutSafetyConfig, apply_rollout_safety_penalties
from src.llm_agent import apply_safety_penalties_to_scored_rollouts

config = GRPORolloutSafetyConfig(
    max_search_rounds=3,       # force answer after 3 search rounds
    max_total_rounds=6,        # soft cap on total agent turns
    allowed_actions=("search", "answer"),   # <plan>/<fetch> = disallowed
    invalid_action_penalty=-0.2,            # per malformed/disallowed XML tag
    repeated_query_penalty=-0.1,            # per repeated search query
    excess_search_penalty=-0.1,             # per round over max_search_rounds
)

# Per-rollout (uses AgentLoopOutput.metrics)
adjusted_reward = apply_rollout_safety_penalties(reward, output, config=config)

# Batch (also checks trajectory steps for disallowed tags, stores breakdown in reward_components)
scored = apply_safety_penalties_to_scored_rollouts(
    scored_rollouts,
    config=config,
    normalize_advantages=True,   # re-derive advantages after penalty adjustment
)
# scored[i].reward_components keys added:
#   "invalid_action_penalty", "repeated_query_penalty",
#   "excess_search_penalty", "disallowed_action_penalty"
```

The `allowed_actions` parser inspects `trajectory.steps` for valid-XML-but-disallowed tags
(e.g. `<fetch>` or `<plan>` when only `["search", "answer"]` are permitted), applying
`invalid_action_penalty` per disallowed tag. This is separate from `invalid_action_count`
in metrics (which only catches malformed XML).

---

## Running the Agent

`src/run_agentic_search.py` is the unified entry point.

### vLLM / server-backed inference

```bash
python3 -m src.run_agentic_search \
    --mode search \
    --question "Compare dense vs sparse retrieval" \
    --model meta-llama/Llama-3.1-8B-Instruct \
    --vllm_url http://localhost:8080 \
    --search_url http://localhost:8000/retrieve
```

### Local inference

```bash
python3 -m src.run_agentic_search \
  --mode single \
  --question "What is FAISS?" \
  --model Qwen/Qwen2.5-1.5B-Instruct \
  --local --device cpu \
  --max_tokens 256 --temperature 0
```

### Key flags

| Flag | Default | Purpose |
|------|---------|---------|
| `--mode` | `search` | `single` / `search` / `tool` |
| `--local` | off | Load model in-process (no vLLM) |
| `--device` | `auto` | `cpu` / `cuda` / `mps` (local only) |
| `--allow_unsafe_mps` | off | Unlock MPS; disabled by default (segfault risk on some models) |
| `--vllm_url` | `http://localhost:8080` | OpenAI-compatible server base URL |
| `--search_url` | `http://localhost:8000/retrieve` | Retrieval endpoint |
| `--max_turns` | `6` | Max agent turns |
| `--max_search_limit` | 0 (= max_turns) | Cap on search rounds |
| `--max_tokens` | `512` | Max new tokens per generation step |
| `--no_evidence_gate` | off | Allow `<answer>` before evidence is sufficient |

---

## Reward Function & RL Training

### Reward presets

| Preset | Formula | When to use |
|--------|---------|-------------|
| `sparse_final_only()` | `correctness` | Phase 0 — baseline |
| `simple_sparse_with_search_penalty()` | `correctness − 0.02 × num_searches` | Phase 1 — recommended first-pass |
| `second_pass()` | Phase 1 + citation support + unsupported-claim penalty + dup-query | Phase 2 — after Phase 1 converges |

```python
from src.agent_loop import SearchRewardConfig, SearchRewardFunction
from src.agent_loop.reward import simple_sparse_correctness_reward

reward_fn = SearchRewardFunction(
    SearchRewardConfig.simple_sparse_with_search_penalty(per_search_penalty=-0.02)
)
reward = reward_fn.compute(output, ground_truth=gt, judge_fn=simple_sparse_correctness_reward)
```

### GRPO advantage presets

```python
from src.agent_loop import GRPOAdvantageConfig

# DeepSeek-R1 style: raw mean-centering
config = GRPOAdvantageConfig.outcome_only()
# advantage_i = reward_i − mean(group)

# Std-normalized (more stable training)
config = GRPOAdvantageConfig.std_normalized()
# advantage_i = (reward_i − mean) / (std + ε)
```

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

### Dense Retrieval Server

The retrieval server defaults to `device="cpu"` so it does not compete with the trainer for GPU memory.

```bash
python3 -m src.search.retrieval_server \
  --model_path intfloat/e5-base-v2 \
  --index_path indexes/e5_Flat.index \
  --corpus_path data/corpus.jsonl \
  --retrieval_method e5 \
  --device cpu \
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
# Dense (E5 / BGE)
python3 -m src.search.index_builder \
  --retrieval_method e5 \
  --model_path intfloat/e5-base-v2 \
  --corpus_path data/corpus.jsonl \
  --save_dir indexes/

# BM25 (requires Java)
python3 -m src.search.index_builder \
  --retrieval_method bm25 \
  --corpus_path data/corpus.jsonl \
  --save_dir indexes/
```

### Retrieval + Rerank Server

```bash
python3 -m src.search.retrieval_rerank_server \
  --retriever_model intfloat/e5-base-v2 \
  --index_path indexes/e5_Flat.index \
  --corpus_path data/corpus.jsonl \
  --retrieval_method e5 \
  --retrieval_topk 10 --rerank_topk 3
```

---

## Testing

```bash
pip install pytest httpx

# All tests
python3 -m pytest

# Unit tests only (no server, no model weights required)
python3 -m pytest tests/unit/ -v

# By module
python3 -m pytest tests/unit/test_llm_agent_generation.py -v
python3 -m pytest tests/unit/test_reward.py -v
python3 -m pytest tests/unit/test_grpo.py -v
```

### PPO / GRPO pipeline tests

```bash
# Trajectory output and log-prob alignment
python3 -m pytest tests/unit/test_llm_agent_generation.py \
  -k "trajectory_log_prob or compute_log_prob or log_probs_alignment" -v

# Policy loss (clipped surrogate, KL penalty, entropy bonus)
python3 -m pytest tests/unit/test_llm_agent_generation.py \
  -k "compute_trajectory_policy_loss or compute_policy_loss" -v

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
| `test_retrieval_server.py` | Single-query + batch-query request shapes; `--device` / `--workers` CLI flags |
| `test_sft.py` | Full action-trace SFT example construction |
| `test_search_client.py` | Session reuse; `results` / `result` response shape normalisation |
| `test_intent_classifier.py` | `IntentPipeline` train / save / load; `resolve_search_settings` |
| `test_run_agentic_search.py` | Local model config validation; MPS guard; `LocalServerManager` |
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
| `"single_turn_agent"` | `SingleTurnAgentLoop` | Single-turn retrieve-then-answer |
| `"search_agent"` | `SearchAgentLoop` | Multi-turn: plan → search → subquestions → fetch → cited answer |
| `"tool_agent"` | `ToolAgentLoop` | Multi-turn with parallel tool execution |

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

```python
from src.agent_loop import SearchAgentLoop, SearchAgentLoopConfig

loop = SearchAgentLoop(
    tokenizer=tokenizer,
    server_manager=server_manager,
    search_config=SearchAgentLoopConfig(
        search_url="http://localhost:8000/retrieve",
        topk=5, max_turns=8, max_search_limit=6,
    ),
)
output = await loop.run(
    messages=[{"role": "user", "content": "Compare dense vs sparse retrieval."}],
    sampling_params={"temperature": 0.7},
)
print(output.final_answer)
print(output.metrics)      # search_rounds, repeated_search_queries, rounds_used, ...
```

### Full-trace SFT

```python
from src.agent_loop import build_search_sft_example

example = build_search_sft_example(
    [{"role": "user", "content": "Compare dense vs sparse retrieval."}],
    output,
)
print(example.completion)  # <plan>...<search>...<answer>...
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

- Google Custom Search and SerpAPI usage are subject to their respective quota and billing rules.
- Some result pages may block scraping or return little usable text.
- Empty or invalid queries return empty result lists.
- `DenseRetriever` defaults to `device="cpu"` to avoid competing with the trainer GPU — set `--device cuda` only on a dedicated retrieval node.
