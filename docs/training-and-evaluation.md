# Training and evaluation

## Serving requests do not train models

`POST /api/agent` and `/api/agent/stream` perform routing, retrieval, tool execution, and model inference only. Even explicit `search_agent` or `tool_agent` mode loads a policy model for generation without updating its weights. A query containing `GRPO` is ordinary search input; it does not invoke the GRPO trainer. SFT, GRPO, and PPO run only through the offline commands in this guide. See [API request routing](request-routing.md) for serving-time dispatch.

Serving still uses indexes built by asynchronous ingestion/indexing jobs. Filter-aware and degraded branches can use the composed session-aware retrieval, ranking/reranking, and evidence-grounded inference pipeline; strong unfiltered auto-search retains its direct ranking, sufficiency gate, and provider fallback path. Neither retrieval nor reranking is a training step. No new serving API or request/response schema was introduced; offline trainers may produce model artifacts, but requests only load and infer with them.

[← Back to README](../README.md)

This guide covers dataset preparation, supervised and reinforcement-learning workflows, and benchmark evaluation.

## Runnable examples

### Agent CLI

| Mode | Loop | Needs retrieval server | Use it for |
|------|------|------------------------|------------|
| `single` | `PlainGenerationLoop` | No | Local generation smoke tests |
| `search` | `SearchAgentLoop` | Yes | Multi-turn RAG, SFT, and RL traces |
| `tool` | `ToolAgentLoop` | Yes | Structured tool-calling experiments |

```bash
# single — no retrieval server needed (plain generation)
# Apple Silicon: use --device mps --allow_unsafe_mps for ~50x faster inference
python3 -m examples.run_agentic_search \
  --mode single --question "What is FAISS?" \
  --model Qwen/Qwen2.5-1.5B-Instruct --local --device mps --allow_unsafe_mps \
  --allow_remote_model_downloads

# single with retrieval server — small models (≤3B) use --mode single; search/tool require 7B+ to emit structured tags
python3 -m examples.run_agentic_search \
  --mode single --question "What is FAISS?" \
  --model Qwen/Qwen2.5-1.5B-Instruct --local --device mps --allow_unsafe_mps \
  --search_url http://localhost:8001/retrieve --allow_remote_model_downloads

# search — 3B is the Mac sweet spot (~6 GB unified memory); 7B needs 16 GB+ and will swap
python3 -m examples.run_agentic_search \
  --mode search --question "What is RAG?" \
  --model Qwen/Qwen2.5-3B-Instruct --local --device mps --allow_unsafe_mps \
  --search_url http://localhost:8001/retrieve --allow_remote_model_downloads

# search — server-backed, requires vLLM on :8080 and retrieval on :8001
python3 -m examples.run_agentic_search \
  --mode search --question "Compare dense and sparse retrieval" \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --vllm_url http://localhost:8080 --search_url http://localhost:8001/retrieve
```

### Bamboogle evaluation

Always requires the retrieval server on port 8001.

```bash
# Smoke test — local model, 1 example, full trace printed
python3 -m examples.run_bamboogle_eval \
  --model Qwen/Qwen2.5-3B-Instruct --local --device mps --allow_unsafe_mps \
  --search_url http://localhost:8001/retrieve --limit 1 --print_trace \
  --allow_remote_model_downloads

# Full benchmark — Apple Silicon, requires SERP_API_KEY in .env
bin/run_bamboogle_eval.sh --limit 125
```

### PPO/GRPO reward

```bash
python3 -m examples.run_grpo_training_pipeline         # end-to-end reward + GRPO smoke test (no GPU, no model)

# Simulated-judge GRPO — actually updates a policy: bamboogle prompts → generate →
# SimulatedPreferenceJudge → GRPO step. No retrieval server; runs on CPU/MPS.
python3 -m examples.run_bamboogle_grpo_train \
  --model Qwen/Qwen2.5-0.5B-Instruct --device cpu \
  --allow_remote_model_downloads --steps 10
```

### Search pipeline with access filters

No live model or retrieval server is required.

```bash
python3 -m examples.run_search_pipeline
```

## Dataset preparation

```bash
# Offline local RAG smoke test (4 examples, existing 30-document demo corpus)
python3 -m examples.prepare_local_rag_smoke_dataset --topk 1 --preview

# Write compact RAG parquet after inspecting the preview
python3 -m examples.prepare_local_rag_smoke_dataset \
  --topk 1 --output_path data/local_rag_smoke.parquet
```

This command requires no retrieval server, network access, FlashRAG dataset, or retrieval caches.

Optional large-dataset workflows:

```bash
# Optional: Search-QA parquet preparation
python3 -m examples.prepare_search_qa_dataset \
  --dataset_name RUC-NLPIR/FlashRAG_datasets --dataset_config nq --local_dir data/nq_search

# Preview before writing
python3 -m examples.prepare_search_qa_dataset \
  --dataset_name RUC-NLPIR/FlashRAG_datasets --dataset_config nq \
  --splits test --max_examples 20 --preview --preview_rows 5

# Optional: full NQ RAG parquet preparation
# Requires an external Wikipedia corpus plus retrieval-cache JSON files keyed
# by NQ question. prepare_search_qa_dataset does not create these inputs.
python3 -m examples.prepare_search_rag_dataset \
  --dataset_name RUC-NLPIR/FlashRAG_datasets --dataset_config nq \
  --corpus_path data/wiki-18.jsonl \
  --train_retrieval_cache data/nq_train_retrieval_cache.json \
  --test_retrieval_cache data/nq_test_retrieval_cache.json \
  --topk 3 --local_dir data/nq_rag
```

## Training

The training pipeline is modular: generate trajectories → score with rewards → compute advantages → optimize.

| Task | Entry point |
|------|-------------|
| QA parquet preparation | `python3 -m examples.prepare_search_qa_dataset` |
| Training data (shell) | `bin/generate_training_data.sh` |
| Reward/GRPO smoke test | `python3 -m examples.run_grpo_training_pipeline` |
| Bamboogle benchmark eval | `python3 -m examples.run_bamboogle_eval` / `bin/run_bamboogle_eval.sh` |
| Reward function | `src/training/reward.py` |
| Simulated preference judge | `src/training/judge.py` |
| GRPO helpers | `src/training/grpo.py` |
| Online GRPO for HF LMs | `src/training/ppo/llm_grpo_trainer.py` |
| Agent-loop GRPO (full reward) | `src/training/ppo/search_agent_grpo_trainer.py` |
| PPO core | `src/training/ppo/core_algos.py` |
| Generation and policy loss | `src/model/generation.py` |
| Feedback-driven GRPO | `python3 -m examples.run_feedback_grpo` |
| SFT warm-start + GRPO | `python3 -m examples.run_sft_grpo` |
| Simulated-judge GRPO (policy update) | `python3 -m examples.run_bamboogle_grpo_train` |

### Fine-tune from user feedback

Train directly on thumbs-up/down sessions collected via `POST /api/feedback` (no GPU required for the smoke path; `--device mps` on Apple Silicon):

```bash
# Feedback-driven GRPO: load rated sessions from the web DB → reward with human_signal → update
python3 -m examples.run_feedback_grpo \
  --db_path data/feedback.sqlite3 \
  --model Qwen/Qwen2.5-1.5B-Instruct \
  --min_ratings 10 --human_feedback_weight 0.5 \
  --num_rollouts 4 --search_url http://localhost:8001/retrieve --device mps \
  --output_dir data/checkpoints/feedback_grpo/

# SFT warm-start (Phase 1, assistant-token-only CE on thumbs-up traces) then GRPO (Phase 2);
# --sft_epochs 0 skips Phase 1 and runs pure GRPO from the base model
python3 -m examples.run_sft_grpo \
  --db_path data/feedback.sqlite3 --model Qwen/Qwen2.5-1.5B-Instruct \
  --jsonl_path data/sft_pairs.jsonl \
  --sft_epochs 3 --sft_lr 2e-5 --sft_output_dir data/checkpoints/sft_warmstart/ \
  --grpo_output_dir data/checkpoints/sft_grpo/ --device mps
```

`load_feedback_examples` raises if fewer than `--min_ratings` rated sessions exist, so collect feedback first (thumbs in the UI, or `POST /api/feedback`). There is **no HTTP training endpoint** — fine-tuning is offline by design; the only backend endpoint in this loop is `POST /api/feedback` (see [Web Backend API](api-reference.md#web-backend-api)).

### Reward components

`SearchRewardFunction` uses these components:

| Component | Config field | What it measures |
|-----------|-------------|-----------------|
| Correctness | `correctness_weight` | Judge score against gold answer (EM / contains-match) |
| Citation support | `citation_support_weight` | Fraction of retrieved docs cited in the final answer |
| Subquestion coverage | `subquestion_coverage_weight` | Fraction of sub-questions with sufficient evidence |
| Search quality | `search_quality_weight` | Evaluator verdict + per-query search quality |
| Unnecessary search | `unnecessary_search_penalty` | Penalty per search round beyond the first |
| Unnecessary fetch | `unnecessary_fetch_penalty` | Penalty per fetched page not cited in the answer |
| Fetch usefulness | `fetch_usefulness_reward` | Bonus when fetched pages are cited in the final answer |
| Format compliance | `format_reward_weight` | Structural compliance in the final answer |
| Human feedback | `human_feedback_weight` | `human_signal` (±1.0) from thumbs-up/down sessions; `0.0` by default (off) |

Reward preset names: `sparse_final_only` | `simple_sparse_with_search_penalty` | `second_pass` | `third_pass_with_format` | `retriever_aware` (see `SearchRewardConfig` in `src/training/reward.py`).

**Four reward dimensions** — `reward_components()` also groups every term into four subtotals via `REWARD_DIMENSIONS`, emitted as `dim_correctness`, `dim_citation_support`, `dim_retrieval_quality`, `dim_search_efficiency` (and available directly via `reward_dimensions()` or the pure `group_reward_components(components)`). Pre-scale, so `sum(dims) == terminal_reward + shaping_total == total / reward_scale`. The rollup is purely additive — no weight, preset, or `total` formula changed.

**GRPO** — `score_prompt_group` scores G rollouts for one prompt and normalises within-group advantages. `compute_grpo_outcome_advantage` computes `reward_i - mean(group)` for a flat rewards list. See `src/training/grpo.py`.

**PPO** — `compute_ppo_policy_loss_core` returns `(pg_loss, pg_clipfrac, ppo_kl, surrogate)`; `compute_value_loss` returns `(vf_loss, vf_clipfrac)`. Both require an `eos_mask` tensor. See `src/training/ppo/core_algos.py`.

### Smoke test

End-to-end reward + GRPO, with no GPU:

```bash
python3 -m examples.run_grpo_training_pipeline
```

### XML search protocol

The ReAct-style trace format used by `SearchAgentLoop` uses these model-output tags:

```xml
<think>decide whether to answer or search</think>
<search>one precise query when external evidence is needed</search>
<fetch>comma- or newline-separated URLs when snippets are insufficient</fetch>
<answer>final grounded answer with citation labels</answer>
```

Optional model-output tags for multi-hop tasks:

```xml
<search_decision>answer</search_decision>   <!-- skip search when internal knowledge suffices -->
<subquestions>one research subquestion per line</subquestions>
<searches>parallel independent queries, one per line</searches>
```

Environment-only tags (injected by the loop — never output by the model):

```xml
<information>search results with citation labels</information>
<search_evaluation>sufficiency verdict and weak-query hints</search_evaluation>
<subquestions_feedback>per-subquestion coverage status</subquestions_feedback>
<full_page>fetched page content</full_page>
```

Mask all environment-only tags from policy/SFT action loss.

## Evaluation

### Bamboogle

Bamboogle is a two-hop QA benchmark that requires chaining retrieval across multiple hops — a strong signal for `SearchAgentLoop` quality.

**CLI (local CPU):**

```bash
python3 -m examples.run_bamboogle_eval \
  --model Qwen/Qwen2.5-1.5B-Instruct --local --limit 5 --print_trace
```

**CLI (server-backed):**

```bash
python3 -m examples.run_bamboogle_eval \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --vllm_url http://localhost:8080 \
  --search_url http://localhost:8001/retrieve \
  --reward_preset second_pass --limit 125
```

Reward presets: `sparse_final_only` | `simple_sparse` | `second_pass` | `third_pass`

**Apple Silicon shell script** (auto-starts SerpAPI retrieval server, reads `SERP_API_KEY` from `.env`):

```bash
bin/run_bamboogle_eval.sh                              # 5 examples, mps device
bin/run_bamboogle_eval.sh --smoke                      # 1 example, quick sanity check
bin/run_bamboogle_eval.sh --limit 125                  # full benchmark
bin/run_bamboogle_eval.sh --device cpu --limit 10
bin/run_bamboogle_eval.sh --limit 125 --concurrency 8  # ~6-8x faster via parallel SerpAPI calls
bin/run_bamboogle_eval.sh --limit 125 --concurrency 8 --resume  # resume an interrupted run
```

The dataset is cached locally after the first download (`~/.cache/agentic_search/bamboogle_test.jsonl`), so subsequent runs skip the network fetch. `--resume` reads the existing output file and skips already-evaluated questions, appending new results.

**Training data generation:**

```bash
bin/generate_training_data.sh                         # Bamboogle → data/bamboogle_train/
bin/generate_training_data.sh --preview               # print 5 sample rows, no write
bin/generate_training_data.sh --dataset nq            # Natural Questions
bin/generate_training_data.sh --dataset trivia_qa     # TriviaQA
bin/generate_training_data.sh --dataset hotpotqa --max_examples 500
```

Each run writes `data/<dataset>_train/train.parquet` and `data/<dataset>_train/test.parquet` ready for `LLMGRPOTrainer` or SFT.

### Activating the eval gates

The `Eval Gate` CI workflow (`.github/workflows/eval-gate.yml`) has two jobs — retrieval and RAGAS regression gates — that are **inactive placeholders** until real baselines are committed:

- The retrieval gate reads `data/eval/baseline_metrics.json`, which ships as a zero placeholder, so no regression can trip it. CI emits an `INACTIVE` warning until a real baseline lands.
- The RAGAS gate needs `data/eval/ragas_baseline.json`, which is not committed, so it reports `INACTIVE` and runs nothing.

To enforce them, generate real baselines against your canonical retrieval stack and commit the results:

```bash
# Retrieval baseline (needs a built index / BM25_INDEX_PATH or a running retrieval server)
python -m src.internal.retrieval.eval_runner \
  --dataset data/eval/qa_pairs.jsonl --top_k 10 \
  --output data/eval/baseline_metrics.json

# RAGAS baseline (needs OPENAI_API_KEY + the retrieval stack)
python -m src.internal.retrieval.ragas_eval \
  --dataset data/eval/ragas_qa.jsonl \
  --metrics faithfulness answer_relevancy \
  --output data/eval/ragas_baseline.json
```

Once a non-zero `baseline_metrics.json` (and/or a `ragas_baseline.json`) is committed, the corresponding gate starts enforcing regressions automatically.
