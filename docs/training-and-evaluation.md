# Training and evaluation

## Serving requests do not train models

`POST /api/agent` and `/api/agent/stream` perform routing, retrieval, tool execution, and model inference only. Even explicit `search_agent` or `tool_agent` mode loads a policy model for generation without updating its weights. A query containing `GRPO` is ordinary search input; it does not invoke the GRPO trainer. SFT, GRPO, and PPO run only through the offline commands in this guide. See [API request routing](request-routing.md) for serving-time dispatch.

Serving still uses indexes built offline by the `index_builder`. Filter-aware and degraded branches can use the composed session-aware retrieval, ranking/reranking, and evidence-grounded inference pipeline; strong unfiltered auto-search retains its direct ranking, sufficiency gate, and provider fallback path. Neither retrieval nor reranking is a training step. No new serving API or request/response schema was introduced; offline trainers may produce model artifacts, but requests only load and infer with them.

## Intent routing by nearest canonical example

**There is no intent training run any more.** The optional three-label (`chat`, `search`, `tool`) request router used to be a small MLP trained on generated examples. It is gone — module, checkpoint format, wordpiece bundle and all. What replaced it compares the incoming request against roughly 280 curated canonical examples and takes the route whose nearest examples are closest. The whole offline workflow is three commands, none of which trains anything:

```bash
# 1. Seed a canonical draft from existing labelled examples (optional; run once).
python -m src.model.intent_index_cli seed \
  --examples data/intent_examples.json --output data/intent_canonical.draft.json

# 2. Build the index the router serves from.
python -m src.model.intent_index_cli build \
  --canonical data/intent_canonical.json --output data/intent_index

# 3. Measure it. Never skip this after editing the canonical set.
python -m src.model.intent_index_cli evaluate \
  --index data/intent_index \
  --eval-queries data/intent_eval_queries.json \
  --hard-queries data/intent_eval_hard.json \
  --out-of-scope data/intent_out_of_scope.json \
  --canonical data/intent_canonical.json \
  --output data/intent_index/evaluation_report.json
```

`data/intent_canonical.json` and the evaluation JSON files are tracked (force-added under an otherwise gitignored `data/`). `data/intent_index/` is a regenerable local build artifact and is not.

### How routing works

Each canonical example is encoded once by `all-MiniLM-L6-v2` and L2-normalized. At serving time the request is encoded the same way, and each route scores as the **mean of its top-k cosine similarities** to that request. The best route wins; the module labels reported alongside it are diagnostics and can never change the route.

`k` (`TOP_K` in `src/model/intent_knn.py`) is `3` by default and is what serves today, but it is a parameter of `IntentIndex.decide()`, not a hardcoded constant — see [the ceiling finding, corrected](#the-ceiling-finding-corrected-top_k-was-never-swept) below for why that distinction matters and `AGENTIC_SEARCH_INTENT_TOP_K` in [Configuration](configuration.md) for the env var.

Two thresholds gate the answer, because two different things go wrong:

- `AGENTIC_SEARCH_INTENT_MODEL_MIN_CONFIDENCE` (default `0.30`) — a low **absolute** similarity means nothing canonical resembles this request at all. That is an out-of-scope request.
- `AGENTIC_SEARCH_INTENT_MIN_ROUTE_MARGIN` (default `0.02`) — a small **gap** between the best and second-best route means two routes fit equally well. That is an ambiguous request.

Either one abstains, and abstention defers to the LLM classifier and the clarification path exactly as before. This is the concrete reason the softmax head was replaced: probabilities sum to one by construction, so a softmax router cannot express "none of these" — it can only say which of the three is least bad. That is why the old model's out-of-scope separation was only `+0.059` (mean in-scope confidence minus mean out-of-scope confidence) against this one's `0.1188`.

### Changing routing behavior

Edit `data/intent_canonical.json` and rebuild. There is no training run, no seed, no schedule, no checkpoint. The canonical examples *are* the model.

Always **append, rebuild, re-measure**, in that order. A badly-phrased canonical example does not fail loudly; it becomes a bad attractor that quietly pulls every nearby query onto its route. The only thing that catches it is the evaluation report, so a canonical edit that has not been re-measured has not been verified. `tests/unit/test_intent_canonical_data.py` additionally guards the set's size band, per-route balance, per-module support, and internal near-duplication; `tests/unit/test_intent_index_eval.py` pins the measured accuracy, out-of-scope separation, and latency bars.

### What it scores, and why it is still dark

Measured 2026-08-13 against the committed canonical set. The evaluation set has two parts: 30 legacy queries that were used as feedback while curating the canonical examples, and 151 `bulk-`prefixed queries written independently. **The clean 151 is the honest number.**

| Slice | This router | MLP | Regex cascade | Majority floor |
|---|---|---|---|---|
| **clean_151** (honest) | **0.6225** | 0.4768 | 0.4238 | 0.3377 |
| legacy_30 (contaminated) | 0.8000 | 0.733 | 0.4000 | 0.3333 |
| bulk_181 (decision-rule input) | 0.6519 | — | 0.4199 | 0.3370 |
| hard_40 (adversarial) | 0.6250 | — | 0.4750 | 0.2750 |

| Other measures | This router | Previous MLP |
|---|---|---|
| Out-of-scope separation margin | `0.1188` | `+0.059` |
| Module macro-F1 / joint accuracy | `0.3471` / `0.2318` | — |
| Leave-one-out over the canonical set | `0.6750` (189/280) | — |
| p50 / p95 routing latency | `5.51ms` / `5.88ms` | `0.16ms` / `0.43ms` |
| Tuned thresholds | `min_confidence=0.30`, `min_margin=0.02` | — |
| Per-route accuracy, clean_151 | chat 24/51, search 25/50, tool 45/50 | — |

The new router beats the MLP on every slice (**+0.146** on the clean instrument), beats the production regex cascade by **+0.199**, and beats the majority-class floor by **+0.285** — **and still misses the `0.75` promotion bar at `0.6519`, so the artifact stays dark.** `AGENTIC_SEARCH_INTENT_INDEX_PATH` remains unset by default and every request falls through the existing LLM/rule cascade.

Read the stop for what it is. The `0.80`/`0.75` promotion bands were calibrated against the legacy-30 instrument, and on legacy-30 this router scores `0.800` — the top band. The hard stop is what happens when a legacy-calibrated constant meets a harder, honest instrument. It is not a regression against anything real; every like-for-like comparison above is a clear win. Latency is a deliberate regression — roughly 13x the MLP's p95 — bought with accuracy and out-of-scope safety, and it clears the 25ms ceiling with wide headroom.

### The ceiling finding, corrected: `TOP_K` was never swept

Leave-one-out accuracy over the canonical set at the shipped `TOP_K=3` is `0.6750`: scoring each of the 280 anchors against the other 279 with the same top-k-mean rule, a third of them cannot recover their own route. An earlier version of this document read that `0.6750` as a representation ceiling — "`all-MiniLM-L6-v2` sentence embeddings with top-3-mean cosine top out near `0.67`–`0.70` no matter how good the examples get" — and named a stronger encoder as the only remaining lever. **That was overstated.** `TOP_K = 3` is an arbitrary constant that was never swept, and sweeping it — same encoder, same 280 anchors — moves both numbers substantially:

| `top_k` | clean_151 accuracy | hard_40 accuracy | out-of-scope separation margin | leave-one-out accuracy |
|---|---|---|---|---|
| **3 (shipped)** | 0.6225 | 0.6250 | **0.1188** | 0.6750 |
| 5 | 0.6358 | 0.6000 | 0.1036 | 0.6893 |
| 8 | 0.6556 | 0.6500 | 0.0918 | 0.7036 |
| **15** | **0.6887** | 0.6500 | 0.0767 | **0.7464** |
| 25 | 0.6755 | 0.6250 | 0.0654 | 0.7643 |

At `k=15`, leave-one-out reaches `0.7464` and clean_151 accuracy reaches `0.6887` — both well above the shipped `k=3` numbers on the same encoder and the same anchors. The `0.6750` figure this document previously called a ceiling reflects `TOP_K`'s arbitrary value at least as much as it reflects the encoder's representation.

**The trade, plainly stated:** out-of-scope separation falls from `0.1188` to `0.0767` as `k` rises from 3 to 15, because averaging more neighbors lifts the confidence floor for every route — including routes the request has nothing to do with — so accuracy and abstention pull against each other. There is no `k` that improves both at once in this table.

**The caveat:** the gap between `k=8` (`0.6556`) and `k=15` (`0.6887`) on clean_151 is about five queries out of 151. That is well within the noise a single held-out slice can produce, so `k` must be chosen on a validation split with the accuracy/abstention trade decided deliberately — not read off this table as if it named a single best value.

**What survives:** the hard stop still stands at every `k` tested. `bulk_181` — the decision-rule input, not clean_151 — lands near `0.72` at `k=15`, still under the `0.75` promotion bar. A stronger encoder and a swept `k` both remain live levers; `k` is the cheaper one to try first, because it costs a config change and a re-run of the evaluation CLI, not a new model. This module's sweep runs report-only (`_sweep_top_k` in `src/model/intent_index_eval.py`, recorded as `top_k_sweep` in `evaluation_report.json`): `TOP_K` stays `3` in serving until that trade is decided once, together with whatever encoder change is tried next — not decided twice.

### Known limitations

- **`top_k` is a live parameter with a measured trade, not a settled constant.** `TOP_K = 3` ships unchanged, but it is now a parameter of `IntentIndex.decide()` (`AGENTIC_SEARCH_INTENT_TOP_K`), and [the corrected ceiling finding](#the-ceiling-finding-corrected-top_k-was-never-swept) above shows sweeping it reaches `0.7464` leave-one-out and `0.6887` clean_151 accuracy at `k=15`, against `0.1188` out-of-scope separation falling to `0.0767`. `evaluation_report.json`'s `top_k_sweep` carries the full table for `k ∈ {3, 5, 8, 15, 25}`. Nothing has picked a new default from it: that decision is deferred to a validation split, alongside whatever encoder change is tried next.
- **Topical concentration.** 47% of the canonical examples carry IR/ML vocabulary, because they were curated from this project's own example set. Of 16 held-out in-scope probes drawn from outside that vocabulary, **13 abstained** rather than routing at all, and **9 had a wrong best-guess route underneath**. The two counts overlap and are not a partition — an abstaining query still has a nearest route; it just does not clear the thresholds. The failure is safe, because abstention defers to the LLM classifier, but off-domain traffic abstains more often than it should.
- **The compose-versus-dispatch boundary.** "write an email to the vendor about the overage" sits at `0.963` cosine to the canonical "email the vendor about the overage". One verb apart, so it routes to `tool` without abstaining, when composing text is arguably `chat`. Adding more compose anchors did not fix it: the two phrasings are near-identical to the encoder and genuinely ambiguous to a human reader.
- **Route imbalance.** The `tool` route scores 45/50 on the clean slice while `search` and `chat` sit near 25/50. Rewriting tool queries into indirect phrasings did not move it, so this is a property of the route — tool requests carry imperative verbs that anchor cleanly — rather than of the instrument.
- **Margin abstentions, module labels, and the composite flag are invisible to production telemetry.** They are recorded through `request_capture`, which only runs under the debug panels; `route_request`'s `telemetry` argument (the one persisted with the session in production) never receives `modules` or `composite`. Composite detection exists precisely to give a future plan-aware router measured data, so this means it is currently unmeasurable in production. Measuring any of this in production would need a `predict_route` or `route_request` signature change.
- **The evaluation set is partly contaminated.** The legacy 30 queries were used as feedback while the canonical set was being curated, so their `0.800` is optimistic and must never be quoted as the router's accuracy. The 151 `bulk-`prefixed queries are clean, and they are the honest measurement.

### Deploying an index

The loaded index is cached by resolved path and is never invalidated, so: **rebuild the index, then restart the web process.** A *failed* load is cached too — starting the web process before the index exists leaves learned routing disabled until the next restart, even after the file appears.

The MiniLM encoder itself loads lazily on the first auto-routed request, separately from the index, and blocks that request for roughly two seconds while the model loads. This is not the promotion-gate activation checklist above; it is a separate one-time cost the first caller pays. A failing model fetch (missing weights, unreachable HuggingFace) is cached as a failure the same way the index's failed load is: the route disables itself and every later request degrades straight to the LLM classifier instead of retrying the download per request.

### The units trap

`AGENTIC_SEARCH_INTENT_MODEL_MIN_CONFIDENCE` is now a **cosine similarity, not a softmax probability**. The two live on entirely different scales: the retired model routinely emitted confidences above `0.9` where this one's in-scope mean is `0.378`. A value carried over from the old model is meaningless, and a plausible-looking `0.6` would abstain on almost every request. Re-tune with the threshold sweep written into `evaluation_report.json` (`threshold_tuning.sweep`, and the chosen pair under `threshold_tuning.selected`) rather than reusing a number. `AGENTIC_SEARCH_INTENT_MIN_ROUTE_MARGIN` and `AGENTIC_SEARCH_INTENT_MIN_MODULE_SCORE` are cosine-scaled in the same way.

`AGENTIC_SEARCH_INTENT_MODEL_PATH` no longer exists. Serving reads `AGENTIC_SEARCH_INTENT_INDEX_PATH`, a directory holding an `index.npz`. Building or evaluating an index never changes a serving setting.

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

Reward preset names: `sparse_final_only` | `simple_sparse_with_search_penalty` | `second_pass` | `third_pass_with_format` | `retriever_aware` (see `SearchRewardConfig` in `src/training/reward.py`). The Bamboogle eval CLI (`run_bamboogle_eval --reward_preset`) exposes the shorthand `sparse_final_only | simple_sparse | second_pass | third_pass`, which map to the first four config presets; `retriever_aware` is config-only.

**The judge.** The `correctness` term is `judge_fn(answer, gold)`. The example trainers pass `simple_sparse_correctness_reward` (exact-normalized match → 1.0, gold contained in prediction → 0.7, else 0.0), which is what the "EM / contains-match" row above refers to. `SimulatedPreferenceJudge` (`src/training/judge.py`) is a separate, **deterministic reference-free heuristic** (length + lexical diversity − hedging) that stands in for a real LLM-as-judge — it ignores the gold answer and is used by the `run_bamboogle_grpo_train` / `run_bamboogle_synthetic_grpo` examples. There is no trained reward model or LLM judge; a real judge would slot in behind the same `BatchJudgeFn` interface.

**Four reward dimensions** — `reward_components()` also groups every term into four subtotals via `REWARD_DIMENSIONS`, emitted as `dim_correctness`, `dim_citation_support`, `dim_retrieval_quality`, `dim_search_efficiency` (and available directly via `reward_dimensions()` or the pure `group_reward_components(components)`). Pre-scale, so `sum(dims) == terminal_reward + shaping_total == total / reward_scale`. The rollup is purely additive — no weight, preset, or `total` formula changed.

**GRPO** — `score_prompt_group` scores G rollouts for one prompt and normalises within-group advantages. `compute_grpo_outcome_advantage` computes `reward_i - mean(group)` for a flat rewards list. See `src/training/grpo.py`.

**PPO core** — `compute_ppo_policy_loss_core` returns `(pg_loss, pg_clipfrac, ppo_kl, surrogate)` and is the clipped surrogate the GRPO trainers use (with a group-relative advantage in place of GAE). `compute_value_loss` and `compute_gae_advantages` implement the PPO-with-critic path but are **not wired into any trainer** — training here is critic-free GRPO (no value/critic model), and those helpers exist for parity/tests only. All require an `eos_mask` tensor. See `src/training/ppo/core_algos.py`.

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
