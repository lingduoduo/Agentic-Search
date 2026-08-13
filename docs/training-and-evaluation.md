# Training and evaluation

## Serving requests do not train models

`POST /api/agent` and `/api/agent/stream` perform routing, retrieval, tool execution, and model inference only. Even explicit `search_agent` or `tool_agent` mode loads a policy model for generation without updating its weights. A query containing `GRPO` is ordinary search input; it does not invoke the GRPO trainer. SFT, GRPO, and PPO run only through the offline commands in this guide. See [API request routing](request-routing.md) for serving-time dispatch.

Serving still uses indexes built offline by the `index_builder`. Filter-aware and degraded branches can use the composed session-aware retrieval, ranking/reranking, and evidence-grounded inference pipeline; strong unfiltered auto-search retains its direct ranking, sufficiency gate, and provider fallback path. Neither retrieval nor reranking is a training step. No new serving API or request/response schema was introduced; offline trainers may produce model artifacts, but requests only load and infer with them.

## Intent-model training and promotion

The supported offline workflow trains and evaluates the optional three-label (`chat`, `search`, `tool`) request router reproducibly. First extract the frozen pretrained wordpiece bundle the model reads with:

```bash
python -m src.model.intent_training embeddings --output data/intent_pretrained
```

The intent model reads requests as **pretrained wordpieces**, not as words from a vocabulary built out of its own training data. `embeddings` extracts MiniLM's tokenizer vocabulary and input embedding matrix once into `data/intent_pretrained/` (a 230KB `vocab.txt` and a 23MB fp16 matrix); the transformer itself never runs, at training time or serving time. This is what removes out-of-vocabulary entirely: an unseen word decomposes into known subwords — *postmortem* becomes `post ##mo ##rte ##m` — instead of being deleted by the padding mask. The previous word-level model read only 47% of the tokens in the evaluation set. The bundle lives under gitignored `data/`, so it is regenerable rather than committed, and a checkpoint carries its own copy so serving needs no separate file.

Then capture the existing classifier/rule result for each ambiguous held-out request in `data/eval/intent_fallback_predictions.json`, and generate the complete baseline:

```bash
python -m src.model.intent_training baseline \
  --examples data/intent_examples.json \
  --fallback-predictions data/eval/intent_fallback_predictions.json \
  --output data/eval/intent_baseline_predictions.json \
  --seed 17
```

The generator reproduces the seed-17 held-out split and runs the production high-precision regex router itself. It requires captured fallback rows for exactly the remaining ambiguous IDs, rejects missing or extra captures, and writes the complete regex → classifier/rule baseline consumed by training. No network or external LLM is invoked by the generator; operators can capture their chosen production classifier separately or supply deterministic rule-based fallback results. Capture `classifier` rows to benchmark against a deployment that runs the LLM classifier: a purely `rule_based` capture reports a zero classifier-fallback rate and no classifier latency, so the fallback-reduction and latency gates have nothing to improve on and the run cannot be promotable.

Both input and output prediction files are JSON arrays. Every record has exactly this schema:

```json
{
  "example_id": "stable-example-id",
  "expected": "chat",
  "predicted": "chat",
  "confidence": 1.0,
  "latency_ms": 42.5,
  "mechanism": "classifier"
}
```

`expected` and `predicted` must be `chat`, `search`, or `tool`; `confidence` is a finite number from `0.0` through `1.0`; `latency_ms` is finite and non-negative. Captured fallback mechanisms are `classifier` or `rule_based`. The generated complete baseline may additionally contain `regex` records, and candidate evaluation adds `model` for candidate-covered predictions. This offline `mechanism` vocabulary (`regex` / `classifier` / `rule_based` / `model`) is separate from the runtime `hook_metadata.route_mechanism` vocabulary used by serving (`explicit_source` / `rules` / `model` / `classifier` / `heuristic_default` / `clarify` / `user_selected`, documented in [Auto-router decision order](request-routing.md#auto-router-decision-order)); the two must not be compared directly.

Then train and evaluate the candidate:

```bash
python -m src.model.intent_training train \
  --examples data/intent_examples.json \
  --baseline data/eval/intent_baseline_predictions.json \
  --out-of-scope data/intent_out_of_scope.json \
  --eval-queries data/intent_eval_queries.json \
  --output-dir models/intent-candidate \
  --seed 17
```

`--out-of-scope` supplies unlabeled requests the router should decline entirely. They carry no label, because the three-label taxonomy cannot express "none of these": out-of-scope safety is measured as the fraction of probes whose confidence falls below the serving threshold. That rate is **reported, not gated**. This model family cannot reach a useful abstention rate at any threshold that leaves coverage, so out-of-scope safety comes from the LLM-classifier fallback and the clarification path, not from the model. What the model can do is score chatter *lower* than real requests: the generated dataset's neutral fillers (roles, times, artifacts, conversational openers) appear equally often under all three labels, so ordinary English carries no class evidence and a request built from it pools toward the centre. Without probes the rate is reported as `null` — unmeasured, never assumed safe.

`--eval-queries` supplies `data/intent_eval_queries.json`, a hand-authored set written independently of the generator. It is never trained on and never split. The report's `realistic_accuracy` block scores it: argmax accuracy and per-label precision/recall/F1 over every query, plus coverage and covered accuracy at the selected threshold. This is the number that says whether the model handles phrasing a person would actually type; the templated test split, whose coverage is `1.00` by construction, measures memorization of the generator instead. Training refuses a set whose queries all appear verbatim in the training examples. Without the flag, `realistic_accuracy` is recorded as `null`.

**What the current model actually scores, and why it is still not promoted.** On the committed dataset at seed 17 with the pretrained bundle, realistic accuracy is `0.733` (macro-F1 `0.731`; chat P/R/F1 `0.875`/`0.700`/`0.778`, search `0.643`/`0.900`/`0.750`, tool `0.750`/`0.600`/`0.667`) and the out-of-scope separation margin (mean in-scope confidence minus mean out-of-scope confidence) is `+0.059`, from mean in-scope `0.941` against mean out-of-scope `0.882`. Routing costs p50 `0.16ms` and p95 `0.43ms` per query. Both bars are pinned by tests in `tests/unit/test_intent_training.py`. Token coverage of the evaluation set, the probes, and the dataset is now **100%** — no query word goes unread, against the word-level model's `47%` — and accuracy rose from `0.567` to `0.733`, clearing the `3/5 = 0.60` hand-scored baseline that motivated this work. Read that headline with the hyperparameters attached: `0.567` was measured at 300 epochs / lr `1e-3`, and the pretrained model at those same settings scores `0.700`, so `0.133` of the gain is the representation change alone and the remaining `0.033` — one query of thirty — comes from the longer schedule the sweep selected. It does not reach the `0.75` bar the spec set for promotion, so the artifact stays dark and the LLM-classifier fallback and clarification path still own out-of-scope safety. The remaining gap is a representation ceiling rather than a vocabulary one: a bag of frozen MiniLM input embeddings discards word order, so running the full MiniLM encoder behind the same tokenizer is the next step. Separation stays positive but narrow, with out-of-scope probes producing 20 distinct confidences across 24 probes, where previously every fully-out-of-vocabulary probe pooled to the zero vector and returned one identical prediction.

Those figures come from `--epochs 800 --lr 3e-3`, the best of a sweep over epochs `100`/`300`/`800` against learning rates `1e-3`/`3e-3` (ties on accuracy broken toward the larger margin). Both are now the CLI defaults, so the `train` command above reproduces the measured and pinned run without extra flags. The epoch count is an optimizer-step count, because training takes one **full-batch** step per epoch: 10 epochs is 10 steps and leaves the model effectively untrained (realistic accuracy `0.333`). There is no longer a `--min-freq`, `--vocab-size`, or `--embedding-dim` knob — the vocabulary and the embedding matrix both come from the extracted bundle at 30522 × 384, and the unknown-token id survives only as the fallback for a word no subword covers, so an unread word maps to its own index rather than the padding id that masked pooling deletes.

The seed controls the source-grouped train, validation, and held-out test split as well as supported deterministic training behavior. Threshold selection uses only validation requests not already decided by regex; the test split is evaluated once after selection. Baseline records must match the test example IDs and expected labels exactly, ensuring the baseline and candidate are compared on the same requests.

The output directory contains three inspection-ready artifacts:

- `intent_model.pt` is the candidate checkpoint, including its ordered labels, preprocessing and architecture metadata, dataset fingerprint, and format version. It carries its own copy of the wordpiece vocabulary and the frozen fp16 embedding matrix, so it is now roughly 23MB and is written at version `4`; versions `1`-`3` predate pretrained wordpieces and are rejected on load, which means an older checkpoint must be retrained rather than migrated.
- `split_manifest.json` records the seed, fingerprint, split sizes, per-label counts, example IDs, and source groups.
- `evaluation_report.json` records the selected threshold, per-split label counts, candidate and baseline metrics, hyperparameters, dataset fingerprint, calibration, realistic accuracy, and every promotion-gate result.

The `calibration` block is what to read before trusting a threshold. Softmax scores are not probabilities, so it reports the full validation sweep — coverage, macro-F1, tool precision, high-confidence errors, and out-of-scope abstention at every candidate threshold — plus reliability bins and the expected calibration error. A model whose out-of-scope confidences overlap its in-domain confidences shows up here as a sweep where abstention only reaches the required rate at a threshold that leaves no coverage.

Evaluation composes the actual cascades on every held-out request: baseline is regex → captured classifier/rule fallback; candidate is the same regex route → covered model → the identical captured fallback on abstention. Regex-owned requests cannot count as model coverage. The promotion gates require: non-decreasing macro-F1; the configured minimum `tool` precision, measured over the model's own covered predictions so deterministic routes cannot dilute it (reported as `null`, which fails the gate, when the model predicted no `tool` route at all); no more than the configured high-confidence error limit; reduced LLM-classifier usage; lower model-resolved latency than LLM classification; and unchanged authoritative regex routes. A failed gate leaves all three candidate artifacts available for inspection but does not activate them or overwrite a serving setting.

Exit codes are `0`, `1`, and `2`: `0` means training and evaluation completed and every promotion gate passed; `1` means invalid input, configuration, I/O, training, or artifact generation prevented a valid completed run; and `2` means the run completed and wrote its artifacts but one or more promotion gates failed.

Operators activate a passing artifact explicitly by reading `selected_threshold` from `evaluation_report.json`, setting `AGENTIC_SEARCH_INTENT_MODEL_PATH` to the resulting `intent_model.pt`, and setting `AGENTIC_SEARCH_INTENT_MODEL_MIN_CONFIDENCE` to that selected threshold before restarting the application. For example, a report with `"selected_threshold": 0.73` requires both `AGENTIC_SEARCH_INTENT_MODEL_PATH=models/intent-candidate/intent_model.pt` and `AGENTIC_SEARCH_INTENT_MODEL_MIN_CONFIDENCE=0.73`. A stricter, higher configured threshold is safe; a lower threshold is rejected because it would serve requests outside the coverage that passed promotion. Non-promotable checkpoints record no approved serving threshold and are rejected by the serving loader, while remaining available for offline inspection. Learned routing is disabled by default when the path is unset or empty. Training never changes either serving setting automatically.

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
