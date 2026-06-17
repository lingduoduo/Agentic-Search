# Feedback-Driven GRPO Fine-Tuning — Design Spec

**Date:** 2026-06-17
**Status:** Approved

---

## 1. Goals & Success Criteria

### Problem

`POST /api/feedback` collects thumbs-up/down signals per session and stores them in `retrieval_feedback`. `GET /api/admin/evals/summary` surfaces aggregate stats. But the signals are never read back into the training loop — the model does not improve from user feedback.

### Success Criteria

- `load_feedback_examples(db_path)` extracts rated sessions from `AgenticSearchStore` as `list[PromptTrainingExample]` with `metadata["human_signal"]` set
- `SearchRewardFunction` adds a `human_feedback` component weighted by `SearchRewardConfig.human_feedback_weight`; default `0.0` produces zero regression on existing reward presets
- `score_prompt_group` and `SearchAgentGRPOTrainer` thread `human_signal` from batch metadata into the reward computation
- `examples/run_feedback_grpo.py` runs end-to-end: load DB → build examples → train → save checkpoint to `data/checkpoints/feedback_grpo/`
- All existing `SearchRewardConfig` presets produce identical scores with `human_feedback_weight=0.0`

### Out of Scope

- DPO / response-level preference pairs
- Auto-triggering training from beat worker or admin API
- HuggingFace Hub push
- New dataclasses — `human_signal` rides in existing `metadata: dict[str, Any]` fields

---

## 2. Architecture

```
AgenticSearchStore (SQLite)
  retrieval_feedback(session_id, signal)   ← POST /api/feedback
  chat_messages(session_id, role, content) ← first user message = training query
         │
         ▼
load_feedback_examples(db_path, min_ratings=10)
  → list[PromptTrainingExample(
        question=<first user message>,
        ground_truth="",
        metadata={"human_signal": +1.0 | -1.0}
    )]
         │
         ▼
PromptOnlyDataset  (existing)
         │
         ▼
SearchAgentGRPOTrainer.rollout_async()
  for each (query, human_signal) prompt:
    → G rollouts from current model via SearchAgentLoop
    → score_prompt_group(
          samples,
          ground_truth="",
          human_signal=signal,         ← extracted from PromptBatch.metadata
          reward_fn=SearchRewardFunction(config),
      )
    → reward = composite + human_feedback_weight * signal
    → PPO-clip update
         │
         ▼
data/checkpoints/feedback_grpo/
  (HuggingFace-format weights)
```

Signal is query-level — all G rollouts for a rated prompt share the same `+1.0` or `-1.0`. Unrated prompts are excluded from `load_feedback_examples`, so the signal is never absent during training.

---

## 3. Components

### 3.1 `load_feedback_examples` — `src/training/data.py`

New function added to the existing module.

```python
def load_feedback_examples(
    db_path: str | Path,
    *,
    min_ratings: int = 10,
) -> list[PromptTrainingExample]:
    """Load rated sessions from AgenticSearchStore as GRPO training examples.

    Each returned example has:
        question     = first user message in the session
        ground_truth = ""  (human signal replaces correctness supervision)
        metadata     = {"human_signal": +1.0 | -1.0}

    Sessions without chat messages are skipped.
    Raises ValueError if fewer than min_ratings rated sessions are found.
    """
```

**Implementation steps:**
1. Open `AgenticSearchStore(db_path)`
2. For each row in `retrieval_feedback`: resolve `session_id` → first `role=user` chat message
3. Skip sessions with no chat messages
4. Map `signal`: `"thumbs_up" → +1.0`, `"thumbs_down" → -1.0`
5. Raise `ValueError(f"Only {n} rated sessions found; need at least {min_ratings}")` before returning if count < threshold

### 3.2 Reward integration — `src/training/reward.py`

**`SearchRewardConfig`** — one new field:

```python
human_feedback_weight: float = 0.0  # disabled by default
```

All existing presets (`second_pass`, `third_pass_with_format`, etc.) are unchanged because the field defaults to `0.0`.

**`SearchRewardFunction._reward_components_from_correctness()`** — add a new `human_signal: float | None = None` keyword parameter:

```python
def _reward_components_from_correctness(
    self,
    output: AgentLoopOutput,
    correctness: float,
    *,
    human_signal: float | None = None,
) -> dict[str, float]:
    ...
    if human_signal is not None:
        components["human_feedback"] = self.config.human_feedback_weight * human_signal
```

When `human_feedback_weight = 0.0` (default), the term is `0.0` and has no effect. The key is only written when `human_signal` is present, so the component does not appear in breakdowns for unrated rollouts. Default of `None` means existing callers need no changes.

### 3.3 `score_prompt_group` — `src/training/grpo.py`

Add `metadata: dict | None = None` parameter. Extract `human_signal` before the per-sample loop and pass it as a keyword arg to `_reward_components_from_correctness`:

```python
def score_prompt_group(
    samples: list[GRPORolloutSample],
    *,
    ground_truth: str,
    judge_fn: JudgeFn,
    reward_fn: SearchRewardFunction | None = None,
    advantage_config: GRPOAdvantageConfig | None = None,
    batch_judge_fn: BatchJudgeFn | None = None,
    metadata: dict | None = None,           # ← new
) -> list[ScoredGRPORollout]:
    ...
    human_signal: float | None = metadata.get("human_signal") if metadata else None
    ...
    for sample, correctness in zip(samples, correctness_scores):
        components = reward_function._reward_components_from_correctness(
            sample.output, correctness, human_signal=human_signal
        )
```

All G rollouts in the group share the same value since they come from the same prompt. Passing `metadata=None` (default) is identical to today's behavior.

### 3.4 `SearchAgentGRPOTrainer` — `src/training/ppo/search_agent_grpo_trainer.py`

In `rollout_async()`, extract `human_signal` from `PromptBatch.metadata` and pass it through to `score_prompt_group`. The trainer already receives the full batch including metadata; this is a one-line extraction before the scoring call.

### 3.5 Script — `examples/run_feedback_grpo.py`

```
--db_path               SQLite DB path  [default: $AGENTIC_SEARCH_WEB_DB_PATH or :memory:]
--model                 HuggingFace model id or local path  [required]
--output_dir            checkpoint destination  [default: data/checkpoints/feedback_grpo/]
--min_ratings           abort if fewer rated sessions  [default: 10]
--human_feedback_weight [default: 0.5]
--num_rollouts          G rollouts per prompt  [default: 4]
--search_url            retrieval server URL  [default: http://localhost:8001/retrieve]
--device                cpu | mps | cuda  [default: mps]
```

Smoke-test (no GPU):
```bash
python3 -m examples.run_feedback_grpo \
  --db_path data/feedback.sqlite3 \
  --model Qwen/Qwen2.5-0.5B-Instruct \
  --num_rollouts 2 --min_ratings 1 --device cpu
```

---

## 4. File Map

| Action | File | Change |
|--------|------|--------|
| **Modify** | `src/training/data.py` | Add `load_feedback_examples(db_path, min_ratings) → list[PromptTrainingExample]` |
| **Modify** | `src/training/reward.py` | Add `human_feedback_weight: float = 0.0` to `SearchRewardConfig`; add `human_signal: float \| None = None` kwarg to `_reward_components_from_correctness()` |
| **Modify** | `src/training/grpo.py` | Add `metadata: dict \| None = None` param to `score_prompt_group`; extract `human_signal` and pass to `_reward_components_from_correctness` |
| **Modify** | `src/training/ppo/search_agent_grpo_trainer.py` | Extract `human_signal` from batch metadata; pass to `score_prompt_group` |
| **Create** | `examples/run_feedback_grpo.py` | CLI entry point |
| **Create** | `tests/unit/training/test_feedback_examples.py` | Unit tests for `load_feedback_examples` |
| **Create** | `tests/unit/training/test_reward_human_signal.py` | Unit tests for human feedback reward component |

No new dataclasses. No schema changes.

---

## 5. Testing

All unit tests — no GPU, no live server, no model downloads.

### `test_feedback_examples.py`

- Returns `PromptTrainingExample` with `metadata["human_signal"] == +1.0` for thumbs-up session
- Returns `metadata["human_signal"] == -1.0` for thumbs-down session
- Sessions with no chat messages are skipped
- Raises `ValueError` when rated count < `min_ratings`
- Sessions without a feedback entry in `retrieval_feedback` are excluded

### `test_reward_human_signal.py`

- `human_feedback_weight=0.0` (default) — total reward identical to baseline; `"human_feedback"` absent from components
- `human_feedback_weight=0.5`, `human_signal=+1.0` → `components["human_feedback"] == +0.5`
- `human_feedback_weight=0.5`, `human_signal=-1.0` → `components["human_feedback"] == -0.5`
- `human_signal` absent from metadata → component is `0`, no `KeyError`
- Existing presets (`second_pass`, `third_pass_with_format`) produce identical scores with default config

---

## 6. Error Handling

| Scenario | Behaviour |
|----------|-----------|
| Fewer than `min_ratings` rated sessions | `ValueError` before model loads — fast fail |
| Session in `retrieval_feedback` has no chat messages | Skipped silently |
| `human_signal` absent from metadata | Reward component contributes `0`; no exception |
| `human_feedback_weight = 0.0` | Human feedback term is `0`; zero regression |
| DB path does not exist | `AgenticSearchStore` raises on open — propagates naturally |
