# Generated Context Pack

# Feedback Grpo

## Sources

- [Specification: 2026-06-17-feedback-grpo-design.md](../specs/2026-06-17-feedback-grpo-design.md)
- [Plan: 2026-06-17-feedback-grpo.md](../plans/2026-06-17-feedback-grpo.md)

## Specification Context

### Out of Scope

- DPO / response-level preference pairs
- Auto-triggering training from beat worker or admin API
- HuggingFace Hub push
- New dataclasses — `human_signal` rides in existing `metadata: dict[str, Any]` fields

---

### 2. Architecture

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

### 5. Testing

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

## Implementation Plan Context

### Task 1: `load_feedback_examples` in `src/training/data.py`

**Files:**
- Modify: `src/training/data.py` (append after last `register_rag_prompt_template` call)
- Test: `tests/unit/test_feedback_examples.py`

### Task 2: Human feedback reward component in `src/training/reward.py`

**Files:**
- Modify: `src/training/reward.py`
- Test: `tests/unit/test_reward_human_signal.py`

### Task 3: `metadata` parameter in `score_prompt_group` (`src/training/grpo.py`)

**Files:**
- Modify: `src/training/grpo.py`
- Test: `tests/unit/test_reward_human_signal.py` (the two `score_prompt_group` tests)

### Task 4: Thread `metadata` through `SearchAgentGRPOTrainer`

**Files:**
- Modify: `src/training/ppo/search_agent_grpo_trainer.py`

### Task 5: CLI script `examples/run_feedback_grpo.py`

**Files:**
- Create: `examples/run_feedback_grpo.py`

### Task 6: Final integration check

- [x] **Step 1: Run all new tests together**

```bash
pytest tests/unit/test_feedback_examples.py tests/unit/test_reward_human_signal.py -v
```
Expected: 15 passed

- [x] **Step 2: Run full unit suite — verify zero regressions**

```bash
pytest tests/unit/ -q
```
Expected: 1830+ passed, 0 failed

- [x] **Step 3: Run linter**

```bash
ruff check . --fix && ruff format .
```
Expected: no errors

- [x] **Step 4: Commit spec + plan**

```bash
git add docs/superpowers/specs/2026-06-17-feedback-grpo-design.md \
        docs/superpowers/plans/2026-06-17-feedback-grpo.md
git commit -m "docs: add spec and plan for feedback-driven GRPO"
```

- [x] **Step 5: Push and open PR**

```bash
git push -u origin <branch>
gh pr create --title "feat(training): feedback-driven GRPO fine-tuning loop" \
  --body "Closes feedback-GRPO spec. 15 new unit tests, 0 regressions."
```

## Context Boundary

This pack summarizes its linked sources. Consult those documents for complete details; no implementation status is inferred here.
