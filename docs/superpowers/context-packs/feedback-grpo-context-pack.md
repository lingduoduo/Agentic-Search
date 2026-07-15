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

Signal is query-level — all G rollouts for a rated prompt share the same `+1.0` or `-1.0`. Unrated prompts are excluded from `load_feedback_examples`, so the signal is never absent during training.

---

### `test_feedback_examples.py`

- Returns `PromptTrainingExample` with `metadata["human_signal"] == +1.0` for thumbs-up session
- Returns `metadata["human_signal"] == -1.0` for thumbs-down session
- Sessions with no chat messages are skipped
- Raises `ValueError` when rated count < `min_ratings`
- Sessions without a feedback entry in `retrieval_feedback` are excluded

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

## Context Boundary

This pack summarizes its linked sources. Consult those documents for complete details; no implementation status is inferred here.
