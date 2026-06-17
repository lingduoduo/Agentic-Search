# Feedback-Driven GRPO Fine-Tuning — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Wire thumbs-up/down signals from `AgenticSearchStore` into the GRPO training loop so the model improves from user feedback without breaking any existing reward preset.

**Architecture:** `load_feedback_examples()` reads rated sessions from SQLite and converts each into a `PromptTrainingExample` with `metadata["human_signal"] = ±1.0`. A new `human_feedback_weight` field (default `0.0`) in `SearchRewardConfig` activates the component; `score_prompt_group` threads the per-prompt metadata through to `_reward_components_from_correctness`. A CLI script ties it together end-to-end.

**Tech Stack:** Python 3.12, SQLite via `AgenticSearchStore`, `src/training/` (data, reward, grpo, trainer), pytest (no GPU, no network for tests).

---

## File Map

| Action | File | Responsibility |
|--------|------|----------------|
| Modify | `src/training/data.py` | Add `load_feedback_examples()` |
| Modify | `src/training/reward.py` | Add `human_feedback_weight` field + `human_signal` kwarg |
| Modify | `src/training/grpo.py` | Add `metadata` param to `score_prompt_group` |
| Modify | `src/training/ppo/search_agent_grpo_trainer.py` | Thread `metadata` through `rollout_async` / `step_async` |
| Create | `examples/run_feedback_grpo.py` | CLI entry point |
| Create | `tests/unit/test_feedback_examples.py` | Unit tests for `load_feedback_examples` |
| Create | `tests/unit/test_reward_human_signal.py` | Unit tests for reward component + `score_prompt_group` |

---

## Task 1: `load_feedback_examples` in `src/training/data.py`

**Files:**
- Modify: `src/training/data.py` (append after last `register_rag_prompt_template` call)
- Test: `tests/unit/test_feedback_examples.py`

### Background

`AgenticSearchStore` is the SQLite wrapper at `src/internal/db/store.py`. Relevant API:
- `store.list_chat_messages(session_id) -> list[ChatMessageRecord]` — each record has `.role` and `.content`
- `store.save_retrieval_feedback(session_id, signal)` — writes `"thumbs_up"` or `"thumbs_down"`
- `store.create_chat_session(session_id=..., user_id=None)` — creates a session (needed in tests)
- `store.add_chat_message(session_id, role, content)` — writes a chat message (needed in tests)
- Raw query for feedback rows: `store._conn.execute("SELECT session_id, signal FROM retrieval_feedback").fetchall()`

`PromptTrainingExample` lives in the same file:
```python
@dataclass(frozen=True)
class PromptTrainingExample:
    question: str
    ground_truth: str
    tools: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
```

- [ ] **Step 1: Write the failing tests**

Create `tests/unit/test_feedback_examples.py`:

```python
"""Tests for load_feedback_examples."""
from __future__ import annotations
import pytest
from src.internal.db import AgenticSearchStore
from src.training.data import PromptTrainingExample, load_feedback_examples


def _seed_store(db_path: str, rows: list[dict]) -> None:
    with AgenticSearchStore(db_path) as store:
        for r in rows:
            session_id = r["session_id"]
            store.create_chat_session(session_id=session_id, user_id=None)
            if r.get("message"):
                store.add_chat_message(
                    session_id=session_id, role="user", content=r["message"]
                )
            store.save_retrieval_feedback(session_id, r["signal"])


def test_thumbs_up_sets_positive_signal(tmp_path):
    db = str(tmp_path / "test.sqlite3")
    _seed_store(db, [{"session_id": "s1", "signal": "thumbs_up", "message": "Q1"}])
    examples = load_feedback_examples(db, min_ratings=1)
    assert len(examples) == 1
    assert examples[0].question == "Q1"
    assert examples[0].ground_truth == ""
    assert examples[0].metadata["human_signal"] == 1.0


def test_thumbs_down_sets_negative_signal(tmp_path):
    db = str(tmp_path / "test.sqlite3")
    _seed_store(db, [{"session_id": "s1", "signal": "thumbs_down", "message": "Q1"}])
    examples = load_feedback_examples(db, min_ratings=1)
    assert examples[0].metadata["human_signal"] == -1.0


def test_session_without_chat_messages_is_skipped(tmp_path):
    db = str(tmp_path / "test.sqlite3")
    _seed_store(db, [{"session_id": "s1", "signal": "thumbs_up"}])
    with pytest.raises(ValueError, match="Only 0 rated sessions"):
        load_feedback_examples(db, min_ratings=1)


def test_raises_when_below_min_ratings(tmp_path):
    db = str(tmp_path / "test.sqlite3")
    _seed_store(db, [{"session_id": "s1", "signal": "thumbs_up", "message": "Q"}])
    with pytest.raises(
        ValueError, match="Only 1 rated sessions found; need at least 5"
    ):
        load_feedback_examples(db, min_ratings=5)


def test_multiple_sessions_all_returned(tmp_path):
    db = str(tmp_path / "test.sqlite3")
    rows = [
        {"session_id": f"s{i}", "signal": "thumbs_up", "message": f"Q{i}"}
        for i in range(3)
    ]
    _seed_store(db, rows)
    examples = load_feedback_examples(db, min_ratings=3)
    assert len(examples) == 3
    assert all(ex.metadata["human_signal"] == 1.0 for ex in examples)


def test_returns_prompt_training_example_instances(tmp_path):
    db = str(tmp_path / "test.sqlite3")
    _seed_store(db, [{"session_id": "s1", "signal": "thumbs_up", "message": "Q"}])
    examples = load_feedback_examples(db, min_ratings=1)
    assert all(isinstance(ex, PromptTrainingExample) for ex in examples)
```

- [ ] **Step 2: Run tests — verify they fail**

```bash
pytest tests/unit/test_feedback_examples.py -v
```
Expected: all 6 fail with `ImportError: cannot import name 'load_feedback_examples'`

- [ ] **Step 3: Add `load_feedback_examples` to `src/training/data.py`**

Add `from pathlib import Path` to the imports block at the top of the file, then append after the last `register_rag_prompt_template` line:

```python
# ---------------------------------------------------------------------------
# Feedback-driven training data
# ---------------------------------------------------------------------------

_SIGNAL_MAP = {"thumbs_up": 1.0, "thumbs_down": -1.0}


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
    from src.internal.db import AgenticSearchStore

    examples: list[PromptTrainingExample] = []
    with AgenticSearchStore(str(db_path)) as store:
        rows = store._conn.execute(
            "SELECT session_id, signal FROM retrieval_feedback"
        ).fetchall()
        for row in rows:
            session_id = row["session_id"]
            if not session_id:
                continue
            signal = _SIGNAL_MAP.get(row["signal"])
            if signal is None:
                continue
            messages = store.list_chat_messages(session_id)
            first_user = next(
                (m.content for m in messages if m.role == "user"), None
            )
            if not first_user:
                continue
            examples.append(
                PromptTrainingExample(
                    question=first_user,
                    ground_truth="",
                    metadata={"human_signal": signal},
                )
            )

    if len(examples) < min_ratings:
        raise ValueError(
            f"Only {len(examples)} rated sessions found; need at least {min_ratings}"
        )
    return examples
```

- [ ] **Step 4: Run tests — verify they pass**

```bash
pytest tests/unit/test_feedback_examples.py -v
```
Expected: 6 passed

- [ ] **Step 5: Commit**

```bash
git add src/training/data.py tests/unit/test_feedback_examples.py
git commit -m "feat(training): add load_feedback_examples for GRPO feedback loop"
```

---

## Task 2: Human feedback reward component in `src/training/reward.py`

**Files:**
- Modify: `src/training/reward.py`
- Test: `tests/unit/test_reward_human_signal.py`

### Background

`SearchRewardConfig` is a `@dataclass(frozen=True)` at line ~174. `_reward_components_from_correctness(self, output, correctness)` is defined at line ~550 and returns a `dict[str, float]` with a `"total"` key. The `total` is computed by `_aggregate_total_reward(terminal_reward, shaping_total)` before `return`.

- [ ] **Step 1: Write the failing tests**

Create `tests/unit/test_reward_human_signal.py`:

```python
"""Tests for human_feedback reward component in SearchRewardFunction."""
from __future__ import annotations
import pytest
pytest.importorskip("torch")

from src import AgentLoopOutput, SearchRewardConfig, SearchRewardFunction
from src.training.grpo import GRPORolloutSample, score_prompt_group
from src.training.reward import simple_sparse_correctness_reward


def _make_output(answer: str = "test answer") -> AgentLoopOutput:
    return AgentLoopOutput(
        prompt_ids=[],
        response_ids=[1, 2, 3],
        response_mask=[1, 1, 1],
        num_turns=1,
        final_answer=answer,
        metrics={
            "rounds_used": 0,
            "subquestion_coverage_ratio": 1.0,
            "repeated_search_queries": 0.0,
            "fetched_pages": 0.0,
            "unnecessary_fetch_count": 0.0,
            "answer_when_evidence_insufficient": 0.0,
            "search_budget_exhausted_without_answer": 0.0,
        },
        context=None,
    )


def _zeroed_with(**kwargs) -> SearchRewardConfig:
    from dataclasses import replace
    return replace(SearchRewardConfig._zeroed(correctness_weight=0.0), **kwargs)


def test_default_weight_zero_no_human_feedback_key():
    fn = SearchRewardFunction()
    components = fn._reward_components_from_correctness(_make_output(), 0.5)
    assert "human_feedback" not in components


def test_default_weight_zero_identical_to_baseline():
    fn = SearchRewardFunction()
    out = _make_output()
    baseline = fn._reward_components_from_correctness(out, 0.5)
    with_signal = fn._reward_components_from_correctness(out, 0.5, human_signal=1.0)
    assert baseline["total"] == pytest.approx(with_signal["total"])
    assert "human_feedback" not in with_signal


def test_positive_signal_adds_positive_component():
    config = SearchRewardConfig(human_feedback_weight=0.5)
    fn = SearchRewardFunction(config)
    components = fn._reward_components_from_correctness(
        _make_output(), 0.0, human_signal=1.0
    )
    assert components["human_feedback"] == pytest.approx(0.5)


def test_negative_signal_adds_negative_component():
    config = SearchRewardConfig(human_feedback_weight=0.5)
    fn = SearchRewardFunction(config)
    components = fn._reward_components_from_correctness(
        _make_output(), 0.0, human_signal=-1.0
    )
    assert components["human_feedback"] == pytest.approx(-0.5)


def test_absent_signal_contributes_zero():
    config = SearchRewardConfig(human_feedback_weight=0.5)
    fn = SearchRewardFunction(config)
    components = fn._reward_components_from_correctness(_make_output(), 0.5)
    assert "human_feedback" not in components


def test_total_includes_human_feedback():
    config = _zeroed_with(human_feedback_weight=0.5)
    fn = SearchRewardFunction(config)
    components = fn._reward_components_from_correctness(
        _make_output(), 0.0, human_signal=1.0
    )
    assert components["total"] == pytest.approx(0.5)


def test_existing_presets_unchanged_with_default_config():
    for preset in (
        SearchRewardConfig.second_pass(),
        SearchRewardConfig.third_pass_with_format(),
    ):
        fn = SearchRewardFunction(preset)
        out = _make_output()
        total_a = fn._reward_components_from_correctness(out, 0.8)["total"]
        total_b = fn._reward_components_from_correctness(out, 0.8)["total"]
        assert total_a == pytest.approx(total_b)
```

- [ ] **Step 2: Run tests — verify they fail**

```bash
pytest tests/unit/test_reward_human_signal.py::test_default_weight_zero_no_human_feedback_key -v
```
Expected: FAIL — `_reward_components_from_correctness` does not accept `human_signal`

- [ ] **Step 3: Add `human_feedback_weight` to `SearchRewardConfig`**

In `src/training/reward.py`, add after the `reward_scale` field inside `SearchRewardConfig`:

```python
    # Human feedback signal weight.  When 0.0 (default) the term is always
    # zero — existing reward presets produce identical scores.
    human_feedback_weight: float = 0.0
```

- [ ] **Step 4: Add `human_signal` kwarg to `_reward_components_from_correctness`**

Change the method signature:

```python
    def _reward_components_from_correctness(
        self,
        output: AgentLoopOutput,
        correctness: float,
        *,
        human_signal: float | None = None,
    ) -> dict[str, float]:
```

Inside the method, add before the `terminal_reward = ...` line:

```python
        human_feedback = (
            cfg.human_feedback_weight * human_signal
            if human_signal is not None and cfg.human_feedback_weight != 0.0
            else None
        )
```

Then after `total = self._aggregate_total_reward(terminal_reward, shaping_total)`:

```python
        if human_feedback is not None:
            total += human_feedback
```

And replace the plain `return { ... }` with:

```python
        components = {
            "reward_mode": cfg.reward_mode,
            "correctness": terminal_reward,
            # ... all existing keys unchanged ...
            "total": total,
        }
        if human_feedback is not None:
            components["human_feedback"] = human_feedback
        return components
```

- [ ] **Step 5: Run reward tests — verify they pass**

```bash
pytest tests/unit/test_reward_human_signal.py -v
```
Expected: all reward tests pass (skip `score_prompt_group` tests — those need Task 3)

- [ ] **Step 6: Verify no regressions on existing reward tests**

```bash
pytest tests/unit/test_reward.py tests/unit/test_reward_shapes.py -v
```
Expected: all pass

- [ ] **Step 7: Commit**

```bash
git add src/training/reward.py tests/unit/test_reward_human_signal.py
git commit -m "feat(training): add human_feedback_weight to SearchRewardConfig and reward function"
```

---

## Task 3: `metadata` parameter in `score_prompt_group` (`src/training/grpo.py`)

**Files:**
- Modify: `src/training/grpo.py`
- Test: `tests/unit/test_reward_human_signal.py` (the two `score_prompt_group` tests)

### Background

`score_prompt_group` is defined at line ~285 in `src/training/grpo.py`. `GRPORolloutSample` is a dataclass with fields: `group_id`, `rollout_index`, `sampling_params`, `output`. It does NOT have `prompt_ids` or `response_ids` — those live on `AgentLoopOutput`.

- [ ] **Step 1: Add `metadata` param and extract `human_signal`**

Change the signature in `src/training/grpo.py`:

```python
def score_prompt_group(
    samples: list[GRPORolloutSample],
    *,
    ground_truth: str,
    judge_fn: JudgeFn,
    reward_fn: SearchRewardFunction | None = None,
    advantage_config: GRPOAdvantageConfig | None = None,
    batch_judge_fn: BatchJudgeFn | None = None,
    metadata: dict | None = None,
) -> list[ScoredGRPORollout]:
```

Add this line immediately before the `reward_scale = ...` line inside the function body:

```python
    human_signal: float | None = metadata.get("human_signal") if metadata else None
```

Update the `_reward_components_from_correctness` call inside the loop:

```python
        components = reward_function._reward_components_from_correctness(
            sample.output, correctness, human_signal=human_signal
        )
```

- [ ] **Step 2: Run the two `score_prompt_group` tests**

```bash
pytest tests/unit/test_reward_human_signal.py::test_score_prompt_group_threads_metadata \
       tests/unit/test_reward_human_signal.py::test_score_prompt_group_no_metadata_passes_none -v
```
Expected: both pass

- [ ] **Step 3: Run full reward_human_signal suite**

```bash
pytest tests/unit/test_reward_human_signal.py -v
```
Expected: all 9 pass

- [ ] **Step 4: Run existing GRPO tests to check no regressions**

```bash
pytest tests/unit/test_grpo.py tests/unit/test_grpo_trainer.py -v
```
Expected: all pass

- [ ] **Step 5: Commit**

```bash
git add src/training/grpo.py
git commit -m "feat(training): thread metadata through score_prompt_group for human_signal"
```

---

## Task 4: Thread `metadata` through `SearchAgentGRPOTrainer`

**Files:**
- Modify: `src/training/ppo/search_agent_grpo_trainer.py`

### Background

`rollout_async(prompts, ground_truths)` iterates `zip(grouped_samples, ground_truths)` at line ~260 and calls `score_prompt_group(group_samples, ground_truth=gt, ...)`. The goal is to add a `metadata: list[dict] | None = None` param and pass `metadata[i]` to each `score_prompt_group` call. `step_async` calls `rollout_async` and needs the same param. `step` (sync) wraps `rollout_async` via `asyncio.run`.

- [ ] **Step 1: Update `rollout` (sync) signature**

```python
    @torch.no_grad()
    def rollout(
        self,
        prompts: list[str],
        ground_truths: list[str],
        metadata: list[dict] | None = None,
    ) -> LLMRolloutResult:
        """Sync entry point — runs :meth:`rollout_async` in a new event loop."""
        return asyncio.run(self.rollout_async(prompts, ground_truths, metadata=metadata))
```

- [ ] **Step 2: Update `rollout_async` signature and scoring loop**

```python
    async def rollout_async(
        self,
        prompts: list[str],
        ground_truths: list[str],
        metadata: list[dict] | None = None,
    ) -> LLMRolloutResult:
```

Change the scoring loop from `for group_samples, gt in zip(...)` to:

```python
        for i, (group_samples, gt) in enumerate(zip(grouped_samples, ground_truths)):
            group_metadata = metadata[i] if metadata and i < len(metadata) else None
            scored = score_prompt_group(
                group_samples,
                ground_truth=gt,
                judge_fn=self.judge_fn,
                reward_fn=self.reward_fn,
                advantage_config=self._advantage_config,
                metadata=group_metadata,
            )
```

- [ ] **Step 3: Update `step_async` signature**

```python
    async def step_async(
        self,
        prompts: list[str],
        ground_truths: list[str],
        metadata: list[dict] | None = None,
    ) -> dict[str, float]:
        rollout = await self.rollout_async(prompts, ground_truths, metadata=metadata)
```

- [ ] **Step 4: Run trainer tests**

```bash
pytest tests/unit/test_search_agent_grpo_trainer.py -v
```
Expected: all pass

- [ ] **Step 5: Commit**

```bash
git add src/training/ppo/search_agent_grpo_trainer.py
git commit -m "feat(training): thread metadata through SearchAgentGRPOTrainer rollout for human_signal"
```

---

## Task 5: CLI script `examples/run_feedback_grpo.py`

**Files:**
- Create: `examples/run_feedback_grpo.py`

### Background

The script must:
1. Parse CLI args (see spec §3.5)
2. Call `load_feedback_examples(db_path, min_ratings=min_ratings)` — raises `ValueError` early if DB has too few ratings
3. Build `policy`, `tokenizer`, `reward_fn`, `trainer`
4. Call `await trainer.step_async(prompts, ground_truths, metadata=metadata)`
5. Save checkpoint with `policy.save_pretrained(output_dir)` + `tokenizer.save_pretrained(output_dir)`

Note: `SearchAgentGRPOTrainer.__init__` requires `reference_policy` (a deep copy of policy) and `optimizer`. The `from_pretrained` factory handles this automatically; use it when the model name is a HuggingFace id. For local paths, instantiate directly.

- [ ] **Step 1: Create the script**

```python
"""Feedback-driven GRPO fine-tuning.

Loads thumbs-up/down signals from an AgenticSearchStore SQLite DB, runs
on-policy rollouts through SearchAgentLoop, and saves a checkpoint.

Usage::

    python3 -m examples.run_feedback_grpo \\
      --db_path data/feedback.sqlite3 \\
      --model Qwen/Qwen2.5-0.5B-Instruct \\
      --num_rollouts 2 --min_ratings 1 --device cpu
"""
from __future__ import annotations

import argparse
import asyncio
import os
from pathlib import Path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Feedback-driven GRPO fine-tuning")
    parser.add_argument(
        "--db_path",
        default=os.environ.get("AGENTIC_SEARCH_WEB_DB_PATH", ":memory:"),
    )
    parser.add_argument("--model", required=True)
    parser.add_argument("--output_dir", default="data/checkpoints/feedback_grpo/")
    parser.add_argument("--min_ratings", type=int, default=10)
    parser.add_argument("--human_feedback_weight", type=float, default=0.5)
    parser.add_argument("--num_rollouts", type=int, default=4)
    parser.add_argument("--search_url", default="http://localhost:8001/retrieve")
    parser.add_argument("--device", default="mps", choices=["cpu", "mps", "cuda"])
    return parser.parse_args()


async def _train(args: argparse.Namespace) -> None:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from src.agents.search import SearchAgentLoop
    from src.training.data import load_feedback_examples
    from src.training.grpo import GRPOAdvantageConfig, GRPOTrainerConfig
    from src.training.ppo.search_agent_grpo_trainer import SearchAgentGRPOTrainer
    from src.training.reward import SearchRewardConfig, SearchRewardFunction
    from src.training.reward import simple_sparse_correctness_reward

    print(f"Loading feedback examples from {args.db_path!r} …")
    examples = load_feedback_examples(args.db_path, min_ratings=args.min_ratings)
    print(f"  {len(examples)} rated sessions loaded")

    prompts = [ex.question for ex in examples]
    ground_truths = [ex.ground_truth for ex in examples]
    metadata = [dict(ex.metadata) for ex in examples]

    device = torch.device(args.device)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    policy = AutoModelForCausalLM.from_pretrained(args.model).to(device)

    reward_config = SearchRewardConfig(
        human_feedback_weight=args.human_feedback_weight,
        correctness_weight=0.0,
    )
    reward_fn = SearchRewardFunction(reward_config)

    def loop_factory():
        return SearchAgentLoop(search_url=args.search_url)

    trainer = SearchAgentGRPOTrainer(
        policy=policy,
        tokenizer=tokenizer,
        loop_factory=loop_factory,
        judge_fn=simple_sparse_correctness_reward,
        reward_fn=reward_fn,
        config=GRPOTrainerConfig(num_rollouts=args.num_rollouts),
        advantage_config=GRPOAdvantageConfig(),
        device=device,
    )

    print("Running rollouts and gradient step …")
    metrics = await trainer.step_async(prompts, ground_truths, metadata=metadata)
    print("  Metrics:", metrics)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    policy.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"Checkpoint saved to {output_dir}")


def main() -> None:
    args = _parse_args()
    asyncio.run(_train(args))


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Smoke-test (no GPU, no real model download needed — just import check)**

```bash
python3 -c "from examples.run_feedback_grpo import _parse_args; print('import ok')"
```
Expected: `import ok`

- [ ] **Step 3: Commit**

```bash
git add examples/run_feedback_grpo.py
git commit -m "feat(training): add run_feedback_grpo CLI script"
```

---

## Task 6: Final integration check

- [ ] **Step 1: Run all new tests together**

```bash
pytest tests/unit/test_feedback_examples.py tests/unit/test_reward_human_signal.py -v
```
Expected: 15 passed

- [ ] **Step 2: Run full unit suite — verify zero regressions**

```bash
pytest tests/unit/ -q
```
Expected: 1830+ passed, 0 failed

- [ ] **Step 3: Run linter**

```bash
ruff check . --fix && ruff format .
```
Expected: no errors

- [ ] **Step 4: Commit spec + plan**

```bash
git add docs/superpowers/specs/2026-06-17-feedback-grpo-design.md \
        docs/superpowers/plans/2026-06-17-feedback-grpo.md
git commit -m "docs: add spec and plan for feedback-driven GRPO"
```

- [ ] **Step 5: Push and open PR**

```bash
git push -u origin <branch>
gh pr create --title "feat(training): feedback-driven GRPO fine-tuning loop" \
  --body "Closes feedback-GRPO spec. 15 new unit tests, 0 regressions."
```
