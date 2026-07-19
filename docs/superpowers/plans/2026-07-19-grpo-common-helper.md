# GRPO Common Helper — Implementation Plan

**Goal:** Extract the duplicated feedback-GRPO step into `examples/_grpo_common.py` and repoint both scripts.

## Global Constraints
- Never commit to `main`; branch `refactor/grpo-common-helper`.
- Behavior-preserving; no CLI/flag changes to either documented script.

### Task 1: Extract helper + repoint scripts (TDD)

- [ ] **Step 1: Write the failing test** — `tests/unit/test_grpo_common.py`

```python
import pytest

pytest.importorskip("torch")


@pytest.mark.asyncio
async def test_run_feedback_grpo_step_wires_and_saves(monkeypatch, tmp_path):
    import examples._grpo_common as gc
    from unittest.mock import AsyncMock, MagicMock
    import types

    ex = [
        types.SimpleNamespace(
            question="q1", ground_truth="a1", metadata={"m": 1}
        ),
        types.SimpleNamespace(
            question="q2", ground_truth="a2", metadata={"m": 2}
        ),
    ]
    monkeypatch.setattr(
        "src.training.data.load_feedback_examples", lambda *a, **k: ex
    )
    trainer = MagicMock()
    trainer.step_async = AsyncMock(return_value={"loss": 0.1})
    trainer.policy = MagicMock()
    trainer.tokenizer = MagicMock()
    monkeypatch.setattr(
        "src.training.ppo.search_agent_grpo_trainer.SearchAgentGRPOTrainer.from_pretrained",
        lambda *a, **k: trainer,
    )

    out = tmp_path / "ckpt"
    metrics = await gc.run_feedback_grpo_step(
        model_path="base",
        db_path=":memory:",
        output_dir=out,
        min_ratings=1,
        human_feedback_weight=0.5,
        num_rollouts=2,
        search_url="http://x/retrieve",
        device="cpu",
    )

    assert metrics == {"loss": 0.1}
    args, kwargs = trainer.step_async.call_args
    assert args[0] == ["q1", "q2"]            # prompts
    assert args[1] == ["a1", "a2"]            # ground_truths
    assert kwargs["metadata"] == [{"m": 1}, {"m": 2}]
    trainer.policy.save_pretrained.assert_called_once_with(out)
    trainer.tokenizer.save_pretrained.assert_called_once_with(out)
```

- [ ] **Step 2: Run to verify it fails**

Run: `python -m pytest tests/unit/test_grpo_common.py -q`
Expected: FAIL — `ModuleNotFoundError: examples._grpo_common`.

- [ ] **Step 3: Create `examples/_grpo_common.py`**

```python
"""Shared GRPO step for the feedback-driven example trainers.

run_feedback_grpo.py and run_sft_grpo.py (Phase 2) both run the same
on-policy GRPO step over thumbs-up/down feedback examples; this is that step.
"""

from __future__ import annotations

from pathlib import Path


async def run_feedback_grpo_step(
    *,
    model_path: str,
    db_path: str,
    output_dir: str | Path,
    min_ratings: int,
    human_feedback_weight: float,
    num_rollouts: int,
    search_url: str,
    device: str,
) -> dict:
    """Load feedback examples, run one GRPO step from ``model_path``, save a
    checkpoint to ``output_dir``, and return the step metrics."""
    from src.agents.search import SearchAgentLoop
    from src.training.data import load_feedback_examples
    from src.training.grpo import GRPOAdvantageConfig
    from src.training.ppo.llm_grpo_trainer import LLMGRPOConfig
    from src.training.ppo.search_agent_grpo_trainer import SearchAgentGRPOTrainer
    from src.training.reward import (
        SearchRewardConfig,
        SearchRewardFunction,
        simple_sparse_correctness_reward,
    )

    feedback_examples = load_feedback_examples(db_path, min_ratings=min_ratings)
    print(f"  {len(feedback_examples)} rated sessions loaded")

    prompts = [ex.question for ex in feedback_examples]
    ground_truths = [ex.ground_truth for ex in feedback_examples]
    metadata = [dict(ex.metadata) for ex in feedback_examples]

    reward_fn = SearchRewardFunction(
        SearchRewardConfig(
            human_feedback_weight=human_feedback_weight,
            correctness_weight=0.0,
        )
    )

    def loop_factory():
        return SearchAgentLoop(search_url=search_url)

    trainer = SearchAgentGRPOTrainer.from_pretrained(
        model_path,
        judge_fn=simple_sparse_correctness_reward,
        loop_factory=loop_factory,
        reward_fn=reward_fn,
        config=LLMGRPOConfig(num_rollouts=num_rollouts),
        advantage_config=GRPOAdvantageConfig(),
        device=device,
    )

    metrics = await trainer.step_async(prompts, ground_truths, metadata=metadata)

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    trainer.policy.save_pretrained(out)
    trainer.tokenizer.save_pretrained(out)
    print(f"  GRPO checkpoint saved to {out}")
    return metrics
```

- [ ] **Step 4: Run the test — GREEN**

Run: `python -m pytest tests/unit/test_grpo_common.py -q`
Expected: PASS.

- [ ] **Step 5: Repoint `run_feedback_grpo.py._train`**

Replace the body of `_train` (its imports + everything from `load_feedback_examples`
through the checkpoint print) with:

```python
async def _train(args: argparse.Namespace) -> None:
    from examples._grpo_common import run_feedback_grpo_step

    print(f"Loading feedback examples from {args.db_path!r} …")
    metrics = await run_feedback_grpo_step(
        model_path=args.model,
        db_path=args.db_path,
        output_dir=args.output_dir,
        min_ratings=args.min_ratings,
        human_feedback_weight=args.human_feedback_weight,
        num_rollouts=args.num_rollouts,
        search_url=args.search_url,
        device=args.device,
    )
    print("  Metrics:", metrics)
```

- [ ] **Step 6: Repoint `run_sft_grpo.py._train` Phase 2**

Keep Phase 1 (SFT) exactly as is. Replace the Phase 2 block (from
`print("[Phase 2] ...")` through the final checkpoint print) with:

```python
    # ── Phase 2: GRPO ────────────────────────────────────────────────────────
    from examples._grpo_common import run_feedback_grpo_step

    print("[Phase 2] Running GRPO over feedback examples …")
    metrics = await run_feedback_grpo_step(
        model_path=grpo_model_path,
        db_path=args.db_path,
        output_dir=args.grpo_output_dir,
        min_ratings=args.min_ratings,
        human_feedback_weight=args.human_feedback_weight,
        num_rollouts=args.num_rollouts,
        search_url=args.search_url,
        device=args.device,
    )
    print(f"  GRPO metrics: {metrics}")
```

Remove any imports in `run_sft_grpo._train` that are now unused (the GRPO-only
imports: `GRPOAdvantageConfig`, `LLMGRPOConfig`, `SearchAgentGRPOTrainer`,
`SearchRewardConfig`, `SearchRewardFunction`, `simple_sparse_correctness_reward`,
and `SearchAgentLoop` if not used by Phase 1). Keep the SFT/torch/transformers
imports and `load_feedback_examples`/`load_sft_examples` as still needed by Phase 1
(`load_sft_examples`) — note `load_feedback_examples` moves into the helper, so
drop it from the script's import if Phase 1 doesn't use it.

- [ ] **Step 7: Verify both scripts import and parse args**

Run: `python -c "import examples.run_feedback_grpo, examples.run_sft_grpo; print('import ok')"`
Run: `python -m examples.run_feedback_grpo --help >/dev/null && python -m examples.run_sft_grpo --help >/dev/null && echo 'help ok'`
Expected: `import ok` then `help ok` (argparse still works, no unused-import crash).

- [ ] **Step 8: ruff + commit**

Run: `ruff check examples/_grpo_common.py examples/run_feedback_grpo.py examples/run_sft_grpo.py tests/unit/test_grpo_common.py`
```bash
git add examples/_grpo_common.py examples/run_feedback_grpo.py examples/run_sft_grpo.py tests/unit/test_grpo_common.py
git commit -m "refactor: extract shared feedback-GRPO step into examples/_grpo_common.py"
```
