"""Unit tests for the checkpointable GRPO outer loop (T-A0.1).

The current GRPO stack is single-step/foreground/no-resume; train_loop adds the
durable multi-step driver: periodic checkpointing, resume-from-checkpoint, and
step-level timeout/skip so a hung rollout does not abort the whole run.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

pytest.importorskip("torch")

from src.training.rl.train_loop import TrainLoopConfig, train_loop


class FakeTrainer:
    """Minimal trainer double: a monotonically advancing 'state' counter."""

    def __init__(self) -> None:
        self.counter = 0
        self.steps_run: list[int] = []
        self.saved: list[str] = []

    async def step_async(self, prompts, ground_truths, metadata=None):
        self.counter += 1
        self.steps_run.append(self.counter)
        return {"loss": 1.0 / self.counter, "mean_reward": float(self.counter)}

    def save_checkpoint(self, path: str) -> None:
        Path(path).mkdir(parents=True, exist_ok=True)
        (Path(path) / "state.txt").write_text(str(self.counter))
        self.saved.append(path)

    def load_checkpoint(self, path: str) -> None:
        self.counter = int((Path(path) / "state.txt").read_text())


def _run(coro):
    return asyncio.run(coro)


def test_runs_requested_number_of_steps() -> None:
    trainer = FakeTrainer()
    history = _run(train_loop(trainer, ["p"], ["gt"], TrainLoopConfig(max_steps=3)))

    assert [h["step"] for h in history] == [0, 1, 2]
    assert trainer.steps_run == [1, 2, 3]


def test_periodic_checkpoint_and_metrics_jsonl(tmp_path) -> None:
    trainer = FakeTrainer()
    cfg = TrainLoopConfig(max_steps=4, ckpt_dir=str(tmp_path), ckpt_every=2)

    _run(train_loop(trainer, ["p"], ["gt"], cfg))

    assert (tmp_path / "step_2").is_dir()
    assert (tmp_path / "step_4").is_dir()
    assert not (tmp_path / "step_3").exists()
    rows = (tmp_path / "metrics.jsonl").read_text().strip().splitlines()
    assert len(rows) == 4
    assert json.loads(rows[0])["step"] == 0


def test_resume_continues_from_saved_step(tmp_path) -> None:
    trainer = FakeTrainer()
    cfg = TrainLoopConfig(max_steps=4, ckpt_dir=str(tmp_path), ckpt_every=2)
    _run(train_loop(trainer, ["p"], ["gt"], cfg))

    # Fresh trainer resumes from the step-4 checkpoint and continues to step 6.
    resumed = FakeTrainer()
    resume_cfg = TrainLoopConfig(max_steps=6, ckpt_dir=str(tmp_path), ckpt_every=2)
    history = _run(
        train_loop(
            resumed, ["p"], ["gt"], resume_cfg, resume_from=str(tmp_path / "step_4")
        )
    )

    assert resumed.counter >= 4  # restored prior state, not from zero
    assert [h["step"] for h in history] == [4, 5]


def test_hung_step_times_out_and_is_skipped() -> None:
    class HangingTrainer(FakeTrainer):
        def __init__(self) -> None:
            super().__init__()
            self.attempts = 0

        async def step_async(self, prompts, ground_truths, metadata=None):
            self.attempts += 1
            if self.attempts == 1:
                await asyncio.sleep(10)  # first attempt hangs; cancelled by timeout
            return await super().step_async(prompts, ground_truths, metadata)

    trainer = HangingTrainer()
    history = _run(
        train_loop(
            trainer, ["p"], ["gt"], TrainLoopConfig(max_steps=3, step_timeout_s=0.02)
        )
    )

    # Step 0 hung and was skipped; steps 1 and 2 ran.
    assert [h["step"] for h in history] == [1, 2]


def test_failing_step_is_skipped_without_aborting() -> None:
    class FlakyTrainer(FakeTrainer):
        async def step_async(self, prompts, ground_truths, metadata=None):
            self.counter += 1
            if self.counter == 1:
                raise RuntimeError("boom")
            self.steps_run.append(self.counter)
            return {"loss": 0.0}

    trainer = FlakyTrainer()
    history = _run(train_loop(trainer, ["p"], ["gt"], TrainLoopConfig(max_steps=3)))

    assert [h["step"] for h in history] == [1, 2]  # step 0 raised, was skipped
