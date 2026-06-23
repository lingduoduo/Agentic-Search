"""Durable multi-step driver for GRPO training.

The trainers expose a single ``step_async`` that runs one rollout+update. On its
own that is single-step and foreground: a crash loses all progress and a hung
rollout stalls the run. ``train_loop`` wraps the step with the durability a
long run needs:

- a checkpointable outer loop (periodic save of model/optimizer + step),
- resume-from-checkpoint (continue at the saved step),
- step-level timeout + skip (a hung or failing step is logged and skipped, not
  fatal),
- per-step metrics persisted to JSONL for observability.

It is trainer-agnostic: any object with ``async step_async(prompts,
ground_truths, metadata=None) -> dict`` works. Model-state persistence is
delegated to the trainer's optional ``save_checkpoint(path)`` /
``load_checkpoint(path)`` methods; the loop owns the step manifest and metrics.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class TrainLoopConfig:
    max_steps: int
    ckpt_dir: str | None = None
    ckpt_every: int = 0  # 0 disables periodic checkpointing
    step_timeout_s: float | None = None  # None disables the per-step timeout


def save_checkpoint(trainer: Any, path: str, step: int) -> None:
    """Persist trainer state plus the step manifest under *path*."""
    Path(path).mkdir(parents=True, exist_ok=True)
    trainer_save = getattr(trainer, "save_checkpoint", None)
    if trainer_save is not None:
        trainer_save(path)
    (Path(path) / "trainer_state.json").write_text(json.dumps({"step": step}))


def load_checkpoint(trainer: Any, path: str) -> int:
    """Restore trainer state from *path*; return the step to resume at."""
    trainer_load = getattr(trainer, "load_checkpoint", None)
    if trainer_load is not None:
        trainer_load(path)
    state = json.loads((Path(path) / "trainer_state.json").read_text())
    return int(state["step"])


async def train_loop(
    trainer: Any,
    prompts: list[str],
    ground_truths: list[str],
    config: TrainLoopConfig,
    *,
    metadata: list[dict] | None = None,
    resume_from: str | None = None,
    on_metrics: Callable[[dict], None] | None = None,
) -> list[dict]:
    """Run up to ``config.max_steps`` training steps, durably.

    Returns the per-step metrics history (skipped steps are absent).
    """
    import asyncio

    start_step = load_checkpoint(trainer, resume_from) if resume_from else 0
    history: list[dict] = []

    for step in range(start_step, config.max_steps):
        try:
            coro = trainer.step_async(prompts, ground_truths, metadata=metadata)
            if config.step_timeout_s is not None:
                metrics = await asyncio.wait_for(coro, config.step_timeout_s)
            else:
                metrics = await coro
        except Exception as exc:  # noqa: BLE001 - a bad step must not abort the run
            logger.warning(
                "Training step %d failed or timed out (%s); skipping.", step, exc
            )
            continue

        record = {**metrics, "step": step}
        history.append(record)
        if on_metrics is not None:
            on_metrics(record)
        if config.ckpt_dir is not None:
            _append_jsonl(Path(config.ckpt_dir) / "metrics.jsonl", record)
        if (
            config.ckpt_dir is not None
            and config.ckpt_every
            and (step + 1) % config.ckpt_every == 0
        ):
            save_checkpoint(
                trainer, str(Path(config.ckpt_dir) / f"step_{step + 1}"), step + 1
            )

    return history


def _append_jsonl(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as fh:
        fh.write(json.dumps(record) + "\n")
