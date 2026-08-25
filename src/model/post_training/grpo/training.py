"""Local PPO / GRPO training controllers.

The controller layer owns orchestration: collect grouped rollouts, score them,
collate trajectories, compute log-probs/loss, and optionally step an optimizer.
Rollout mechanics stay on ``LLMGenerationManager`` and tensor math stays in
``core_algos``.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..ppo.core_algos import PPOPolicyLossConfig
from ..reward import group_relative_advantages
from .generation import (
    GRPOPromptGroupResult,
    GRPORolloutSafetyConfig,
    GRPOTrainingStepResult,
    LLMGenerationManager,
    LogProbCapable,
    _single_prompt_batch,
    apply_safety_penalties_to_scored_rollouts,
    async_run_prompt_rollout_group,
    score_group_rollout,
)
from .rollouts import GRPOAdvantageConfig

logger = logging.getLogger(__name__)


@dataclass
class RolloutResult:
    """Small generic rollout result for local controller experiments."""

    prompt_id: int
    rollout_id: int
    trajectory: Any
    reward: float = 0.0
    advantage: float = 0.0


class LocalGRPOController:
    """Mac-friendly, single-process GRPO controller.

    No Ray, RPC workers, CUDA ranks, or distributed worker groups.  This class
    coordinates existing local components through an ``LLMGenerationManager``.

    The sync ``collect_rollouts`` / ``training_step`` are sequential and safe
    to mock in tests.  The ``async_collect_rollouts`` / ``async_training_step``
    variants run all ``N_prompts × N_rollouts`` trajectories concurrently,
    overlapping HTTP search I/O — use these from an async training loop.
    """

    def __init__(
        self,
        manager: LLMGenerationManager,
        *,
        num_rollouts: int = 4,
        max_workers: int | None = None,
    ) -> None:
        self.manager = manager
        self.num_rollouts = int(num_rollouts)
        self.max_workers = max_workers

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def assign_group_advantages(group: list[RolloutResult]) -> list[RolloutResult]:
        """Assign std-normalized advantages to a simple rollout group."""
        if not group:
            return group
        advantages = group_relative_advantages(
            [float(item.reward) for item in group], normalize=True
        )
        for item, advantage in zip(group, advantages):
            item.advantage = advantage
        return group

    def _build_single_batches(self, prompt_batch: Any) -> list[Any]:
        return [
            _single_prompt_batch(prompt_batch, i)
            for i in range(len(prompt_batch.questions))
        ]

    def _resolve_configs(
        self,
        advantage_config: Any,
        safety_config: Any,
    ) -> tuple[Any, Any, bool]:
        """Return (advantage_config, safety_config, normalize_advantages)."""
        resolved_adv = advantage_config or GRPOAdvantageConfig(
            mode="group_outcome",
            reward_component="total",
        )
        resolved_safety = safety_config or GRPORolloutSafetyConfig(
            max_search_rounds=self.manager.config.max_search_rounds,
            max_total_rounds=self.manager.config.max_total_rounds,
            allowed_actions=tuple(self.manager.config.allowed_actions),
        )
        normalize = resolved_adv.mode == "group_std_normalized"
        return resolved_adv, resolved_safety, normalize

    def _score_grouped_rollouts(
        self,
        prompt_batch: Any,
        single_batches: list[Any],
        grouped_results: list[list[Any]],
        *,
        judge_fn: Callable[[str, str], float],
        reward_fn: Any,
        advantage_config: Any,
        batch_judge_fn: Any,
        safety_config: Any,
    ) -> tuple[list[Any], list[Any]]:
        """Score, apply safety penalties, and flatten grouped rollouts.

        Shared by both the sync and async collect paths so scoring logic
        lives in exactly one place.
        """
        resolved_adv, resolved_safety, normalize = self._resolve_configs(
            advantage_config, safety_config
        )

        group_results: list[Any] = []
        scored_rollouts: list[Any] = []
        for question, single_batch, grouped in zip(
            prompt_batch.questions, single_batches, grouped_results
        ):
            scored = score_group_rollout(
                grouped,
                ground_truth=single_batch.ground_truths[0],
                judge_fn=judge_fn,
                reward_fn=reward_fn,
                advantage_config=resolved_adv,
                batch_judge_fn=batch_judge_fn,
            )
            scored = apply_safety_penalties_to_scored_rollouts(
                scored,
                config=resolved_safety,
                normalize_advantages=normalize,
            )
            group_results.append(
                GRPOPromptGroupResult(
                    question=question,
                    ground_truth=single_batch.ground_truths[0],
                    grouped_rollouts=grouped,
                    scored_rollouts=scored,
                )
            )
            scored_rollouts.extend(scored)

        return group_results, scored_rollouts

    def _apply_loss_and_step(
        self,
        group_results: list[Any],
        scored_rollouts: list[Any],
        *,
        old_backend: Any,
        new_backend: Any,
        ref_backend: Any,
        loss_config: PPOPolicyLossConfig | None,
        optimizer: Any,
    ) -> Any:
        """Collate trajectories, compute log-probs + policy loss, step optimizer."""
        training_batch = self.manager.collate_scored_rollouts_for_training(
            scored_rollouts
        )

        if "old_log_probs" not in training_batch.batch or old_backend is not None:
            self.manager.compute_log_prob(
                training_batch,
                backend=old_backend or self.manager.generation_backend,
                store_key="old_log_probs",
                overwrite=(
                    old_backend is not None
                    or "old_log_probs" not in training_batch.batch
                ),
            )
        self.manager.compute_log_prob(
            training_batch,
            backend=new_backend or self.manager.generation_backend,
            store_key="new_log_probs",
        )
        if ref_backend is not None:
            self.manager.compute_log_prob(
                training_batch,
                backend=ref_backend,
                store_key="ref_log_probs",
            )

        loss = self.manager.compute_policy_loss(training_batch, config=loss_config)

        optimizer_stepped = False
        if optimizer is not None:
            try:
                optimizer.zero_grad(set_to_none=True)
            except TypeError:
                optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            optimizer_stepped = True

        rewards = [s.reward for s in scored_rollouts]
        advantages = [s.advantage for s in scored_rollouts]
        return GRPOTrainingStepResult(
            group_results=group_results,
            scored_rollouts=scored_rollouts,
            training_batch=training_batch,
            loss=loss,
            optimizer_stepped=optimizer_stepped,
            mean_reward=(sum(rewards) / len(rewards)) if rewards else 0.0,
            mean_advantage=(sum(advantages) / len(advantages)) if advantages else 0.0,
        )

    # ------------------------------------------------------------------
    # Async rollout collection
    # ------------------------------------------------------------------

    async def async_collect_rollouts(
        self,
        prompt_batch: Any,
        *,
        search_mode: str,
        sampling_params: dict[str, Any],
        judge_fn: Callable[[str, str], float],
        num_rollouts: int | None = None,
        reward_fn: Any = None,
        advantage_config: Any = None,
        batch_judge_fn: Any = None,
        safety_config: Any = None,
        base_seed: int | None = None,
        current_step: int = 0,
        total_steps: int = 1,
    ) -> tuple[list[Any], list[Any]]:
        """Collect all rollouts for every prompt concurrently.

        All ``N_prompts × N_rollouts`` trajectories run in parallel, overlapping
        HTTP search I/O.  Returns ``(group_results, scored_rollouts)``.
        """
        resolved_num = int(num_rollouts or self.num_rollouts)
        single_batches = self._build_single_batches(prompt_batch)

        grouped_results = await async_run_prompt_rollout_group(
            self.manager,
            single_batches,
            search_mode=search_mode,
            sampling_params=sampling_params,
            num_rollouts=resolved_num,
            base_seed=base_seed,
            current_step=current_step,
            total_steps=total_steps,
            max_workers=self.max_workers,
        )

        return self._score_grouped_rollouts(
            prompt_batch,
            single_batches,
            grouped_results,
            judge_fn=judge_fn,
            reward_fn=reward_fn,
            advantage_config=advantage_config,
            batch_judge_fn=batch_judge_fn,
            safety_config=safety_config,
        )

    # ------------------------------------------------------------------
    # Sync collect (sequential, uses manager.run_prompt_rollout_group)
    # ------------------------------------------------------------------

    def collect_rollouts(
        self,
        prompt_batch: Any,
        *,
        search_mode: str,
        sampling_params: dict[str, Any],
        judge_fn: Callable[[str, str], float],
        num_rollouts: int | None = None,
        reward_fn: Any = None,
        advantage_config: Any = None,
        batch_judge_fn: Any = None,
        safety_config: Any = None,
        base_seed: int | None = None,
        current_step: int = 0,
        total_steps: int = 1,
    ) -> tuple[list[Any], list[Any]]:
        """Collect rollouts sequentially, one prompt at a time.

        Calls ``self.manager.run_prompt_rollout_group`` per prompt so the
        manager can be mocked in tests.  For concurrent rollout collection
        use ``async_collect_rollouts`` instead.
        """
        resolved_num = int(num_rollouts or self.num_rollouts)
        single_batches = self._build_single_batches(prompt_batch)

        grouped_results: list[list[Any]] = []
        for i, single_batch in enumerate(single_batches):
            seed = None if base_seed is None else base_seed + i * resolved_num
            grouped = self.manager.run_prompt_rollout_group(
                single_batch,
                search_mode=search_mode,
                sampling_params=sampling_params,
                num_rollouts=resolved_num,
                base_seed=seed,
                current_step=current_step,
                total_steps=total_steps,
            )
            grouped_results.append(grouped)

        return self._score_grouped_rollouts(
            prompt_batch,
            single_batches,
            grouped_results,
            judge_fn=judge_fn,
            reward_fn=reward_fn,
            advantage_config=advantage_config,
            batch_judge_fn=batch_judge_fn,
            safety_config=safety_config,
        )

    # ------------------------------------------------------------------
    # Async training step
    # ------------------------------------------------------------------

    async def async_training_step(
        self,
        prompt_batch: Any,
        *,
        search_mode: str,
        sampling_params: dict[str, Any],
        judge_fn: Callable[[str, str], float],
        num_rollouts: int | None = None,
        reward_fn: Any = None,
        advantage_config: Any = None,
        batch_judge_fn: Any = None,
        old_backend: Any = None,
        new_backend: Any = None,
        ref_backend: Any = None,
        loss_config: PPOPolicyLossConfig | None = None,
        safety_config: Any = None,
        optimizer: Any = None,
        base_seed: int | None = None,
        current_step: int = 0,
        total_steps: int = 1,
    ) -> Any:
        """Run one GRPO training step with concurrent rollout collection."""
        group_results, scored_rollouts = await self.async_collect_rollouts(
            prompt_batch,
            search_mode=search_mode,
            sampling_params=sampling_params,
            judge_fn=judge_fn,
            num_rollouts=num_rollouts,
            reward_fn=reward_fn,
            advantage_config=advantage_config,
            batch_judge_fn=batch_judge_fn,
            safety_config=safety_config,
            base_seed=base_seed,
            current_step=current_step,
            total_steps=total_steps,
        )
        return self._apply_loss_and_step(
            group_results,
            scored_rollouts,
            old_backend=old_backend,
            new_backend=new_backend,
            ref_backend=ref_backend,
            loss_config=loss_config,
            optimizer=optimizer,
        )

    # ------------------------------------------------------------------
    # Sync training step
    # ------------------------------------------------------------------

    def training_step(
        self,
        prompt_batch: Any,
        *,
        search_mode: str,
        sampling_params: dict[str, Any],
        judge_fn: Callable[[str, str], float],
        num_rollouts: int | None = None,
        reward_fn: Any = None,
        advantage_config: Any = None,
        batch_judge_fn: Any = None,
        old_backend: Any = None,
        new_backend: Any = None,
        ref_backend: Any = None,
        loss_config: PPOPolicyLossConfig | None = None,
        safety_config: Any = None,
        optimizer: Any = None,
        base_seed: int | None = None,
        current_step: int = 0,
        total_steps: int = 1,
    ) -> Any:
        """Sequential training step using ``collect_rollouts``.

        Uses ``manager.run_prompt_rollout_group`` under the hood so the manager
        can be mocked in tests.  For concurrent rollout collection use
        ``async_training_step`` instead.
        """
        group_results, scored_rollouts = self.collect_rollouts(
            prompt_batch,
            search_mode=search_mode,
            sampling_params=sampling_params,
            judge_fn=judge_fn,
            num_rollouts=num_rollouts,
            reward_fn=reward_fn,
            advantage_config=advantage_config,
            batch_judge_fn=batch_judge_fn,
            safety_config=safety_config,
            base_seed=base_seed,
            current_step=current_step,
            total_steps=total_steps,
        )
        return self._apply_loss_and_step(
            group_results,
            scored_rollouts,
            old_backend=old_backend,
            new_backend=new_backend,
            ref_backend=ref_backend,
            loss_config=loss_config,
            optimizer=optimizer,
        )


async def async_run_grpo_training_step(
    manager: LLMGenerationManager,
    prompt_batch: Any,
    *,
    search_mode: str,
    sampling_params: dict[str, Any],
    judge_fn: Callable[[str, str], float],
    num_rollouts: int = 4,
    reward_fn: Any = None,
    advantage_config: Any = None,
    batch_judge_fn: Any = None,
    old_backend: LogProbCapable | None = None,
    new_backend: LogProbCapable | None = None,
    ref_backend: LogProbCapable | None = None,
    loss_config: PPOPolicyLossConfig | None = None,
    safety_config: GRPORolloutSafetyConfig | None = None,
    optimizer: Any = None,
    base_seed: int | None = None,
    current_step: int = 0,
    total_steps: int = 1,
    max_workers: int | None = None,
) -> GRPOTrainingStepResult:
    """Run one GRPO trainer step with concurrent rollout collection.

    Delegates to ``LocalGRPOController.async_training_step`` which runs all
    ``N_prompts × N_rollouts`` trajectories in parallel, overlapping HTTP
    search I/O, then performs one learner-side update.
    """
    return await LocalGRPOController(
        manager, num_rollouts=num_rollouts, max_workers=max_workers
    ).async_training_step(
        prompt_batch,
        search_mode=search_mode,
        sampling_params=sampling_params,
        judge_fn=judge_fn,
        num_rollouts=num_rollouts,
        reward_fn=reward_fn,
        advantage_config=advantage_config,
        batch_judge_fn=batch_judge_fn,
        old_backend=old_backend,
        new_backend=new_backend,
        ref_backend=ref_backend,
        loss_config=loss_config,
        safety_config=safety_config,
        optimizer=optimizer,
        base_seed=base_seed,
        current_step=current_step,
        total_steps=total_steps,
    )


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
