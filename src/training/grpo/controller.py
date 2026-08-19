"""Local PPO / GRPO training controllers.

The controller layer owns orchestration: collect grouped rollouts, score them,
collate trajectories, compute log-probs/loss, and optionally step an optimizer.
Rollout mechanics stay on ``LLMGenerationManager`` and tensor math stays in
``core_algos``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable

if TYPE_CHECKING:
    from src.model.generation import LLMGenerationManager

from ..ppo.core_algos import PPOPolicyLossConfig


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
        rewards = [float(item.reward) for item in group]
        mean_reward = sum(rewards) / len(rewards)
        variance = sum((r - mean_reward) ** 2 for r in rewards) / max(len(rewards), 1)
        std = variance**0.5
        for item in group:
            item.advantage = (float(item.reward) - mean_reward) / (std + 1e-8)
        return group

    def _build_single_batches(self, prompt_batch: Any) -> list[Any]:
        from src.model.generation import _single_prompt_batch

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
        from src import GRPOAdvantageConfig
        from src.model.generation import GRPORolloutSafetyConfig

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
        from src.model.generation import (
            GRPOPromptGroupResult,
            apply_safety_penalties_to_scored_rollouts,
            score_group_rollout,
        )

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
        from src.model.generation import GRPOTrainingStepResult

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
        from src.model.generation import async_run_prompt_rollout_group

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
