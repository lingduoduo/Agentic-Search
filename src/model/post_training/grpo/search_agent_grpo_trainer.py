"""GRPO trainer backed by a live SearchAgentLoop for fully shaped rewards.

Extends :class:`LLMGRPOTrainer` by replacing the ``model.generate()``-based
rollout with real multi-turn ``SearchAgentLoop`` executions.  This unlocks the
full ``SearchRewardFunction`` signal — citation support, search quality,
unnecessary-fetch penalty, fetch-usefulness reward — because each rollout
produces a genuine :class:`AgentLoopOutput` with populated ``metrics`` and
``context`` rather than the hollow ``pseudo_output`` used by the parent class.

Concept mapping (extends llm_grpo_trainer.py):

    LLMGRPOTrainer                SearchAgentGRPOTrainer
    ─────────────────────────     ──────────────────────────────────────────
    model.generate()              SearchAgentLoop.run()   (async, multi-turn)
    judge_fn(answer, gt)          score_prompt_group()    (full shaped reward)
    pseudo_output                 real AgentLoopOutput    (citations + metrics)
    compute_loss() unchanged      compute_loss() inherited as-is
    step() sync                   step() / step_async()   both available

Typical usage::

    from transformers import AutoModelForCausalLM, AutoTokenizer
    from src.agents.search import SearchAgentLoop
    from src.model.post_training.reward import SearchRewardFunction, SearchRewardConfig

    def loop_factory() -> SearchAgentLoop:
        return SearchAgentLoop(tokenizer=tok, server_manager=server)

    trainer = SearchAgentGRPOTrainer.from_pretrained(
        "Qwen/Qwen2.5-1.5B-Instruct",
        judge_fn=exact_match,
        loop_factory=loop_factory,
        reward_fn=SearchRewardFunction(SearchRewardConfig.second_pass()),
        sampling_params={"temperature": 0.8, "max_tokens": 512},
    )
    metrics = trainer.step(prompts=["What is FAISS?"], ground_truths=["..."])
"""

from __future__ import annotations

import asyncio
import copy
from typing import TYPE_CHECKING, Any, Callable

import torch
import torch.nn as nn

from src.agents.core.base import AgentLoopBase
from src.model.post_training.grpo.rollouts import (
    GRPOAdvantageConfig,
    score_prompt_group,
    sample_prompt_group,
)

from .core_algos import compute_grpo_outcome_advantage
from .llm_grpo_trainer import (
    LLMGRPOConfig,
    LLMGRPOTrainer,
    LLMRolloutResult,
    _sparse_reward_at_eos,
    get_response_log_probs,
)

if TYPE_CHECKING:
    from src.model.post_training.reward import JudgeFn, SearchRewardFunction


# Default ceiling on concurrent agent rollouts. Bounding this (rather than
# leaving it unbounded) keeps a batch of B×G live search rollouts from
# saturating / rate-limiting the retrieval server during long runs.
DEFAULT_MAX_CONCURRENT = 8


def _resolve_max_concurrent(value: int | None) -> int:
    """Always return a positive bound; None falls back to the default ceiling."""
    return DEFAULT_MAX_CONCURRENT if value is None else value


class SearchAgentGRPOTrainer(LLMGRPOTrainer):
    """GRPO trainer that samples via live agent loops for full shaped reward.

    Inherits ``compute_loss()`` and ``step()`` from :class:`LLMGRPOTrainer`
    unchanged — only ``rollout()`` is replaced with an async agent-loop version
    that produces real :class:`~src.agents.core.base.AgentLoopOutput` objects.

    Args:
        loop_factory: Zero-argument callable that returns a fresh
            :class:`~src.agents.core.base.AgentLoopBase` instance per rollout.
            Called ``num_rollouts`` times per prompt group concurrently.
        sampling_params: Default sampling parameters forwarded to
            ``AgentLoopBase.run()``.  Individual rollouts receive per-rollout
            variants (temperature sweep) built by ``build_grpo_sampling_params``
            unless ``sampling_variants`` is provided explicitly.
        sampling_variants: If given, must have length ``num_rollouts``; overrides
            the automatic temperature-sweep logic.
        max_concurrent: Maximum number of agent loops running concurrently
            across the whole batch.  ``None`` (default) launches everything at
            once.  Reduce this if the inference server is capacity-limited.
        advantage_config: Controls within-group advantage normalisation passed
            to ``score_prompt_group``.
    """

    def __init__(
        self,
        policy: nn.Module,
        reference_policy: nn.Module,
        tokenizer: Any,
        optimizer: torch.optim.Optimizer,
        judge_fn: JudgeFn,
        loop_factory: Callable[[], AgentLoopBase],
        *,
        config: LLMGRPOConfig | None = None,
        reward_fn: SearchRewardFunction | None = None,
        sampling_params: dict[str, Any] | None = None,
        sampling_variants: list[dict[str, Any]] | None = None,
        max_concurrent: int | None = None,
        advantage_config: GRPOAdvantageConfig | None = None,
        device: str | torch.device | None = None,
    ) -> None:
        super().__init__(
            policy=policy,
            reference_policy=reference_policy,
            tokenizer=tokenizer,
            optimizer=optimizer,
            judge_fn=judge_fn,
            config=config,
            reward_fn=reward_fn,
            device=device,
        )
        self.loop_factory = loop_factory
        self._sampling_params: dict[str, Any] = sampling_params or {
            "temperature": 0.7,
            "max_tokens": 512,
        }
        self._sampling_variants = sampling_variants
        self._max_concurrent = _resolve_max_concurrent(max_concurrent)
        self._advantage_config = advantage_config or GRPOAdvantageConfig()

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def from_pretrained(
        cls,
        model_name_or_path: str,
        judge_fn: JudgeFn,
        loop_factory: Callable[[], AgentLoopBase],
        *,
        optimizer_cls: type = torch.optim.AdamW,
        lr: float = 1e-5,
        config: LLMGRPOConfig | None = None,
        reward_fn: SearchRewardFunction | None = None,
        sampling_params: dict[str, Any] | None = None,
        sampling_variants: list[dict[str, Any]] | None = None,
        max_concurrent: int | None = None,
        advantage_config: GRPOAdvantageConfig | None = None,
        device: str | None = None,
        **hf_kwargs: Any,
    ) -> SearchAgentGRPOTrainer:
        """Load policy + tokenizer from *model_name_or_path* and build trainer."""
        from transformers import AutoModelForCausalLM, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, **hf_kwargs)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = tokenizer.eos_token_id

        policy = AutoModelForCausalLM.from_pretrained(model_name_or_path, **hf_kwargs)
        reference = copy.deepcopy(policy)

        resolved_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        policy = policy.to(resolved_device)
        reference = reference.to(resolved_device)

        optimizer = optimizer_cls(policy.parameters(), lr=lr)
        return cls(
            policy=policy,
            reference_policy=reference,
            tokenizer=tokenizer,
            optimizer=optimizer,
            judge_fn=judge_fn,
            loop_factory=loop_factory,
            config=config,
            reward_fn=reward_fn,
            sampling_params=sampling_params,
            sampling_variants=sampling_variants,
            max_concurrent=max_concurrent,
            advantage_config=advantage_config,
            device=resolved_device,
        )

    # ------------------------------------------------------------------
    # Rollout  (overrides LLMGRPOTrainer.rollout)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def rollout(
        self,
        prompts: list[str],
        ground_truths: list[str],
        metadata: list[dict] | None = None,
    ) -> LLMRolloutResult:
        """Sync entry point — runs :meth:`rollout_async` in a new event loop."""
        return asyncio.run(
            self.rollout_async(prompts, ground_truths, metadata=metadata)
        )

    async def rollout_async(
        self,
        prompts: list[str],
        ground_truths: list[str],
        metadata: list[dict] | None = None,
    ) -> LLMRolloutResult:
        """Run live agent loops for each prompt and assemble an LLMRolloutResult.

        Each prompt spawns ``num_rollouts`` concurrent :class:`AgentLoopBase`
        executions, each producing a real :class:`~src.agents.core.base.AgentLoopOutput`
        with populated ``metrics`` and ``context``.  These are scored by
        ``score_prompt_group`` using the full :class:`SearchRewardFunction`
        signal (citations, search quality, etc.).

        Args:
            prompts: List of B raw text prompts.
            ground_truths: List of B gold answers.

        Returns:
            :class:`LLMRolloutResult` with ``(B*G, ...)`` tensors ready for
            :meth:`compute_loss`.
        """
        B = len(prompts)
        G = self.config.num_rollouts
        pad_id = (
            self.tokenizer.pad_token_id
            if hasattr(self.tokenizer, "pad_token_id")
            else 0
        ) or 0

        # ── 1. Run G agent loops per prompt (concurrent, always bounded) ──
        semaphore = asyncio.Semaphore(self._max_concurrent)

        async def _sample_one(messages: list[dict[str, Any]]):
            if semaphore is not None:
                async with semaphore:
                    return await sample_prompt_group(
                        self.loop_factory,
                        messages=messages,
                        sampling_params=self._sampling_params,
                        num_rollouts=G,
                        sampling_variants=self._sampling_variants,
                    )
            return await sample_prompt_group(
                self.loop_factory,
                messages=messages,
                sampling_params=self._sampling_params,
                num_rollouts=G,
                sampling_variants=self._sampling_variants,
            )

        batch_messages = [[{"role": "user", "content": p}] for p in prompts]
        grouped_samples = list(
            await asyncio.gather(*[_sample_one(m) for m in batch_messages])
        )

        # ── 2. Score each group with the full reward function ─────────────────
        # score_prompt_group calls reward_fn._reward_components_from_correctness()
        # with the real AgentLoopOutput — citations and metrics are populated.
        rewards_list: list[float] = []
        all_response_ids: list[list[int]] = []
        all_prompt_ids: list[list[int]] = []  # last-turn prompt per rollout

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
            for s in scored:
                rewards_list.append(s.reward)
                all_response_ids.append(s.output.response_ids)
                all_prompt_ids.append(s.output.prompt_ids)

        # ── 3. Pad response_ids to uniform length ─────────────────────────────
        max_resp_len = max(len(r) for r in all_response_ids) if all_response_ids else 1

        response_ids = torch.tensor(
            [r + [pad_id] * (max_resp_len - len(r)) for r in all_response_ids],
            dtype=torch.long,
            device=self.device,
        )  # (B*G, T)

        response_mask = torch.tensor(
            [[1] * len(r) + [0] * (max_resp_len - len(r)) for r in all_response_ids],
            dtype=torch.long,
            device=self.device,
        )  # (B*G, T)

        rewards = torch.tensor(
            rewards_list, dtype=torch.float32, device=self.device
        )  # (B*G,)

        # ── 4. Group IDs and sparse token advantages ──────────────────────────
        group_ids = torch.arange(B, device=self.device).repeat_interleave(G)

        token_rewards = _sparse_reward_at_eos(rewards, response_mask.cpu()).to(
            self.device
        )  # (B*G, T)

        advantages, _ = compute_grpo_outcome_advantage(
            token_rewards,
            response_mask,
            group_ids,
            epsilon=self.config.advantage_epsilon,
            clip_advantages=self.config.advantage_clip,
        )  # (B*G, T)

        # ── 5. Prompt IDs — one per prompt (not per rollout) ─────────────────
        # Use the first rollout's prompt_ids for each prompt as the representative
        # context.  All rollouts for the same prompt share the same initial prompt.
        prompt_ids_per_prompt = [all_prompt_ids[i * G] for i in range(B)]
        max_prompt_len = max(len(p) for p in prompt_ids_per_prompt)

        prompt_ids = torch.tensor(
            [[pad_id] * (max_prompt_len - len(p)) + p for p in prompt_ids_per_prompt],
            dtype=torch.long,
            device=self.device,
        )  # (B, P)

        # ── 6. Old log-probs under current policy (snapshot before update) ────
        tiled_prompt = prompt_ids.repeat_interleave(G, dim=0)  # (B*G, P)
        prompt_len = tiled_prompt.shape[1]
        full_ids = torch.cat([tiled_prompt, response_ids], dim=1)  # (B*G, P+T)

        old_log_probs = get_response_log_probs(
            self.policy, full_ids, prompt_len, response_mask
        )  # (B*G, T)

        return LLMRolloutResult(
            prompt_ids=prompt_ids,
            response_ids=response_ids,
            response_mask=response_mask,
            old_log_probs=old_log_probs,
            rewards=rewards,
            advantages=advantages,
            group_ids=group_ids,
        )

    # ------------------------------------------------------------------
    # Async step convenience  (compute_loss + gradient update)
    # ------------------------------------------------------------------

    async def step_async(
        self,
        prompts: list[str],
        ground_truths: list[str],
        metadata: list[dict] | None = None,
    ) -> dict[str, float]:
        """Async version of :meth:`step` — avoids nested ``asyncio.run()`` calls.

        Use this when the caller already runs inside an event loop (e.g. a
        training loop driven by ``asyncio.run(train())``).
        """
        rollout = await self.rollout_async(prompts, ground_truths, metadata=metadata)

        self.policy.train()
        loss, metrics = self.compute_loss(rollout)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.config.grad_clip)
        self.optimizer.step()

        return metrics
