"""Reward function and GRPO advantage computation for SearchAgentLoop rollouts."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable

from .agent_loop import AgentLoopOutput
from .context import AgentContext


@dataclass(frozen=True)
class SearchRewardConfig:
    # Primary signal — weight applied to the judge score in [0, 1].
    correctness_weight: float = 1.0

    # Fraction of retrieved documents cited in the final answer, weighted by this.
    citation_support_weight: float = 0.3

    # Fraction of declared subquestions whose evidence was marked sufficient.
    subquestion_coverage_weight: float = 0.2

    # Applied per search round beyond the first (encourages efficiency).
    unnecessary_search_penalty: float = -0.05

    # Applied per repeated query issued across turns.
    duplicate_query_penalty: float = -0.05

    # Applied once if rounds_used / max_search_rounds exceeds this threshold.
    budget_penalty_threshold: float = 0.8
    budget_penalty: float = -0.1

    # Reward when fetched pages are cited in the final answer.
    fetch_usefulness_reward: float = 0.1

    # Reference budget for the budget-penalty calculation.  Set to the same
    # value as SearchAgentLoopConfig.max_search_limit when known.
    max_search_rounds: int = 5


class SearchRewardFunction:
    """Computes per-rollout rewards and GRPO advantages for search agent training.

    Usage::

        reward_fn = SearchRewardFunction()
        rewards = [
            reward_fn.compute(output, ground_truth=gt, judge_fn=exact_match)
            for output, gt in zip(outputs, ground_truths)
        ]
        advantages = reward_fn.compute_batch_advantages(rewards, group_ids)
    """

    def __init__(self, config: SearchRewardConfig | None = None) -> None:
        self.config = config if config is not None else SearchRewardConfig()

    # ------------------------------------------------------------------
    # Per-rollout reward
    # ------------------------------------------------------------------

    def compute(
        self,
        output: AgentLoopOutput,
        ground_truth: str,
        judge_fn: Callable[[str, str], float],
    ) -> float:
        """Return a scalar reward for one rollout.

        Args:
            output: The :class:`AgentLoopOutput` produced by
                :meth:`SearchAgentLoop.run`.
            ground_truth: The reference answer string.
            judge_fn: ``(answer, ground_truth) -> float`` in ``[0, 1]``.
                Common choices: exact-match, token-F1, or an LLM judge.

        Returns:
            A scalar reward (may be negative when penalties dominate).
        """
        return self.reward_components(output, ground_truth, judge_fn)["total"]

    def reward_components(
        self,
        output: AgentLoopOutput,
        ground_truth: str,
        judge_fn: Callable[[str, str], float],
    ) -> dict[str, float]:
        """Same as :meth:`compute` but returns a labelled breakdown for logging.

        The ``"total"`` key equals what :meth:`compute` returns.
        """
        cfg = self.config
        answer = output.final_answer or ""
        metrics = output.metrics
        ctx: AgentContext | None = output.context

        # 1. Answer correctness (primary signal).
        correctness = judge_fn(answer, ground_truth) if answer else 0.0

        # 2. Citation support: fraction of retrieved docs cited in the answer.
        citation_support = self._citation_support(answer, ctx)

        # 3. Subquestion coverage: fraction of tasks with sufficient evidence.
        coverage = metrics.get("subquestion_coverage_ratio", 1.0)

        # 4. Unnecessary-search penalty: each round beyond the first costs a little.
        rounds_used = int(metrics.get("rounds_used", 0))
        unnecessary_pen = cfg.unnecessary_search_penalty * max(0, rounds_used - 1)

        # 5. Duplicate-query penalty.
        dup_pen = cfg.duplicate_query_penalty * metrics.get(
            "repeated_search_queries", 0.0
        )

        # 6. Budget penalty: fired once when rounds consumed exceed the threshold.
        budget_fraction = rounds_used / max(cfg.max_search_rounds, 1)
        budget_pen = (
            cfg.budget_penalty
            if budget_fraction >= cfg.budget_penalty_threshold
            else 0.0
        )

        # 7. Fetch usefulness reward: pages were fetched AND at least one is cited.
        # The reward requires an actual citation — a non-empty answer alone is not
        # sufficient, because that would grant a free bonus for any <fetch> call.
        fetch_reward = 0.0
        if metrics.get("fetched_pages", 0.0) > 0 and ctx is not None:
            if ctx.cited_result_ids(answer):
                fetch_reward = cfg.fetch_usefulness_reward

        total = (
            cfg.correctness_weight * correctness
            + cfg.citation_support_weight * citation_support
            + cfg.subquestion_coverage_weight * coverage
            + unnecessary_pen
            + dup_pen
            + budget_pen
            + fetch_reward
        )
        return {
            "correctness": cfg.correctness_weight * correctness,
            "citation_support": cfg.citation_support_weight * citation_support,
            "subquestion_coverage": cfg.subquestion_coverage_weight * coverage,
            "unnecessary_search_penalty": unnecessary_pen,
            "duplicate_query_penalty": dup_pen,
            "budget_penalty": budget_pen,
            "fetch_usefulness_reward": fetch_reward,
            "total": total,
        }

    # ------------------------------------------------------------------
    # GRPO advantage computation
    # ------------------------------------------------------------------

    def compute_batch_advantages(
        self,
        rewards: list[float],
        group_ids: list[str],
    ) -> list[float]:
        """Normalise rewards within each prompt group for GRPO training.

        Each advantage is ``(reward - group_mean) / (group_std + eps)``.
        Groups with a single sample get advantage 0.0 (no within-group signal).

        Variance uses the population formula (N denominator) so that advantages
        are scaled consistently regardless of group size.  If the surrounding
        GRPO trainer uses sample variance (N-1), pass ``group_std`` from there
        instead of calling this method directly.

        Args:
            rewards: One scalar reward per rollout.
            group_ids: Prompt-group identifier for each rollout.  Rollouts that
                share a ``group_id`` were generated from the same prompt and are
                normalised together.

        Returns:
            A list of advantages aligned with *rewards*.
        """
        if len(rewards) != len(group_ids):
            raise ValueError("rewards and group_ids must have the same length.")

        groups: dict[str, list[tuple[int, float]]] = {}
        for idx, (gid, r) in enumerate(zip(group_ids, rewards)):
            groups.setdefault(gid, []).append((idx, r))

        advantages = [0.0] * len(rewards)
        for group in groups.values():
            if len(group) == 1:
                continue  # single-sample group: no within-group signal
            indices, group_rewards = zip(*group)
            mean = sum(group_rewards) / len(group_rewards)
            # Population variance (N denominator) — see docstring for tradeoffs.
            variance = sum((r - mean) ** 2 for r in group_rewards) / len(group_rewards)
            std = math.sqrt(variance)
            for idx, r in zip(indices, group_rewards):
                advantages[idx] = (r - mean) / (std + 1e-8)
        return advantages

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _citation_support(answer: str, ctx: AgentContext | None) -> float:
        """Fraction of retrieved results that are cited in *answer*."""
        if ctx is None or ctx.num_results == 0 or not answer:
            return 0.0
        cited = len(ctx.cited_result_ids(answer))
        return cited / ctx.num_results
