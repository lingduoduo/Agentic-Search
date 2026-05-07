"""Reward function and GRPO advantage computation for SearchAgentLoop rollouts."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable

from .agent_loop import AgentLoopOutput
from .context import AgentContext


@dataclass(frozen=True)
class SearchRewardConfig:
    # Reward aggregation mode:
    # - "shaped": final-answer reward plus process-level shaping terms
    # - "sparse_final_only": only the final-answer judge contributes to total
    reward_mode: str = "shaped"

    # Primary signal — weight applied to the judge score in [0, 1].
    correctness_weight: float = 1.0

    # Fraction of retrieved documents cited in the final answer, weighted by this.
    citation_support_weight: float = 0.3

    # Fraction of declared subquestions whose evidence was marked sufficient.
    subquestion_coverage_weight: float = 0.2

    # Reward for good evaluator verdicts and strong per-query search quality.
    search_quality_weight: float = 0.15

    # Applied per search round beyond the first (encourages efficiency).
    unnecessary_search_penalty: float = -0.05

    # Applied per repeated query issued across turns.
    duplicate_query_penalty: float = -0.1

    # Applied once if rounds_used / max_search_rounds exceeds this threshold.
    budget_penalty_threshold: float = 0.8
    budget_penalty: float = -0.1

    # Applied per fetched page that does not contribute to cited evidence.
    unnecessary_fetch_penalty: float = -0.1

    # Applied once when the agent answers despite insufficient evidence.
    answer_when_evidence_insufficient_penalty: float = -0.2

    # Applied once when the search budget is exhausted and no answer is produced.
    search_budget_exhausted_without_answer_penalty: float = -0.2

    # Reward when fetched pages are cited in the final answer.
    fetch_usefulness_reward: float = 0.1

    # Reference budget for the budget-penalty calculation.  Set to the same
    # value as SearchAgentLoopConfig.max_search_limit when known.
    max_search_rounds: int = 5

    @classmethod
    def sparse_final_only(
        cls,
        *,
        correctness_weight: float = 1.0,
    ) -> "SearchRewardConfig":
        """Build a strict sparse-reward config for agent RL.

        In this mode the rollout gets reward only from the final answer score.
        Search, reasoning, and retrieval traces remain available as labelled
        diagnostics, but they do not contribute to the optimisation target.
        """
        return cls(
            reward_mode="sparse_final_only",
            correctness_weight=correctness_weight,
            citation_support_weight=0.0,
            subquestion_coverage_weight=0.0,
            search_quality_weight=0.0,
            unnecessary_search_penalty=0.0,
            duplicate_query_penalty=0.0,
            budget_penalty_threshold=1.0,
            budget_penalty=0.0,
            unnecessary_fetch_penalty=0.0,
            answer_when_evidence_insufficient_penalty=0.0,
            search_budget_exhausted_without_answer_penalty=0.0,
            fetch_usefulness_reward=0.0,
        )


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

    def compute_token_rewards(
        self,
        output: AgentLoopOutput,
        ground_truth: str,
        judge_fn: Callable[[str, str], float],
    ) -> list[float]:
        """Sparse token-level reward vector for Agent RL training.

        Places the scalar rollout reward at the last model-generated token
        (last position where response_mask == 1) and 0 everywhere else.
        This mirrors the standard sparse-reward formulation: search tokens,
        reasoning tokens, and retrieval tokens receive no reward signal — only
        the final <answer> token does.

        Falls back to the last token when response_mask is absent or all-zero.
        """
        scalar = self.compute(output, ground_truth, judge_fn)
        n = len(output.response_ids)
        if n == 0:
            return []
        token_rewards = [0.0] * n
        mask = output.response_mask
        last_action_idx = next(
            (i for i in range(n - 1, -1, -1) if i < len(mask) and mask[i]),
            n - 1,
        )
        token_rewards[last_action_idx] = scalar
        return token_rewards

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

        # 4. Search quality: combines the evaluator's final sufficiency verdict
        # with the average fraction of strong queries across search rounds.
        search_quality = self._search_quality(metrics)

        # 5. Unnecessary-search penalty: each round beyond the first costs a little.
        rounds_used = int(metrics.get("rounds_used", 0))
        unnecessary_pen = cfg.unnecessary_search_penalty * max(0, rounds_used - 1)

        # 6. Duplicate-query penalty.
        dup_pen = cfg.duplicate_query_penalty * metrics.get(
            "repeated_search_queries", 0.0
        )

        # 7. Budget penalty: fired once when rounds consumed exceed the threshold.
        budget_fraction = rounds_used / max(cfg.max_search_rounds, 1)
        budget_pen = (
            cfg.budget_penalty
            if budget_fraction >= cfg.budget_penalty_threshold
            else 0.0
        )

        # 8. Explicit penalties for fetch waste, premature answering, and budget
        # exhaustion without an answer.
        unnecessary_fetch_pen = cfg.unnecessary_fetch_penalty * metrics.get(
            "unnecessary_fetch_count", 0.0
        )
        insufficient_answer_pen = (
            cfg.answer_when_evidence_insufficient_penalty
            * metrics.get("answer_when_evidence_insufficient", 0.0)
        )
        exhausted_without_answer_pen = (
            cfg.search_budget_exhausted_without_answer_penalty
            * metrics.get("search_budget_exhausted_without_answer", 0.0)
        )

        # 9. Fetch usefulness reward: fetched pages help only when the answer cites
        # search results whose URLs were later fetched for deeper inspection.
        fetch_reward = self._fetch_usefulness_reward(answer, ctx)

        terminal_reward = cfg.correctness_weight * correctness
        shaping_total = (
            cfg.citation_support_weight * citation_support
            + cfg.subquestion_coverage_weight * coverage
            + cfg.search_quality_weight * search_quality
            + unnecessary_pen
            + dup_pen
            + budget_pen
            + unnecessary_fetch_pen
            + insufficient_answer_pen
            + exhausted_without_answer_pen
            + fetch_reward
        )
        total = self._aggregate_total_reward(terminal_reward, shaping_total)
        return {
            "reward_mode": cfg.reward_mode,
            "correctness": terminal_reward,
            "citation_support": cfg.citation_support_weight * citation_support,
            "subquestion_coverage": cfg.subquestion_coverage_weight * coverage,
            "search_quality": cfg.search_quality_weight * search_quality,
            "unnecessary_search_penalty": unnecessary_pen,
            "duplicate_query_penalty": dup_pen,
            "budget_penalty": budget_pen,
            "unnecessary_fetch_penalty": unnecessary_fetch_pen,
            "answer_when_evidence_insufficient_penalty": insufficient_answer_pen,
            "search_budget_exhausted_without_answer_penalty": (
                exhausted_without_answer_pen
            ),
            "fetch_usefulness_reward": fetch_reward,
            "terminal_reward": terminal_reward,
            "shaping_total": shaping_total,
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

    @staticmethod
    def _search_quality(metrics: dict[str, float]) -> float:
        """Blend final evidence sufficiency with average per-round query quality."""
        final_sufficient = metrics.get("final_evidence_sufficient", 0.0)
        avg_quality = metrics.get("search_quality_score", 0.0)
        search_rounds = metrics.get("search_rounds", 0.0)

        if search_rounds <= 0:
            return metrics.get("answer_allowed", 0.0)
        return (final_sufficient + avg_quality) / 2.0

    def _fetch_usefulness_reward(
        self,
        answer: str,
        ctx: AgentContext | None,
    ) -> float:
        """Reward fetches only when fetched URLs are actually used in cited evidence."""
        if not answer or ctx is None or not ctx.fetched_pages:
            return 0.0

        fetched_urls = {page.url for page in ctx.fetched_pages if page.url}
        if not fetched_urls:
            return 0.0

        cited_urls = {result.url for result in ctx.cited_results(answer) if result.url}
        if cited_urls & fetched_urls:
            return self.config.fetch_usefulness_reward
        return 0.0

    def _aggregate_total_reward(
        self,
        terminal_reward: float,
        shaping_total: float,
    ) -> float:
        """Combine terminal reward and shaping according to the configured mode."""
        mode = self.config.reward_mode
        if mode == "shaped":
            return terminal_reward + shaping_total
        if mode == "sparse_final_only":
            return terminal_reward
        raise ValueError(
            f"Unsupported reward_mode: {mode!r}. "
            "Expected 'shaped' or 'sparse_final_only'."
        )
