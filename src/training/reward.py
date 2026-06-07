"""Reward function and GRPO outcome-advantage computation for SearchAgentLoop rollouts."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Callable

from ..agents.base import AgentLoopOutput
from ..context.search import AgentContext

# Type aliases for judge callables.
JudgeFn = Callable[[str, str], float]
BatchJudgeFn = Callable[[list[str], list[str]], list[float]]
_WHITESPACE_PATTERN = re.compile(r"\s+")


def normalize_answer_text(text: str) -> str:
    """Normalize an answer string for simple sparse-reward matching."""
    lowered = text.strip().lower()
    lowered = _WHITESPACE_PATTERN.sub(" ", lowered)
    return lowered


def simple_sparse_correctness_reward(
    pred: str,
    gold: str,
    *,
    partial_match_reward: float = 0.7,
) -> float:
    """Simple first-pass sparse reward for final answers.

    Rules:
    - exact normalized match -> 1.0
    - normalized gold is contained in normalized prediction -> 0.7
    - otherwise -> 0.0
    """
    norm_pred = normalize_answer_text(pred)
    norm_gold = normalize_answer_text(gold)
    if not norm_pred or not norm_gold:
        return 0.0
    if norm_pred == norm_gold:
        return 1.0
    if norm_gold in norm_pred:
        return float(partial_match_reward)
    return 0.0


def _score_answers(
    judge_fn: JudgeFn,
    answers: list[str],
    ground_truths: list[str],
    *,
    batch_judge_fn: BatchJudgeFn | None = None,
) -> list[float]:
    """Score answer/ground-truth pairs, preferring the batch judge when provided.

    Pass `batch_judge_fn` to have an LLM judge score all pairs in a single
    API call instead of making N separate round-trips.
    """
    if batch_judge_fn is not None:
        return [float(s) for s in batch_judge_fn(answers, ground_truths)]
    return [float(judge_fn(a, g)) for a, g in zip(answers, ground_truths)]


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

    # Simple first-pass efficiency penalty: subtract this per search round used.
    # Example: -0.02 * num_searches
    per_search_penalty: float = 0.0

    # Applied per search round beyond the first (encourages efficiency).
    unnecessary_search_penalty: float = -0.05

    # Applied per repeated query issued across turns.
    duplicate_query_penalty: float = -0.1

    # Applied once if rounds_used / max_search_rounds exceeds this threshold.
    budget_penalty_threshold: float = 0.8
    budget_penalty: float = -0.1

    # Applied per fetched page that does not contribute to cited evidence.
    unnecessary_fetch_penalty: float = -0.1

    # Applied once when the agent answers with no retrieval citations despite
    # having issued searches that returned results.  A negative value penalises
    # answers that make claims without any retrieved evidence — i.e. the agent
    # searched, got documents, and then ignored them.
    # Phase 2 recommended: -0.1
    unsupported_claim_penalty: float = 0.0

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
            per_search_penalty=0.0,
            unnecessary_search_penalty=0.0,
            duplicate_query_penalty=0.0,
            budget_penalty_threshold=1.0,
            budget_penalty=0.0,
            unnecessary_fetch_penalty=0.0,
            answer_when_evidence_insufficient_penalty=0.0,
            search_budget_exhausted_without_answer_penalty=0.0,
            fetch_usefulness_reward=0.0,
        )

    @classmethod
    def simple_sparse_with_search_penalty(
        cls,
        *,
        correctness_weight: float = 1.0,
        per_search_penalty: float = -0.02,
    ) -> "SearchRewardConfig":
        """Build the recommended first-pass reward for early agent RL.

        Objective:

            reward = correctness_reward - 0.02 * num_searches

        Everything else is disabled so the signal stays easy to reason about.
        Use with :func:`simple_sparse_correctness_reward` as the judge for a
        straightforward first training phase.
        """
        return cls(
            reward_mode="shaped",
            correctness_weight=correctness_weight,
            citation_support_weight=0.0,
            subquestion_coverage_weight=0.0,
            search_quality_weight=0.0,
            per_search_penalty=per_search_penalty,
            unnecessary_search_penalty=0.0,
            duplicate_query_penalty=0.0,
            budget_penalty_threshold=1.0,
            budget_penalty=0.0,
            unnecessary_fetch_penalty=0.0,
            answer_when_evidence_insufficient_penalty=0.0,
            search_budget_exhausted_without_answer_penalty=0.0,
            fetch_usefulness_reward=0.0,
        )

    @classmethod
    def second_pass(
        cls,
        *,
        correctness_weight: float = 1.0,
        per_search_penalty: float = -0.02,
        citation_support_weight: float = 0.1,
        unsupported_claim_penalty: float = -0.1,
        duplicate_query_penalty: float = -0.05,
    ) -> "SearchRewardConfig":
        """Phase 2 reward: first-pass formula plus citation and query-quality signals.

        Adds three terms to the Phase 1 (simple sparse + search penalty) objective:

            reward = correctness
                   - 0.02 * num_searches            ← search efficiency
                   + 0.1  * citation_support         ← fraction of retrieved docs cited
                   - 0.1  * unsupported_claim        ← searched but cited nothing
                   - 0.05 * repeated_query_count     ← duplicate search penalty

        Use :func:`simple_sparse_correctness_reward` as the judge unless you
        already have a trained LLM evaluator.  Move to Phase 2 only after the
        model reliably produces answers in Phase 1 (policy has converged enough
        to benefit from the finer-grained signal).
        """
        return cls(
            reward_mode="shaped",
            correctness_weight=correctness_weight,
            per_search_penalty=per_search_penalty,
            citation_support_weight=citation_support_weight,
            unsupported_claim_penalty=unsupported_claim_penalty,
            duplicate_query_penalty=duplicate_query_penalty,
            subquestion_coverage_weight=0.0,
            search_quality_weight=0.0,
            unnecessary_search_penalty=0.0,
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

    def compute_terminal_reward(
        self,
        output: AgentLoopOutput,
        ground_truth: str,
        judge_fn: Callable[[str, str], float],
    ) -> float:
        """Return the final-answer reward with no process shaping.

        This is the strict sparse-reward signal used in many Agent RL setups:

            reward = judge_fn(final_answer, ground_truth)

        It ignores search, retrieval, reasoning, and fetch behaviour entirely.
        Use this when you want an explicit terminal-only optimisation target
        regardless of the configured reward aggregation mode.
        """
        answer = output.final_answer or ""
        if not answer:
            return 0.0
        return float(self.config.correctness_weight) * judge_fn(answer, ground_truth)

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
        token_rewards[self._last_action_idx(output)] = scalar
        return token_rewards

    def compute_sparse_token_rewards(
        self,
        output: AgentLoopOutput,
        ground_truth: str,
        judge_fn: Callable[[str, str], float],
    ) -> list[float]:
        """Strict sparse token rewards using only the terminal answer score.

        Unlike :meth:`compute_token_rewards`, this helper never includes any
        process shaping. The final-answer reward is placed only at the last
        model-generated token and every earlier token gets 0.0.
        """
        scalar = self.compute_terminal_reward(output, ground_truth, judge_fn)
        n = len(output.response_ids)
        if n == 0:
            return []
        token_rewards = [0.0] * n
        token_rewards[self._last_action_idx(output)] = scalar
        return token_rewards

    def compute_batch_sparse_token_rewards(
        self,
        outputs: list[AgentLoopOutput],
        ground_truths: list[str],
        judge_fn: JudgeFn,
        *,
        batch_judge_fn: BatchJudgeFn | None = None,
    ) -> list[list[float]]:
        """Strict sparse token rewards for an entire batch in one call.

        The standard Agent RL sparse-reward loop:

            for each rollout i:
                token_rewards[i][last_action_token] = correctness_weight
                                                      * judge(answer_i, gt_i)
                token_rewards[i][all other tokens]  = 0.0

        Search tokens, reasoning tokens, and retrieval tokens always receive
        zero reward — only the final ``<answer>`` token carries the signal.

        Args:
            outputs: One :class:`AgentLoopOutput` per rollout.
            ground_truths: Reference answer for each rollout.
            judge_fn: ``(answer, ground_truth) -> float`` in ``[0, 1]``.
            batch_judge_fn: Optional batch variant
                ``(list[answers], list[ground_truths]) -> list[float]``.
                Supply this for LLM judges so all N pairs are scored in a
                single API call rather than N separate round-trips.

        Returns:
            One token-level reward vector per rollout.
        """
        if len(outputs) != len(ground_truths):
            raise ValueError("outputs and ground_truths must have the same length.")
        answers = [output.final_answer or "" for output in outputs]
        scores = _score_answers(
            judge_fn, answers, ground_truths, batch_judge_fn=batch_judge_fn
        )
        result: list[list[float]] = []
        for output, score in zip(outputs, scores):
            n = len(output.response_ids)
            if n == 0:
                result.append([])
                continue
            terminal = (
                float(self.config.correctness_weight) * score
                if output.final_answer
                else 0.0
            )
            token_rewards = [0.0] * n
            token_rewards[self._last_action_idx(output)] = terminal
            result.append(token_rewards)
        return result

    def reward_components(
        self,
        output: AgentLoopOutput,
        ground_truth: str,
        judge_fn: Callable[[str, str], float],
    ) -> dict[str, float]:
        """Same as :meth:`compute` but returns a labelled breakdown for logging.

        The ``"total"`` key equals what :meth:`compute` returns.
        """
        answer = output.final_answer or ""
        correctness = judge_fn(answer, ground_truth) if answer else 0.0
        return self._reward_components_from_correctness(output, correctness)

    def _reward_components_from_correctness(
        self,
        output: AgentLoopOutput,
        correctness: float,
    ) -> dict[str, float]:
        """Compute reward breakdown with a pre-scored correctness value.

        Separating the judge call from the arithmetic lets callers batch all
        judge calls upfront (e.g. using a single LLM API request for N rollouts)
        and then compute the full breakdown per sample without re-calling the judge.
        """
        cfg = self.config
        answer = output.final_answer or ""
        metrics = output.metrics
        ctx: AgentContext | None = output.context

        # 2. Citation support: fraction of retrieved docs cited in the answer.
        citation_support = self._citation_support(answer, ctx)

        # 3. Subquestion coverage: fraction of tasks with sufficient evidence.
        coverage = metrics.get("subquestion_coverage_ratio", 1.0)

        # 4. Search quality: combines the evaluator's final sufficiency verdict
        # with the average fraction of strong queries across search rounds.
        search_quality = self._search_quality(metrics)

        # 5. Unnecessary-search penalty: each round beyond the first costs a little.
        rounds_used = int(metrics.get("rounds_used", 0))
        per_search_pen = cfg.per_search_penalty * rounds_used
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

        # 9. Unsupported-claim penalty: fires when the agent searched and got
        # results but the answer cites none of them — the model is fabricating.
        unsupported_claim_pen = self._unsupported_claim_penalty(answer, ctx, metrics)

        # 10. Fetch usefulness reward: fetched pages help only when the answer cites
        # search results whose URLs were later fetched for deeper inspection.
        fetch_reward = self._fetch_usefulness_reward(answer, ctx)

        terminal_reward = cfg.correctness_weight * correctness
        shaping_total = (
            cfg.citation_support_weight * citation_support
            + cfg.subquestion_coverage_weight * coverage
            + cfg.search_quality_weight * search_quality
            + per_search_pen
            + unnecessary_pen
            + dup_pen
            + budget_pen
            + unnecessary_fetch_pen
            + unsupported_claim_pen
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
            "per_search_penalty": per_search_pen,
            "unnecessary_search_penalty": unnecessary_pen,
            "duplicate_query_penalty": dup_pen,
            "budget_penalty": budget_pen,
            "unnecessary_fetch_penalty": unnecessary_fetch_pen,
            "unsupported_claim_penalty": unsupported_claim_pen,
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

    def assign_grpo_outcome_token_advantages(
        self,
        outputs: list[AgentLoopOutput],
        ground_truths: list[str],
        judge_fn: JudgeFn,
        group_ids: list[str],
        *,
        batch_judge_fn: BatchJudgeFn | None = None,
    ) -> list[list[float]]:
        """End-to-end GRPO outcome advantages as sparse token-level vectors.

        Full pipeline:

            1. Terminal reward per rollout (sparse — no process shaping):
                   r_i = correctness_weight * judge_fn(answer_i, gt_i)

            2. Group-relative centering (no value model):
                   A_i = r_i - mean({r_j : group_j == group_i})

            3. Std normalization within the group:
                   A_i = A_i / (std(group) + ε)

            4. Sparse token assignment — place A_i at the last
               action token (last response_mask == 1 position) and 0
               everywhere else.  All search, reasoning, and retrieval
               tokens carry no gradient signal.

        This is the DeepSeek-R1 / GRPO training loop in one call:
        no separate value model, no token-level critic, no cross-prompt mixing.

        Args:
            outputs: One AgentLoopOutput per rollout.
            ground_truths: Reference answer for each rollout.
            judge_fn: ``(answer, ground_truth) -> float`` in ``[0, 1]``.
            group_ids: Prompt-group identifier.  Rollouts sharing an ID were
                sampled from the same prompt and are normalised together.
            batch_judge_fn: Optional batch variant
                ``(list[answers], list[ground_truths]) -> list[float]``.
                Supply this for LLM judges to score all N rollouts in a
                single API call instead of N separate calls.

        Returns:
            One token-level advantage vector per rollout, aligned with outputs.
            Each vector has length ``len(output.response_ids)`` and is all-zero
            except at the last action token.
        """
        if len(outputs) != len(ground_truths) or len(outputs) != len(group_ids):
            raise ValueError(
                "outputs, ground_truths, and group_ids must have the same length."
            )

        answers = [output.final_answer or "" for output in outputs]
        scores = _score_answers(
            judge_fn, answers, ground_truths, batch_judge_fn=batch_judge_fn
        )
        rewards = [
            float(self.config.correctness_weight) * score
            if output.final_answer
            else 0.0
            for output, score in zip(outputs, scores)
        ]
        scalar_advantages = self.compute_batch_advantages(rewards, group_ids)

        result: list[list[float]] = []
        for output, adv in zip(outputs, scalar_advantages):
            n = len(output.response_ids)
            if n == 0:
                result.append([])
                continue
            token_advs = [0.0] * n
            token_advs[self._last_action_idx(output)] = adv
            result.append(token_advs)
        return result

    def compute_grpo_outcome_advantages(
        self,
        rewards: list[float],
        group_ids: list[str],
    ) -> list[float]:
        """Compute GRPO outcome advantages without a critic value model.

        For each prompt group, advantage is centered by the mean reward of the
        trajectories sampled from that same prompt:

            advantage_i = reward_i - mean(group_rewards)

        This is the core outcome-based GRPO signal: no separate value model, no
        token-level critic targets, and no cross-prompt mixing.
        Single-sample groups get advantage 0.0 because there is no relative
        within-group comparison signal.
        """
        if len(rewards) != len(group_ids):
            raise ValueError("rewards and group_ids must have the same length.")

        groups: dict[str, list[tuple[int, float]]] = {}
        for idx, (gid, reward) in enumerate(zip(group_ids, rewards)):
            groups.setdefault(gid, []).append((idx, reward))

        advantages = [0.0] * len(rewards)
        for group in groups.values():
            if len(group) == 1:
                continue
            indices, group_rewards = zip(*group)
            mean = sum(group_rewards) / len(group_rewards)
            for idx, reward in zip(indices, group_rewards):
                advantages[idx] = reward - mean
        return advantages

    def compute_batch_advantages(
        self,
        rewards: list[float],
        group_ids: list[str],
    ) -> list[float]:
        """Compute std-normalized GRPO advantages within each prompt group.

        For each prompt group:

            advantage_i = (reward_i - mean(group)) / (std(group) + ε)

        Groups with a single sample get advantage 0.0 (no within-group signal).
        Variance uses the population formula (N denominator).  If your trainer
        uses sample variance (N-1), normalise at that layer instead.

        Single-pass: mean and std are computed together in one traversal of each
        group rather than first centering then re-scanning for std.
        """
        if len(rewards) != len(group_ids):
            raise ValueError("rewards and group_ids must have the same length.")

        groups: dict[str, list[tuple[int, float]]] = {}
        for idx, (gid, reward) in enumerate(zip(group_ids, rewards)):
            groups.setdefault(gid, []).append((idx, reward))

        advantages = [0.0] * len(rewards)
        for group in groups.values():
            if len(group) == 1:
                continue
            indices, group_rewards = zip(*group)
            n = len(group_rewards)
            mean = sum(group_rewards) / n
            variance = sum((r - mean) ** 2 for r in group_rewards) / n
            std = math.sqrt(variance)
            for idx, reward in zip(indices, group_rewards):
                advantages[idx] = (reward - mean) / (std + 1e-8)
        return advantages

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _last_action_idx(output: AgentLoopOutput) -> int:
        """Index of the last model-generated token in the response sequence.

        Returns the last position where response_mask == 1 (truthy).
        Falls back to ``len(response_ids) - 1`` when the mask is absent or
        all-zero (e.g. padding-only or observation-only sequences).
        """
        n = len(output.response_ids)
        mask = output.response_mask
        return next(
            (i for i in range(n - 1, -1, -1) if i < len(mask) and mask[i]),
            n - 1,
        )

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

    def _unsupported_claim_penalty(
        self,
        answer: str,
        ctx: AgentContext | None,
        metrics: dict,
    ) -> float:
        """Penalty fired when the agent searched but cited none of the results.

        Conditions for the penalty to fire (all must be true):
        - The answer is non-empty.
        - The agent issued at least one search (rounds_used > 0).
        - Retrieved results exist (ctx.num_results > 0).
        - The answer cites zero of those results.

        This catches the fabrication pattern: the model searched, received
        documents, and then wrote an answer that ignores all of them — a sign
        that it is drawing on prior knowledge (or hallucinating) rather than
        grounding claims in the retrieved evidence.
        """
        if not answer or self.config.unsupported_claim_penalty == 0.0:
            return 0.0
        if metrics.get("rounds_used", 0) <= 0:
            return 0.0
        if ctx is None or ctx.num_results == 0:
            return 0.0
        if len(ctx.cited_result_ids(answer)) > 0:
            return 0.0
        return self.config.unsupported_claim_penalty

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
