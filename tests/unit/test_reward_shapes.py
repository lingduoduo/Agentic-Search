"""Tests for composite reward shaping, group-relative advantages, and algorithm helpers."""

from __future__ import annotations

import math

import pytest

pytest.importorskip("torch")
import torch

from src.model.post_training.grpo.algorithms import (
    GRPOAdvantageConfig,
    compute_dapo_advantages,
    score_prompt_group,
)
from src.model.post_training.grpo.algorithms import (
    compute_grpo_policy_loss,
    compute_grpo_token_advantages,
)
from src.model.post_training.ppo.core_algos import PPOPolicyLossConfig
from src.model.post_training.reward import (
    CompositeRewardConfig,
    REWARD_DIMENSIONS,
    SearchRewardConfig,
    SearchRewardFunction,
    _NON_DIMENSION_KEYS,
    format_compliance_reward,
    group_reward_components,
    simple_sparse_correctness_reward,
    token_f1_score,
)


def _dim_output(answer: str, metrics: dict | None = None):
    from src.agents.core.base import AgentLoopOutput

    return AgentLoopOutput(
        prompt_ids=[1],
        response_ids=[2, 3],
        response_mask=[1, 1],
        num_turns=2,
        final_answer=answer,
        metrics=metrics or {},
    )


def test_group_reward_components_sums_each_bucket():
    flat = {
        "correctness": 1.0,
        "citation_support": 0.3,
        "unsupported_claim_penalty": -0.1,
        "fetch_usefulness_reward": 0.1,
        "format_reward": 0.05,
        "search_quality": 0.15,
        "subquestion_coverage": 0.2,
        "evidence_gain": 0.1,
        "early_stop_bonus": 0.0,
        "answer_when_evidence_insufficient_penalty": -0.2,
        "forced_final_answer_penalty": -0.05,
        "search_budget_exhausted_without_answer_penalty": -0.2,
        "per_search_penalty": -0.02,
        "unnecessary_search_penalty": -0.05,
        "duplicate_query_penalty": -0.1,
        "budget_penalty": -0.1,
        "unnecessary_fetch_penalty": -0.1,
        "retriever_cost": -0.05,
        "rerank_cost": -0.02,
    }
    dims = group_reward_components(flat)
    assert set(dims) == {
        "correctness",
        "citation_support",
        "retrieval_quality",
        "search_efficiency",
    }
    assert dims["correctness"] == pytest.approx(1.0)
    assert dims["citation_support"] == pytest.approx(0.3 - 0.1 + 0.1 + 0.05)
    assert dims["retrieval_quality"] == pytest.approx(
        0.15 + 0.2 + 0.1 + 0.0 - 0.2 - 0.05 - 0.2
    )
    assert dims["search_efficiency"] == pytest.approx(
        -0.02 - 0.05 - 0.1 - 0.1 - 0.1 - 0.05 - 0.02
    )


def test_group_reward_components_tolerates_missing_keys():
    dims = group_reward_components({"correctness": 0.7})
    assert dims["correctness"] == pytest.approx(0.7)
    assert dims["citation_support"] == pytest.approx(0.0)
    assert dims["retrieval_quality"] == pytest.approx(0.0)
    assert dims["search_efficiency"] == pytest.approx(0.0)


def test_reward_components_includes_dimension_keys_and_partition_invariant():
    output = _dim_output(
        "Paris [R1Q1D1]",
        metrics={
            "rounds_used": 2.0,
            "search_rounds": 2.0,
            "repeated_search_queries": 1.0,
            "subquestion_coverage_ratio": 1.0,
            "final_evidence_sufficient": 1.0,
            "search_quality_score": 1.0,
            "answer_allowed": 1.0,
        },
    )
    fn = SearchRewardFunction(SearchRewardConfig.second_pass())
    comps = fn.reward_components(output, "Paris", lambda p, g: 1.0)

    for key in (
        "dim_correctness",
        "dim_citation_support",
        "dim_retrieval_quality",
        "dim_search_efficiency",
    ):
        assert key in comps

    dim_sum = (
        comps["dim_correctness"]
        + comps["dim_citation_support"]
        + comps["dim_retrieval_quality"]
        + comps["dim_search_efficiency"]
    )
    assert dim_sum == pytest.approx(comps["terminal_reward"] + comps["shaping_total"])

    members = {k for ks in REWARD_DIMENSIONS.values() for k in ks}
    dim_keys = {
        "dim_correctness",
        "dim_citation_support",
        "dim_retrieval_quality",
        "dim_search_efficiency",
    }
    for key in comps:
        if key in _NON_DIMENSION_KEYS or key in dim_keys:
            continue
        assert key in members, f"reward_components key {key!r} has no dimension"


def test_reward_dimensions_matches_dim_keys_and_scale_invariant():
    output = _dim_output(
        "Paris [R1Q1D1]",
        metrics={
            "rounds_used": 2.0,
            "search_rounds": 2.0,
            "repeated_search_queries": 1.0,
            "subquestion_coverage_ratio": 1.0,
            "final_evidence_sufficient": 1.0,
            "search_quality_score": 1.0,
            "answer_allowed": 1.0,
        },
    )
    from dataclasses import replace

    cfg = replace(SearchRewardConfig.second_pass(), reward_scale=2.0)
    fn = SearchRewardFunction(cfg)
    comps = fn.reward_components(output, "Paris", lambda p, g: 1.0)
    dims = fn.reward_dimensions(output, "Paris", lambda p, g: 1.0)
    assert dims == {
        "correctness": comps["dim_correctness"],
        "citation_support": comps["dim_citation_support"],
        "retrieval_quality": comps["dim_retrieval_quality"],
        "search_efficiency": comps["dim_search_efficiency"],
    }
    # Pre-scale dimensions sum to total / reward_scale.
    assert sum(dims.values()) == pytest.approx(comps["total"] / 2.0)


# ---------------------------------------------------------------------------
# token_f1_score
# ---------------------------------------------------------------------------


class TestTokenF1Score:
    def test_exact_match(self):
        assert token_f1_score("paris", "paris") == pytest.approx(1.0)

    def test_case_insensitive(self):
        assert token_f1_score("Paris", "paris") == pytest.approx(1.0)

    def test_partial_overlap(self):
        # pred: {a, b, c}, gold: {b, c, d} → overlap=2, prec=2/3, rec=2/3
        f1 = token_f1_score("a b c", "b c d")
        assert 0.5 < f1 < 1.0

    def test_no_overlap_returns_zero(self):
        assert token_f1_score("cat", "dog") == pytest.approx(0.0)

    def test_both_empty_returns_one(self):
        assert token_f1_score("", "") == pytest.approx(1.0)

    def test_one_empty_returns_zero(self):
        assert token_f1_score("", "answer") == pytest.approx(0.0)
        assert token_f1_score("answer", "") == pytest.approx(0.0)

    def test_gold_subset_of_pred(self):
        # pred: {the, capital, of, france, is, paris}
        # gold: {paris}
        # overlap=1, prec=1/6, rec=1/1, f1 = 2*prec*rec/(prec+rec)
        f1 = token_f1_score("the capital of france is paris", "paris")
        assert 0 < f1 < 1.0

    def test_symmetric_up_to_precision_recall(self):
        f1_a = token_f1_score("a b c", "a b")
        f1_b = token_f1_score("a b", "a b c")
        # Both should be > 0, and they needn't be equal (precision ≠ recall)
        assert f1_a > 0 and f1_b > 0

    def test_multi_word_exact(self):
        assert token_f1_score("New York City", "new york city") == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# format_compliance_reward
# ---------------------------------------------------------------------------


class TestFormatComplianceReward:
    def test_empty_returns_zero(self):
        assert format_compliance_reward("") == pytest.approx(0.0)

    def test_has_citation_and_require_citations(self):
        assert format_compliance_reward("The answer is X. [D1]") == pytest.approx(1.0)

    def test_no_citation_returns_zero_when_required(self):
        assert format_compliance_reward("The answer is X.") == pytest.approx(0.0)

    def test_answer_tag_required_and_present(self):
        r = format_compliance_reward(
            "<answer>Paris</answer>",
            require_citations=False,
            require_answer_tag=True,
        )
        assert r == pytest.approx(1.0)

    def test_answer_tag_required_but_absent(self):
        r = format_compliance_reward(
            "Paris",
            require_citations=False,
            require_answer_tag=True,
        )
        assert r == pytest.approx(0.0)

    def test_both_checks_half_credit(self):
        # Has citation but no answer tag → 0.5
        r = format_compliance_reward(
            "See [D1].",
            require_citations=True,
            require_answer_tag=True,
        )
        assert r == pytest.approx(0.5)

    def test_no_requirements_returns_one(self):
        r = format_compliance_reward(
            "anything",
            require_citations=False,
            require_answer_tag=False,
        )
        assert r == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# CompositeRewardConfig
# ---------------------------------------------------------------------------


class TestCompositeRewardConfig:
    def test_empty_judges_returns_zero(self):
        cfg = CompositeRewardConfig(judges=[])
        assert cfg.compute("a", "a") == pytest.approx(0.0)

    def test_single_judge_exact_weight(self):
        cfg = CompositeRewardConfig(
            judges=[("em", 0.8, simple_sparse_correctness_reward)]
        )
        assert cfg.compute("paris", "paris") == pytest.approx(0.8)
        assert cfg.compute("berlin", "paris") == pytest.approx(0.0)

    def test_blended_exact_and_f1(self):
        cfg = CompositeRewardConfig(
            judges=[
                ("em", 0.5, simple_sparse_correctness_reward),
                ("f1", 0.5, token_f1_score),
            ]
        )
        score = cfg.compute("paris france", "paris")
        assert 0 < score < 1.0

    def test_as_judge_fn_produces_callable(self):
        cfg = CompositeRewardConfig(
            judges=[("em", 1.0, simple_sparse_correctness_reward)]
        )
        fn = cfg.as_judge_fn()
        assert callable(fn)
        assert fn("paris", "paris") == pytest.approx(1.0)

    def test_compute_breakdown_keys(self):
        cfg = CompositeRewardConfig(
            judges=[
                ("exact", 0.7, simple_sparse_correctness_reward),
                ("f1", 0.3, token_f1_score),
            ]
        )
        breakdown = cfg.compute_breakdown("paris", "paris")
        assert set(breakdown.keys()) == {"exact", "f1"}
        assert breakdown["exact"] == pytest.approx(0.7)
        assert breakdown["f1"] == pytest.approx(0.3)


# ---------------------------------------------------------------------------
# SearchRewardConfig new fields
# ---------------------------------------------------------------------------


class TestSearchRewardConfigNewFields:
    def _make_output(self, answer: str = ""):
        from src.agents.core.base import AgentLoopOutput

        return AgentLoopOutput(
            prompt_ids=[0],
            response_ids=[1, 2, 3],
            response_mask=[1, 1, 1],
            num_turns=1,
            final_answer=answer,
            metrics={},
        )

    def test_format_reward_weight_fires_with_citation(self):
        cfg = SearchRewardConfig.third_pass_with_format(
            correctness_weight=1.0,
            format_reward_weight=0.1,
        )
        fn = SearchRewardFunction(cfg)
        output = self._make_output("The answer is X. [D1]")
        components = fn.reward_components(output, "X", lambda p, g: 1.0)
        assert components["format_reward"] == pytest.approx(0.1)

    def test_format_reward_zero_without_citation(self):
        cfg = SearchRewardConfig.third_pass_with_format(
            correctness_weight=1.0,
            format_reward_weight=0.1,
        )
        fn = SearchRewardFunction(cfg)
        output = self._make_output("The answer is X.")
        components = fn.reward_components(output, "X", lambda p, g: 1.0)
        assert components["format_reward"] == pytest.approx(0.0)

    def test_reward_scale_multiplies_total(self):
        cfg = SearchRewardConfig(reward_scale=2.0)
        fn = SearchRewardFunction(cfg)
        output = self._make_output("paris")
        r_scaled = fn.compute(output, "paris", lambda p, g: 1.0)
        cfg1 = SearchRewardConfig(reward_scale=1.0)
        fn1 = SearchRewardFunction(cfg1)
        r_base = fn1.compute(output, "paris", lambda p, g: 1.0)
        assert r_scaled == pytest.approx(r_base * 2.0)

    def test_third_pass_with_format_preset_fields(self):
        cfg = SearchRewardConfig.third_pass_with_format()
        assert cfg.format_reward_weight == pytest.approx(0.1)
        assert cfg.citation_support_weight == pytest.approx(0.1)
        assert cfg.per_search_penalty == pytest.approx(-0.02)


# ---------------------------------------------------------------------------
# GRPOAdvantageConfig — new presets
# ---------------------------------------------------------------------------


class TestGRPOAdvantageConfigPresets:
    def test_reinforce_with_baseline_mode(self):
        cfg = GRPOAdvantageConfig.reinforce_with_baseline()
        assert cfg.mode == "reinforce_baseline"
        assert cfg.reward_component == "total"

    def test_dapo_mode(self):
        cfg = GRPOAdvantageConfig.dapo()
        assert cfg.mode == "dapo"
        assert cfg.clip_range == pytest.approx(1.0)

    def test_clip_range_stored(self):
        cfg = GRPOAdvantageConfig(mode="group_std_normalized", clip_range=2.5)
        assert cfg.clip_range == pytest.approx(2.5)

    def test_reward_scale_stored(self):
        cfg = GRPOAdvantageConfig(mode="group_outcome", reward_scale=0.5)
        assert cfg.reward_scale == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# compute_dapo_advantages
# ---------------------------------------------------------------------------


class TestComputeDapoAdvantages:
    def test_returns_rewards_unchanged(self):
        rewards = [0.3, 0.7, 1.0, 0.0]
        adv = compute_dapo_advantages(rewards)
        assert adv == pytest.approx(rewards)

    def test_clip_range_applied(self):
        rewards = [-2.0, 0.5, 3.0]
        adv = compute_dapo_advantages(rewards, clip_range=1.0)
        assert adv == pytest.approx([-1.0, 0.5, 1.0])

    def test_empty(self):
        assert compute_dapo_advantages([]) == []


# ---------------------------------------------------------------------------
# reinforce_baseline mode in _compute_advantages (via score_prompt_group)
# ---------------------------------------------------------------------------


class TestReinforceBaselineMode:
    def _make_sample(self, answer: str, group_id: str = "g0"):
        from src.agents.core.base import AgentLoopOutput
        from src.model.post_training.grpo.algorithms import GRPORolloutSample

        output = AgentLoopOutput(
            prompt_ids=[0],
            response_ids=[1],
            response_mask=[1],
            num_turns=1,
            final_answer=answer,
            metrics={},
        )
        output.group_id = group_id
        output.rollout_index = 0
        return GRPORolloutSample(
            group_id=group_id,
            rollout_index=0,
            sampling_params={},
            output=output,
        )

    def test_reinforce_baseline_centers_by_mean(self):
        def _judge(p: str, g: str) -> float:
            return 1.0 if p == g else 0.0

        samples = [
            self._make_sample("correct"),
            self._make_sample("wrong"),
        ]
        scored = score_prompt_group(
            samples,
            ground_truth="correct",
            judge_fn=_judge,
            advantage_config=GRPOAdvantageConfig.reinforce_with_baseline(),
        )
        # rewards: [1.0, 0.0], mean=0.5 → advantages: [0.5, -0.5]
        advs = [s.advantage for s in scored]
        assert advs[0] == pytest.approx(0.5)
        assert advs[1] == pytest.approx(-0.5)

    def test_reward_scale_applied(self):
        def _judge(p: str, g: str) -> float:
            return 1.0 if p == g else 0.0

        samples = [self._make_sample("correct"), self._make_sample("wrong")]
        cfg = GRPOAdvantageConfig(  # noqa: E501
            mode="reinforce_baseline",
            reward_component="terminal_reward",
            reward_scale=2.0,
        )
        scored = score_prompt_group(
            samples, ground_truth="correct", judge_fn=_judge, advantage_config=cfg
        )
        rewards = [s.reward for s in scored]
        assert rewards[0] == pytest.approx(2.0)
        assert rewards[1] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# compute_grpo_token_advantages — clip_advantages
# ---------------------------------------------------------------------------


class TestComputeGRPOOutcomeAdvantageClipping:
    def test_no_clipping(self):
        rewards = torch.tensor([[0.0, 1.0], [0.0, 0.0]])
        eos_mask = torch.ones(2, 2)
        index = torch.tensor([0, 0])
        adv, _ = compute_grpo_token_advantages(rewards, eos_mask, index)
        # Both rollouts in same group → advantages should be non-zero at last token
        assert adv.shape == (2, 2)

    def test_clip_advantages_applied(self):
        # Create rewards that produce extreme group-normalized advantages
        rewards = torch.tensor([[0.0, 10.0], [0.0, -10.0]])
        eos_mask = torch.ones(2, 2)
        index = torch.tensor([0, 0])
        adv_clipped, _ = compute_grpo_token_advantages(
            rewards, eos_mask, index, clip_advantages=0.5
        )
        assert adv_clipped.abs().max().item() <= 0.5 + 1e-5


# ---------------------------------------------------------------------------
# compute_grpo_policy_loss (PPOPolicyLossConfig wrapper)
# ---------------------------------------------------------------------------


class TestComputeGRPOPolicyLoss:
    def _basic_inputs(self):
        new_lp = [-0.5, -0.4, -0.6, -0.5]
        old_lp = [-0.5, -0.5, -0.5, -0.5]
        adv = [1.0, 1.0, 1.0, 1.0]
        mask = [1, 1, 1, 1]
        return new_lp, old_lp, adv, mask

    def test_returns_expected_keys(self):
        new_lp, old_lp, adv, mask = self._basic_inputs()
        result = compute_grpo_policy_loss(
            new_log_probs=new_lp,
            old_log_probs=old_lp,
            advantages=adv,
            response_mask=mask,
        )
        assert "grpo_policy_loss" in result
        assert "kl_penalty" in result
        assert "entropy_bonus" in result
        assert "total_loss" in result
        assert "clip_fraction" in result

    def test_no_kl_no_entropy_matches_base(self):
        from src.model.post_training.ppo.core_algos import (
            compute_trajectory_policy_loss,
        )

        new_lp, old_lp, adv, mask = self._basic_inputs()
        base = compute_trajectory_policy_loss(
            new_log_probs=new_lp,
            old_log_probs=old_lp,
            advantages=adv,
            response_mask=mask,
            clip_epsilon=0.2,
            kl_beta=0.0,
        )
        wrapped = compute_grpo_policy_loss(
            new_log_probs=new_lp,
            old_log_probs=old_lp,
            advantages=adv,
            response_mask=mask,
            config=PPOPolicyLossConfig(clip_epsilon=0.2, kl_coefficient=0.0),
        )
        assert wrapped["total_loss"] == pytest.approx(base["total_loss"], abs=1e-5)

    def test_entropy_bonus_reduces_loss(self):
        new_lp, old_lp, adv, mask = self._basic_inputs()
        no_ent = compute_grpo_policy_loss(
            new_log_probs=new_lp,
            old_log_probs=old_lp,
            advantages=adv,
            response_mask=mask,
            config=PPOPolicyLossConfig(entropy_coefficient=0.0),
        )
        with_ent = compute_grpo_policy_loss(
            new_log_probs=new_lp,
            old_log_probs=old_lp,
            advantages=adv,
            response_mask=mask,
            config=PPOPolicyLossConfig(entropy_coefficient=0.01),
        )
        assert with_ent["entropy_bonus"] != pytest.approx(0.0)
        assert with_ent["total_loss"] < no_ent["total_loss"]

    def test_whiten_advantages_does_not_raise(self):
        new_lp, old_lp, adv, mask = self._basic_inputs()
        result = compute_grpo_policy_loss(
            new_log_probs=new_lp,
            old_log_probs=old_lp,
            advantages=adv,
            response_mask=mask,
            config=PPOPolicyLossConfig(whiten_advantages=True),
        )
        assert math.isfinite(result["total_loss"])
