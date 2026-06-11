"""Tests for grouped rollout helpers used by GRPO-style training."""

from __future__ import annotations

import asyncio

import pytest

from src import (
    AgentLoopBase,
    AgentLoopOutput,
    GRPOAdvantageConfig,
    OnPolicyBatchStats,
    OnPolicyGRPOConfig,
    SearchRewardConfig,
    SearchRewardFunction,
    ScoredGRPORollout,
    assemble_on_policy_batch,
    build_grpo_sampling_params,
    compute_grpo_outcome_advantage,
    compute_on_policy_batch_stats,
    filter_zero_advantage_groups,
    sample_prompt_group,
    score_prompt_group,
)


class _DummyLoop(AgentLoopBase):
    def __init__(self, answer: str, metrics: dict[str, float] | None = None) -> None:
        super().__init__(tokenizer=object(), server_manager=object())
        self._answer = answer
        self._metrics = {
            "rounds_used": 0.0,
            "search_rounds": 0.0,
            "subquestion_coverage_ratio": 1.0,
            "repeated_search_queries": 0.0,
            "fetched_pages": 0.0,
            "unnecessary_fetch_count": 0.0,
            "answer_when_evidence_insufficient": 0.0,
            "search_budget_exhausted_without_answer": 0.0,
            "final_evidence_sufficient": 1.0,
            "answer_allowed": 1.0,
            "search_quality_score": 1.0,
        }
        if metrics is not None:
            self._metrics.update(metrics)

    async def run(self, messages, sampling_params):
        del messages, sampling_params
        return AgentLoopOutput(
            prompt_ids=[],
            response_ids=[],
            response_mask=[],
            num_turns=1,
            metrics=dict(self._metrics),
            final_answer=self._answer,
        )


def test_build_grpo_sampling_params_creates_one_variant_per_rollout():
    variants = build_grpo_sampling_params(
        {"temperature": 0.4, "top_p": 0.8, "max_tokens": 64},
        num_rollouts=4,
    )
    assert len(variants) == 4
    assert variants[0]["temperature"] == pytest.approx(0.4)
    assert variants[1]["temperature"] > variants[0]["temperature"]
    assert variants[3]["top_p"] >= variants[0]["top_p"]
    assert variants[0]["max_tokens"] == 64


def test_compute_grpo_outcome_advantage_centers_rewards_by_group_mean():
    advantages = compute_grpo_outcome_advantage([1.0, 0.7, 0.1])
    assert advantages[0] == pytest.approx(0.4)
    assert advantages[1] == pytest.approx(0.1)
    assert advantages[2] == pytest.approx(-0.5)


def test_compute_grpo_outcome_advantage_single_sample_group_returns_zero():
    assert compute_grpo_outcome_advantage([2.0]) == [0.0]


def test_sample_prompt_group_assigns_shared_group_id_and_rollout_indices():
    answers = iter(["direct", "search once", "decompose", "fetch"])

    def loop_factory():
        return _DummyLoop(next(answers))

    samples = asyncio.run(
        sample_prompt_group(
            loop_factory,
            question="What is the answer?",
            sampling_params={"temperature": 0.7, "top_p": 0.9},
            num_rollouts=4,
            group_id="q1",
        )
    )

    assert [sample.group_id for sample in samples] == ["q1", "q1", "q1", "q1"]
    assert [sample.rollout_index for sample in samples] == [0, 1, 2, 3]
    assert [sample.output.group_id for sample in samples] == ["q1", "q1", "q1", "q1"]
    assert [sample.output.rollout_index for sample in samples] == [0, 1, 2, 3]


def test_sample_prompt_group_accepts_prebuilt_messages():
    class _RecordingLoop(_DummyLoop):
        def __init__(self):
            super().__init__("ok")
            self.seen_messages = None

        async def run(self, messages, sampling_params):
            self.seen_messages = list(messages)
            return await super().run(messages, sampling_params)

    holder: dict[str, _RecordingLoop] = {}

    def loop_factory():
        loop = _RecordingLoop()
        holder["loop"] = loop
        return loop

    samples = asyncio.run(
        sample_prompt_group(
            loop_factory,
            messages=[
                {"role": "system", "content": "Available tools: search"},
                {
                    "role": "user",
                    "content": "Who won the Nobel Prize in Physics in 2024?",
                },
            ],
            sampling_params={"temperature": 0.7, "top_p": 0.9},
            num_rollouts=1,
            group_id="q-messages",
        )
    )

    assert samples[0].group_id == "q-messages"
    assert holder["loop"].seen_messages == [
        {"role": "system", "content": "Available tools: search"},
        {"role": "user", "content": "Who won the Nobel Prize in Physics in 2024?"},
    ]


def test_score_prompt_group_normalizes_rewards_within_shared_prompt_group():
    samples = [
        asyncio.run(
            sample_prompt_group(
                lambda: _DummyLoop(
                    "Paris",
                    metrics={"search_rounds": 0.0, "answer_allowed": 1.0},
                ),
                question="Capital of France?",
                sampling_params={"temperature": 0.7, "top_p": 0.9},
                num_rollouts=1,
                group_id="capital",
                sampling_variants=[{"temperature": 0.7, "top_p": 0.9}],
            )
        )[0],
        asyncio.run(
            sample_prompt_group(
                lambda: _DummyLoop(
                    "Wrong",
                    metrics={
                        "search_rounds": 2.0,
                        "rounds_used": 2.0,
                        "repeated_search_queries": 1.0,
                        "answer_when_evidence_insufficient": 1.0,
                        "final_evidence_sufficient": 0.0,
                        "answer_allowed": 0.0,
                        "search_quality_score": 0.2,
                    },
                ),
                question="Capital of France?",
                sampling_params={"temperature": 0.85, "top_p": 0.93},
                num_rollouts=1,
                group_id="capital",
                sampling_variants=[{"temperature": 0.85, "top_p": 0.93}],
            )
        )[0],
    ]

    scored = score_prompt_group(
        samples,
        ground_truth="Paris",
        judge_fn=lambda answer, truth: 1.0 if answer == truth else 0.0,
        reward_fn=SearchRewardFunction(
            SearchRewardConfig(
                citation_support_weight=0.0,
                subquestion_coverage_weight=0.0,
                fetch_usefulness_reward=0.0,
            )
        ),
    )

    assert scored[0].reward > scored[1].reward
    assert scored[0].advantage > 0.0
    assert scored[1].advantage < 0.0


def test_score_prompt_group_supports_group_outcome_advantage_mode():
    samples = [
        asyncio.run(
            sample_prompt_group(
                lambda: _DummyLoop("Paris"),
                question="Capital of France?",
                sampling_params={"temperature": 0.7, "top_p": 0.9},
                num_rollouts=1,
                group_id="capital",
                sampling_variants=[{"temperature": 0.7, "top_p": 0.9}],
            )
        )[0],
        asyncio.run(
            sample_prompt_group(
                lambda: _DummyLoop("Wrong"),
                question="Capital of France?",
                sampling_params={"temperature": 0.85, "top_p": 0.93},
                num_rollouts=1,
                group_id="capital",
                sampling_variants=[{"temperature": 0.85, "top_p": 0.93}],
            )
        )[0],
    ]

    scored = score_prompt_group(
        samples,
        ground_truth="Paris",
        judge_fn=lambda answer, truth: 1.0 if answer == truth else 0.0,
        reward_fn=SearchRewardFunction(SearchRewardConfig.sparse_final_only()),
        advantage_config=GRPOAdvantageConfig(mode="group_outcome"),
    )

    assert scored[0].reward == pytest.approx(1.0)
    assert scored[0].reward_component == "total"
    assert scored[1].reward == pytest.approx(0.0)
    assert scored[0].advantage == pytest.approx(0.5)
    assert scored[1].advantage == pytest.approx(-0.5)


def test_score_prompt_group_can_use_terminal_reward_for_group_outcome():
    samples = [
        asyncio.run(
            sample_prompt_group(
                lambda: _DummyLoop("Paris"),
                question="Capital of France?",
                sampling_params={"temperature": 0.7, "top_p": 0.9},
                num_rollouts=1,
                group_id="capital",
                sampling_variants=[{"temperature": 0.7, "top_p": 0.9}],
            )
        )[0],
        asyncio.run(
            sample_prompt_group(
                lambda: _DummyLoop(
                    "Wrong",
                    metrics={
                        "rounds_used": 4.0,
                        "repeated_search_queries": 5.0,
                        "answer_when_evidence_insufficient": 1.0,
                        "final_evidence_sufficient": 0.0,
                        "answer_allowed": 0.0,
                        "search_quality_score": 0.1,
                    },
                ),
                question="Capital of France?",
                sampling_params={"temperature": 0.85, "top_p": 0.93},
                num_rollouts=1,
                group_id="capital",
                sampling_variants=[{"temperature": 0.85, "top_p": 0.93}],
            )
        )[0],
    ]

    reward_fn = SearchRewardFunction(
        SearchRewardConfig(
            reward_mode="shaped",
            citation_support_weight=0.0,
            subquestion_coverage_weight=0.0,
            fetch_usefulness_reward=0.0,
        )
    )
    scored = score_prompt_group(
        samples,
        ground_truth="Paris",
        judge_fn=lambda answer, truth: 1.0 if answer == truth else 0.0,
        reward_fn=reward_fn,
        advantage_config=GRPOAdvantageConfig(
            mode="group_outcome",
            reward_component="terminal_reward",
        ),
    )

    assert scored[0].reward_component == "terminal_reward"
    assert scored[0].reward == pytest.approx(1.0)
    assert scored[1].reward == pytest.approx(0.0)
    assert scored[0].reward_components["total"] != pytest.approx(scored[0].reward)
    assert scored[1].reward_components["total"] != pytest.approx(scored[1].reward)
    assert scored[0].advantage == pytest.approx(0.5)
    assert scored[1].advantage == pytest.approx(-0.5)


def test_grpo_advantage_config_outcome_only_preset_uses_terminal_reward():
    cfg = GRPOAdvantageConfig.outcome_only()
    assert cfg.mode == "group_outcome"
    assert cfg.reward_component == "terminal_reward"


def test_compute_reinforce_policy_loss_masks_and_baselines_tokens():
    from src.training.ppo import compute_reinforce_policy_loss

    result = compute_reinforce_policy_loss(
        log_probs=[-0.2, -0.4, -0.8],
        rewards=[1.0, 0.5, 2.0],
        response_mask=[1, 0, 1],
        baseline=0.5,
    )

    expected = -((-0.2 * 0.5) + (-0.8 * 1.5)) / 2
    assert result["reinforce_policy_loss"] == pytest.approx(expected)
    assert result["total_loss"] == pytest.approx(expected)
    assert result["mean_reward"] == pytest.approx(1.5)
    assert result["mean_advantage"] == pytest.approx(1.0)


def test_compute_reinforce_policy_loss_rejects_length_mismatch():
    from src.training.ppo import compute_reinforce_policy_loss

    with pytest.raises(ValueError, match="same length"):
        compute_reinforce_policy_loss(
            log_probs=[-0.2],
            rewards=[1.0, 0.5],
            response_mask=[1],
        )


def test_score_prompt_group_rejects_unknown_advantage_mode():
    samples = asyncio.run(
        sample_prompt_group(
            lambda: _DummyLoop("Paris"),
            question="Capital of France?",
            sampling_params={"temperature": 0.7, "top_p": 0.9},
            num_rollouts=1,
            group_id="capital",
            sampling_variants=[{"temperature": 0.7, "top_p": 0.9}],
        )
    )

    with pytest.raises(ValueError, match="Unsupported GRPO advantage mode"):
        score_prompt_group(
            samples,
            ground_truth="Paris",
            judge_fn=lambda answer, truth: 1.0 if answer == truth else 0.0,
            advantage_config=GRPOAdvantageConfig(mode="unknown"),
        )


def test_score_prompt_group_rejects_unknown_reward_component():
    samples = asyncio.run(
        sample_prompt_group(
            lambda: _DummyLoop("Paris"),
            question="Capital of France?",
            sampling_params={"temperature": 0.7, "top_p": 0.9},
            num_rollouts=1,
            group_id="capital",
            sampling_variants=[{"temperature": 0.7, "top_p": 0.9}],
        )
    )

    with pytest.raises(ValueError, match="Unsupported GRPO reward component"):
        score_prompt_group(
            samples,
            ground_truth="Paris",
            judge_fn=lambda answer, truth: 1.0 if answer == truth else 0.0,
            advantage_config=GRPOAdvantageConfig(reward_component="unknown_component"),
        )


def test_sample_prompt_group_rejects_mismatched_sampling_variants():
    with pytest.raises(ValueError, match="sampling_variants length must equal"):
        asyncio.run(
            sample_prompt_group(
                lambda: _DummyLoop("x"),
                question="hello",
                sampling_params={"temperature": 0.7},
                num_rollouts=2,
                sampling_variants=[{"temperature": 0.7}],
            )
        )


def test_grpo_advantage_config_std_normalized_preset():
    cfg = GRPOAdvantageConfig.std_normalized()
    assert cfg.mode == "group_std_normalized"
    assert cfg.reward_component == "total"


def test_score_prompt_group_batch_judge_fn_called_once_not_per_rollout():
    """With batch_judge_fn, the judge is called once per group, not once per rollout."""
    from src import score_prompt_group

    call_count = {"n": 0}

    def batch_judge(answers: list[str], gts: list[str]) -> list[float]:
        call_count["n"] += 1
        return [1.0 if a.strip() == g.strip() else 0.0 for a, g in zip(answers, gts)]

    samples = asyncio.run(
        sample_prompt_group(
            lambda: _DummyLoop("Paris"),
            question="Capital of France?",
            sampling_params={"temperature": 0.7, "top_p": 0.9},
            num_rollouts=3,
            group_id="capital",
        )
    )

    scored = score_prompt_group(
        samples,
        ground_truth="Paris",
        judge_fn=lambda a, g: 1.0 if a.strip() == g.strip() else 0.0,
        reward_fn=SearchRewardFunction(SearchRewardConfig.sparse_final_only()),
        advantage_config=GRPOAdvantageConfig.outcome_only(),
        batch_judge_fn=batch_judge,
    )

    assert call_count["n"] == 1, "batch_judge_fn must be called once per group"
    assert len(scored) == 3
    for rollout in scored:
        assert rollout.reward == pytest.approx(1.0)


def test_compute_batch_advantages_single_pass_matches_two_pass_result():
    """Single-pass compute_batch_advantages must give the same result as before."""
    rf = SearchRewardFunction()
    rewards = [1.0, 0.2, 0.8, 0.5, 0.5]
    group_ids = ["g1", "g1", "g1", "g2", "g2"]

    # Ground truth: per-group (reward - mean) / std
    adv = rf.compute_batch_advantages(rewards, group_ids)

    # g1: mean=0.666, std=sqrt(((0.333)^2+(0.466)^2+(0.133)^2)/3)
    g1_mean = (1.0 + 0.2 + 0.8) / 3
    g1_var = sum((r - g1_mean) ** 2 for r in [1.0, 0.2, 0.8]) / 3
    import math

    g1_std = math.sqrt(g1_var)
    assert adv[0] == pytest.approx((1.0 - g1_mean) / (g1_std + 1e-8), abs=1e-6)
    assert adv[1] == pytest.approx((0.2 - g1_mean) / (g1_std + 1e-8), abs=1e-6)
    assert adv[2] == pytest.approx((0.8 - g1_mean) / (g1_std + 1e-8), abs=1e-6)

    # g2: identical rewards → std=0 → all advantages 0
    assert adv[3] == pytest.approx(0.0, abs=1e-6)
    assert adv[4] == pytest.approx(0.0, abs=1e-6)


def test_sample_prompt_group_requires_exactly_one_prompt_input():
    with pytest.raises(
        ValueError, match="Either `question` or `messages` must be provided."
    ):
        asyncio.run(
            sample_prompt_group(
                lambda: _DummyLoop("x"),
                sampling_params={"temperature": 0.7},
                num_rollouts=1,
            )
        )

    with pytest.raises(
        ValueError, match="Provide only one of `question` or `messages`."
    ):
        asyncio.run(
            sample_prompt_group(
                lambda: _DummyLoop("x"),
                question="hello",
                messages=[{"role": "user", "content": "hello"}],
                sampling_params={"temperature": 0.7},
                num_rollouts=1,
            )
        )


# ---------------------------------------------------------------------------
# On-policy GRPO helpers
# ---------------------------------------------------------------------------


def _make_scored(group_id: str, reward: float, advantage: float) -> ScoredGRPORollout:
    output = AgentLoopOutput(
        prompt_ids=[],
        response_ids=[],
        response_mask=[],
        num_turns=1,
        metrics={},
        final_answer="answer",
    )
    return ScoredGRPORollout(
        group_id=group_id,
        rollout_index=0,
        sampling_params={"temperature": 0.7},
        output=output,
        reward=reward,
        reward_component="total",
        reward_components={"total": reward},
        advantage=advantage,
    )


def test_filter_zero_advantage_groups_removes_uniform_reward_groups():
    groups = [
        [_make_scored("g1", 1.0, 0.5), _make_scored("g1", 0.5, -0.5)],  # varied
        [_make_scored("g2", 0.8, 0.0), _make_scored("g2", 0.8, 0.0)],  # uniform
        [_make_scored("g3", 0.3, 0.4), _make_scored("g3", 0.9, -0.4)],  # varied
    ]
    kept = filter_zero_advantage_groups(groups)
    assert len(kept) == 2
    assert {kept[0][0].group_id, kept[1][0].group_id} == {"g1", "g3"}


def test_filter_zero_advantage_groups_respects_min_reward_range():
    groups = [
        [_make_scored("g1", 1.0, 0.0), _make_scored("g1", 0.9, 0.0)],  # range=0.1
        [_make_scored("g2", 1.0, 0.0), _make_scored("g2", 0.5, 0.0)],  # range=0.5
    ]
    kept = filter_zero_advantage_groups(groups, min_reward_range=0.2)
    assert len(kept) == 1
    assert kept[0][0].group_id == "g2"


def test_assemble_on_policy_batch_flattens_and_filters():
    groups = [
        [_make_scored("g1", 1.0, 0.5), _make_scored("g1", 0.5, -0.5)],
        [_make_scored("g2", 0.8, 0.0), _make_scored("g2", 0.8, 0.0)],
    ]
    flat = assemble_on_policy_batch(groups)
    assert len(flat) == 2
    assert all(r.group_id == "g1" for r in flat)


def test_assemble_on_policy_batch_max_groups_truncates():
    groups = [
        [_make_scored("g1", 1.0, 0.5), _make_scored("g1", 0.0, -0.5)],
        [_make_scored("g2", 1.0, 0.5), _make_scored("g2", 0.0, -0.5)],
        [_make_scored("g3", 1.0, 0.5), _make_scored("g3", 0.0, -0.5)],
    ]
    flat = assemble_on_policy_batch(groups, OnPolicyGRPOConfig(max_groups=2))
    group_ids = {r.group_id for r in flat}
    assert len(group_ids) == 2


def test_assemble_on_policy_batch_global_normalization_centers_advantages():
    import math

    groups = [
        [_make_scored("g1", 1.0, 2.0), _make_scored("g1", 0.0, -2.0)],
        [_make_scored("g2", 1.0, 4.0), _make_scored("g2", 0.0, -4.0)],
    ]
    flat = assemble_on_policy_batch(groups, OnPolicyGRPOConfig(normalize_globally=True))
    adv_values = [r.advantage for r in flat]
    mean = sum(adv_values) / len(adv_values)
    assert mean == pytest.approx(0.0, abs=1e-6)
    # std should be 1 after z-score normalization
    std = math.sqrt(sum(a**2 for a in adv_values) / len(adv_values))
    assert std == pytest.approx(1.0, abs=1e-4)


def test_compute_on_policy_batch_stats_returns_correct_counts():
    groups = [
        [_make_scored("g1", 1.0, 0.5), _make_scored("g1", 0.0, -0.5)],
        [
            _make_scored("g2", 0.8, 0.0),
            _make_scored("g2", 0.8, 0.0),
        ],  # will be filtered
        [_make_scored("g3", 0.9, 0.3), _make_scored("g3", 0.3, -0.3)],
    ]
    flat = assemble_on_policy_batch(groups)
    stats = compute_on_policy_batch_stats(groups, flat)

    assert isinstance(stats, OnPolicyBatchStats)
    assert stats.n_groups_total == 3
    assert stats.n_groups_kept == 2
    assert stats.n_rollouts_kept == 4
    assert stats.pct_groups_kept == pytest.approx(2 / 3)


def test_compute_on_policy_batch_stats_empty_batch():
    groups = [
        [_make_scored("g1", 0.5, 0.0), _make_scored("g1", 0.5, 0.0)],
    ]
    flat = assemble_on_policy_batch(groups)  # all filtered
    stats = compute_on_policy_batch_stats(groups, flat)

    assert stats.n_rollouts_kept == 0
    assert stats.mean_reward == 0.0
    assert stats.reward_std == 0.0
