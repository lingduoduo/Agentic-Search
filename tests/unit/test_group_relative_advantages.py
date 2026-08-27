"""The shared scalar advantage primitive, pinned against every implementation it replaces.

These tests originally pinned the primitive against six independent
implementations at the moment of replacement -- that comparison is what made
the dedupe safe. The call sites have since been repointed to delegate to the
primitive, so now most `test_matches_*` cases compare a wrapper to the
primitive through the same code and are retained as delegation-wiring checks
rather than formula checks. The formula itself is pinned by the hand-derived
tests earlier in this file (the ones that re-derive mean/std in the test
body) and by independent pins in `tests/unit/test_reward.py` and
`tests/unit/test_grpo.py`.
"""

from __future__ import annotations

import pytest

from src.model.post_training.reward import (
    SearchRewardConfig,
    SearchRewardFunction,
    group_relative_advantages,
    grouped_relative_advantages,
)

# Shared input table. Covers: empty, singleton, all-equal (std == 0, exercising
# the 1e-8 floor), a typical spread, a two-element group, and a zero-heavy group.
CASES: list[list[float]] = [
    [],
    [1.0],
    [1.0, 1.0, 1.0],
    [1.0, 0.7, 0.0, 0.0],
    [-2.5, 3.5],
    [0.0, 0.0, 1.0],
]

GROUPED_REWARDS = [1.0, 0.0, 0.5, 1.0, 0.25]
GROUPED_IDS = ["a", "b", "a", "b", "a"]


@pytest.mark.parametrize("rewards", CASES)
def test_mean_centering_matches_the_documented_formula(rewards: list[float]):
    result = group_relative_advantages(rewards)
    if len(rewards) <= 1:
        assert result == [0.0] * len(rewards)
        return
    mean = sum(rewards) / len(rewards)
    assert result == pytest.approx([r - mean for r in rewards])


@pytest.mark.parametrize("rewards", CASES)
def test_normalized_divides_by_population_std_with_epsilon(rewards: list[float]):
    result = group_relative_advantages(rewards, normalize=True)
    if len(rewards) <= 1:
        assert result == [0.0] * len(rewards)
        return
    n = len(rewards)
    mean = sum(rewards) / n
    centered = [r - mean for r in rewards]
    std = (sum(c * c for c in centered) / n) ** 0.5
    assert result == pytest.approx([c / (std + 1e-8) for c in centered])


def test_all_equal_rewards_give_zero_advantage_not_nan():
    # std == 0; the epsilon floor is what keeps this finite.
    result = group_relative_advantages([2.0, 2.0, 2.0], normalize=True)
    assert result == pytest.approx([0.0, 0.0, 0.0])


def test_grouped_partitions_by_id_and_preserves_input_order():
    result = grouped_relative_advantages(GROUPED_REWARDS, GROUPED_IDS)
    # group "a" = indices 0, 2, 4 -> rewards 1.0, 0.5, 0.25, mean 0.583333...
    # group "b" = indices 1, 3    -> rewards 0.0, 1.0, mean 0.5
    mean_a = (1.0 + 0.5 + 0.25) / 3
    assert result == pytest.approx(
        [1.0 - mean_a, -0.5, 0.5 - mean_a, 0.5, 0.25 - mean_a]
    )


def test_grouped_gives_a_lone_member_zero():
    assert grouped_relative_advantages([5.0, 1.0, 2.0], ["solo", "x", "x"]) == (
        pytest.approx([0.0, -0.5, 0.5])
    )


def test_grouped_rejects_mismatched_lengths():
    with pytest.raises(ValueError):
        grouped_relative_advantages([1.0, 2.0], ["a"])


# --------------------------------------------------------------------------
# Equivalence with the implementations this primitive replaces.
# --------------------------------------------------------------------------


@pytest.mark.parametrize("rewards", CASES)
def test_matches_rollouts_compute_grpo_outcome_advantage(rewards: list[float]):
    from src.model.post_training.grpo.algorithms import compute_grpo_outcome_advantage

    assert group_relative_advantages(rewards) == pytest.approx(
        compute_grpo_outcome_advantage(list(rewards))
    )


def test_matches_reward_function_outcome_advantages():
    fn = SearchRewardFunction(SearchRewardConfig.second_pass())
    assert grouped_relative_advantages(GROUPED_REWARDS, GROUPED_IDS) == pytest.approx(
        fn.compute_grpo_outcome_advantages(list(GROUPED_REWARDS), list(GROUPED_IDS))
    )


def test_matches_reward_function_batch_advantages():
    fn = SearchRewardFunction(SearchRewardConfig.second_pass())
    assert grouped_relative_advantages(
        GROUPED_REWARDS, GROUPED_IDS, normalize=True
    ) == pytest.approx(
        fn.compute_batch_advantages(list(GROUPED_REWARDS), list(GROUPED_IDS))
    )


@pytest.mark.parametrize("rewards", CASES)
def test_matches_controller_assign_group_advantages(rewards: list[float]):
    pytest.importorskip("torch", exc_type=ImportError)
    from src.model.post_training.grpo.training import (
        LocalGRPOController,
        RolloutResult,
    )

    group = [
        RolloutResult(prompt_id=0, rollout_id=i, trajectory=None, reward=r)
        for i, r in enumerate(rewards)
    ]
    LocalGRPOController.assign_group_advantages(group)
    assert [item.advantage for item in group] == pytest.approx(
        group_relative_advantages(rewards, normalize=True)
    )
