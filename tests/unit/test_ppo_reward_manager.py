"""Tests for PPO reward-manager helpers."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from src.training.ppo import (
    PPORewardManager,
    qa_exact_match_score,
    select_reward_score_fn,
)


class _Tokenizer:
    def decode(self, ids):
        return " ".join(str(int(token)) for token in ids.tolist())


def test_ppo_reward_manager_returns_existing_rm_scores():
    rm_scores = torch.tensor([[0.0, 1.0]])
    data = SimpleNamespace(
        batch={
            "responses": torch.tensor([[7, 8]]),
            "attention_mask": torch.ones(1, 4, dtype=torch.long),
            "rm_scores": rm_scores,
        },
        non_tensor_batch={},
    )

    rewards = PPORewardManager(_Tokenizer())(data)

    assert rewards is rm_scores


def test_ppo_reward_manager_places_score_on_last_valid_response_token():
    def _score(solution: str, ground_truth: str, format_score: float) -> float:
        assert format_score == 0.25
        assert (solution, ground_truth) in {
            ("1 2 7 8", "answer"),
            ("3 9", "other"),
        }
        return 0.75

    data = SimpleNamespace(
        batch={
            "prompts": torch.tensor([[1, 2], [0, 3]]),
            "responses": torch.tensor([[7, 8, 0], [9, 0, 0]]),
            "attention_mask": torch.tensor(
                [
                    [1, 1, 1, 1, 0],
                    [0, 1, 1, 0, 0],
                ],
                dtype=torch.long,
            ),
        },
        non_tensor_batch={
            "data_source": ["custom", "custom"],
            "ground_truth": ["answer", "other"],
        },
    )

    manager = PPORewardManager(
        _Tokenizer(),
        format_score=0.25,
        score_fns={"custom": _score},
    )
    rewards = manager(data)

    assert rewards.tolist() == [[0.0, 0.75, 0.0], [0.75, 0.0, 0.0]]


def test_ppo_reward_manager_supports_reward_model_ground_truth_shape():
    def _score(solution: str, ground_truth: str, format_score: float) -> float:
        del solution, format_score
        return 1.0 if ground_truth == "Paris" else 0.0

    data = SimpleNamespace(
        batch={
            "prompts": torch.tensor([[1]]),
            "responses": torch.tensor([[2, 3]]),
            "attention_mask": torch.ones(1, 3, dtype=torch.long),
        },
        non_tensor_batch={
            "data_source": ["custom"],
            "reward_model": [{"ground_truth": "Paris"}],
        },
    )

    rewards = PPORewardManager(_Tokenizer(), score_fns={"custom": _score})(data)

    assert rewards.tolist() == [[0.0, 1.0]]


def test_select_reward_score_fn_supports_known_qa_sources():
    assert select_reward_score_fn("nq") is qa_exact_match_score
    assert qa_exact_match_score("<answer>Paris</answer>", "Paris") == 1.0


def test_select_reward_score_fn_rejects_unknown_source():
    with pytest.raises(NotImplementedError, match="No reward score function"):
        select_reward_score_fn("unknown")
