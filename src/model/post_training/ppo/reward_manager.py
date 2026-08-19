"""Reward-manager helpers for PPO-style token rewards.

The training loop in this repo is intentionally local and lightweight, but it
still benefits from the same reward-tensor convention used by distributed PPO
trainers: compute one scalar outcome score per rollout and place it on the last
valid response token.
"""

from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping

import torch

RewardScoreFn = Callable[[str, str, float], float]

DEFAULT_QA_DATA_SOURCES = frozenset(
    {
        "nq",
        "triviaqa",
        "popqa",
        "hotpotqa",
        "2wikimultihopqa",
        "musique",
        "bamboogle",
    }
)

_RE_ARTICLES = re.compile(r"\b(a|an|the)\b")
_RE_NON_ALNUM = re.compile(r"[^a-z0-9\s]")
_RE_ANSWER_TAG = re.compile(r"<answer>.*?</answer>", re.DOTALL)


def normalize_qa_answer(text: str) -> str:
    text = text.lower()
    text = _RE_ARTICLES.sub(" ", text)
    text = _RE_NON_ALNUM.sub(" ", text)
    return " ".join(text.split())


def qa_exact_match_score(
    solution_str: str,
    ground_truth: str,
    format_score: float = 0.0,
) -> float:
    """Return exact-match reward for normalized QA answers.

    ``format_score`` is added when the completion contains an ``<answer>`` tag.
    """
    normalized_solution = normalize_qa_answer(solution_str)
    normalized_truth = normalize_qa_answer(ground_truth)
    if not normalized_solution or not normalized_truth:
        score = 0.0
    else:
        score = 1.0 if normalized_truth in normalized_solution else 0.0
    if format_score and _RE_ANSWER_TAG.search(solution_str):
        score += format_score
    return score


def select_reward_score_fn(
    data_source: str,
    registry: Mapping[str, RewardScoreFn] | None = None,
) -> RewardScoreFn:
    if registry and data_source in registry:
        return registry[data_source]
    if data_source in DEFAULT_QA_DATA_SOURCES:
        return qa_exact_match_score
    raise NotImplementedError(
        f"No reward score function registered for {data_source!r}."
    )


def _as_list(value: Any, length: int, default: Any = None) -> list[Any]:
    if value is None:
        return [default for _ in range(length)]
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    return [value for _ in range(length)]


def _ground_truth_at(non_tensor_batch: Mapping[str, Any], index: int) -> str:
    reward_model = non_tensor_batch.get("reward_model")
    if isinstance(reward_model, list) and index < len(reward_model):
        item = reward_model[index]
        if isinstance(item, Mapping) and "ground_truth" in item:
            return str(item["ground_truth"])
    if isinstance(reward_model, Mapping) and "ground_truth" in reward_model:
        gt = reward_model["ground_truth"]
        if isinstance(gt, (list, tuple)):
            return str(gt[index]) if index < len(gt) else ""
        return str(gt)

    for key in ("ground_truth", "ground_truths", "golden_answers"):
        values = non_tensor_batch.get(key)
        if values is None:
            continue
        if isinstance(values, (list, tuple)):
            value = values[index] if index < len(values) else ""
        else:
            value = values
        if isinstance(value, (list, tuple)):
            return str(value[0]) if value else ""
        return str(value)
    return ""


@dataclass(frozen=True)
class PPORewardManager:
    """Build token-level PPO reward tensors from batch rollouts.

    If ``batch["rm_scores"]`` exists, it is returned directly. Otherwise the
    manager decodes prompt+valid-response tokens, scores the decoded text
    against per-example ground truth, and writes that scalar score to the last
    valid response token.
    """

    tokenizer: Any
    num_examine: int = 0
    format_score: float = 0.0
    score_fns: Mapping[str, RewardScoreFn] = field(default_factory=dict)
    default_data_source: str = "nq"

    def __call__(self, data: Any) -> torch.Tensor:
        batch: Mapping[str, torch.Tensor] = (
            data.batch if hasattr(data, "batch") else data
        )
        non_tensor_batch: Mapping[str, Any] = getattr(data, "non_tensor_batch", {})

        if "rm_scores" in batch:
            return batch["rm_scores"]

        responses = batch["responses"]
        attention_mask = batch["attention_mask"]
        prompts = batch.get("prompts")
        if prompts is None:
            prompt_length = attention_mask.shape[-1] - responses.shape[-1]
            prompts = batch.get(
                "input_ids", attention_mask.new_zeros((len(responses), 0))
            )[:, :prompt_length]
        prompt_length = prompts.shape[-1]

        reward_tensor = torch.zeros_like(responses, dtype=torch.float32)
        printed_by_source: dict[str, int] = defaultdict(int)
        score_fn_cache: dict[str, RewardScoreFn] = {}
        data_sources = _as_list(
            non_tensor_batch.get("data_source"),
            len(responses),
            self.default_data_source,
        )

        for index in range(len(responses)):
            valid_prompt_length = int(
                attention_mask[index, :prompt_length].sum().item()
            )
            response_mask = attention_mask[
                index, prompt_length : prompt_length + responses.shape[-1]
            ]
            valid_response_length = int(response_mask.sum().item())
            if valid_response_length < 1:
                continue

            valid_prompt_ids = (
                prompts[index, -valid_prompt_length:]
                if valid_prompt_length
                else prompts[index, :0]
            )
            valid_response_ids = responses[index, :valid_response_length]
            sequence_ids = torch.cat((valid_prompt_ids, valid_response_ids))
            sequence_str = self.tokenizer.decode(sequence_ids)

            data_source = str(data_sources[index] or self.default_data_source)
            if data_source not in score_fn_cache:
                score_fn_cache[data_source] = select_reward_score_fn(
                    data_source, self.score_fns
                )
            score = score_fn_cache[data_source](
                sequence_str,
                _ground_truth_at(non_tensor_batch, index),
                self.format_score,
            )
            reward_tensor[index, valid_response_length - 1] = score

            if printed_by_source[data_source] < self.num_examine:
                printed_by_source[data_source] += 1
                print(sequence_str)

        return reward_tensor
