"""GRPO: group-relative policy optimization, and the trainers that run it.

Three trainers -- bandit, HuggingFace causal-LM, and one backed by a live
SearchAgentLoop -- plus grouped rollout sampling, a controller and a durable
train loop.

This package was called ``rl`` until the things in it that were not GRPO moved
out: the tabular Q-learning demo to ``src.model.post_training.qlearning`` and
``PPORewardManager`` to ``src.model.post_training.ppo``, where its name already pointed.
What remains that is not strictly GRPO is the pair of REINFORCE losses in
``core_algos`` -- kept here deliberately, as the ancestor policy-gradient
algorithm the group-relative variant descends from, and far too small to earn a
package.

The clipped surrogate itself, the KL controllers and the masked-tensor
primitives live one layer down in ``src.model.post_training.ppo``: GRPO is that surrogate
with a group-relative advantage in place of GAE. They are deliberately not
re-exported here -- a variant should not advertise its base layer's API as its
own.

Re-exports are lazy, and that is load-bearing: importing any submodule of a
package executes this file first, so an eager ``from .core_algos import ...``
would put ``import torch`` in front of every module here. That is exactly how
this package broke CI when the Q-learning demo still lived in it (#536), and the
repo has shipped that failure four times.

``rollouts`` is not re-exported at all -- it pulls in the agent loop. Import it
by module path.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

_LAZY_EXPORTS: dict[str, str] = {
    # core_algos — the GRPO/REINFORCE variants. The clipped surrogate itself,
    # the KL controllers and the masked-tensor primitives are one layer down in
    # `src.model.post_training.ppo`, and are NOT re-exported here: a variant should not
    # advertise its base layer's API as its own.
    "compute_grpo_token_advantages": "core_algos",
    "compute_grpo_policy_loss": "core_algos",
    "compute_reinforce_policy_loss": "core_algos",
    "compute_reinforce_policy_loss_core": "core_algos",
    # training — local controller and durable loop
    "LocalGRPOController": "training",
    "RolloutResult": "training",
    # trainers — bandit, HuggingFace causal-LM, and live SearchAgentLoop rollouts
    "GRPOTrainer": "trainers",
    "Policy": "trainers",
    "compute_group_advantages": "trainers",
    "grpo_clipped_policy_loss": "trainers",
    "make_grpo_trainer": "trainers",
    "reverse_kl_penalty": "trainers",
    "LLMGRPOConfig": "trainers",
    "LLMGRPOTrainer": "trainers",
    "LLMRolloutResult": "trainers",
    "get_response_log_probs": "trainers",
    "SearchAgentGRPOTrainer": "trainers",
    # judge — RLAIF scoring, moved in from the shared top level. Deferred
    # like everything else here: it reaches ``reward``, which reaches the
    # agent loop.
    "GoldAgreementJudge": "judge",
    "JudgeParseError": "judge",
    "LLMJudge": "judge",
    "SimulatedPreferenceJudge": "judge",
    "is_degenerate_group": "judge",
    "judge_gold_agreement": "judge",
    "parse_judge_score": "judge",
}

__all__ = sorted(_LAZY_EXPORTS)


def __getattr__(name: str) -> Any:
    try:
        module_name = _LAZY_EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc
    value = getattr(import_module(f".{module_name}", __name__), name)
    globals()[name] = value
    return value
