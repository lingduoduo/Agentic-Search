"""GRPO: group-relative policy optimization, and the trainers that run it.

Three trainers -- bandit, HuggingFace causal-LM, and one backed by a live
SearchAgentLoop -- plus grouped rollout sampling, a controller and a durable
train loop.

This package was called ``rl`` until the things in it that were not GRPO moved
out: the tabular Q-learning demo to ``src.model.post_training.qlearning`` and
``PPORewardManager`` to ``src.model.post_training.ppo``, where its name already pointed.
What remains that is not strictly GRPO is the pair of REINFORCE losses in
``algorithms`` -- kept here deliberately, as the ancestor policy-gradient
algorithm the group-relative variant descends from, and far too small to earn a
package.

The clipped surrogate itself, the KL controllers and the masked-tensor
primitives live one layer down in ``src.model.post_training.ppo``: GRPO is that surrogate
with a group-relative advantage in place of GAE. They are deliberately not
re-exported here -- a variant should not advertise its base layer's API as its
own.

Re-exports are lazy, and that is load-bearing: importing any submodule of a
package executes this file first, so an eager ``from .algorithms import ...``
would put ``import torch`` in front of every module here. That is exactly how
this package broke CI when the Q-learning demo still lived in it (#536), and the
repo has shipped that failure four times.

Rollout helpers remain available by their root exports and consolidated module
path.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

_LAZY_EXPORTS: dict[str, str] = {
    # algorithms — GRPO/REINFORCE variants, rollout scoring, and judges.
    # the KL controllers and the masked-tensor primitives are one layer down in
    # `src.model.post_training.ppo`, and are NOT re-exported here: a variant should not
    # advertise its base layer's API as its own.
    "compute_grpo_outcome_advantage": "algorithms",
    "compute_grpo_policy_loss": "algorithms",
    "compute_reinforce_policy_loss": "algorithms",
    "compute_reinforce_policy_loss_core": "algorithms",
    # training — trainers, local controller, and durable loop
    "LocalGRPOController": "training",
    "RolloutResult": "training",
    # bandit, HuggingFace causal-LM, and live SearchAgentLoop rollouts
    "GRPOTrainer": "training",
    "Policy": "training",
    "compute_group_advantages": "training",
    "grpo_clipped_policy_loss": "training",
    "make_grpo_trainer": "training",
    "reverse_kl_penalty": "training",
    "LLMGRPOConfig": "training",
    "LLMGRPOTrainer": "training",
    "LLMRolloutResult": "training",
    "get_response_log_probs": "..log_probs",
    "SearchAgentGRPOTrainer": "training",
    # judge — RLAIF scoring, moved in from the shared top level. Deferred
    # like everything else here: it reaches ``reward``, which reaches the
    # agent loop.
    "GoldAgreementJudge": "algorithms",
    "JudgeParseError": "algorithms",
    "LLMJudge": "algorithms",
    "SimulatedPreferenceJudge": "algorithms",
    "is_degenerate_group": "algorithms",
    "judge_gold_agreement": "algorithms",
    "parse_judge_score": "algorithms",
}

__all__ = sorted(_LAZY_EXPORTS)


def __getattr__(name: str) -> Any:
    try:
        module_name = _LAZY_EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc
    module_path = module_name if module_name.startswith("..") else f".{module_name}"
    value = getattr(import_module(module_path, __name__), name)
    globals()[name] = value
    return value
