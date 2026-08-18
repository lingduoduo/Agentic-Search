"""Reinforcement learning: the GRPO stack, and the shared PPO/GRPO math.

Despite the historical ``ppo`` name this package carried, almost everything here
is GRPO: three trainers (bandit, HuggingFace causal-LM, and one backed by a live
SearchAgentLoop), a controller, and a durable train loop. ``core_algos`` holds
the genuinely shared PPO/GRPO tensor math, which ``src/model/generation`` also
uses.

**Every re-export below is lazy, and that is load-bearing.** Importing any
submodule of a package first executes this file, so an eager
``from .core_algos import ...`` here would put ``import torch`` in front of
*every* module in the package — including ``qlearning`` and
``search_environment``, a self-contained tabular Q-learning demo that has
nothing to do with torch or the LLM stack. That is exactly what broke the
unit-test job when these modules moved in: CI installs no heavy ML packages, so
``import src.training.rl.qlearning`` failed at collection with
``ModuleNotFoundError: No module named 'torch'``. This repo has shipped
torch-in-CI collection failures twice before (#356, re-fixed in #418).

Deferring the imports rather than wrapping them in ``try/except ImportError``
is deliberate: a swallowed ImportError turns a typo in ``core_algos`` into a
silently missing export, whereas ``__getattr__`` raises the real error at the
moment someone actually asks for a torch-backed name.

``rollouts``, ``qlearning`` and ``search_environment`` are not re-exported at
all — ``rollouts`` pulls in the agent loop, and the Q-learning pair is a demo.
Import those by module path.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

_LAZY_EXPORTS: dict[str, str] = {
    # core_algos — shared PPO/GRPO tensor math (torch)
    "AdaptiveKLController": "core_algos",
    "FixedKLController": "core_algos",
    "PPOPolicyLossConfig": "core_algos",
    "clip_by_value": "core_algos",
    "compute_entropy_loss": "core_algos",
    "compute_grpo_outcome_advantage": "core_algos",
    "compute_grpo_policy_loss": "core_algos",
    "compute_ppo_policy_loss_core": "core_algos",
    "compute_reinforce_policy_loss": "core_algos",
    "compute_reinforce_policy_loss_core": "core_algos",
    "compute_rewards": "core_algos",
    "compute_trajectory_policy_loss": "core_algos",
    "entropy_from_logits": "core_algos",
    "kl_penalty": "core_algos",
    "masked_mean": "core_algos",
    "masked_whiten": "core_algos",
    # controller
    "LocalGRPOController": "controller",
    "RolloutResult": "controller",
    # grpo_trainer — bandit / grouped-rollout
    "GRPOTrainer": "grpo_trainer",
    "Policy": "grpo_trainer",
    "compute_group_advantages": "grpo_trainer",
    "grpo_clipped_policy_loss": "grpo_trainer",
    "make_grpo_trainer": "grpo_trainer",
    "reverse_kl_penalty": "grpo_trainer",
    # llm_grpo_trainer — HuggingFace causal-LM policies
    "LLMGRPOConfig": "llm_grpo_trainer",
    "LLMGRPOTrainer": "llm_grpo_trainer",
    "LLMRolloutResult": "llm_grpo_trainer",
    "get_response_log_probs": "llm_grpo_trainer",
    # search_agent_grpo_trainer — live SearchAgentLoop rollouts
    "SearchAgentGRPOTrainer": "search_agent_grpo_trainer",
    # reward_manager — PPO-style token rewards
    "PPORewardManager": "reward_manager",
    "qa_exact_match_score": "reward_manager",
    "select_reward_score_fn": "reward_manager",
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
