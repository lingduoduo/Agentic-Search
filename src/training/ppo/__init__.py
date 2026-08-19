"""PPO: the clipped-surrogate algorithm layer that `grpo` is built on.

A base algorithm layer, **not** a training method. There is no PPO trainer here
-- no critic, no value head, no GAE -- because training in this repo is
critic-free. What this package owns is the clipped surrogate, the KL
controllers, and the masked-tensor primitives they need.

Read the dependency direction as layering rather than inversion: GRPO is the PPO
surrogate with a group-relative advantage in place of GAE, so `grpo` importing
`ppo` is a variant importing its base. `sft/`, `dpo/` and `grpo/` are methods you
run; `ppo/` is arithmetic they run on, which is why nothing imports it *from*
`grpo` in the other direction.

Re-exports are lazy for the same reason they are in `grpo`: importing any
submodule of a package executes this file first, and an eager
`from .core_algos import ...` would put `import torch` in front of every module
placed here later. That has broken CI in this repo four times.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

_LAZY_EXPORTS: dict[str, str] = {
    "LOG_RATIO_CLAMP": "core_algos",
    # masked-tensor primitives
    "masked_mean": "core_algos",
    "masked_whiten": "core_algos",
    "entropy_from_logits": "core_algos",
    "clip_by_value": "core_algos",
    # KL control
    "AdaptiveKLController": "core_algos",
    "FixedKLController": "core_algos",
    "kl_penalty": "core_algos",
    "compute_rewards": "core_algos",
    # the clipped surrogate
    "compute_ppo_policy_loss_core": "core_algos",
    "compute_entropy_loss": "core_algos",
    "PPOPolicyLossConfig": "core_algos",
    "compute_trajectory_policy_loss": "core_algos",
    # reward_manager — PPO-style token rewards (VERL heritage). Moved here
    # from the GRPO package: its name always pointed at this layer.
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
