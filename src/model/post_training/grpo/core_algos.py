"""GRPO and REINFORCE: the variants built on the PPO surrogate.

The clipped surrogate itself, the KL controllers and the masked-tensor
primitives live one layer down in `src/model/post_training/ppo`; this module holds what is
specific to the algorithms this package actually trains with -- GRPO's group-relative
advantage and its config wrapper, and the REINFORCE losses.

`compute_grpo_policy_loss` delegates the arithmetic to
`compute_trajectory_policy_loss` in that base layer, which is the honest shape:
GRPO is the PPO surrogate with a different advantage.
"""

from __future__ import annotations

import torch

from ..ppo.core_algos import (
    PPOPolicyLossConfig,
    compute_trajectory_policy_loss,
    masked_mean,
    masked_whiten,
)


def compute_grpo_token_advantages(
    token_level_rewards: torch.Tensor,
    eos_mask: torch.Tensor,
    index: torch.Tensor,
    epsilon: float = 1e-6,
    clip_advantages: float | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Group-normalized outcome advantages expanded over response tokens.

    Named for its shape: unlike the scalar ``compute_grpo_outcome_advantage``
    in :mod:`src.model.post_training.grpo.rollouts`, this returns a
    ``(batch, seq_len)`` tensor with each rollout's advantage broadcast across
    its response tokens.

    Args:
        token_level_rewards: ``(batch, seq_len)`` sparse reward tensor.
        eos_mask: ``(batch, seq_len)`` binary mask over response tokens.
        index: ``(batch,)`` integer group IDs.
        epsilon: Denominator stabiliser for std normalization.
        clip_advantages: If given, clip normalized advantages to
            ``[-clip_advantages, +clip_advantages]`` before expansion.
    """
    response_length = token_level_rewards.shape[-1]
    scores = token_level_rewards.sum(dim=-1)
    advantages = torch.zeros_like(token_level_rewards)

    with torch.no_grad():
        for group_id in torch.unique(index).tolist():
            group_mask = index == int(group_id)
            group_scores = scores[group_mask]
            if group_scores.numel() == 0:
                continue
            mean = group_scores.mean()
            std = (
                group_scores.std(unbiased=False)
                if group_scores.numel() > 1
                else torch.tensor(1.0, device=scores.device, dtype=scores.dtype)
            )
            normalized = (group_scores - mean) / (std + epsilon)
            if clip_advantages is not None:
                normalized = normalized.clamp(
                    -float(clip_advantages), float(clip_advantages)
                )
            advantages[group_mask] = (
                normalized.unsqueeze(-1).expand(-1, response_length)
                * eos_mask[group_mask]
            )

    return advantages, advantages


def compute_reinforce_policy_loss_core(
    log_prob: torch.Tensor,
    advantages: torch.Tensor,
    eos_mask: torch.Tensor,
) -> torch.Tensor:
    """Unclipped REINFORCE policy-gradient loss over masked response tokens."""

    return -masked_mean(log_prob * advantages, eos_mask)


def compute_reinforce_policy_loss(
    *,
    log_probs: list[float],
    rewards: list[float],
    response_mask: list[int],
    baseline: float = 0.0,
) -> dict[str, float]:
    """Compute a trajectory-level REINFORCE loss from aligned token lists."""

    n = len(log_probs)
    if not (len(rewards) == len(response_mask) == n):
        raise ValueError(
            "log_probs, rewards, and response_mask must all have the same length, "
            f"got lengths {n}, {len(rewards)}, {len(response_mask)}."
        )

    log_prob_tensor = torch.tensor(log_probs, dtype=torch.float32)
    reward_tensor = torch.tensor(rewards, dtype=torch.float32)
    mask = torch.tensor(response_mask, dtype=torch.float32)
    advantages = reward_tensor - float(baseline)
    normalizer = max(float(mask.sum()), 1.0)
    loss = float(-((log_prob_tensor * advantages * mask).sum() / normalizer))
    return {
        "reinforce_policy_loss": loss,
        "total_loss": loss,
        "mean_reward": float((reward_tensor * mask).sum() / normalizer),
        "mean_advantage": float((advantages * mask).sum() / normalizer),
    }


def compute_grpo_policy_loss(
    *,
    new_log_probs: list[float],
    old_log_probs: list[float],
    advantages: list[float],
    response_mask: list[int],
    ref_log_probs: list[float] | None = None,
    config: PPOPolicyLossConfig | None = None,
) -> dict[str, float]:
    """Convenience wrapper that accepts a :class:`PPOPolicyLossConfig`.

    Combines the clipped policy loss, optional KL penalty, optional entropy
    bonus, and optional advantage whitening in a single call.  Delegates to
    :func:`compute_trajectory_policy_loss` for the core arithmetic.

    Args:
        new_log_probs: Current policy log-probs for response tokens.
        old_log_probs: Behaviour-policy log-probs (from rollout time).
        advantages: Per-token advantage values (aligned with response_mask).
        response_mask: 1 for model-generated tokens, 0 for prompt / padding.
        ref_log_probs: Reference model log-probs for KL penalty.
            If ``None`` and ``config.kl_coefficient > 0``, ``old_log_probs``
            is used as the reference.
        config: Loss configuration.  Defaults to ``PPOPolicyLossConfig()``.

    Returns:
        Dict with ``grpo_policy_loss``, ``kl_penalty``, ``entropy_bonus``,
        ``total_loss``, ``clip_fraction``, and ``mean_ratio`` keys.
    """
    cfg = config or PPOPolicyLossConfig()

    adv = list(advantages)
    if cfg.whiten_advantages:
        adv_t = torch.tensor(adv, dtype=torch.float32)
        mask_t = torch.tensor(response_mask, dtype=torch.float32)
        adv_t = masked_whiten(adv_t, mask_t)
        adv = adv_t.tolist()

    base = compute_trajectory_policy_loss(
        new_log_probs=new_log_probs,
        old_log_probs=old_log_probs,
        advantages=adv,
        response_mask=response_mask,
        ref_log_probs=ref_log_probs,
        clip_epsilon=cfg.clip_epsilon,
        kl_beta=cfg.kl_coefficient,
    )

    entropy_bonus = 0.0
    if cfg.entropy_coefficient != 0.0:
        new_lp = torch.tensor(new_log_probs, dtype=torch.float32)
        mask_t = torch.tensor(response_mask, dtype=torch.float32)
        # Entropy approximation: H ≈ -E[log p], averaged over masked tokens.
        h = masked_mean(-new_lp, mask_t)
        entropy_bonus = float(cfg.entropy_coefficient * h)

    total = base["total_loss"] - entropy_bonus
    return {
        **base,
        "entropy_bonus": entropy_bonus,
        "total_loss": total,
    }
