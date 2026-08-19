"""PPO: the clipped-surrogate algorithm, and the tensor layer it needs.

This is a **base algorithm layer, not a training method**. It has no trainer, no
critic, no value head and no GAE -- that path was deliberately removed, since
training here is critic-free. What lives here is the clipped surrogate itself
plus the small numerical primitives it is built from.

`grpo` depends on this package, and that direction is deliberate: GRPO *is* the
PPO clipped surrogate with a group-relative advantage substituted for GAE, so
`grpo/llm_grpo_trainer.py` and `grpo/controller.py` call straight into
`compute_ppo_policy_loss_core` and `PPOPolicyLossConfig`. `grpo/generation.py`
uses this layer directly too.

So do not read `ppo/` as a sibling of `sft/`, `dpo/` and `grpo/`. Those are
methods you run; this is the arithmetic they run on.

One naming wart, preserved rather than fixed: `compute_trajectory_policy_loss`
returns its loss under a `"grpo_policy_loss"` key. It is the PPO surrogate --
GRPO only supplies different advantages -- but the key is a published contract
that `examples/run_grpo_training_pipeline.py` and `grpo/generation.py` read,
so renaming it belongs in its own change.

Contains no rollout-manager or search-agent state.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

LOG_RATIO_CLAMP = 20.0


def masked_mean(x: torch.Tensor, mask: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Mean over masked positions, used by PPO/GRPO core losses."""
    resolved_mask = mask.to(dtype=x.dtype, device=x.device)
    return (x * resolved_mask).sum() / (resolved_mask.sum() + eps)


def masked_whiten(
    values: torch.Tensor,
    mask: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Whiten values over masked positions while preserving tensor shape."""
    resolved_mask = mask.to(dtype=values.dtype, device=values.device)
    mean = masked_mean(values, resolved_mask, eps)
    var = masked_mean((values - mean).square(), resolved_mask, eps)
    return (values - mean) / torch.sqrt(var + eps)


def entropy_from_logits(logits: torch.Tensor) -> torch.Tensor:
    """Per-token categorical entropy from unnormalized logits."""
    probs = torch.softmax(logits, dim=-1)
    log_probs = torch.log_softmax(logits, dim=-1)
    return -(probs * log_probs).sum(dim=-1)


def clip_by_value(
    x: torch.Tensor,
    min_value: torch.Tensor | float,
    max_value: torch.Tensor | float,
) -> torch.Tensor:
    """Clamp with tensor-compatible bounds."""
    return torch.max(torch.min(x, max_value), min_value)


class AdaptiveKLController:
    """Adaptive KL coefficient controller for PPO-style training."""

    def __init__(self, init_kl_coef: float, target_kl: float, horizon: int) -> None:
        self.value = float(init_kl_coef)
        self.target = float(target_kl)
        self.horizon = int(horizon)

    def update(self, current_kl: float, n_steps: int) -> None:
        if self.target <= 0 or self.horizon <= 0:
            return
        proportional_error = max(
            min(float(current_kl) / self.target - 1.0, 0.2),
            -0.2,
        )
        mult = 1.0 + proportional_error * int(n_steps) / self.horizon
        self.value *= mult


class FixedKLController:
    """Fixed KL coefficient controller with the same interface as adaptive KL."""

    def __init__(self, kl_coef: float) -> None:
        self.value = float(kl_coef)

    def update(self, current_kl: float, n_steps: int) -> None:
        del current_kl, n_steps


def compute_rewards(
    token_level_scores: torch.Tensor,
    old_log_prob: torch.Tensor,
    ref_log_prob: torch.Tensor,
    kl_ratio: float,
) -> torch.Tensor:
    """Token rewards after subtracting old-vs-reference KL penalty."""
    kl = old_log_prob - ref_log_prob
    return token_level_scores - kl * float(kl_ratio)


def compute_ppo_policy_loss_core(
    old_log_prob: torch.Tensor,
    log_prob: torch.Tensor,
    advantages: torch.Tensor,
    eos_mask: torch.Tensor,
    cliprange: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Clipped PPO policy loss core.

    Returns ``(pg_loss, pg_clipfrac, ppo_kl, surrogate)`` where:

    - ``pg_loss`` is ready to minimize.
    - ``pg_clipfrac`` is the masked fraction of clipped tokens.
    - ``ppo_kl`` is the masked approximate KL for monitoring.
    - ``surrogate`` is per-token ``max(pg_losses, pg_losses_clipped)``.
    """
    negative_approx_kl = log_prob - old_log_prob
    ratio = torch.exp(
        torch.clamp(negative_approx_kl, -LOG_RATIO_CLAMP, LOG_RATIO_CLAMP)
    )
    ppo_kl = masked_mean(-negative_approx_kl, eos_mask)

    pg_losses = -advantages * ratio
    pg_losses_clipped = -advantages * torch.clamp(
        ratio,
        1.0 - float(cliprange),
        1.0 + float(cliprange),
    )
    surrogate = torch.maximum(pg_losses, pg_losses_clipped)
    pg_loss = masked_mean(surrogate, eos_mask)
    pg_clipfrac = masked_mean(
        (pg_losses_clipped > pg_losses).to(dtype=log_prob.dtype),
        eos_mask,
    )
    return pg_loss, pg_clipfrac, ppo_kl, surrogate


def compute_entropy_loss(logits: torch.Tensor, eos_mask: torch.Tensor) -> torch.Tensor:
    """Masked mean policy entropy."""
    return masked_mean(entropy_from_logits(logits), eos_mask)


def kl_penalty(
    logprob: torch.Tensor,
    ref_logprob: torch.Tensor,
    kl_penalty_type: str,
) -> torch.Tensor:
    """Per-token KL penalty variants commonly used by PPO/GRPO trainers."""
    if kl_penalty_type == "kl":
        return logprob - ref_logprob
    if kl_penalty_type == "abs":
        return (logprob - ref_logprob).abs()
    if kl_penalty_type == "mse":
        return 0.5 * (logprob - ref_logprob).square()
    if kl_penalty_type == "low_var_kl":
        kl = ref_logprob - logprob
        ratio = torch.exp(torch.clamp(kl, -LOG_RATIO_CLAMP, LOG_RATIO_CLAMP))
        kld = ratio - kl - 1.0
        return torch.clamp(kld, min=-10.0, max=10.0)
    raise NotImplementedError(f"Unknown KL penalty type: {kl_penalty_type}")


@dataclass(frozen=True)
class PPOPolicyLossConfig:
    """Configuration for PPO / GRPO clipped policy optimization."""

    clip_epsilon: float = 0.2
    kl_coefficient: float = 0.0
    # Coefficient for the entropy bonus.  Positive values keep search /
    # reasoning / stopping policies diverse and prevent mode collapse.
    # Requires new_log_probs to be present in the batch.
    entropy_coefficient: float = 0.0
    # Per-action-type loss multipliers.  Keys: "search", "think", "fetch", "answer".
    # Tokens with no matching key keep weight 1.0.  None means uniform weighting.
    action_type_weights: dict[str, float] | None = None
    # Whiten advantages over action tokens before computing the PPO loss.
    # Reduces gradient variance by zero-meaning and unit-scaling the advantage
    # signal across the batch before it enters the clipped surrogate.
    whiten_advantages: bool = False


def compute_trajectory_policy_loss(
    *,
    new_log_probs: list[float],
    old_log_probs: list[float],
    advantages: list[float],
    response_mask: list[int],
    ref_log_probs: list[float] | None = None,
    clip_epsilon: float = 0.2,
    kl_beta: float = 0.0,
) -> dict[str, float]:
    """Compute the GRPO / PPO clipped policy loss for one trajectory.

    Operates on the aligned lists produced by ``trajectory_log_prob_pack``
    so it requires no SearchBatch, no tokenizer, and no GPU batch.

    Formula::

        ratio      = exp(new_log_probs - old_log_probs)
        loss1      = ratio * advantages
        loss2      = clamp(ratio, 1-ε, 1+ε) * advantages
        grpo_loss  = -mean(min(loss1, loss2) * response_mask)

        kl         = KL(ref || new)  using ref_log_probs if given,
                     else KL(old || new)
        kl_penalty = kl_beta * mean(kl * response_mask)

        total_loss = grpo_loss + kl_penalty

    ``response_mask`` must be 1 only for model-generated action tokens.
    Prompt positions and env-injected observation tokens must be 0.

    Returns:
        Dict with ``grpo_policy_loss``, ``kl_penalty``, ``total_loss``,
        ``clip_fraction``, and ``mean_ratio`` for logging.
    """
    n = len(new_log_probs)
    if not (len(old_log_probs) == len(advantages) == len(response_mask) == n):
        raise ValueError(
            "new_log_probs, old_log_probs, advantages, and response_mask "
            f"must all have the same length, got lengths "
            f"{n}, {len(old_log_probs)}, {len(advantages)}, {len(response_mask)}."
        )

    new_lp = torch.tensor(new_log_probs, dtype=torch.float32)
    old_lp = torch.tensor(old_log_probs, dtype=torch.float32)
    adv = torch.tensor(advantages, dtype=torch.float32)
    mask = torch.tensor(response_mask, dtype=torch.float32)

    log_ratio = torch.clamp(new_lp - old_lp, min=-LOG_RATIO_CLAMP, max=LOG_RATIO_CLAMP)
    ratio = torch.exp(log_ratio)
    clipped_ratio = torch.clamp(ratio, 1.0 - clip_epsilon, 1.0 + clip_epsilon)
    surrogate = torch.minimum(ratio * adv, clipped_ratio * adv)

    normalizer = max(float(mask.sum()), 1.0)
    grpo_policy_loss = float(-(surrogate * mask).sum() / normalizer)

    kl_penalty_val = 0.0
    if kl_beta:
        baseline = (
            torch.tensor(ref_log_probs, dtype=torch.float32)
            if ref_log_probs is not None
            else old_lp
        )
        kl_log_ratio = torch.clamp(
            (baseline - new_lp) * mask, min=-LOG_RATIO_CLAMP, max=LOG_RATIO_CLAMP
        )
        kl = (torch.exp(kl_log_ratio) - kl_log_ratio - 1.0) * mask
        kl_penalty_val = kl_beta * float(kl.sum() / normalizer)

    clip_fraction = float(
        ((ratio != clipped_ratio).to(dtype=torch.float32) * mask).sum() / normalizer
    )
    return {
        "grpo_policy_loss": grpo_policy_loss,
        "kl_penalty": kl_penalty_val,
        "total_loss": grpo_policy_loss + kl_penalty_val,
        "clip_fraction": clip_fraction,
        "mean_ratio": float((ratio * mask).sum() / normalizer),
    }
