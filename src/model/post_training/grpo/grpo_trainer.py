"""Self-contained GRPO trainer for bandit / grouped-rollout settings.

Complements the token-level utilities in core_algos.py with a batch-grouped
variant suitable for discrete-action environments and synthetic benchmarks.
"""

from __future__ import annotations

import copy

import torch
import torch.nn as nn
import torch.optim as optim


# ---------------------------------------------------------------------------
# Tensor utilities (2-D grouped variants — different from the scalar
# masked_mean in core_algos.py which reduces over all positions)
# ---------------------------------------------------------------------------


def _group_masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Row-wise masked mean. Returns shape ``(batch, 1)``."""
    mask = mask.float()
    denom = mask.sum(dim=1, keepdim=True).clamp_min(1e-8)
    return (values * mask).sum(dim=1, keepdim=True) / denom


def compute_group_advantages(
    rewards: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """GRPO group advantage: normalise rewards within each prompt group.

    Args:
        rewards: ``(batch_size, group_size)`` — one scalar reward per rollout.
        mask: ``(batch_size, group_size)`` — 1 for valid rollouts, 0 for padding.

    Returns:
        ``(batch_size, group_size)`` advantages detached from the graph.
    """
    if rewards.ndim != 2 or mask.ndim != 2:
        raise ValueError("rewards and mask must both be 2-D grouped tensors")
    if rewards.shape != mask.shape:
        raise ValueError("rewards and mask must have the same shape")
    if rewards.shape[0] == 0 or rewards.shape[1] == 0:
        raise ValueError("rewards and mask must have non-empty group dimensions")

    mask = mask.to(device=rewards.device, dtype=rewards.dtype)
    if torch.any(mask.sum(dim=1) <= 0):
        raise ValueError("each group must contain at least one valid rollout")

    group_mean = _group_masked_mean(rewards, mask)
    centered = (rewards - group_mean) * mask
    group_var = _group_masked_mean(centered.pow(2), mask)
    group_std = torch.sqrt(group_var + 1e-8)
    return (centered / group_std).detach()


def grpo_clipped_policy_loss(
    new_log_probs: torch.Tensor,
    old_log_probs: torch.Tensor,
    advantages: torch.Tensor,
    clip_epsilon: float = 0.2,
) -> tuple[torch.Tensor, torch.Tensor]:
    """PPO-style clipped objective for a flat batch of rollouts.

    Returns:
        ``(per_token_loss, ratios)`` — caller applies mask and reduces.
    """
    ratios = torch.exp(new_log_probs - old_log_probs)
    clipped = torch.clamp(ratios, 1.0 - clip_epsilon, 1.0 + clip_epsilon)
    loss = -torch.min(ratios * advantages, clipped * advantages)
    return loss, ratios


def reverse_kl_penalty(
    policy_log_probs: torch.Tensor,
    ref_log_probs: torch.Tensor,
) -> torch.Tensor:
    """Sample-based reverse-KL penalty: ``exp(Δlog p) - Δlog p - 1``."""
    log_ratio = policy_log_probs - ref_log_probs
    return torch.exp(log_ratio) - log_ratio - 1.0


# ---------------------------------------------------------------------------
# Policy network
# ---------------------------------------------------------------------------


class Policy(nn.Module):
    """Simple MLP policy for discrete action spaces."""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 64,
        lr: float = 1e-3,
    ) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )
        self.optimizer = optim.Adam(self.parameters(), lr=lr)

    def forward(self, states: torch.Tensor) -> torch.Tensor:
        return self.network(states)

    def get_distribution(self, states: torch.Tensor) -> torch.distributions.Categorical:
        return torch.distributions.Categorical(logits=self.forward(states))

    def get_log_probs(
        self, states: torch.Tensor, actions: torch.Tensor
    ) -> torch.Tensor:
        return self.get_distribution(states).log_prob(actions)


# ---------------------------------------------------------------------------
# GRPO Trainer
# ---------------------------------------------------------------------------


class GRPOTrainer:
    """Self-contained GRPO update loop for grouped rollout environments.

    Args:
        policy: Trainable policy.
        reference_policy: Frozen anchor for the KL penalty.
        clip_epsilon: PPO clip range.
        beta: KL penalty coefficient (``0`` disables the penalty).
        grad_clip: Max-norm for gradient clipping.
    """

    def __init__(
        self,
        policy: Policy,
        reference_policy: Policy,
        clip_epsilon: float = 0.2,
        beta: float = 0.1,
        grad_clip: float = 1.0,
    ) -> None:
        self.policy = policy
        self.reference_policy = reference_policy
        self.clip_epsilon = clip_epsilon
        self.beta = beta
        self.grad_clip = grad_clip

        self.reference_policy.eval()
        for param in self.reference_policy.parameters():
            param.requires_grad = False

    def compute_loss(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        old_log_probs: torch.Tensor,
        rewards: torch.Tensor,
        mask: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Compute the masked GRPO loss with KL penalty.

        Args:
            states: ``(batch_size * group_size, state_dim)``
            actions: ``(batch_size * group_size,)``
            old_log_probs: ``(batch_size * group_size,)``
            rewards: ``(batch_size, group_size)``
            mask: ``(batch_size, group_size)``

        Returns:
            ``(mean_loss, metrics_dict)``
        """
        expected_rollouts = rewards.numel()
        if (
            states.shape[0] != expected_rollouts
            or actions.shape != (expected_rollouts,)
            or old_log_probs.shape != (expected_rollouts,)
        ):
            raise ValueError(
                "states, actions, and old_log_probs must contain "
                "batch_size * group_size rollouts"
            )

        flat_mask = mask.reshape(-1).to(device=rewards.device, dtype=rewards.dtype)
        advantages = compute_group_advantages(rewards, mask).reshape(-1)

        new_log_probs = self.policy.get_log_probs(states, actions)

        policy_loss, ratios = grpo_clipped_policy_loss(
            new_log_probs=new_log_probs,
            old_log_probs=old_log_probs,
            advantages=advantages,
            clip_epsilon=self.clip_epsilon,
        )
        if self.beta == 0:
            kl = torch.zeros_like(new_log_probs)
        else:
            with torch.no_grad():
                ref_log_probs = self.reference_policy.get_log_probs(states, actions)
            kl = reverse_kl_penalty(new_log_probs, ref_log_probs)
        total = policy_loss + self.beta * kl
        normalizer = flat_mask.sum().clamp_min(1e-8)
        mean_loss = (total * flat_mask).sum() / normalizer

        with torch.no_grad():
            metrics: dict[str, float] = {
                "loss": mean_loss.item(),
                "mean_reward": float(
                    (rewards.reshape(-1) * flat_mask).sum() / normalizer
                ),
                "mean_advantage": float((advantages * flat_mask).sum() / normalizer),
                "mean_ratio": float((ratios.detach() * flat_mask).sum() / normalizer),
                "mean_kl": float((kl.detach() * flat_mask).sum() / normalizer),
            }
        return mean_loss, metrics

    def update(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        old_log_probs: torch.Tensor,
        rewards: torch.Tensor,
        mask: torch.Tensor,
    ) -> dict[str, float]:
        """Run one gradient step; returns the metrics dict."""
        loss, metrics = self.compute_loss(states, actions, old_log_probs, rewards, mask)
        self.policy.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.grad_clip)
        self.policy.optimizer.step()
        return metrics


def make_grpo_trainer(
    state_dim: int,
    action_dim: int,
    hidden_dim: int = 64,
    lr: float = 1e-3,
    clip_epsilon: float = 0.2,
    beta: float = 0.1,
    grad_clip: float = 1.0,
) -> GRPOTrainer:
    """Convenience factory that builds policy + reference and returns a trainer."""
    policy = Policy(state_dim, action_dim, hidden_dim=hidden_dim, lr=lr)
    reference_policy = copy.deepcopy(policy)
    return GRPOTrainer(
        policy=policy,
        reference_policy=reference_policy,
        clip_epsilon=clip_epsilon,
        beta=beta,
        grad_clip=grad_clip,
    )
