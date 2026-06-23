"""Online GRPO trainer for HuggingFace causal-LM policies.

Maps the toy bandit demo directly onto real LLM training:

    Demo concept          LLM equivalent
    ──────────────────── ──────────────────────────────────────────────
    state                prompt  (tokenized, shape: prompt_len)
    action               response tokens  (shape: response_len)
    reward_function      SearchRewardFunction + judge_fn  → scalar
    group_size           num_rollouts  — G completions per prompt
    reference_policy     frozen SFT model  (deepcopy, no grad)
    policy               current trainable LLM
    group advantage      compute_grpo_outcome_advantage  (token-level)
    KL penalty           low_var_kl(new_log_probs, ref_log_probs)

The full step mirrors GRPOTrainer.update() in grpo_trainer.py but
operates on token sequences instead of discrete bandit actions:

    1. rollout  — tile prompts G times, model.generate → response_ids
    2. score    — decode + judge_fn + SearchRewardFunction → rewards
    3. advantage — compute_grpo_outcome_advantage (group-relative, token-expanded)
    4. old_log_probs — forward pass under rollout policy (frozen snapshot)
    5. loss     — PPO clip + β·KL(policy ‖ reference)
    6. step     — zero_grad → backward → clip_grad → optimizer.step
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import torch
import torch.nn as nn

from .core_algos import (
    compute_grpo_outcome_advantage,
    compute_ppo_policy_loss_core,
    kl_penalty,
    masked_mean,
)

if TYPE_CHECKING:
    from src.training.reward import JudgeFn, SearchRewardFunction


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LLMGRPOConfig:
    """Hyperparameters for one online GRPO training step."""

    num_rollouts: int = 4
    max_new_tokens: int = 256
    temperature: float = 0.7
    temperature_step: float = 0.1
    clip_epsilon: float = 0.2
    kl_beta: float = 0.04
    grad_clip: float = 1.0
    kl_penalty_type: str = "low_var_kl"
    advantage_epsilon: float = 1e-6
    advantage_clip: float | None = None


# ---------------------------------------------------------------------------
# Rollout output
# ---------------------------------------------------------------------------


@dataclass
class LLMRolloutResult:
    """All per-group data produced by one rollout step.

    Shape convention (B = batch_size, G = num_rollouts, T = response_len):

        prompt_ids     (B, P)
        response_ids   (B*G, T)  — flattened; group i occupies rows [i*G : (i+1)*G]
        response_mask  (B*G, T)  — 1 for non-pad response tokens
        old_log_probs  (B*G, T)  — log π_old(token | context) at each response position
        rewards        (B*G,)    — scalar reward per rollout
        advantages     (B*G, T)  — token-expanded group-relative advantage
        group_ids      (B*G,)    — integer group index (same for G rows in a group)
    """

    prompt_ids: torch.Tensor
    response_ids: torch.Tensor
    response_mask: torch.Tensor
    old_log_probs: torch.Tensor
    rewards: torch.Tensor
    advantages: torch.Tensor
    group_ids: torch.Tensor
    metadata: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Token log-probability helper
# ---------------------------------------------------------------------------


def get_response_log_probs(
    model: nn.Module,
    input_ids: torch.Tensor,
    prompt_len: int,
    response_mask: torch.Tensor,
) -> torch.Tensor:
    """Forward pass through *model* and return per-token log probs over the response.

    Args:
        model: Causal-LM model.  Called in whatever grad context the caller sets.
        input_ids: ``(B, prompt_len + response_len)`` — concatenated prompt + response.
        prompt_len: Number of prompt tokens to skip in the output.
        response_mask: ``(B, response_len)`` — 1 for valid response tokens.

    Returns:
        ``(B, response_len)`` — masked log probabilities (0 at padding positions).
    """
    outputs = model(input_ids=input_ids)
    # Logits at position t predict token t+1.
    # Slice prompt positions so response logits align with response_ids.
    logits = outputs.logits[:, prompt_len - 1 : -1, :]  # (B, response_len, vocab)
    log_probs = torch.log_softmax(logits.float(), dim=-1)

    response_ids = input_ids[:, prompt_len:]  # (B, response_len)
    token_log_probs = log_probs.gather(
        dim=-1, index=response_ids.unsqueeze(-1)
    ).squeeze(-1)  # (B, response_len)

    return token_log_probs * response_mask.to(dtype=token_log_probs.dtype)


# ---------------------------------------------------------------------------
# Rollout builder
# ---------------------------------------------------------------------------


def _build_response_mask(response_ids: torch.Tensor, pad_token_id: int) -> torch.Tensor:
    """1 for every non-pad response token, 0 otherwise."""
    return (response_ids != pad_token_id).long()


def _sparse_reward_at_eos(
    rewards: torch.Tensor,
    response_mask: torch.Tensor,
) -> torch.Tensor:
    """Place each scalar reward at the last valid response token (EOS position).

    This is the standard sparse outcome-reward layout for GRPO: the entire
    reward signal lives at the position where the model stopped generating.

    Args:
        rewards: ``(N,)`` scalar reward per rollout.
        response_mask: ``(N, T)`` — 1 for non-pad positions.

    Returns:
        ``(N, T)`` sparse tensor with reward at the last 1-position per row.
    """
    token_rewards = torch.zeros_like(response_mask, dtype=rewards.dtype)
    eos_positions = (response_mask.sum(dim=1) - 1).clamp(min=0)  # last non-pad idx
    for i, pos in enumerate(eos_positions.tolist()):
        token_rewards[i, int(pos)] = rewards[i]
    return token_rewards


# ---------------------------------------------------------------------------
# Online LLM GRPO Trainer
# ---------------------------------------------------------------------------


class LLMGRPOTrainer:
    """HuggingFace-native online GRPO trainer.

    Mirrors GRPOTrainer (grpo_trainer.py) but operates on token sequences:

        policy         — AutoModelForCausalLM being optimized
        reference      — frozen SFT copy (deepcopy at init)
        judge_fn       — (response_str, ground_truth_str) → float
        reward_fn      — optional SearchRewardFunction for shaped rewards

    Typical usage::

        trainer = LLMGRPOTrainer.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct",
                      judge_fn=exact_match, optimizer=optim.AdamW(policy.parameters()))
        metrics = trainer.step(prompts=["What is FAISS?"], ground_truths=["..."])
    """

    def __init__(
        self,
        policy: nn.Module,
        reference_policy: nn.Module,
        tokenizer: Any,
        optimizer: torch.optim.Optimizer,
        judge_fn: JudgeFn,
        config: LLMGRPOConfig | None = None,
        reward_fn: SearchRewardFunction | None = None,
        device: str | torch.device | None = None,
    ) -> None:
        self.policy = policy
        self.reference = reference_policy
        self.tokenizer = tokenizer
        self.optimizer = optimizer
        self.judge_fn = judge_fn
        self.reward_fn = reward_fn
        self.config = config or LLMGRPOConfig()
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )

        # Freeze reference; it is never updated.
        self.reference.eval()
        for p in self.reference.parameters():
            p.requires_grad = False

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def from_pretrained(
        cls,
        model_name_or_path: str,
        judge_fn: JudgeFn,
        optimizer_cls: type = torch.optim.AdamW,
        lr: float = 1e-5,
        config: LLMGRPOConfig | None = None,
        reward_fn: SearchRewardFunction | None = None,
        device: str | None = None,
        **hf_kwargs: Any,
    ) -> LLMGRPOTrainer:
        """Load policy + tokenizer from *model_name_or_path* and build trainer."""
        from transformers import AutoModelForCausalLM, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, **hf_kwargs)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = tokenizer.eos_token_id

        policy = AutoModelForCausalLM.from_pretrained(model_name_or_path, **hf_kwargs)
        reference = copy.deepcopy(policy)

        resolved_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        policy = policy.to(resolved_device)
        reference = reference.to(resolved_device)

        optimizer = optimizer_cls(policy.parameters(), lr=lr)
        return cls(
            policy=policy,
            reference_policy=reference,
            tokenizer=tokenizer,
            optimizer=optimizer,
            judge_fn=judge_fn,
            config=config,
            reward_fn=reward_fn,
            device=resolved_device,
        )

    # ------------------------------------------------------------------
    # Rollout  (demo: generate_dummy_group_data)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def rollout(
        self,
        prompts: list[str],
        ground_truths: list[str],
    ) -> LLMRolloutResult:
        """Sample G responses per prompt and compute rewards and advantages.

        Analogous to ``generate_dummy_group_data`` in the bandit demo, but for
        token sequences.  All G rollouts per prompt share the same prompt ids;
        group membership is tracked via ``group_ids``.

        Args:
            prompts: List of B raw text prompts.
            ground_truths: List of B gold answers aligned with *prompts*.

        Returns:
            :class:`LLMRolloutResult` with shape ``(B*G, ...)`` tensors.
        """
        cfg = self.config
        B = len(prompts)
        G = cfg.num_rollouts
        pad_id = self.tokenizer.pad_token_id or 0

        # ── 1. Tokenize prompts ───────────────────────────────────────────
        enc = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
        )
        prompt_ids: torch.Tensor = enc["input_ids"].to(self.device)  # (B, P)
        prompt_len = prompt_ids.shape[1]

        # Tile: (B, P) → (B*G, P) — each prompt repeated G times.
        tiled_prompt = prompt_ids.repeat_interleave(G, dim=0)  # (B*G, P)
        tiled_attention = (
            enc["attention_mask"].to(self.device).repeat_interleave(G, dim=0)
        )

        # ── 2. Generate G completions per prompt with varied temperature ──
        all_responses: list[torch.Tensor] = []
        for rollout_idx in range(G):
            temp = cfg.temperature + rollout_idx * cfg.temperature_step
            # Select rows for this rollout index: 0, G, 2G, ... (B rows)
            rows = tiled_prompt[rollout_idx::G]
            attn = tiled_attention[rollout_idx::G]
            out = self.policy.generate(
                input_ids=rows,
                attention_mask=attn,
                do_sample=True,
                temperature=temp,
                max_new_tokens=cfg.max_new_tokens,
                pad_token_id=pad_id,
            )
            # Strip prompt prefix → pure response tokens
            response = out[:, prompt_len:]  # (B, T_i)
            all_responses.append(response)

        # Pad all rollout responses to a common length T.
        max_resp_len = max(r.shape[1] for r in all_responses)

        def _pad(t: torch.Tensor) -> torch.Tensor:
            pad = max_resp_len - t.shape[1]
            if pad > 0:
                t = torch.cat(
                    [t, torch.full((t.shape[0], pad), pad_id, device=t.device)], dim=1
                )
            return t

        # Interleave back to (B*G, T): row order [prompt_0_roll_0, prompt_0_roll_1, ...]
        # i.e. group 0 → rows 0..G-1, group 1 → rows G..2G-1, ...
        padded = [_pad(r) for r in all_responses]  # G tensors of (B, T)
        # Stack to (G, B, T) then transpose to (B, G, T) then reshape to (B*G, T)
        response_ids = (
            torch.stack(padded, dim=0)  # (G, B, T)
            .permute(1, 0, 2)  # (B, G, T)
            .reshape(B * G, max_resp_len)  # (B*G, T)
        )

        response_mask = _build_response_mask(response_ids, pad_id)  # (B*G, T)

        # ── 3. Score responses (demo: reward_function) ────────────────────
        decoded = self.tokenizer.batch_decode(response_ids, skip_special_tokens=True)

        # Align ground truths: ground_truths[i] applies to rows [i*G : (i+1)*G]
        rewards_list: list[float] = []
        for flat_idx, response_str in enumerate(decoded):
            prompt_idx = flat_idx // G
            gt = ground_truths[prompt_idx]
            if self.reward_fn is not None:
                from src.agents.base import AgentLoopOutput

                pseudo_output = AgentLoopOutput(
                    prompt_ids=[],
                    response_ids=[],
                    response_mask=[],
                    num_turns=1,
                    metrics={},
                    final_answer=response_str,
                )
                components = self.reward_fn.reward_components(
                    pseudo_output,
                    ground_truth=gt,
                    judge_fn=self.judge_fn,
                )
                rewards_list.append(float(components["total"]))
            else:
                rewards_list.append(float(self.judge_fn(response_str, gt)))

        rewards = torch.tensor(rewards_list, dtype=torch.float32, device=self.device)

        # ── 4. Group advantage (demo: compute_group_advantages) ───────────
        # Group IDs: rollouts from prompt i all share group id i.
        group_ids = torch.arange(B, device=self.device).repeat_interleave(G)

        # Sparse outcome reward: scalar reward at EOS token position.
        token_rewards = _sparse_reward_at_eos(rewards, response_mask.cpu()).to(
            self.device
        )

        advantages, _ = compute_grpo_outcome_advantage(
            token_rewards,
            response_mask,
            group_ids,
            epsilon=cfg.advantage_epsilon,
            clip_advantages=cfg.advantage_clip,
        )  # (B*G, T)

        # ── 5. Old log-probs under the rollout policy (frozen snapshot) ───
        full_ids = torch.cat([tiled_prompt, response_ids], dim=1)  # (B*G, P+T)
        old_log_probs = get_response_log_probs(
            self.policy, full_ids, prompt_len, response_mask
        )  # (B*G, T)

        return LLMRolloutResult(
            prompt_ids=prompt_ids,
            response_ids=response_ids,
            response_mask=response_mask,
            old_log_probs=old_log_probs,
            rewards=rewards,
            advantages=advantages,
            group_ids=group_ids,
        )

    # ------------------------------------------------------------------
    # Loss  (demo: GRPOTrainer.compute_loss)
    # ------------------------------------------------------------------

    def compute_loss(
        self,
        rollout: LLMRolloutResult,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Compute GRPO clip loss + KL penalty from a rollout result.

        Analogous to ``GRPOTrainer.compute_loss`` but operating on the
        token-level log-prob tensors in *rollout*.

        Steps:
            new_log_probs  = log π_θ(token | context)             current policy
            ref_log_probs  = log π_ref(token | context)           frozen SFT
            pg_loss        = clipped surrogate (old vs new ratio × advantage)
            kl             = low_var_kl(new ‖ ref) averaged over response tokens
            total          = pg_loss + β · kl
        """
        cfg = self.config
        tiled_prompt = rollout.prompt_ids.repeat_interleave(cfg.num_rollouts, dim=0)
        prompt_len = tiled_prompt.shape[1]
        full_ids = torch.cat([tiled_prompt, rollout.response_ids], dim=1)
        mask = rollout.response_mask

        # Current policy: new_log_probs
        new_log_probs = get_response_log_probs(self.policy, full_ids, prompt_len, mask)

        # Reference policy: ref_log_probs (no grad)
        with torch.no_grad():
            ref_log_probs = get_response_log_probs(
                self.reference, full_ids, prompt_len, mask
            )

        # PPO-clipped policy loss (demo: grpo_clipped_policy_loss)
        pg_loss, clip_frac, approx_kl, _ = compute_ppo_policy_loss_core(
            old_log_prob=rollout.old_log_probs,
            log_prob=new_log_probs,
            advantages=rollout.advantages,
            eos_mask=mask,
            cliprange=cfg.clip_epsilon,
        )

        # KL penalty vs SFT reference (demo: reverse_kl_penalty → here: low_var_kl)
        kl_per_token = kl_penalty(new_log_probs, ref_log_probs, cfg.kl_penalty_type)
        mean_kl = masked_mean(kl_per_token, mask)
        total_loss = pg_loss + cfg.kl_beta * mean_kl

        with torch.no_grad():
            normalizer = mask.float().sum().clamp_min(1e-8)
            metrics: dict[str, float] = {
                "loss": total_loss.item(),
                "pg_loss": pg_loss.item(),
                "mean_kl": mean_kl.item(),
                "mean_reward": (rollout.rewards.sum() / len(rollout.rewards)).item(),
                "clip_fraction": clip_frac.item(),
                "approx_kl": approx_kl.item(),
                "mean_advantage": (
                    (rollout.advantages * mask.float()).sum() / normalizer
                ).item(),
            }
        return total_loss, metrics

    # ------------------------------------------------------------------
    # Full update step  (demo: GRPOTrainer.update)
    # ------------------------------------------------------------------

    def step(
        self,
        prompts: list[str],
        ground_truths: list[str],
    ) -> dict[str, float]:
        """One complete online GRPO step: rollout → loss → gradient update.

        Analogous to ``GRPOTrainer.update`` in grpo_trainer.py.

        Args:
            prompts: B raw text prompts.
            ground_truths: B gold answers.

        Returns:
            Metrics dict with loss, reward, KL, clip_fraction, etc.
        """
        rollout = self.rollout(prompts, ground_truths)

        self.policy.train()
        loss, metrics = self.compute_loss(rollout)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.config.grad_clip)
        self.optimizer.step()

        return metrics

    def save_checkpoint(self, path: str) -> None:
        """Persist policy weights, tokenizer, and optimizer state for resume.

        Used by ``train_loop`` for periodic checkpointing of long runs. The
        reference policy is frozen and reconstructable, so it is not saved.
        """
        import os

        os.makedirs(path, exist_ok=True)
        self.policy.save_pretrained(path)
        self.tokenizer.save_pretrained(path)
        torch.save(self.optimizer.state_dict(), os.path.join(path, "optimizer.pt"))

    def load_checkpoint(self, path: str) -> None:
        """Restore policy weights and optimizer state saved by save_checkpoint."""
        import os

        from transformers import AutoModelForCausalLM

        loaded = AutoModelForCausalLM.from_pretrained(path)
        self.policy.load_state_dict(loaded.state_dict())
        self.policy.to(self.device)
        optimizer_path = os.path.join(path, "optimizer.pt")
        if os.path.exists(optimizer_path):
            self.optimizer.load_state_dict(torch.load(optimizer_path))
