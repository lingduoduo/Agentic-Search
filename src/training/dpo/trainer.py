"""Direct Preference Optimization: the loss, and a trainer around it.

DPO trains a policy directly on preference pairs. There is no reward model, no
critic, and no online sampling -- the reference policy plays the role the KL
penalty plays in GRPO, anchoring the policy to where it started.

The loss for one pair, with ``y_w`` the chosen response and ``y_l`` the rejected
one::

    L = -log sigmoid( beta * [ (log pi(y_w|x) - log pi_ref(y_w|x))
                             - (log pi(y_l|x) - log pi_ref(y_l|x)) ] )

``log pi(y|x)`` is the **sum** of per-token log probabilities over response
tokens only. Summed, not averaged: length-normalizing turns this into SimPO,
which is a different method with different behaviour, not a refinement.

Per-token log probabilities come from ``get_response_log_probs`` in
``src.training.rl.llm_grpo_trainer`` rather than a second implementation here.
That function owns the ``logits[:, prompt_len - 1 : -1]`` shift aligning logits
with the tokens they predict; an off-by-one there is silent and ruins training,
so it must exist in exactly one place.
"""

from __future__ import annotations

import copy
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..rl.llm_grpo_trainer import get_response_log_probs
from .data import PreferenceExample


@dataclass(frozen=True)
class DPOConfig:
    """Hyperparameters for a DPO run.

    ``beta`` controls how far the policy may drift from the reference: it scales
    the implicit reward, so a small beta permits large drift and a large beta
    keeps the policy close. 0.1 is the value the DPO paper uses.
    """

    beta: float = 0.1
    epochs: int = 1
    batch_size: int = 4
    max_length: int = 2048
    grad_clip: float = 1.0


@dataclass(frozen=True)
class DPOStep:
    """What one optimizer step did.

    ``loss`` alone cannot distinguish learning from collapse. ``margin`` is the
    implicit reward gap (positive means the policy prefers the chosen response
    more than the reference does) and ``accuracy`` is the fraction of pairs with
    a positive gap -- those two are what show whether training is working. Both
    start at 0 by construction, since the policy starts at the reference.
    """

    loss: float
    margin: float
    accuracy: float


def dpo_loss(
    policy_chosen: torch.Tensor,
    policy_rejected: torch.Tensor,
    ref_chosen: torch.Tensor,
    ref_rejected: torch.Tensor,
    beta: float,
) -> tuple[torch.Tensor, torch.Tensor, float]:
    """Return ``(loss, mean_margin, accuracy)`` for a batch of preference pairs.

    Every argument is a ``(B,)`` tensor of **summed** sequence log probabilities.
    Keeping this a pure function of four scalars per pair -- rather than folding
    it into the trainer -- is what makes the formula testable against known
    values with no model involved.
    """
    policy_delta = policy_chosen - policy_rejected
    ref_delta = ref_chosen - ref_rejected
    margin = policy_delta - ref_delta

    # logsigmoid rather than log(sigmoid(x)): it is numerically stable for large
    # |x|, where sigmoid saturates to exactly 0 or 1 and log would return -inf.
    loss = -F.logsigmoid(beta * margin).mean()

    # Implicit-reward accuracy: the fraction of pairs whose IMPLICIT REWARD
    # gap favours the chosen response, i.e. margin > 0. This is the standard
    # DPO metric, and it is the right one because it measures what DPO
    # optimizes. Using raw policy likelihood (policy_delta > 0) instead would
    # report the base model's pre-existing preference at init -- when the
    # policy still equals the reference and nothing has been learned at all.
    accuracy = (margin > 0).float().mean().item()
    return loss, margin.mean(), accuracy


class DPOTrainer:
    """Trains a causal-LM policy on preference pairs against a frozen reference.

    Shaped like ``SFTTrainer``: the optimizer is injected rather than
    constructed, the device defaults to CPU, and nothing here loads a model.
    """

    def __init__(
        self,
        policy: nn.Module,
        tokenizer: Any,
        optimizer: torch.optim.Optimizer,
        config: DPOConfig | None = None,
        device: str | torch.device = "cpu",
        reference_policy: nn.Module | None = None,
    ) -> None:
        self.policy = policy
        self.tokenizer = tokenizer
        self.optimizer = optimizer
        self.config = config or DPOConfig()
        self.device = torch.device(device)

        # Default reference is a frozen snapshot of the policy at init, matching
        # LLMGRPOTrainer. That is correct when the policy was initialized from
        # the SFT checkpoint, which is the usual DPO setup; a caller pointing at
        # a different checkpoint injects it instead.
        self.reference = (
            reference_policy if reference_policy is not None else copy.deepcopy(policy)
        )
        self.reference.to(self.device)
        self.reference.eval()
        for parameter in self.reference.parameters():
            parameter.requires_grad_(False)

    def _encode(self, prompt: str, response: str) -> tuple[torch.Tensor, int]:
        """Tokenize ``prompt + response``; return ids and the prompt's length.

        Assumes the prompt's ids are a prefix of the joined sequence -- the same
        assumption ``SFTTrainer``'s non-chat-template path makes.
        """
        prompt_enc = self.tokenizer(prompt, return_tensors="pt")
        full_enc = self.tokenizer(
            prompt + response,
            return_tensors="pt",
            max_length=self.config.max_length,
            truncation=True,
        )
        input_ids = full_enc.input_ids.to(self.device)
        # Guard the case where the prompt alone exceeds max_length: without this
        # the response slice would be empty and the sum silently zero.
        prompt_len = min(prompt_enc.input_ids.shape[-1], input_ids.shape[-1] - 1)
        return input_ids, max(prompt_len, 1)

    def sequence_log_prob(
        self, prompt: str, response: str, *, model: nn.Module | None = None
    ) -> torch.Tensor:
        """Summed log prob of ``response`` given ``prompt``, as a 0-d tensor.

        Prompt tokens are excluded: they are conditioning, not targets.
        """
        target = self.policy if model is None else model
        input_ids, prompt_len = self._encode(prompt, response)
        response_mask = torch.ones(
            (1, input_ids.shape[-1] - prompt_len),
            dtype=torch.long,
            device=self.device,
        )
        token_log_probs = get_response_log_probs(
            target, input_ids, prompt_len, response_mask
        )
        return token_log_probs.sum()

    def _batch_log_probs(
        self, batch: list[PreferenceExample], model: nn.Module
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Summed chosen/rejected log probs for a batch, as ``(B,)`` tensors."""
        chosen = torch.stack(
            [self.sequence_log_prob(p.prompt, p.chosen, model=model) for p in batch]
        )
        rejected = torch.stack(
            [self.sequence_log_prob(p.prompt, p.rejected, model=model) for p in batch]
        )
        return chosen, rejected

    def train(self, examples: list[PreferenceExample]) -> list[DPOStep]:
        """Train for ``config.epochs``. Returns one record per optimizer step."""
        if self.config.epochs == 0 or not examples:
            return []

        self.policy.train()
        history: list[DPOStep] = []

        for _epoch in range(self.config.epochs):
            epoch_examples = list(examples)
            random.shuffle(epoch_examples)
            for start in range(0, len(epoch_examples), self.config.batch_size):
                batch = epoch_examples[start : start + self.config.batch_size]
                self.optimizer.zero_grad()

                policy_chosen, policy_rejected = self._batch_log_probs(
                    batch, self.policy
                )
                with torch.no_grad():
                    ref_chosen, ref_rejected = self._batch_log_probs(
                        batch, self.reference
                    )

                loss, margin, accuracy = dpo_loss(
                    policy_chosen,
                    policy_rejected,
                    ref_chosen,
                    ref_rejected,
                    self.config.beta,
                )
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.policy.parameters(), self.config.grad_clip
                )
                self.optimizer.step()
                history.append(
                    DPOStep(
                        loss=loss.item(),
                        margin=margin.item(),
                        accuracy=accuracy,
                    )
                )
        return history

    def save(self, output_dir: str | Path) -> None:
        """Save the policy and tokenizer in HuggingFace format.

        The reference is deliberately not saved: it is a frozen copy of a
        checkpoint that already exists on disk.
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        self.policy.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)
