"""Shared causal-LM response-token log-probability arithmetic."""

from __future__ import annotations

import torch
import torch.nn as nn


def get_response_log_probs(
    model: nn.Module,
    input_ids: torch.Tensor,
    prompt_len: int,
    response_mask: torch.Tensor,
) -> torch.Tensor:
    """Return masked log probabilities for a causal-LM response sequence.

    Logits at position ``t`` predict token ``t + 1``. The slice beginning at
    ``prompt_len - 1`` therefore aligns each response token with its predictor.
    """
    logits = model(input_ids=input_ids).logits
    response_logits = logits[:, prompt_len - 1 : -1]
    response_ids = input_ids[:, prompt_len:]
    log_probs = torch.log_softmax(response_logits.float(), dim=-1)
    selected = log_probs.gather(-1, response_ids.unsqueeze(-1)).squeeze(-1)
    return selected * response_mask.to(dtype=selected.dtype)
