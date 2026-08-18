"""Helpers for turning search-agent trajectories into supervised examples."""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from ...agents.core.base import AgentLoopOutput


@dataclass(frozen=True)
class SFTExample:
    """A supervised training example derived from one search-agent rollout."""

    prompt_messages: list[dict[str, Any]]
    completion: str
    trajectory_messages: list[dict[str, Any]]


def build_search_sft_example(
    input_messages: list[dict[str, Any]],
    output: AgentLoopOutput,
    *,
    include_environment_messages: bool = False,
) -> SFTExample:
    """Create an SFT example from a search-agent rollout.

    By default, the completion is the full assistant action trace, e.g.
    ``<plan>...<searches>...<fetch>...<answer>...`` joined across turns.

    When ``include_environment_messages`` is true, the returned
    ``trajectory_messages`` includes the full multi-turn conversation after the
    original prompt so a downstream trainer can reconstruct the whole dialogue.
    """
    completion = output.action_trace or output.final_answer or ""
    if not completion:
        raise ValueError("Search rollout does not contain an assistant action trace.")

    if include_environment_messages:
        trajectory_messages = list(output.trajectory_messages)
    else:
        trajectory_messages = [
            {"role": "assistant", "content": completion},
        ]

    return SFTExample(
        prompt_messages=list(input_messages),
        completion=completion,
        trajectory_messages=trajectory_messages,
    )


@dataclass(frozen=True)
class SFTConfig:
    epochs: int = 3
    lr: float = 2e-5
    batch_size: int = 4
    max_length: int = 2048
    grad_clip: float = 1.0


class SFTTrainer:
    """Supervised fine-tuning trainer for search-agent trajectories.

    Applies cross-entropy loss on assistant tokens only. Prompt tokens
    (system + user + tool-result) are masked to -100 so they do not
    contribute to the loss.
    """

    def __init__(
        self,
        policy: nn.Module,
        tokenizer: Any,
        optimizer: torch.optim.Optimizer,
        config: SFTConfig | None = None,
        device: str | torch.device = "cpu",
    ) -> None:
        self.policy = policy
        self.tokenizer = tokenizer
        self.optimizer = optimizer
        self.config = config or SFTConfig()
        self.device = torch.device(device)

    def _tokenize_example(self, example: SFTExample) -> dict[str, torch.Tensor]:
        """Return input_ids, attention_mask, and labels for one example."""
        cfg = self.config

        full_messages = list(example.prompt_messages) + [
            {"role": "assistant", "content": example.completion}
        ]

        # Tokenize prompt only to find where assistant tokens begin.
        if callable(getattr(self.tokenizer, "apply_chat_template", None)):
            prompt_enc = self.tokenizer.apply_chat_template(
                example.prompt_messages,
                tokenize=True,
                return_tensors="pt",
                add_generation_prompt=True,
            )
            prompt_len = prompt_enc["input_ids"].shape[-1]
            full_enc = self.tokenizer.apply_chat_template(
                full_messages,
                tokenize=True,
                return_tensors="pt",
                max_length=cfg.max_length,
                truncation=True,
            )
            input_ids = full_enc["input_ids"]
            attention_mask = full_enc["attention_mask"]
        else:
            # Fallback: join content strings, track prompt length by token count.
            prompt_text = "\n".join(m["content"] for m in example.prompt_messages)
            full_text = prompt_text + "\n" + example.completion
            prompt_enc = self.tokenizer(prompt_text, return_tensors="pt")
            full_enc = self.tokenizer(
                full_text,
                return_tensors="pt",
                max_length=cfg.max_length,
                truncation=True,
            )
            prompt_len = prompt_enc.input_ids.shape[1]
            input_ids = full_enc.input_ids
            attention_mask = full_enc.attention_mask

        labels = input_ids.clone()
        prompt_len = min(
            prompt_len, input_ids.shape[-1]
        )  # guard when prompt exceeds max_length
        labels[:, :prompt_len] = -100  # mask prompt tokens
        return {
            "input_ids": input_ids.to(self.device),
            "attention_mask": attention_mask.to(self.device),
            "labels": labels.to(self.device),
        }

    def train(self, examples: list[SFTExample]) -> list[float]:
        """Train for config.epochs on examples. Returns per-step loss history."""
        if self.config.epochs == 0 or not examples:
            return []

        self.policy.train()
        history: list[float] = []

        for _epoch in range(self.config.epochs):
            epoch_examples = list(examples)
            random.shuffle(epoch_examples)
            for i in range(0, len(epoch_examples), self.config.batch_size):
                batch = epoch_examples[i : i + self.config.batch_size]
                self.optimizer.zero_grad()
                batch_loss = torch.tensor(0.0, device=self.device)
                for example in batch:
                    enc = self._tokenize_example(example)
                    output = self.policy(**enc)
                    batch_loss = batch_loss + output.loss
                batch_loss = batch_loss / len(batch)
                batch_loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.policy.parameters(), self.config.grad_clip
                )
                self.optimizer.step()
                history.append(batch_loss.item())
        return history

    def save(self, output_dir: str | Path) -> None:
        """Save policy and tokenizer in HuggingFace format."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        self.policy.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)
