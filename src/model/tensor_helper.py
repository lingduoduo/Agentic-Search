"""Tensor helpers shared by the LLM agent generation loop."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class TensorConfig:
    pad_token_id: int
    max_prompt_length: int
    max_obs_length: int
    max_start_length: int


class TensorHelper:
    def __init__(self, config: TensorConfig):
        self.config = config

    def cut_to_effective_len(
        self,
        tensor_dict: dict[str, torch.Tensor],
        keys: list[str],
        cut_left: bool = True,
    ) -> dict[str, torch.Tensor]:
        """Trim tensors to the maximum non-padding length in the batch."""

        effective_len = int(tensor_dict["attention_mask"].sum(dim=1).max().item())
        result = tensor_dict.copy()
        for key in keys:
            result[key] = (
                tensor_dict[key][:, -effective_len:]
                if cut_left
                else tensor_dict[key][:, :effective_len]
            )
        return result

    def convert_pad_structure(
        self,
        tensor: torch.Tensor,
        pad_to_left: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Move padding to the desired side while preserving token order."""

        if pad_to_left:
            sorted_indices = (
                (tensor != self.config.pad_token_id)
                .to(torch.int64)
                .argsort(
                    dim=1,
                    stable=True,
                )
            )
        else:
            sorted_indices = (
                (tensor == self.config.pad_token_id)
                .to(torch.int64)
                .argsort(
                    dim=1,
                    stable=True,
                )
            )
        return tensor.gather(1, sorted_indices), sorted_indices

    def create_attention_mask(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Create an attention mask from input ids."""

        return (input_ids != self.config.pad_token_id).to(dtype=torch.long)

    def create_position_ids(self, attention_mask: torch.Tensor) -> torch.Tensor:
        """Create incremental position ids for non-padding tokens."""

        return (torch.cumsum(attention_mask, dim=1) - 1) * attention_mask

    def concatenate_with_padding(
        self,
        tensors: list[torch.Tensor],
        pad_to_left: bool = True,
    ) -> torch.Tensor:
        """Concatenate tensors then normalize the padding layout."""

        concatenated = torch.cat(tensors, dim=1)
        padded_tensor, _ = self.convert_pad_structure(
            concatenated, pad_to_left=pad_to_left
        )
        return padded_tensor

    def example_level_pad(
        self,
        responses: torch.Tensor,
        responses_str: list[str],
        active_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, list[str]]:
        """Re-expand active-only outputs back to the original batch size."""

        if int(active_mask.sum().item()) != responses.shape[0]:
            raise ValueError("active_mask must match the number of active responses.")

        batch_size = int(active_mask.shape[0])
        seq_len = int(responses.shape[1])
        padded_responses = torch.full(
            (batch_size, seq_len),
            self.config.pad_token_id,
            dtype=responses.dtype,
            device=responses.device,
        )
        padded_responses[active_mask] = responses

        padded_responses_str = [""] * batch_size
        active_index = 0
        for batch_index, is_active in enumerate(active_mask.tolist()):
            if is_active:
                padded_responses_str[batch_index] = responses_str[active_index]
                active_index += 1

        return padded_responses, padded_responses_str
