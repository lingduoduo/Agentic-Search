"""Prompt-only training data helpers for tool-use / agentic RL workflows."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Sequence

import torch
from torch.utils.data import DataLoader, Dataset

DEFAULT_TOOL_SYSTEM_PROMPT = (
    "You are a tool-using assistant. "
    "Use tools when they help, and answer directly when the question is already solved."
)


@dataclass(frozen=True)
class PromptTrainingExample:
    """Raw training example for prompt-only rollout generation."""

    question: str
    ground_truth: str
    tools: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class PromptSample:
    """One tokenized prompt example ready for rollout generation."""

    question: str
    messages: list[dict[str, Any]]
    prompt_ids: list[int]
    ground_truth: str
    tools: list[str]
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class PromptBatch:
    """A padded batch of prompt-only inputs for RL/SFT rollout generation."""

    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    prompt_lengths: list[int]
    questions: list[str]
    messages: list[list[dict[str, Any]]]
    ground_truths: list[str]
    tools: list[list[str]]
    metadata: list[dict[str, Any]]


def normalize_prompt_training_example(
    example: PromptTrainingExample | Mapping[str, Any],
) -> PromptTrainingExample:
    """Convert a dict-like record into a validated PromptTrainingExample."""

    if isinstance(example, PromptTrainingExample):
        return example

    question = str(example.get("question", "")).strip()
    ground_truth = str(example.get("ground_truth", "")).strip()
    tools = [str(tool) for tool in example.get("tools", [])]
    metadata = {
        key: value
        for key, value in dict(example).items()
        if key not in {"question", "ground_truth", "tools"}
    }

    if not question:
        raise ValueError("Training example is missing a non-empty `question`.")
    if not ground_truth:
        raise ValueError("Training example is missing a non-empty `ground_truth`.")

    return PromptTrainingExample(
        question=question,
        ground_truth=ground_truth,
        tools=tools,
        metadata=metadata,
    )


def build_prompt_messages(
    question: str,
    *,
    tools: Sequence[str] | None = None,
    system_prompt: str | None = DEFAULT_TOOL_SYSTEM_PROMPT,
) -> list[dict[str, str]]:
    """Build a rollout prompt from question + available tool names.

    The dataset intentionally contains only the prompt-side context, not the
    intermediate reasoning or action trace.
    """

    messages: list[dict[str, str]] = []
    resolved_tools = [tool for tool in (tools or []) if tool]
    if system_prompt:
        content = system_prompt.strip()
        if resolved_tools:
            content += "\n\nAvailable tools: " + ", ".join(resolved_tools)
        messages.append({"role": "system", "content": content})
    messages.append({"role": "user", "content": question})
    return messages


def build_prompt_ids_from_messages(
    tokenizer: Any,
    messages: list[dict[str, Any]],
    *,
    prompt_length: int = 4096,
) -> list[int]:
    """Tokenize a chat prompt using the same fallback logic as AgentLoopBase."""

    chat_template = getattr(tokenizer, "chat_template", "__missing__")
    if hasattr(tokenizer, "apply_chat_template") and chat_template is not None:
        prompt_ids = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
        )
        return list(prompt_ids)[-prompt_length:]

    joined = "\n".join(message.get("content", "") for message in messages)
    if hasattr(tokenizer, "encode"):
        return list(tokenizer.encode(joined))[-prompt_length:]
    raise TypeError("tokenizer must implement apply_chat_template(...) or encode(...).")


class PromptOnlyDataset(Dataset[PromptSample]):
    """Dataset that maps raw examples to tokenized prompt-only rollout inputs."""

    def __init__(
        self,
        examples: Sequence[PromptTrainingExample | Mapping[str, Any]],
        *,
        tokenizer: Any,
        prompt_length: int = 4096,
        prompt_builder: Callable[[PromptTrainingExample], list[dict[str, Any]]]
        | None = None,
    ) -> None:
        self.examples = [
            normalize_prompt_training_example(example) for example in examples
        ]
        self.tokenizer = tokenizer
        self.prompt_length = prompt_length
        self.prompt_builder = prompt_builder or (
            lambda example: build_prompt_messages(
                example.question,
                tools=example.tools,
            )
        )

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> PromptSample:
        example = self.examples[index]
        messages = list(self.prompt_builder(example))
        prompt_ids = build_prompt_ids_from_messages(
            self.tokenizer,
            messages,
            prompt_length=self.prompt_length,
        )
        return PromptSample(
            question=example.question,
            messages=messages,
            prompt_ids=prompt_ids,
            ground_truth=example.ground_truth,
            tools=list(example.tools),
            metadata=dict(example.metadata),
        )


def collate_prompt_batch(
    samples: Sequence[PromptSample],
    *,
    pad_token_id: int = 0,
) -> PromptBatch:
    """Pad tokenized prompt samples into a rollout batch."""

    if not samples:
        raise ValueError("Cannot collate an empty prompt batch.")

    max_length = max(len(sample.prompt_ids) for sample in samples)
    input_ids: list[list[int]] = []
    attention_masks: list[list[int]] = []
    prompt_lengths: list[int] = []

    for sample in samples:
        length = len(sample.prompt_ids)
        prompt_lengths.append(length)
        padding = max_length - length
        # Left-pad so every prompt ends at position [-1], aligning generation
        # starts without requiring per-sample index arithmetic.
        input_ids.append([pad_token_id] * padding + list(sample.prompt_ids))
        attention_masks.append([0] * padding + [1] * length)

    return PromptBatch(
        input_ids=torch.tensor(input_ids, dtype=torch.long),
        attention_mask=torch.tensor(attention_masks, dtype=torch.long),
        prompt_lengths=prompt_lengths,
        questions=[sample.question for sample in samples],
        messages=[list(sample.messages) for sample in samples],
        ground_truths=[sample.ground_truth for sample in samples],
        tools=[list(sample.tools) for sample in samples],
        metadata=[dict(sample.metadata) for sample in samples],
    )


def prompt_batch_to_search_batch(
    batch: PromptBatch,
) -> Any:
    """Convert a PromptBatch into a rollout-ready SearchBatch.

    The result matches the `gen_batch` shape expected by
    `LLMGenerationManager.run_llm_loop(...)`: prompt token tensors live in
    `.batch`, while prompt-only supervision fields stay in `.non_tensor_batch`.
    Intermediate reasoning or tool traces are intentionally absent.
    """

    from src.agent_loop import SearchBatch

    search_batch = SearchBatch.from_dict(
        {
            "input_ids": batch.input_ids,
            "attention_mask": batch.attention_mask,
            "position_ids": torch.arange(
                batch.input_ids.shape[1],
                dtype=torch.long,
            )
            .unsqueeze(0)
            .expand_as(batch.input_ids),
        }
    )
    search_batch.non_tensor_batch = {
        "question": list(batch.questions),
        "golden_answers": [[ground_truth] for ground_truth in batch.ground_truths],
        "ground_truth": list(batch.ground_truths),
        "messages": [list(messages) for messages in batch.messages],
        "tools": [list(tools) for tools in batch.tools],
        "metadata": [dict(item) for item in batch.metadata],
    }
    search_batch.meta_info["prompt_lengths"] = list(batch.prompt_lengths)
    return search_batch


def build_prompt_dataloader(
    examples: Sequence[PromptTrainingExample | Mapping[str, Any]],
    *,
    tokenizer: Any,
    batch_size: int,
    shuffle: bool = False,
    prompt_length: int = 4096,
    pad_token_id: int = 0,
    num_workers: int = 0,
    drop_last: bool = False,
    prompt_builder: Callable[[PromptTrainingExample], list[dict[str, Any]]]
    | None = None,
) -> DataLoader:
    """Create a DataLoader that yields left-padded prompt-only rollout batches."""

    dataset = PromptOnlyDataset(
        examples,
        tokenizer=tokenizer,
        prompt_length=prompt_length,
        prompt_builder=prompt_builder,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        drop_last=drop_last,
        collate_fn=lambda batch: collate_prompt_batch(batch, pad_token_id=pad_token_id),
    )
