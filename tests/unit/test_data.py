"""Tests for prompt-only rollout datasets and batch collation."""

from __future__ import annotations

from src.agent_loop import (
    PromptOnlyDataset,
    build_prompt_dataloader,
    build_prompt_messages,
    normalize_prompt_training_example,
)


class _TokenizerWithEncode:
    chat_template = None

    def encode(self, text: str) -> list[int]:
        return [ord(char) for char in text]


def test_normalize_prompt_training_example_extracts_known_fields():
    example = normalize_prompt_training_example(
        {
            "question": "Who won the Nobel Prize in Physics in 2024?",
            "ground_truth": "John Hopfield and Geoffrey Hinton.",
            "tools": ["search"],
            "source": "demo",
        }
    )

    assert example.question == "Who won the Nobel Prize in Physics in 2024?"
    assert example.ground_truth == "John Hopfield and Geoffrey Hinton."
    assert example.tools == ["search"]
    assert example.metadata == {"source": "demo"}


def test_build_prompt_messages_keeps_prompt_only_and_lists_tools():
    messages = build_prompt_messages(
        "Who won the Nobel Prize in Physics in 2024?",
        tools=["search"],
    )

    assert messages[0]["role"] == "system"
    assert "Available tools: search" in messages[0]["content"]
    assert messages[1] == {
        "role": "user",
        "content": "Who won the Nobel Prize in Physics in 2024?",
    }


def test_prompt_only_dataset_returns_tokenized_prompt_sample():
    dataset = PromptOnlyDataset(
        [
            {
                "question": "Who won the Nobel Prize in Physics in 2024?",
                "ground_truth": "John Hopfield and Geoffrey Hinton.",
                "tools": ["search"],
            }
        ],
        tokenizer=_TokenizerWithEncode(),
        prompt_length=1024,
    )

    sample = dataset[0]

    assert sample.ground_truth == "John Hopfield and Geoffrey Hinton."
    assert sample.tools == ["search"]
    assert (
        sample.messages[-1]["content"] == "Who won the Nobel Prize in Physics in 2024?"
    )
    assert len(sample.prompt_ids) > 0


def test_build_prompt_dataloader_pads_prompt_ids_and_preserves_metadata():
    loader = build_prompt_dataloader(
        [
            {
                "question": "Short question?",
                "ground_truth": "A",
                "tools": [],
                "id": "q1",
            },
            {
                "question": "A much longer question that should create a longer prompt?",
                "ground_truth": "B",
                "tools": ["search"],
                "id": "q2",
            },
        ],
        tokenizer=_TokenizerWithEncode(),
        batch_size=2,
        shuffle=False,
        pad_token_id=0,
    )

    batch = next(iter(loader))

    assert tuple(batch.input_ids.shape) == tuple(batch.attention_mask.shape)
    assert len(batch.prompt_lengths) == 2
    assert batch.prompt_lengths[1] >= batch.prompt_lengths[0]
    assert batch.ground_truths == ["A", "B"]
    assert batch.tools == [[], ["search"]]
    assert batch.metadata == [{"id": "q1"}, {"id": "q2"}]
    assert batch.attention_mask[0].sum().item() == batch.prompt_lengths[0]
    assert batch.attention_mask[1].sum().item() == batch.prompt_lengths[1]
