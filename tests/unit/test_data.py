"""Tests for prompt-only rollout datasets and batch collation."""

from __future__ import annotations

from src import (
    PromptOnlyDataset,
    build_prompt_dataloader,
    build_prompt_messages,
    normalize_prompt_training_example,
    prompt_batch_to_search_batch,
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

    assert sample.question == "Who won the Nobel Prize in Physics in 2024?"
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
    assert batch.questions == [
        "Short question?",
        "A much longer question that should create a longer prompt?",
    ]
    assert batch.ground_truths == ["A", "B"]
    assert batch.tools == [[], ["search"]]
    assert batch.metadata == [{"id": "q1"}, {"id": "q2"}]
    assert batch.attention_mask[0].sum().item() == batch.prompt_lengths[0]
    assert batch.attention_mask[1].sum().item() == batch.prompt_lengths[1]


def test_collate_prompt_batch_left_pads_shorter_sequences():
    """Padding must be on the LEFT so every prompt ends at the final position.

    The generation loop slices with [:, -max_start_length:] which only works
    correctly when real tokens are right-aligned (i.e. left-padded).
    """
    loader = build_prompt_dataloader(
        [
            {"question": "Hi?", "ground_truth": "Hello.", "tools": []},
            {
                "question": "A much longer question to force padding?",
                "ground_truth": "B.",
            },
        ],
        tokenizer=_TokenizerWithEncode(),
        batch_size=2,
        shuffle=False,
        pad_token_id=0,
    )
    batch = next(iter(loader))
    short_len = batch.prompt_lengths[0]
    long_len = batch.prompt_lengths[1]
    padding = long_len - short_len

    # Leading positions of the shorter sequence must all be pad
    assert batch.input_ids[0, :padding].tolist() == [0] * padding
    assert batch.attention_mask[0, :padding].tolist() == [0] * padding
    # Trailing positions of the shorter sequence must be real tokens
    assert batch.attention_mask[0, padding:].tolist() == [1] * short_len
    # The longer sequence has no leading padding
    assert batch.attention_mask[1, 0].item() == 1


def test_build_prompt_dataloader_drop_last_omits_incomplete_final_batch():
    examples = [{"question": f"Q{i}?", "ground_truth": f"A{i}."} for i in range(3)]
    loader_keep = build_prompt_dataloader(
        examples, tokenizer=_TokenizerWithEncode(), batch_size=2, drop_last=False
    )
    loader_drop = build_prompt_dataloader(
        examples, tokenizer=_TokenizerWithEncode(), batch_size=2, drop_last=True
    )
    assert sum(1 for _ in loader_keep) == 2  # batches: [2, 1]
    assert sum(1 for _ in loader_drop) == 1  # batches: [2]


def test_prompt_batch_to_search_batch_builds_rollout_ready_gen_batch():
    loader = build_prompt_dataloader(
        [
            {
                "question": "Who won the Nobel Prize in Physics in 2024?",
                "ground_truth": "John Hopfield and Geoffrey Hinton.",
                "tools": ["search"],
                "id": "physics-2024",
            }
        ],
        tokenizer=_TokenizerWithEncode(),
        batch_size=1,
        shuffle=False,
        pad_token_id=0,
    )

    prompt_batch = next(iter(loader))
    gen_batch = prompt_batch_to_search_batch(prompt_batch)

    assert tuple(gen_batch.batch["input_ids"].shape) == (
        1,
        prompt_batch.input_ids.shape[1],
    )
    assert tuple(gen_batch.batch["attention_mask"].shape) == tuple(
        prompt_batch.attention_mask.shape
    )
    assert tuple(gen_batch.batch["position_ids"].shape) == tuple(
        prompt_batch.input_ids.shape
    )
    assert gen_batch.non_tensor_batch["question"] == [
        "Who won the Nobel Prize in Physics in 2024?"
    ]
    assert gen_batch.non_tensor_batch["golden_answers"] == [
        ["John Hopfield and Geoffrey Hinton."]
    ]
    assert gen_batch.non_tensor_batch["ground_truth"] == [
        "John Hopfield and Geoffrey Hinton."
    ]
    assert gen_batch.non_tensor_batch["tools"] == [["search"]]
    assert gen_batch.non_tensor_batch["metadata"] == [{"id": "physics-2024"}]
    assert gen_batch.meta_info["prompt_lengths"] == prompt_batch.prompt_lengths
