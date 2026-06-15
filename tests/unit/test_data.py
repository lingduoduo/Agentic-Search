"""Tests for prompt-only rollout datasets and batch collation."""

from __future__ import annotations

import json

import pytest

pytest.importorskip("torch")
from examples.prepare_search_qa_dataset import preview_records
from src import (
    PromptOnlyDataset,
    build_search_rag_record,
    build_prompt_dataloader,
    build_prompt_messages,
    build_search_qa_messages,
    build_search_qa_prompt,
    build_search_qa_record,
    format_rag_reference,
    make_search_rag_map_fn,
    make_search_qa_map_fn,
    normalize_prompt_training_example,
    normalize_question_text,
    prompt_batch_to_search_batch,
)


class _TokenizerWithEncode:
    chat_template = None

    def encode(self, text: str) -> list[int]:
        return [ord(char) for char in text]


class _MiniDataset(list):
    def select(self, indexes):
        return _MiniDataset([self[index] for index in indexes])


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


def test_normalize_question_text_cleans_whitespace_and_question_mark():
    assert normalize_question_text("  Who   wrote Dune.  ") == "Who wrote Dune?"
    assert normalize_question_text("What is FAISS?") == "What is FAISS?"


def test_normalize_prompt_training_example_accepts_golden_answer_aliases():
    example = normalize_prompt_training_example(
        {
            "question": "who wrote dune",
            "golden_answers": ["Frank Herbert", "frank herbert", "F. Herbert"],
            "source": "nq",
        }
    )

    assert example.question == "who wrote dune?"
    assert example.ground_truth == "Frank Herbert"
    assert example.metadata["answer_aliases"] == ["Frank Herbert", "F. Herbert"]
    assert example.metadata["source"] == "nq"


def test_build_search_qa_prompt_contains_agent_search_contract():
    assert build_search_qa_prompt("who wrote dune") == "who wrote dune?"


def test_build_search_qa_messages_reuses_search_agent_contract():
    messages = build_search_qa_messages("who wrote dune")

    assert messages[0]["role"] == "system"
    assert "<think>" in messages[0]["content"]
    assert "<search>" in messages[0]["content"]
    assert "<information>" in messages[0]["content"]
    assert "<answer>" in messages[0]["content"]
    assert "Never write or fabricate this block" in messages[0]["content"]
    assert messages[1] == {"role": "user", "content": "who wrote dune?"}


def test_build_search_qa_record_preserves_aliases_and_reward_target():
    record = build_search_qa_record(
        {
            "question": " who wrote dune ",
            "golden_answers": ["Frank Herbert", "F. Herbert"],
            "difficulty": "easy",
        },
        split="train",
        index=7,
        data_source="nq",
    )

    assert record["data_source"] == "nq"
    assert set(record) == {
        "data_source",
        "prompt",
        "ability",
        "reward_model",
        "extra_info",
    }
    assert record["prompt"] == [{"role": "user", "content": "who wrote dune?"}]
    assert record["ability"] == "fact-reasoning"
    assert record["reward_model"]["ground_truth"] == {
        "target": ["Frank Herbert", "F. Herbert"]
    }
    assert record["extra_info"] == {
        "split": "train",
        "index": 7,
    }


def test_make_search_qa_map_fn_matches_dataset_map_signature():
    process_fn = make_search_qa_map_fn(
        "test",
        data_source="nq",
        ability="fact-reasoning",
    )

    record = process_fn(
        {"question": " where is the louvre ", "golden_answers": ["Paris"]},
        3,
    )

    assert record["data_source"] == "nq"
    assert record["prompt"] == [{"role": "user", "content": "where is the louvre?"}]
    assert record["reward_model"]["ground_truth"] == {"target": ["Paris"]}
    assert record["extra_info"] == {"split": "test", "index": 3}


def test_format_rag_reference_uses_numbered_docs():
    text = format_rag_reference(
        [
            {"id": "d1", "contents": "Dune\nDune was written by Frank Herbert."},
            {"id": "d2", "title": "Louvre", "contents": "Located in Paris."},
        ]
    )

    assert "Doc 1(Title: Dune) Dune was written by Frank Herbert." in text
    assert "Doc 2(Title: Louvre) Located in Paris." in text


def test_build_search_rag_record_uses_compact_shape_with_context():
    record = build_search_rag_record(
        {"question": "who wrote dune", "golden_answers": ["Frank Herbert"]},
        context="Doc 1(Title: Dune) Dune was written by Frank Herbert.",
        split="train",
        index=4,
        data_source="nq",
    )

    assert set(record) == {
        "data_source",
        "prompt",
        "ability",
        "reward_model",
        "extra_info",
    }
    assert record["prompt"][0]["role"] == "user"
    assert "Question: who wrote dune?" in record["prompt"][0]["content"]
    assert "Context:\nDoc 1(Title: Dune)" in record["prompt"][0]["content"]
    assert record["reward_model"]["ground_truth"] == {"target": ["Frank Herbert"]}
    assert record["extra_info"] == {"split": "train", "index": 4}


def test_make_search_rag_map_fn_uses_cached_doc_ids():
    process_fn = make_search_rag_map_fn(
        "test",
        retrieval_cache={
            "who wrote dune": [{"id": "d1"}, {"id": "d2"}],
        },
        corpus={
            "d1": {"id": "d1", "contents": "Dune\nFrank Herbert wrote Dune."},
            "d2": {"id": "d2", "contents": "Novel\nDune is a novel."},
        },
        topk=1,
        data_source="nq",
    )

    record = process_fn(
        {"question": "who wrote dune", "golden_answers": ["Frank Herbert"]},
        2,
    )

    assert record["data_source"] == "nq"
    assert "Doc 1(Title: Dune)" in record["prompt"][0]["content"]
    assert "Novel" not in record["prompt"][0]["content"]
    assert record["reward_model"]["ground_truth"] == {"target": ["Frank Herbert"]}
    assert record["extra_info"] == {"split": "test", "index": 2}


def test_preview_records_prints_question_answer_audit_lines(capsys):
    rows = _MiniDataset(
        [
            build_search_qa_record(
                {"question": "who wrote dune", "golden_answers": ["Frank Herbert"]},
                split="test",
                index=0,
                data_source="nq",
            )
        ]
    )

    preview_records(rows, split="test", limit=1)

    output_lines = capsys.readouterr().out.strip().splitlines()
    preview = json.loads(output_lines[-1])
    assert output_lines[0] == "[test] converted preview"
    assert preview["question"] == "who wrote dune?"
    assert preview["golden_answers"] == ["Frank Herbert"]
    assert preview["reward_target"] == ["Frank Herbert"]
    assert preview["prompt_roles"] == ["user"]
    assert preview["extra_info"] == {"split": "test", "index": 0}


def test_prompt_only_dataset_uses_stored_search_qa_prompt():
    record = build_search_qa_record(
        {"question": "who wrote dune", "golden_answers": ["Frank Herbert"]},
        data_source="nq",
    )
    dataset = PromptOnlyDataset(
        [record],
        tokenizer=_TokenizerWithEncode(),
        prompt_length=4096,
    )

    sample = dataset[0]

    assert sample.question == "who wrote dune?"
    assert sample.messages == [{"role": "user", "content": "who wrote dune?"}]
    assert sample.ground_truth == "Frank Herbert"
    assert sample.tools == []


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
