"""Tests for load_sft_examples."""

from __future__ import annotations

import json
import pytest

from src.internal.db import AgenticSearchStore
from src.training.data import load_sft_examples
from src.training.sft import SFTExample


def _seed_store(db_path: str, rows: list[dict]) -> None:
    with AgenticSearchStore(db_path) as store:
        for r in rows:
            sid = r["session_id"]
            store.create_chat_session(session_id=sid, user_id=None)
            if r.get("user_msg"):
                store.add_chat_message(sid, role="user", content=r["user_msg"])
            if r.get("assistant_msg"):
                store.add_chat_message(
                    sid, role="assistant", content=r["assistant_msg"]
                )
            if r.get("signal"):
                store.save_retrieval_feedback(sid, r["signal"])


def test_thumbs_up_session_becomes_sft_example(tmp_path):
    db = str(tmp_path / "t.sqlite3")
    _seed_store(
        db,
        [
            {
                "session_id": "s1",
                "signal": "thumbs_up",
                "user_msg": "Q?",
                "assistant_msg": "A.",
            }
        ],
    )
    examples = load_sft_examples(db, min_ratings=1)
    assert len(examples) == 1
    ex = examples[0]
    assert ex.prompt_messages == [{"role": "user", "content": "Q?"}]
    assert ex.completion == "A."


def test_thumbs_down_session_excluded(tmp_path):
    db = str(tmp_path / "t.sqlite3")
    _seed_store(
        db,
        [
            {
                "session_id": "s1",
                "signal": "thumbs_down",
                "user_msg": "Q?",
                "assistant_msg": "A.",
            }
        ],
    )
    with pytest.raises(ValueError, match="Only 0 SFT examples"):
        load_sft_examples(db, min_ratings=1)


def test_session_without_assistant_turn_skipped(tmp_path):
    db = str(tmp_path / "t.sqlite3")
    _seed_store(db, [{"session_id": "s1", "signal": "thumbs_up", "user_msg": "Q?"}])
    with pytest.raises(ValueError, match="Only 0 SFT examples"):
        load_sft_examples(db, min_ratings=1)


def test_jsonl_row_becomes_sft_example(tmp_path):
    db = str(tmp_path / "t.sqlite3")
    AgenticSearchStore(db).__exit__(None, None, None)  # init schema only
    jsonl = tmp_path / "pairs.jsonl"
    jsonl.write_text(json.dumps({"question": "Q?", "response": "R."}) + "\n")
    examples = load_sft_examples(db, jsonl_path=str(jsonl), min_ratings=1)
    assert len(examples) == 1
    assert examples[0].prompt_messages == [{"role": "user", "content": "Q?"}]
    assert examples[0].completion == "R."


def test_jsonl_row_missing_response_skipped(tmp_path):
    db = str(tmp_path / "t.sqlite3")
    AgenticSearchStore(db).__exit__(None, None, None)
    jsonl = tmp_path / "pairs.jsonl"
    jsonl.write_text(json.dumps({"question": "Q?"}) + "\n")
    with pytest.raises(ValueError, match="Only 0 SFT examples"):
        load_sft_examples(db, jsonl_path=str(jsonl), min_ratings=1)


def test_db_and_jsonl_merged(tmp_path):
    db = str(tmp_path / "t.sqlite3")
    _seed_store(
        db,
        [
            {
                "session_id": "s1",
                "signal": "thumbs_up",
                "user_msg": "Q1?",
                "assistant_msg": "A1.",
            }
        ],
    )
    jsonl = tmp_path / "pairs.jsonl"
    jsonl.write_text(json.dumps({"question": "Q2?", "response": "R2."}) + "\n")
    examples = load_sft_examples(db, jsonl_path=str(jsonl), min_ratings=2)
    assert len(examples) == 2


def test_raises_when_below_min_ratings(tmp_path):
    db = str(tmp_path / "t.sqlite3")
    _seed_store(
        db,
        [
            {
                "session_id": "s1",
                "signal": "thumbs_up",
                "user_msg": "Q?",
                "assistant_msg": "A.",
            }
        ],
    )
    with pytest.raises(ValueError, match="Only 1 SFT examples found; need at least 3"):
        load_sft_examples(db, min_ratings=3)


def test_returns_sft_example_instances(tmp_path):
    db = str(tmp_path / "t.sqlite3")
    _seed_store(
        db,
        [
            {
                "session_id": "s1",
                "signal": "thumbs_up",
                "user_msg": "Q?",
                "assistant_msg": "A.",
            }
        ],
    )
    examples = load_sft_examples(db, min_ratings=1)
    assert all(isinstance(ex, SFTExample) for ex in examples)
