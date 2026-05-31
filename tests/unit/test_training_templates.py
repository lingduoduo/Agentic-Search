"""Tests for multi-template support in training/data.py."""

from __future__ import annotations

import pytest

from src.training.data import (
    build_search_qa_messages,
    build_search_qa_prompt,
    build_search_rag_prompt,
    register_qa_messages_template,
    register_qa_prompt_template,
    register_rag_prompt_template,
)


# ---------------------------------------------------------------------------
# build_search_qa_prompt
# ---------------------------------------------------------------------------


def test_qa_prompt_base():
    result = build_search_qa_prompt("what is paris?", template_type="base")
    assert result == "what is paris?"


def test_qa_prompt_chat():
    result = build_search_qa_prompt("what is paris", template_type="chat")
    assert result.startswith("Please help me answer:")
    assert "paris" in result.lower()


def test_qa_prompt_instruct():
    result = build_search_qa_prompt("what is paris", template_type="instruct")
    assert "step" in result.lower()
    assert "paris" in result.lower()


def test_qa_prompt_unknown_raises():
    with pytest.raises(ValueError, match="Unknown QA prompt template"):
        build_search_qa_prompt("q", template_type="nonexistent")


def test_qa_prompt_error_lists_available_templates():
    with pytest.raises(ValueError) as exc_info:
        build_search_qa_prompt("q", template_type="missing")
    assert "base" in str(exc_info.value)
    assert "chat" in str(exc_info.value)
    assert "instruct" in str(exc_info.value)


def test_qa_prompt_custom_registration():
    register_qa_prompt_template("upper", lambda q: q.upper())
    result = build_search_qa_prompt("hello world?", template_type="upper")
    assert result == "HELLO WORLD?"


# ---------------------------------------------------------------------------
# build_search_qa_messages
# ---------------------------------------------------------------------------


def test_qa_messages_base_has_system_and_user():
    msgs = build_search_qa_messages("what is faiss?", template_type="base")
    roles = [m["role"] for m in msgs]
    assert roles == ["system", "user"]
    assert "faiss" in msgs[1]["content"].lower()


def test_qa_messages_base_uses_custom_system_prompt():
    msgs = build_search_qa_messages(
        "question",
        template_type="base",
        system_prompt="Custom system.",
    )
    assert msgs[0]["content"] == "Custom system."


def test_qa_messages_chat():
    msgs = build_search_qa_messages("what is faiss?", template_type="chat")
    assert msgs[0]["role"] == "system"
    assert "Please help me answer" in msgs[1]["content"]


def test_qa_messages_instruct():
    msgs = build_search_qa_messages("what is faiss?", template_type="instruct")
    assert "step" in msgs[1]["content"].lower()


def test_qa_messages_unknown_raises():
    with pytest.raises(ValueError, match="Unknown QA messages template"):
        build_search_qa_messages("q", template_type="unknown")


def test_qa_messages_custom_registration():
    register_qa_messages_template(
        "minimal", lambda q, **_: [{"role": "user", "content": q}]
    )
    msgs = build_search_qa_messages("hi", template_type="minimal")
    assert msgs == [{"role": "user", "content": "hi"}]


# ---------------------------------------------------------------------------
# build_search_rag_prompt
# ---------------------------------------------------------------------------


def test_rag_prompt_base_contains_xml_markers():
    result = build_search_rag_prompt(
        "what is FAISS?", "FAISS is a library.", template_type="base"
    )
    assert "<think>" in result
    assert "<answer>" in result
    assert "faiss" in result.lower()


def test_rag_prompt_chat():
    result = build_search_rag_prompt(
        "what is FAISS?", "FAISS is a library.", template_type="chat"
    )
    assert "Here is some information" in result
    assert "faiss" in result.lower()
    assert "<think>" not in result


def test_rag_prompt_instruct():
    result = build_search_rag_prompt(
        "what is FAISS?", "FAISS is a library.", template_type="instruct"
    )
    assert "step" in result.lower()
    assert "faiss" in result.lower()


def test_rag_prompt_unknown_raises():
    with pytest.raises(ValueError, match="Unknown RAG prompt template"):
        build_search_rag_prompt("q", "ctx", template_type="nope")


def test_rag_prompt_custom_registration():
    register_rag_prompt_template("compact", lambda q, ctx: f"Q:{q} C:{ctx}")
    result = build_search_rag_prompt("hi?", "ctx here", template_type="compact")
    assert result == "Q:hi? C:ctx here"
