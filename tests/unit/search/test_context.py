from src.context import ChatMessage
from src.internal.search.context import build_retrieval_context
import pytest


def test_standalone_query_is_not_rewritten():
    history = [ChatMessage(role="user", content="Tell me about PostgreSQL")]

    context = build_retrieval_context("How does Redis persistence work?", history)

    assert context.query == "How does Redis persistence work?"
    assert context.retrieval_query == "How does Redis persistence work?"


def test_standalone_why_question_is_not_rewritten():
    history = [ChatMessage(role="user", content="Tell me about PostgreSQL")]

    context = build_retrieval_context("Why is the sky blue?", history)

    assert context.retrieval_query == "Why is the sky blue?"


def test_standalone_topic_containing_pronoun_is_not_rewritten():
    history = [ChatMessage(role="user", content="Tell me about PostgreSQL")]

    context = build_retrieval_context(
        "How did Ada Lovelace shape computing, and what did she write?", history
    )

    assert context.retrieval_query == (
        "How did Ada Lovelace shape computing, and what did she write?"
    )


def test_latest_standalone_topic_may_contain_a_pronoun():
    history = [
        ChatMessage(role="user", content="Tell me about PostgreSQL"),
        ChatMessage(
            role="user",
            content="How did Ada Lovelace shape computing, and what did she write?",
        ),
    ]

    context = build_retrieval_context("Where did she publish it?", history)

    assert context.retrieval_query == (
        "How did Ada Lovelace shape computing, and what did she write?\n"
        "Where did she publish it?"
    )


def test_pronoun_follow_up_uses_most_recent_user_topic():
    history = [
        ChatMessage(role="user", content="Explain the Voyager 1 mission"),
        ChatMessage(role="assistant", content="Voyager 1 launched in 1977."),
    ]

    context = build_retrieval_context("Where is it now?", history)

    assert context.query == "Where is it now?"
    assert context.retrieval_query == "Explain the Voyager 1 mission\nWhere is it now?"


def test_continuation_cue_uses_latest_standalone_user_topic():
    history = [
        ChatMessage(role="user", content="Compare Kafka and RabbitMQ"),
        ChatMessage(role="assistant", content="They differ in several ways."),
        ChatMessage(role="user", content="What about throughput?"),
        ChatMessage(role="assistant", content="Kafka is generally optimized for it."),
    ]

    context = build_retrieval_context("And operational complexity?", history)

    assert context.retrieval_query == (
        "Compare Kafka and RabbitMQ\nAnd operational complexity?"
    )


def test_history_is_bounded_to_most_recent_messages():
    history = [
        ChatMessage(role="user", content=f"message {index}") for index in range(8)
    ]

    context = build_retrieval_context("A standalone query", history, max_messages=3)

    assert [message.content for message in context.history] == [
        "message 5",
        "message 6",
        "message 7",
    ]


def test_assistant_tool_and_evidence_markup_is_excluded_from_history():
    history = [
        ChatMessage(role="user", content="Tell me about lunar ice"),
        ChatMessage(
            role="assistant", content="<tool_call>search lunar ice</tool_call>"
        ),
        ChatMessage(role="assistant", content="<evidence>secret raw result</evidence>"),
        ChatMessage(role="assistant", content="Ice exists in shadowed craters."),
    ]

    context = build_retrieval_context("Where was it found?", history)

    assert [message.content for message in context.history] == [
        "Tell me about lunar ice",
        "Ice exists in shadowed craters.",
    ]
    assert "secret raw result" not in context.retrieval_query
    assert "search lunar ice" not in context.retrieval_query


@pytest.mark.parametrize(
    "tag",
    ["search", "searches", "fetch", "information", "search_decision"],
)
def test_repository_native_assistant_markup_is_excluded(tag):
    history = [
        ChatMessage(role="user", content="Tell me about lunar ice"),
        ChatMessage(role="assistant", content=f"<{tag}>internal data</{tag}>"),
        ChatMessage(role="assistant", content="Ice exists in shadowed craters."),
    ]

    context = build_retrieval_context("Where was it found?", history)

    assert [message.content for message in context.history] == [
        "Tell me about lunar ice",
        "Ice exists in shadowed craters.",
    ]
