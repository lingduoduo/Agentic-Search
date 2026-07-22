import asyncio

from src.internal.db.models import UserRecord
from src.internal.db.store import AgenticSearchStore
from src.internal.llm.model_response import (
    ChatCompletionDeltaToolCall,
    Delta,
    FunctionCall,
    ModelResponseStream,
    StreamingChoice,
)
from src.internal.memory import service


class _FakeConfig:
    model_name = "fake-model"


class _FakeLLM:
    """Yields scripted stream turns: a list of chunk-lists, one per stream() call."""

    def __init__(self, turns):
        self._turns = list(turns)
        self.config = _FakeConfig()

    def stream(self, prompt, tools=None, tool_choice=None, max_tokens=None, **kwargs):
        return iter(self._turns.pop(0))


def _tool_chunk(index, call_id, name, arguments):
    return ModelResponseStream(
        id="x",
        created="0",
        choice=StreamingChoice(
            delta=Delta(
                tool_calls=[
                    ChatCompletionDeltaToolCall(
                        id=call_id,
                        index=index,
                        function=FunctionCall(name=name, arguments=arguments),
                    )
                ]
            )
        ),
    )


def _text_chunk(text):
    return ModelResponseStream(
        id="x", created="0", choice=StreamingChoice(delta=Delta(content=text))
    )


def test_curation_applies_tool_calls_and_persists_trajectory():
    store = AgenticSearchStore(":memory:")
    store.upsert_user(UserRecord(id="u1"))
    session = store.create_chat_session(user_id="u1")
    store.add_chat_message(session.id, role="user", content="I just moved to Shanghai.")
    store.add_chat_message(session.id, role="assistant", content="Noted!")

    llm = _FakeLLM(
        turns=[
            [
                _tool_chunk(
                    0, "c1", "add_memory", '{"content": "User moved to Shanghai"}'
                )
            ],
            [_text_chunk("STOP")],  # second turn: no tool calls -> loop ends
        ]
    )

    summary = asyncio.run(service.curate_from_conversation(store, "u1", llm))
    assert summary["status"] == "ok"
    assert summary["counts"]["add"] == 1
    texts = [r.memory_text for r in store.get_user_memory_records("u1")]
    assert texts == ["User moved to Shanghai"]

    traj = store.list_memory_trajectories("u1")
    assert len(traj) == 1
    assert traj[0].trajectory["counts"]["add"] == 1
    assert traj[0].trajectory["memory_after"] == ["User moved to Shanghai"]
    store.close()


def test_curation_empty_sources_returns_message():
    store = AgenticSearchStore(":memory:")
    llm = _FakeLLM(turns=[])
    summary = asyncio.run(service.curate_from_conversation(store, "nobody", llm))
    assert summary["status"] == "empty"
    store.close()


def test_curation_logs_malformed_tool_call_in_trajectory():
    store = AgenticSearchStore(":memory:")
    store.upsert_user(UserRecord(id="u1"))
    session = store.create_chat_session(user_id="u1")
    store.add_chat_message(session.id, role="user", content="I just moved to Shanghai.")
    store.add_chat_message(session.id, role="assistant", content="Noted!")

    llm = _FakeLLM(
        turns=[
            [_tool_chunk(0, "c1", "add_memory", "{not json")],
            [_text_chunk("STOP")],  # second turn: no tool calls -> loop ends
        ]
    )

    summary = asyncio.run(service.curate_from_conversation(store, "u1", llm))
    assert summary["status"] == "ok"

    traj = store.list_memory_trajectories("u1")
    assert len(traj) == 1
    tool_calls = traj[0].trajectory["tool_calls"]
    assert len(tool_calls) == 1
    assert tool_calls[0]["name"] == "add_memory"
    assert tool_calls[0]["arguments"] == "{not json"
    assert "invalid JSON arguments" in tool_calls[0]["result"]
    store.close()
