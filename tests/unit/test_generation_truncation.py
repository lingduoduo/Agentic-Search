"""A truncated generation must reach the caller, not just stdout.

The wall-clock stop cuts long local answers mid-word. The manager already
detected it and printed a warning, but nothing carried the fact into the API
response, so the user saw half a sentence with no explanation.
"""

from __future__ import annotations

import types

import pytest

from src.agents.core.base import AgentLoopOutput


def _manager(**kw):
    from src.model.serving import LocalServerManager

    mgr = LocalServerManager.__new__(LocalServerManager)
    mgr.generation_timeout_seconds = kw.get("timeout", 120.0)
    mgr._truncated = {}
    return mgr


# ---------------------------------------------------------------------------
# Manager records it, keyed by request so concurrent runs don't cross wires
# ---------------------------------------------------------------------------


def test_pop_truncated_is_false_when_nothing_was_recorded():
    assert _manager().pop_truncated("req-1") is False


def test_recorded_truncation_pops_once():
    mgr = _manager()
    mgr._record_truncation("req-1")
    assert mgr.pop_truncated("req-1") is True
    assert mgr.pop_truncated("req-1") is False  # consumed


def test_truncation_is_keyed_per_request():
    mgr = _manager()
    mgr._record_truncation("req-1")
    assert mgr.pop_truncated("req-2") is False
    assert mgr.pop_truncated("req-1") is True


def test_records_are_bounded_so_unpopped_entries_cannot_grow_forever():
    from src.model.serving import _TRUNCATION_RECORD_LIMIT

    mgr = _manager()
    for i in range(_TRUNCATION_RECORD_LIMIT + 50):
        mgr._record_truncation(f"req-{i}")
    assert len(mgr._truncated) <= _TRUNCATION_RECORD_LIMIT


# ---------------------------------------------------------------------------
# The loop carries it out
# ---------------------------------------------------------------------------


def test_agent_loop_output_defaults_to_not_truncated():
    out = AgentLoopOutput(prompt_ids=[], response_ids=[], response_mask=[], num_turns=1)
    assert out.truncated is False


@pytest.mark.asyncio
async def test_tool_agent_loop_reports_a_truncated_generation():
    from src.agents.tool import ToolAgentLoop, ToolAgentLoopConfig

    class _Manager:
        def __init__(self):
            self.popped = []

        async def generate(self, request_id, prompt_ids, sampling_params):
            return [1, 2, 3]

        def pop_truncated(self, request_id):
            self.popped.append(request_id)
            return True

    class _Tok:
        def apply_chat_template(self, messages, **kw):
            return [1]

        def decode(self, ids, **kw):
            return "half a sen"

    loop = ToolAgentLoop(
        tokenizer=_Tok(),
        server_manager=_Manager(),
        tools=[],
        config=ToolAgentLoopConfig(tool_parser_format="json"),
    )
    out = await loop.run(
        [{"role": "user", "content": "q"}], sampling_params={"max_tokens": 8}
    )
    assert out.truncated is True
    assert out.final_answer == "half a sen"


@pytest.mark.asyncio
async def test_untruncated_generation_reports_false():
    from src.agents.tool import ToolAgentLoop, ToolAgentLoopConfig

    class _Manager:
        async def generate(self, request_id, prompt_ids, sampling_params):
            return [1]

        def pop_truncated(self, request_id):
            return False

    class _Tok:
        def apply_chat_template(self, messages, **kw):
            return [1]

        def decode(self, ids, **kw):
            return "done."

    loop = ToolAgentLoop(
        tokenizer=_Tok(),
        server_manager=_Manager(),
        tools=[],
        config=ToolAgentLoopConfig(tool_parser_format="json"),
    )
    out = await loop.run([{"role": "user", "content": "q"}], sampling_params={})
    assert out.truncated is False


@pytest.mark.asyncio
async def test_a_manager_without_the_hook_is_tolerated():
    # OpenAI-compatible and test-double managers do not implement pop_truncated.
    from src.agents.tool import ToolAgentLoop, ToolAgentLoopConfig

    class _Manager:
        async def generate(self, request_id, prompt_ids, sampling_params):
            return [1]

    class _Tok:
        def apply_chat_template(self, messages, **kw):
            return [1]

        def decode(self, ids, **kw):
            return "fine"

    loop = ToolAgentLoop(
        tokenizer=_Tok(),
        server_manager=_Manager(),
        tools=[],
        config=ToolAgentLoopConfig(tool_parser_format="json"),
    )
    out = await loop.run([{"role": "user", "content": "q"}], sampling_params={})
    assert out.truncated is False


# ---------------------------------------------------------------------------
# The runner hands it to the web layer
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_run_tool_agent_surfaces_truncation_in_extra(monkeypatch):
    from unittest.mock import AsyncMock, MagicMock
    from src.internal.servers.web import app as web_app

    output = AgentLoopOutput(
        prompt_ids=[],
        response_ids=[],
        response_mask=[],
        num_turns=1,
        final_answer="half a sen",
        action_trace="",
        trajectory_messages=[],
        truncated=True,
    )
    monkeypatch.setattr(
        "src.agents.tool.tool_calling.ToolAgentLoop.run", AsyncMock(return_value=output)
    )
    _a, _c, _d, _i, extra = await web_app._run_tool_agent(
        "q",
        manager=MagicMock(),
        tokenizer=MagicMock(),
        search_url="http://x/retrieve",
        history=[],
        resolved=types.SimpleNamespace(tool_agent_parser="json"),
        on_turn=None,
        with_search_tool=False,
    )
    assert extra["truncated"] is True


# ---------------------------------------------------------------------------
# The API says so
# ---------------------------------------------------------------------------


def test_tool_agent_response_model_carries_truncated():
    from src.internal.servers.query_and_chat.models import ToolAgentMessageResponse

    assert ToolAgentMessageResponse(session_id="s", answer="a").truncated is False
    assert (
        ToolAgentMessageResponse(session_id="s", answer="a", truncated=True).truncated
        is True
    )


@pytest.mark.asyncio
async def test_send_tool_message_reports_truncation(monkeypatch):
    from src.internal.db import AgenticSearchStore
    from src.internal.servers.query_and_chat import tool_backend
    from src.internal.servers.query_and_chat.models import SendToolMessageRequest

    store = AgenticSearchStore(":memory:")
    router = tool_backend.create_tool_router(
        store, search_url="http://x/retrieve", resolved=types.SimpleNamespace()
    )
    endpoint = next(
        r.endpoint
        for r in router.routes
        if getattr(r, "path", "") == "/tool/send-tool-message"
    )

    async def fake_run_tool_agent(query, **kw):
        return (
            "half a sen",
            [],
            [],
            "tool",
            {"tool_calls": [], "num_turns": 2, "truncated": True},
        )

    monkeypatch.setattr(
        "src.internal.servers.web.tool_agent_runner._run_tool_agent",
        fake_run_tool_agent,
    )
    request = types.SimpleNamespace(
        app=types.SimpleNamespace(
            state=types.SimpleNamespace(
                search_agent_manager=object(),
                search_agent_tokenizer=object(),
                tool_approval_broker=None,
            )
        ),
        headers={},
        cookies={},
        challenge_unused=None,
    )
    body = SendToolMessageRequest(message="q", stream=False)
    response = await endpoint(body, request)
    assert response.truncated is True
    assert response.answer == "half a sen"


# ---------------------------------------------------------------------------
# The real call path. Mocking the manager hides a signature break between
# generate() and _generate_sync(), which is exactly how one slipped through.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_generate_records_truncation_through_the_real_call_path():
    torch = pytest.importorskip("torch")
    from src.model.serving import LocalServerManager

    class _Tokenizer:
        pad_token_id = 0
        eos_token_id = 0

    class _Model:
        generation_config = types.SimpleNamespace(
            pad_token_id=None, do_sample=True, temperature=0.7, top_p=0.9, top_k=20
        )

        def generate(self, inputs, **kwargs):
            import time

            time.sleep(0.05)  # outlast the timeout below
            return torch.tensor([[1, 2, 3]], dtype=torch.long)

    manager = LocalServerManager(
        model_path="dummy",
        device="cpu",
        generation_timeout_seconds=0.01,
        generation_heartbeat_seconds=999.0,
    )
    manager._tokenizer = _Tokenizer()
    manager._model = _Model()

    ids = await manager.generate(
        request_id="req-x", prompt_ids=[1, 2], sampling_params={"max_tokens": 64}
    )

    assert ids == [3]
    assert manager.pop_truncated("req-x") is True


@pytest.mark.asyncio
async def test_generate_records_nothing_when_it_finishes_in_time():
    torch = pytest.importorskip("torch")
    from src.model.serving import LocalServerManager

    class _Tokenizer:
        pad_token_id = 0
        eos_token_id = 0

    class _Model:
        generation_config = types.SimpleNamespace(
            pad_token_id=None, do_sample=True, temperature=0.7, top_p=0.9, top_k=20
        )

        def generate(self, inputs, **kwargs):
            return torch.tensor([[1, 2, 3]], dtype=torch.long)

    manager = LocalServerManager(
        model_path="dummy",
        device="cpu",
        generation_timeout_seconds=60.0,
        generation_heartbeat_seconds=999.0,
    )
    manager._tokenizer = _Tokenizer()
    manager._model = _Model()

    await manager.generate(
        request_id="req-y", prompt_ids=[1, 2], sampling_params={"max_tokens": 64}
    )
    assert manager.pop_truncated("req-y") is False
