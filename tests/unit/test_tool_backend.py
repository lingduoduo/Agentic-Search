import json

from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.internal.db import AgenticSearchStore
from src.internal.configs import load_app_settings
from src.internal.servers.query_and_chat.tool_backend import create_tool_router


def _make_app(*, with_model: bool) -> FastAPI:
    store = AgenticSearchStore(":memory:")
    app = FastAPI()
    app.include_router(
        create_tool_router(
            store, search_url="http://x/retrieve", resolved=load_app_settings()
        )
    )
    app.state.search_agent_manager = object() if with_model else None
    app.state.search_agent_tokenizer = object() if with_model else None
    app.state.tool_approval_broker = None
    app.state._store = store  # keep a handle for assertions
    return app


def test_send_tool_message_no_model_returns_400():
    client = TestClient(_make_app(with_model=False))
    resp = client.post(
        "/tool/send-tool-message", json={"message": "hi", "stream": False}
    )
    assert resp.status_code == 400
    assert "requires a local model" in resp.json()["detail"]


def test_tool_history_anonymous_is_empty():
    client = TestClient(_make_app(with_model=True))
    resp = client.get("/tool/tool-history")
    assert resp.status_code == 200
    assert resp.json() == {"sessions": []}


def test_send_tool_message_streams_progress_then_done(monkeypatch):
    from src.internal.servers.web import tool_agent_runner
    from src.internal.servers.web.tool_agent_runner import ToolCallView

    async def fake_run_tool_agent(query, *, on_turn=None, **kw):
        if on_turn is not None:
            await on_turn(1, "search", 3)
        tc = ToolCallView(
            tool_name="search",
            status="completed",
            arguments={"q": query},
            result_summary="3 items",
            latency_ms=12,
            error=None,
        )
        return ("the answer", [], [], "tool", {"tool_calls": [tc], "num_turns": 1})

    # tool_backend now imports _run_tool_agent function-locally (per-call) from
    # tool_agent_runner, so patch the source module rather than tool_backend's
    # (now nonexistent) module-level attribute.
    monkeypatch.setattr(tool_agent_runner, "_run_tool_agent", fake_run_tool_agent)

    client = TestClient(_make_app(with_model=True))
    with client.stream(
        "POST", "/tool/send-tool-message", json={"message": "find X", "stream": True}
    ) as resp:
        assert resp.status_code == 200
        events = [
            json.loads(line[len("data:") :].strip())
            for line in resp.iter_lines()
            if line.startswith("data:")
        ]

    types = [e["type"] for e in events]
    assert "progress" in types
    assert types[-1] == "done"
    tool_call = next(e for e in events if e["type"] == "tool_call")
    assert tool_call["tool_name"] == "search"
    done = events[-1]
    assert done["num_turns"] == 1 and done["session_id"]
