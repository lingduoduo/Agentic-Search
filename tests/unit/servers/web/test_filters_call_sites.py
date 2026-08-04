"""`filters` must reach `_run_tool_agent` at every call site that offers it a
corpus search tool, or the tool agent falls back to `_run_tool_agent`'s
`filters=None` default — which `tool_agent_runner._run_tool_agent` treats as
"pass no filters to the request-bound corpus search", i.e. unfiltered.

This mirrors `test_user_present_call_sites.py`'s pattern: fake out
`_run_tool_agent`, capture the kwargs it was called with, and assert on the
actual object rather than trusting the call site reads correctly. Only the two
call sites that pass `with_search_tool=True` are covered here — the auto route
(`app.py`, `with_search_tool=False`) never builds the corpus search tool, so
its `filters` argument is inert and pinning it would just test dead code.
"""

from __future__ import annotations

import types

import pytest
from fastapi.testclient import TestClient

import src.internal.servers.web.app as web_app
from src.context.models import SearchFilters
from src.internal.access.access import PUBLIC_ACL
from src.internal.servers.web.app import SearchExperienceSettings, create_web_app


@pytest.fixture
def seen() -> dict:
    return {}


@pytest.fixture
def patched_tool_agent(monkeypatch, seen):
    async def _fake_tool_agent(query, **kwargs):
        seen["filters"] = kwargs.get("filters")
        return ("tool answer", [], [], "tool", {})

    monkeypatch.setattr(web_app, "_run_tool_agent", _fake_tool_agent)
    monkeypatch.setattr(
        "src.internal.servers.web.tool_agent_runner._run_tool_agent", _fake_tool_agent
    )
    return _fake_tool_agent


def _client(tmp_path, name: str) -> TestClient:
    app = create_web_app(SearchExperienceSettings(db_path=tmp_path / f"{name}.sqlite3"))
    app.state.search_agent_manager = types.SimpleNamespace()
    app.state.search_agent_tokenizer = types.SimpleNamespace()
    return TestClient(app)


def test_tool_agent_mode_passes_the_caller_s_filters(
    patched_tool_agent, seen, tmp_path
):
    response = _client(tmp_path, "tool_mode_filters").post(
        "/api/agent",
        json={
            "query": "search the corpus",
            "mode": "tool_agent",
            "user_id": "victim",
        },
    )

    assert response.status_code == 200, response.text
    # Unauthenticated resolves to the anonymous capability, which carries
    # ["public"] rather than an unfiltered `None` — so a leak here would show
    # up as `filters is None`, not merely as the wrong ACL.
    assert seen["filters"] == SearchFilters(access_acl=[PUBLIC_ACL])


def test_send_tool_message_passes_the_caller_s_filters(
    patched_tool_agent, seen, tmp_path
):
    # The tool surface resolves its own caller rather than inheriting
    # /api/agent's, so it needs its own proof rather than reuse of the above.
    response = _client(tmp_path, "tool_router_filters").post(
        "/tool/send-tool-message",
        json={"message": "search the corpus", "stream": False, "run_search_tool": True},
    )

    assert response.status_code == 200, response.text
    assert seen["filters"] == SearchFilters(access_acl=[PUBLIC_ACL])
