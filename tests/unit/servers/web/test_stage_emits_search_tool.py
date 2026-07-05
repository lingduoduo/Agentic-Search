from __future__ import annotations

from src.agents.search.agentic_rag import AgenticRAGConfig, AgenticRAGLoop
from src.agents.search.search import SearchAgentLoop
from src.agents.tool.tool_calling import ToolAgentLoop
from src.context.models import ContextDocument
from src.context.search import SearchResult
from src.internal.servers.web import request_capture as rc


def test_agentic_rag_record_search_stage_emits_search_stage():
    loop = AgenticRAGLoop(AgenticRAGConfig())
    docs = [
        ContextDocument(
            id="D1", title="Doc One", content="alpha", url="http://a", score=0.9
        ),
        ContextDocument(
            id="D2", title="Doc Two", content="beta", url="http://b", score=0.5
        ),
    ]
    token = rc.start_capture("r", "q")
    try:
        loop._record_search_stage("dense retrieval", 5, 1, docs)
        cap = rc.active()
        search_stages = [s for s in cap.stages if s.stage == "search"]
        assert search_stages and len(search_stages) == 1
        payload = search_stages[0].payload
        assert payload["query"] == "dense retrieval"
        assert payload["top_k"] == 5
        assert payload["round"] == 1
        assert len(payload["documents"]) == 2
        assert payload["documents"][0] == {
            "id": "D1",
            "title": "Doc One",
            "text": "alpha",
            "score": 0.9,
            "source": "http://a",
        }
    finally:
        rc.reset_capture(token)


def test_search_agent_loop_record_search_stage_emits_search_stage():
    docs = [
        SearchResult(contents="alpha", score=0.9, title="Doc One", url="http://a"),
        SearchResult(contents="beta", score=0.5, title="Doc Two", url="http://b"),
    ]
    token = rc.start_capture("r", "q")
    try:
        SearchAgentLoop._record_search_stage(None, ["dense retrieval"], 5, 2, docs)
        cap = rc.active()
        search_stages = [s for s in cap.stages if s.stage == "search"]
        assert search_stages and len(search_stages) == 1
        payload = search_stages[0].payload
        assert payload["query"] == ["dense retrieval"]
        assert payload["top_k"] == 5
        assert payload["round"] == 2
        assert len(payload["documents"]) == 2
        assert payload["documents"][0]["text"] == "alpha"
        assert payload["documents"][0]["source"] == "http://a"
    finally:
        rc.reset_capture(token)


def test_tool_agent_loop_record_tool_stage_emits_tool_stage():
    token = rc.start_capture("r", "q")
    try:
        ToolAgentLoop._record_tool_stage(
            None, "search", {"query": "faiss"}, "some result text"
        )
        cap = rc.active()
        tool_stages = [s for s in cap.stages if s.stage == "tool"]
        assert tool_stages and len(tool_stages) == 1
        payload = tool_stages[0].payload
        assert payload == {
            "name": "search",
            "args": {"query": "faiss"},
            "result": "some result text",
        }
        assert tool_stages[0].label == "search"
    finally:
        rc.reset_capture(token)


def test_no_active_capture_does_not_raise():
    loop = AgenticRAGLoop(AgenticRAGConfig())
    assert rc.active() is None
    loop._record_search_stage("q", 5, 0, [])
    SearchAgentLoop._record_search_stage(None, ["q"], 5, 0, [])
    ToolAgentLoop._record_tool_stage(None, "search", {}, None)
    assert rc.active() is None
