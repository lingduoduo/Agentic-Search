def test_symbols_importable_from_new_module():
    from src.internal.servers.web.tool_agent_runner import (
        ToolCallView,
        _extract_tool_calls_and_docs,
        _run_tool_agent,
    )

    assert ToolCallView.__name__ == "ToolCallView"
    assert callable(_extract_tool_calls_and_docs)
    assert callable(_run_tool_agent)


def test_app_reexports_same_objects():
    # app.py must import (not redefine) the relocated symbols.
    from src.internal.servers.web import app, tool_agent_runner

    assert app.ToolCallView is tool_agent_runner.ToolCallView
    assert app._run_tool_agent is tool_agent_runner._run_tool_agent


def test_extract_empty_trace_returns_empty():
    from src.internal.servers.web.tool_agent_runner import _extract_tool_calls_and_docs

    class _Out:
        action_trace = ""

    calls, docs = _extract_tool_calls_and_docs(_Out())
    assert calls == [] and docs == []


def test_citeable_tool_output_becomes_documents():
    """A citeable tool's {title, content, url} array turns into source cards."""
    import json

    from src.internal.servers.web.tool_agent_runner import (
        _extract_tool_calls_and_docs,
    )

    class _Out:
        action_trace = json.dumps(
            {
                "tool_name": "search_wikipedia",
                "status": "completed",
                "arguments": {"query": "faiss"},
                "result": json.dumps(
                    [{"title": "FAISS", "content": "A library.", "url": "u1"}]
                ),
            }
        )

    _calls, docs = _extract_tool_calls_and_docs(_Out(), frozenset({"search_wikipedia"}))

    assert len(docs) == 1
    assert docs[0].title == "FAISS"
    assert docs[0].metadata["source"] == "search_wikipedia"


def test_non_citeable_tool_output_is_not_cited():
    import json

    from src.internal.servers.web.tool_agent_runner import (
        _extract_tool_calls_and_docs,
    )

    class _Out:
        action_trace = json.dumps(
            {
                "tool_name": "get_weather",
                "status": "completed",
                "arguments": {"location": "Berlin"},
                "result": json.dumps({"temperature": 14.2}),
            }
        )

    _calls, docs = _extract_tool_calls_and_docs(_Out(), frozenset({"search_wikipedia"}))

    assert docs == []


def test_document_ids_are_unique_across_citeable_tools():
    """Two citeable tools in one turn must not both emit D1."""
    import json

    from src.internal.servers.web.tool_agent_runner import (
        _extract_tool_calls_and_docs,
    )

    def _record(name, url):
        return json.dumps(
            {
                "tool_name": name,
                "status": "completed",
                "arguments": {},
                "result": json.dumps([{"title": name, "content": "c", "url": url}]),
            }
        )

    class _Out:
        action_trace = "\n".join(
            [_record("search_wikipedia", "u1"), _record("search_arxiv", "u2")]
        )

    _calls, docs = _extract_tool_calls_and_docs(
        _Out(), frozenset({"search_wikipedia", "search_arxiv"})
    )

    assert [d.id for d in docs] == ["D1", "D2"]


def test_citeable_tool_with_non_conforming_result_yields_no_documents():
    """web_search is citeable but answers with prose; that must not error."""
    import json

    from src.internal.servers.web.tool_agent_runner import (
        _extract_tool_calls_and_docs,
    )

    class _Out:
        action_trace = json.dumps(
            {
                "tool_name": "web_search",
                "status": "completed",
                "arguments": {},
                "result": "1. Some Page\nA prose summary.",
            }
        )

    calls, docs = _extract_tool_calls_and_docs(_Out(), frozenset({"web_search"}))

    assert docs == []
    assert calls[0].tool_name == "web_search"
