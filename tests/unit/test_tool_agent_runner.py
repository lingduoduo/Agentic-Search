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
