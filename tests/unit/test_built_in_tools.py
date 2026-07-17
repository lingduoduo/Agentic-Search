from __future__ import annotations

import src.internal.tools.built_in_tools as bit


def test_live_symbols_present():
    assert bit.CITEABLE_TOOLS_NAMES == {"search", "web_search", "open_url"}
    assert bit.STOPPING_TOOLS_NAMES == {"image_generation"}
    assert bit.TOOL_NAME_TO_CLASS == {}


def test_dead_stub_symbols_removed():
    for name in (
        "SearchTool",
        "WebSearchTool",
        "PythonTool",
        "OpenURLTool",
        "ImageGenerationTool",
        "MemoryTool",
        "run_tool_calls",
        "extract_url_snippet_map",
        "_ParallelToolCallResults",
    ):
        assert not hasattr(bit, name), f"{name} should be removed"


def test_consumers_still_import():
    import src.internal.chat.tool_call_args_streaming  # noqa: F401
    import src.internal.observability.admin_surface  # noqa: F401
