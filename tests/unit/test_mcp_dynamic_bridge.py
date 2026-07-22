"""Regression test: the dynamic MCP tool bridge must not crash on import
when the global tool_registry holds a tool whose wrapper fastmcp rejects.

fastmcp raises ``ValueError: Functions with **kwargs are not supported as
tools`` for any function that only accepts **kwargs. dynamic._make_mcp_wrapper
builds exactly such a function, so registering ANY real tool from the global
tool_registry currently crashes _sync_all() (and thus the import of
src.internal.mcp_server.api, which every other MCP test transitively imports).
"""

from __future__ import annotations

from src.internal.db.store import AgenticSearchStore
from src.internal.memory.tools import build_memory_registry
from src.internal.mcp_server.tools import dynamic
from src.tools.registry import tool_registry


def test_sync_all_skips_unmirrorable_tool_without_raising():
    store = AgenticSearchStore(":memory:")
    registry, _counts, _schemas = build_memory_registry(store, "u1")
    tool = registry.get("add_memory")
    assert tool is not None

    tool_registry.register(tool)
    try:
        count = dynamic._sync_all()
    finally:
        tool_registry.unregister(tool.name)
        store.close()

    assert isinstance(count, int)
