"""Bridge: expose all tools from the global ToolRegistry as FastMCP tools.

Import-time effect: imports this module (from api.py) registers every tool that
is already in ``tool_registry`` as an MCP tool.

Runtime effect: call ``sync_tool_to_mcp(name)`` after registering a new tool
at runtime (e.g. from the REST tools API) to make it available over MCP without
restarting the server.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

# Deferred imports to avoid circular dependency:
# api.py creates mcp_server first, then imports this module.
from src.internal.mcp_server.api import mcp_server  # noqa: E402
from src.internal.tools.registry import tool_registry  # noqa: E402


def _make_mcp_wrapper(tool_name: str):
    """Factory that creates a wrapper function for one tool.

    Using a factory (instead of a closure capturing a loop variable) prevents
    the classic late-binding bug where all wrappers refer to the last tool.
    """

    async def _wrapper(**kwargs: Any) -> str:
        response, _raw, errors = await tool_registry.invoke(tool_name, kwargs)
        if errors:
            return "Error: " + "; ".join(errors)
        return response

    _wrapper.__name__ = tool_name
    return _wrapper


def sync_tool_to_mcp(name: str) -> bool:
    """Register a single tool from tool_registry into the MCP server.

    Returns True if the tool was found and registered, False otherwise.
    """
    tool = tool_registry.get(name)
    if tool is None:
        logger.warning("sync_tool_to_mcp: tool %r not found in registry", name)
        return False
    wrapper = _make_mcp_wrapper(name)
    wrapper.__doc__ = tool.schema.description or name
    try:
        mcp_server.tool()(wrapper)
    except Exception as exc:
        logger.warning("sync_tool_to_mcp: could not mirror tool %r: %s", name, exc)
        return False
    logger.debug("MCP tool registered dynamically: %s", name)
    return True


def _exportable_entries(registry=None) -> list:
    """Registry entries this bridge may mirror out to MCP.

    Tools pulled *in* from a remote MCP server (``source="mcp"``) are excluded:
    re-exporting them would offer a server its own tools back and duplicate the
    catalog on every restart.
    """
    from src.internal.tools.mcp_client import MCP_SOURCE

    entries = (registry or tool_registry).list()
    return [e for e in entries if e.source != MCP_SOURCE]


def _sync_all() -> int:
    """Register all tools currently in tool_registry with MCP. Called at import time."""
    count = 0
    for entry in _exportable_entries():
        wrapper = _make_mcp_wrapper(entry.tool.name)
        wrapper.__doc__ = entry.tool.schema.description or entry.tool.name
        try:
            mcp_server.tool()(wrapper)
        except Exception as exc:
            logger.warning(
                "_sync_all: could not mirror tool %r: %s", entry.tool.name, exc
            )
            continue
        count += 1
    if count:
        logger.info(
            "MCP bridge: registered %d dynamic tool(s) from ToolRegistry", count
        )
    return count


# Register tools that already exist in the registry at import time.
_sync_all()
