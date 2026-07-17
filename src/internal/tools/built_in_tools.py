"""Built-in tool name sets used by the chat and observability surfaces.

These constants classify tool behavior (citeable / stopping) for the
observability admin surface and chat streaming. ``TOOL_NAME_TO_CLASS`` is a
placeholder name→class map, currently empty (no built-in tool classes are
registered through it in this repo).
"""

from __future__ import annotations

# Tool names that produce citable documents (trigger citation reminders)
CITEABLE_TOOLS_NAMES: set[str] = {"search", "web_search", "open_url"}

# Tool names that stop the loop after running (e.g. image generation)
STOPPING_TOOLS_NAMES: set[str] = {"image_generation"}

# Placeholder name→class map (empty; no built-in tool classes are registered)
TOOL_NAME_TO_CLASS: dict[str, type] = {}
