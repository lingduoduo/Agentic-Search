# Tool Category Flags (citeable / stopping) — Design

Date: 2026-07-09
Status: Approved (brainstorming)
Branch/PR: feat/tool-category-flags
Related: [[project_chat_orchestration]]; first step of reconciling `src/internal/tools/` (stub layer) onto the canonical `src/tools/` registry

## Context (the big picture, plain)

There are two tool systems: **`src/tools/`** (the real one — `Tool`/`ToolRegistry`,
used by the REST API, web app, MCP, and `ToolAgentLoop`) and **`src/internal/tools/`**
(mostly-empty stub scaffolding — `ChatTool`, `built_in_tools.py`). We are making
`src/tools/` the single source of truth.

The one real concept the stub layer has that the canonical `Tool` lacks: **tool
categories** — `CITEABLE_TOOLS_NAMES = {search, web_search, open_url}` (produce
citable docs) and `STOPPING_TOOLS_NAMES = {image_generation}` (stop the loop),
hardcoded name-sets in `built_in_tools.py`.

## This PR (scoped, additive)

Teach the canonical `Tool` those two facts, and have the real tools declare them.
**Nothing consumes the flags yet** — this is the safe foundation. (The natural
consumer, `admin_surface.py` deriving the sets from the registry, is deferred:
today the built-in tools are NOT registered in the global `tool_registry`, so a
registry-derived count would be wrong. That migration is a separate, larger step.)

## Non-goals

- No change to `admin_surface.py` / `tool_call_args_streaming.py` (deferred).
- No removal of `built_in_tools.py` name-sets yet (still the source of truth for
  admin until the migration).
- Flags are **not** added to `ToolSchema.to_dict()` — they are tool metadata, not
  part of the JSON function-definition the model sees (like `effect`).

## Approach (all in `src/tools/`)

1. **`Tool` ABC** (`base.py`) — add two properties defaulting to `False`, mirroring
   the existing `effect` property:
   ```python
   @property
   def citeable(self) -> bool:
       return False

   @property
   def stopping(self) -> bool:
       return False
   ```

2. **`FunctionTool`** (`base.py`) — accept `citeable: bool = False`,
   `stopping: bool = False` in `__init__`; expose via the properties.

3. **`@tool_registry.tool(...)`** (`registry.py`) — add `citeable`/`stopping`
   params, threaded into the `FunctionTool` it builds.

4. **Declare on the real tools** (`search.py`):
   - `MultiQueryWebSearchTool.citeable` → `True` (name `web_search`).
   - `build_search_tool` → build with `citeable=True` (name `search`).

## Success criteria

- `Tool` exposes `citeable`/`stopping` (default `False`); `FunctionTool` and the
  `@tool` decorator accept them.
- The real `search` and `web_search` tools report `citeable is True`.
- `ToolSchema.to_dict()` is unchanged (flags not leaked to the model).
- Existing tool tests stay green.

## Testing (no model)

- `FunctionTool(..., citeable=True, stopping=True)` → the properties reflect it;
  a default `FunctionTool` → both `False`.
- `@tool_registry.tool(citeable=True)` registers a tool whose `.citeable is True`.
- `build_search_tool(...).citeable is True`; a `MultiQueryWebSearchTool().citeable
  is True`.
- `ToolSchema.to_dict()` has no `citeable`/`stopping` keys.

## Risks

- Minimal — purely additive metadata with `False` defaults. The only "value" is
  that the canonical `search`/`web_search` tools now self-describe their category,
  ready for the later admin migration.
