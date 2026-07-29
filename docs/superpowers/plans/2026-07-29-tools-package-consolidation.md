# Tools Package Consolidation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Move the 11-module agent tool framework from `src/tools/` into `src/internal/tools/`, repoint all 94 references, and delete `src/tools/`.

**Architecture:** A three-task sequence, each ending green. Task 1 is the atomic move — the files, every reference, and the merged `__init__.py` must land together or the suite is red, so they form one task. Task 2 removes the dead re-exports the move exposes; a reviewer can reject it while keeping the move. Task 3 syncs the two live doc lines.

**Tech Stack:** Python 3.12, pytest 2,732-test regression suite, ruff, `git mv` for rename detection, BSD `sed` (macOS).

Spec: `docs/superpowers/specs/2026-07-29-tools-package-consolidation-design.md`

## Global Constraints

- **Behavior-preserving.** No functional change in any task. The existing suite is the regression net.
- **Baseline:** `pytest` → **2,732 passed** in ~77s at `ff091bc`. Task 1 adds 15 tests (→ **2,747**), Task 2 adds 10 more (→ **2,757**). No pre-existing test may be lost; a count below `2,732 + new` means a test module failed to import.
- **`from .tools import ...` is not always this package.** `src/internal/mcp_server/api.py:34-39` has six legitimate relative imports of `src/internal/mcp_server/tools/`, an unrelated package. Never blanket-rewrite the relative form — only `src/__init__.py` lines 64–71 need it.
- **Branch:** `chore/consolidate-tools-package`. Never commit to `main`.
- **BSD sed:** on macOS, in-place edit is `sed -i ''` (with the empty-string argument). BSD sed does **not** support `\b` word boundaries — do not use them.
- **Out of scope** (do not do these): pruning the export list beyond the 5 named dead re-exports; reconciling the two same-named `OpenAPISchema` types (`openapi_schema.py` Pydantic model vs. `api.py:33` frozen dataclass); relocating `interface.py`/`built_in_tools.py`; editing anything under `docs/superpowers/` other than checking off this plan; touching `src/internal/servers/tools/` or `src/internal/mcp_server/tools/` beyond their import lines.
- **Leave alone:** the untracked files `src/internal/document_index/base.py` and `src/internal/document_index/document_processing.py`. They belong to a parallel agent and are unrelated to this work. Never `git add -A`.
- **Concurrent work:** a parallel agent is live in `.worktrees/mcp-document-extraction`. Do not touch that worktree.

## Spec deviation (deliberate)

The spec says "`__all__` is updated to match the merged surface." This plan instead **drops `__all__`** from the merged `__init__.py` and uses the explicit `from .x import Y as Y` re-export idiom throughout. Reason: the 33 lines being carried over from `src/tools/__init__.py` already use that idiom and have no `__all__`; ruff treats `X as X` as an explicit re-export, so `__all__` would be a second, redundant list to keep in sync. Task 2 amends that one sentence in the spec.

---

### Task 1: Move the package and repoint every reference

The whole move is atomic. `git mv` the 11 modules, rewrite every `src.tools` reference repo-wide, merge the two `__init__.py` files, and delete `src/tools/`. The suite must be green at the end of this task.

**Files:**
- Move (11): `src/tools/{api,base,html_text,knowledge_base,openapi_schema,parsers,registry,routing_tools,search,semantic_router,validation}.py` → `src/internal/tools/`
- Delete: `src/tools/__init__.py`, then the `src/tools/` directory
- Modify: `src/internal/tools/__init__.py` (merged surface)
- Modify (11 production): `src/__init__.py`, `src/agents/tool/tool_calling.py`, `src/context/retrieval/search_runner.py`, `src/internal/mcp_server/tools/dynamic.py`, `src/internal/mcp_server/tools/search.py`, `src/internal/memory/tools.py`, `src/internal/servers/tools/api.py`, `src/internal/servers/web/app.py`, `src/internal/servers/web/debug_router.py`, `src/internal/servers/web/tool_agent_runner.py`, `src/internal/servers/web_search/api.py`
- Modify (21 tests): `tests/unit/servers/web/{test_browser_pipeline,test_debug_tools,test_hybrid_web_fallback,test_reranking,test_tool_admin_api,test_web_experience_app}.py`, `tests/unit/{test_api_tools,test_intent_routing,test_knowledge_base,test_mcp_dynamic_bridge,test_mcp_server,test_on_turn_callback,test_search_filters_plumbing,test_search_tools,test_semantic_router,test_state_models,test_tool_approval,test_tool_arg_validation,test_tool_categories,test_tool_registry,test_web_cascade_search}.py`
- Test: `tests/unit/test_tools_package_layout.py` (create)

**Interfaces:**
- Consumes: nothing (first task).
- Produces: the package `src.internal.tools`, re-exporting `Tool`, `FunctionTool`, `ToolSchema`, `ToolEffect`, `ToolEntry`, `ToolRegistry`, `tool`, `tool_registry`, `FunctionCall`, `ToolParser`, `HermesToolParser`, `JSONToolParser`, `Llama3ToolParser`, `ApiRequestTool`, `ApiToolError`, `ApiToolNotFoundError`, `ApiToolProviderSpec`, `ApiToolRegistry`, `ApiToolSpec`, `SearchPage`, `build_search_tool`, `fetch_pages_concurrently`, `fetch_url`, `format_search_pages`, `search_tool`, `MultiQueryWebSearchTool`, `serper_dev_search`, `OpenAPISchema`, `ParameterIn`, `ParameterType`, `ParameterTypeMap`, `build_search_routing_tool`, `build_rag_routing_tool`, `ChatTool`. Submodules `src.internal.tools.{api,base,html_text,knowledge_base,openapi_schema,parsers,registry,routing_tools,search,semantic_router,validation,interface,built_in_tools}`.

- [ ] **Step 1: Confirm the starting point**

```bash
git branch --show-current   # must print: chore/consolidate-tools-package
git status --short          # only the 2 untracked document_index files
python3 -m pytest -q 2>&1 | tail -2   # must print: 2732 passed
```

- [ ] **Step 2: Write the failing test**

Create `tests/unit/test_tools_package_layout.py`:

```python
"""Pins the consolidated tools package surface at src.internal.tools."""

import importlib

import pytest

_RE_EXPORTS = (
    "Tool",
    "FunctionTool",
    "ToolSchema",
    "ToolEffect",
    "ToolEntry",
    "ToolRegistry",
    "tool",
    "tool_registry",
    "FunctionCall",
    "ToolParser",
    "HermesToolParser",
    "JSONToolParser",
    "Llama3ToolParser",
    "ApiRequestTool",
    "ApiToolError",
    "ApiToolNotFoundError",
    "ApiToolProviderSpec",
    "ApiToolRegistry",
    "ApiToolSpec",
    "SearchPage",
    "build_search_tool",
    "fetch_pages_concurrently",
    "fetch_url",
    "format_search_pages",
    "search_tool",
    "MultiQueryWebSearchTool",
    "serper_dev_search",
    "OpenAPISchema",
    "ParameterIn",
    "ParameterType",
    "ParameterTypeMap",
    "build_search_routing_tool",
    "build_rag_routing_tool",
    "ChatTool",
)

_SUBMODULES = (
    "api",
    "base",
    "built_in_tools",
    "html_text",
    "interface",
    "knowledge_base",
    "openapi_schema",
    "parsers",
    "registry",
    "routing_tools",
    "search",
    "semantic_router",
    "validation",
)


def test_framework_surface_re_exported_from_internal_tools():
    mod = importlib.import_module("src.internal.tools")
    missing = [name for name in _RE_EXPORTS if not hasattr(mod, name)]
    assert missing == [], f"missing re-exports: {missing}"


@pytest.mark.parametrize("name", _SUBMODULES)
def test_submodule_lives_under_internal_tools(name):
    importlib.import_module(f"src.internal.tools.{name}")


def test_legacy_src_tools_package_is_gone():
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("src.tools")
```

- [ ] **Step 3: Run it and watch it fail**

```bash
python3 -m pytest tests/unit/test_tools_package_layout.py -q 2>&1 | tail -5
```

Expected: FAIL. `test_framework_surface_re_exported_from_internal_tools` fails on missing re-exports, the `submodule` cases fail with `ModuleNotFoundError` for the 11 not-yet-moved names, and `test_legacy_src_tools_package_is_gone` fails because `src.tools` still imports.

- [ ] **Step 4: Move the 11 modules with `git mv`**

`git mv` (not `mv`) so the diff records renames instead of 2,800 added + 2,800 deleted lines.

```bash
cd /Users/linghuang/Git/Agentic-Search
for f in api base html_text knowledge_base openapi_schema parsers registry \
         routing_tools search semantic_router validation; do
  git mv "src/tools/$f.py" "src/internal/tools/$f.py"
done
git rm -q src/tools/__init__.py
rm -rf src/tools            # removes the directory and its stale __pycache__
test ! -d src/tools && echo "src/tools gone"
```

- [ ] **Step 5: Repoint all 94 references**

One pass over every `.py` file that mentions `src.tools`. This catches ordinary
imports **and** the 13 string-literal `patch()` / `monkeypatch.setattr()`
targets in the same sweep, which is the point — those are invisible to
import-based tooling.

`src.tools` cannot match inside `src.internal.tools`, so the substitution is not
self-applying and is safe to run once.

```bash
cd /Users/linghuang/Git/Agentic-Search
grep -rl 'src\.tools' --include='*.py' src/ tests/ examples/ \
  | grep -v __pycache__ \
  | xargs sed -i '' 's/src\.tools/src.internal.tools/g'
```

This also fixes the two absolute self-references inside the moved
`registry.py` (line 12, a docstring usage example; line 47, a `TYPE_CHECKING`
import of `OpenAPISchema`), and the 8 re-export lines in `src/__init__.py`
(`from .tools.base` / `from .tools.parsers`) — note those use the *relative*
form, so handle them in the next step.

- [ ] **Step 6: Fix the relative re-exports in `src/__init__.py`**

The sweep in Step 5 only rewrites the absolute `src.tools` form. `src/__init__.py`
uses relative imports, which it misses. Replace lines 64–71:

```python
from .tools.base import FunctionTool as FunctionTool
from .tools.base import Tool as Tool
from .tools.base import ToolSchema as ToolSchema
from .tools.parsers import FunctionCall as FunctionCall
from .tools.parsers import HermesToolParser as HermesToolParser
from .tools.parsers import JSONToolParser as JSONToolParser
from .tools.parsers import Llama3ToolParser as Llama3ToolParser
from .tools.parsers import ToolParser as ToolParser
```

with:

```python
from .internal.tools.base import FunctionTool as FunctionTool
from .internal.tools.base import Tool as Tool
from .internal.tools.base import ToolSchema as ToolSchema
from .internal.tools.parsers import FunctionCall as FunctionCall
from .internal.tools.parsers import HermesToolParser as HermesToolParser
from .internal.tools.parsers import JSONToolParser as JSONToolParser
from .internal.tools.parsers import Llama3ToolParser as Llama3ToolParser
from .internal.tools.parsers import ToolParser as ToolParser
```

Verify no relative form survives **in this file only**:

```bash
grep -n 'from \.tools' src/__init__.py && echo "FAIL: relative refs remain" || echo "PASS: src/__init__.py clean"
```

Scope the check to `src/__init__.py`. A repo-wide `from \.tools` grep matches
`src/internal/mcp_server/api.py:34-39`, which relatively imports
`src/internal/mcp_server/tools/` — a different package that must not be touched.

- [ ] **Step 7: Write the merged `__init__.py`**

Overwrite `src/internal/tools/__init__.py` with the 33 re-exports carried over
from the old `src/tools/__init__.py`, plus `ChatTool`. The five chat-model
re-exports (`ChatFile`, `SearchToolUsage`, `ToolCallInfo`, `ToolCallKickoff`,
`ToolResponse`) are **kept for now** — Task 2 removes them, so that the move and
the cleanup are separately reviewable.

```python
"""Agent tool framework: schemas, registry, parsers, and built-in tools.

Also hosts the chat-loop ``ChatTool`` interface (``interface``) and the
built-in tool name sets (``built_in_tools``), which consumers import as
submodules.
"""

from .base import FunctionTool as FunctionTool
from .base import Tool as Tool
from .base import ToolSchema as ToolSchema
from .base import ToolEffect as ToolEffect
from .parsers import FunctionCall as FunctionCall
from .parsers import HermesToolParser as HermesToolParser
from .parsers import JSONToolParser as JSONToolParser
from .parsers import Llama3ToolParser as Llama3ToolParser
from .parsers import ToolParser as ToolParser
from .api import ApiRequestTool as ApiRequestTool
from .api import ApiToolError as ApiToolError
from .api import ApiToolNotFoundError as ApiToolNotFoundError
from .api import ApiToolProviderSpec as ApiToolProviderSpec
from .api import ApiToolRegistry as ApiToolRegistry
from .api import ApiToolSpec as ApiToolSpec
from .search import SearchPage as SearchPage
from .search import build_search_tool as build_search_tool
from .search import fetch_pages_concurrently as fetch_pages_concurrently
from .search import fetch_url as fetch_url
from .search import format_search_pages as format_search_pages
from .search import search_tool as search_tool
from .search import MultiQueryWebSearchTool as MultiQueryWebSearchTool
from .search import serper_dev_search as serper_dev_search
from .registry import ToolEntry as ToolEntry
from .registry import ToolRegistry as ToolRegistry
from .registry import tool as tool
from .registry import tool_registry as tool_registry
from .openapi_schema import OpenAPISchema as OpenAPISchema
from .openapi_schema import ParameterIn as ParameterIn
from .openapi_schema import ParameterType as ParameterType
from .openapi_schema import ParameterTypeMap as ParameterTypeMap
from .routing_tools import build_search_routing_tool as build_search_routing_tool
from .routing_tools import build_rag_routing_tool as build_rag_routing_tool
from .interface import ChatTool as ChatTool

# Removed in Task 2 — dead re-exports, canonical home is
# src.internal.chat.tool_models.
from src.internal.chat.tool_models import (
    ChatFile as ChatFile,
    SearchToolUsage as SearchToolUsage,
    ToolCallInfo as ToolCallInfo,
    ToolCallKickoff as ToolCallKickoff,
    ToolResponse as ToolResponse,
)
```

- [ ] **Step 8: Run the new test — it must pass**

```bash
python3 -m pytest tests/unit/test_tools_package_layout.py -q 2>&1 | tail -3
```

Expected: PASS (15 tests — 1 surface + 13 parametrized submodules + 1 legacy-gone).

- [ ] **Step 9: Prove no reference survives**

```bash
cd /Users/linghuang/Git/Agentic-Search
grep -rn 'src\.tools' --include='*.py' src/ tests/ examples/ | grep -v __pycache__ && echo "FAIL: references remain" || echo "PASS: zero src.tools references"
test ! -d src/tools && echo "PASS: src/tools removed"
python3 -c "import src; print('import src OK')"
```

- [ ] **Step 10: Run the full suite**

```bash
python3 -m pytest -q 2>&1 | tail -3
```

Expected: **2,747 passed** (2,732 baseline + 15 new). If the number is lower than 2,732 + new, a test module failed to import — find it with `python3 -m pytest --collect-only -q 2>&1 | tail -5`.

- [ ] **Step 11: Lint**

```bash
ruff check . --fix && ruff format .
python3 -m pytest -q 2>&1 | tail -2   # re-confirm after any autofix
```

- [ ] **Step 12: Commit**

Stage with `git add -u` (tracked modifications, renames and deletions only)
plus the one new file by name. Do **not** use `git add src/` or `git add -A` —
either would sweep in the parallel agent's untracked
`src/internal/document_index/{base,document_processing}.py`.

```bash
cd /Users/linghuang/Git/Agentic-Search
git add -u
git add tests/unit/test_tools_package_layout.py
git status --short   # the 2 document_index files MUST still show as ?? untracked
git commit -m "refactor: move src/tools into src/internal/tools

Relocate the 11-module agent tool framework under src/internal/, merging it
with the chat-heritage ChatTool interface and built-in tool name sets already
there. Repoint 94 references across 32 files, including 13 string-literal
patch() targets. Delete src/tools.

Behavior-preserving; the merged __init__.py carries the same surface.

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>"
```

---

### Task 2: Drop the dead chat-model re-exports

`src/internal/tools/__init__.py` re-exports five names from
`src.internal.chat.tool_models`. Nothing imports the `src.internal.tools`
package root for them — every consumer imports a submodule directly
(`src.internal.tools.built_in_tools`, `src.internal.tools.interface`). Remove
them; they stay importable from their canonical home.

This is the seam where a reviewer can keep Task 1 and reject the cleanup.

**Files:**
- Modify: `src/internal/tools/__init__.py`
- Modify: `docs/superpowers/specs/2026-07-29-tools-package-consolidation-design.md` (the `__all__` sentence)
- Test: `tests/unit/test_tools_package_layout.py` (extend)

**Interfaces:**
- Consumes: `src.internal.tools` package from Task 1.
- Produces: no new names. `ChatFile`, `SearchToolUsage`, `ToolCallInfo`, `ToolCallKickoff`, `ToolResponse` are no longer attributes of `src.internal.tools`; they remain importable from `src.internal.chat.tool_models`.

- [ ] **Step 1: Write the failing test**

Append to `tests/unit/test_tools_package_layout.py`:

```python
_CHAT_MODELS = (
    "ChatFile",
    "SearchToolUsage",
    "ToolCallInfo",
    "ToolCallKickoff",
    "ToolResponse",
)


@pytest.mark.parametrize("name", _CHAT_MODELS)
def test_chat_models_not_re_exported_from_tools(name):
    """These belong to src.internal.chat.tool_models, not the tools package."""
    mod = importlib.import_module("src.internal.tools")
    assert not hasattr(mod, name), f"{name} should not be re-exported here"


@pytest.mark.parametrize("name", _CHAT_MODELS)
def test_chat_models_importable_from_canonical_home(name):
    mod = importlib.import_module("src.internal.chat.tool_models")
    assert hasattr(mod, name)
```

- [ ] **Step 2: Run it and watch it fail**

```bash
python3 -m pytest tests/unit/test_tools_package_layout.py -q 2>&1 | tail -5
```

Expected: the 5 `test_chat_models_not_re_exported_from_tools` cases FAIL (the names are still re-exported); the 5 `canonical_home` cases PASS.

- [ ] **Step 3: Remove the re-export block**

Delete these final lines from `src/internal/tools/__init__.py`:

```python
# Removed in Task 2 — dead re-exports, canonical home is
# src.internal.chat.tool_models.
from src.internal.chat.tool_models import (
    ChatFile as ChatFile,
    SearchToolUsage as SearchToolUsage,
    ToolCallInfo as ToolCallInfo,
    ToolCallKickoff as ToolCallKickoff,
    ToolResponse as ToolResponse,
)
```

The file now ends at `from .interface import ChatTool as ChatTool`.

- [ ] **Step 4: Run the tests**

```bash
python3 -m pytest tests/unit/test_tools_package_layout.py -q 2>&1 | tail -3
python3 -m pytest -q 2>&1 | tail -3
```

Expected: layout file PASS (25 tests); full suite **2,757 passed** (2,747 from Task 1 + 10 new).

- [ ] **Step 5: Amend the spec's `__all__` sentence**

In `docs/superpowers/specs/2026-07-29-tools-package-consolidation-design.md`,
replace:

```
`__all__` is updated to match the merged surface.
```

with:

```
The merged file uses the explicit `from .x import Y as Y` re-export idiom
already used by the 33 carried-over lines, and carries no `__all__` — ruff
treats `Y as Y` as an explicit re-export, so a second list would only be one
more thing to keep in sync.
```

- [ ] **Step 6: Lint and commit**

```bash
cd /Users/linghuang/Git/Agentic-Search
ruff check . --fix && ruff format .
git add src/internal/tools/__init__.py tests/unit/test_tools_package_layout.py docs/superpowers/specs/2026-07-29-tools-package-consolidation-design.md
git commit -m "refactor(tools): drop dead chat-model re-exports from the tools package

ChatFile, SearchToolUsage, ToolCallInfo, ToolCallKickoff and ToolResponse were
re-exported from src.internal.tools with no importer — every consumer uses a
submodule directly. They remain importable from src.internal.chat.tool_models.

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>"
```

---

### Task 3: Sync the two live doc lines

Only two documentation lines describe the old layout. Everything else matching
`src/tools` under `docs/superpowers/` is a point-in-time record of already-merged
work — rewriting it would falsify history. Leave it.

**Files:**
- Modify: `docs/mcp.md:82`, `docs/tool-engine.md:56`

**Interfaces:**
- Consumes: the final layout from Tasks 1–2.
- Produces: nothing consumed by later tasks.

- [ ] **Step 1: Update `docs/mcp.md`**

Line 82 — replace:

```
runnable tools. `src/tools/knowledge_base.py` provides the built-in seed set, and
```

with:

```
runnable tools. `src/internal/tools/knowledge_base.py` provides the built-in seed set, and
```

- [ ] **Step 2: Update `docs/tool-engine.md`**

Line 56 — replace:

```
runnable tools. `src/tools/knowledge_base.py` provides the built-in seed set, and
```

with:

```
runnable tools. `src/internal/tools/knowledge_base.py` provides the built-in seed set, and
```

- [ ] **Step 3: Verify no live doc still points at the old path**

```bash
cd /Users/linghuang/Git/Agentic-Search
grep -rn 'src/tools' --include='*.md' docs/ README.md 2>/dev/null | grep -v 'docs/superpowers/' && echo "FAIL: live doc refs remain" || echo "PASS: live docs clean"
```

- [ ] **Step 4: Commit**

```bash
git add docs/mcp.md docs/tool-engine.md
git commit -m "docs: point tool-engine and mcp docs at src/internal/tools

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>"
```

---

## Final verification

Run after all three tasks:

```bash
cd /Users/linghuang/Git/Agentic-Search
python3 -c "import src; print('import src OK')"
python3 -c "import src.internal.tools as t; print(t.Tool, t.ChatTool)"
python3 -m pytest -q 2>&1 | tail -2                     # 2,757 passed
grep -rn 'src\.tools' --include='*.py' src/ tests/ examples/ | grep -v __pycache__ || echo "zero src.tools refs"
test ! -d src/tools && echo "src/tools removed"
ruff check . && ruff format --check .
cd web && npm run typecheck                              # unaffected; confirms no incidental breakage
```

Then push and open a PR against `main`.
