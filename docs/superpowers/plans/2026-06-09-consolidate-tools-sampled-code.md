# Tools Sampled-Code Consolidation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Remove the orphaned Onyx-sampled dead code from `src/tools/` so the directory contains only the repo-native tool abstractions that are actually used.

**Architecture:** `src/tools/` currently holds two coexisting systems. The clean system (`base.py`, `api.py`, `parsers.py`, `search.py`) is fully integrated, tested, and used by `src/agents/tool_calling.py`. The sampled system (ten files plus two sub-directories) was copied from the Onyx project and imports `from onyx.*` — a package that does not exist in this repo. Nothing in the working codebase imports from the sampled files; they are dead weight. This plan deletes the dead code and leaves `src/tools/` in a consistent state.

**Tech Stack:** Python 3.12, pytest

---

## File Map

### Will be deleted from `src/tools/`

| File / directory | Reason |
|---|---|
| `interface.py` | Imports `onyx.chat.emitter`, `onyx.server.*`, `onyx.tools.models` — all broken |
| `models.py` | Imports 10+ `onyx.*` symbols — broken |
| `built_in_tools.py` | Imports `onyx.tools.tool_implementations.*` — broken |
| `utils.py` | Imports `onyx.configs.*`, `onyx.db.*` — broken |
| `tool_constructor.py` | Imports `onyx.*` throughout — broken |
| `tool_runner.py` | Imports `onyx.*` throughout — broken |
| `tool_name.py` | Standalone but unused; `src/internal/chat/llm_step.py` has its own copy |
| `constants.py` | Standalone but unused; no working-code consumer |
| `tool_implementations/` | All files import `from onyx.*` — broken |
| `fake_tools/` | All files import `from onyx.*` — broken |

### Will be kept in `src/tools/`

| File | Role |
|---|---|
| `base.py` | `Tool`, `ToolSchema`, `FunctionTool` — core abstractions |
| `api.py` | `ApiToolRegistry` / `ApiRequestTool` — OpenAPI-backed tools (newly added, clean) |
| `parsers.py` | Tool-call parsers used by `ToolAgentLoop` |
| `search.py` | `build_search_tool` / `SearchPage` |
| `__init__.py` | Public re-exports of base + api + parsers + search |

### Unaffected

`src/internal/tools/` — the web-backend tool layer (`models.py`, `built_in_tools.py`, `interface.py`) has no `onyx.*` imports and is used by `src/internal/chat/`. No changes needed there.

---

## Task 1: Verify test baseline before touching anything

**Files:** none changed

- [ ] **Step 1: Run the full unit test suite**

```bash
pytest tests/unit/ -v --tb=short 2>&1 | tail -30
```

Expected: all tests pass (including `tests/unit/test_api_tools.py`).

- [ ] **Step 2: Confirm the sampled files are truly unreferenced**

```bash
grep -r "from.*tools\.interface\|from.*tools\.models\|from.*tools\.tool_constructor\|from.*tools\.tool_runner\|from.*tools\.built_in_tools\|from.*tools\.utils\|from.*tools\.constants\|from.*tool_name\|from.*tool_implementations\|from.*fake_tools" \
  src/ --include="*.py" | grep -v "__pycache__" | grep -v "from onyx\."
```

Expected: output matches only the sampled files themselves cross-referencing each other (no hits inside `src/agents/`, `src/internal/`, or `src/context/`).

- [ ] **Step 3: Commit baseline marker**

```bash
git add -A
git commit -m "chore: confirm baseline before tools dead-code cleanup"
```

---

## Task 2: Delete the Onyx-sampled top-level files

**Files modified:** `src/tools/interface.py`, `models.py`, `built_in_tools.py`, `utils.py`, `tool_constructor.py`, `tool_runner.py`, `constants.py`, `tool_name.py` — all deleted

- [ ] **Step 1: Delete the eight standalone sampled files**

```bash
rm src/tools/interface.py \
   src/tools/models.py \
   src/tools/built_in_tools.py \
   src/tools/utils.py \
   src/tools/tool_constructor.py \
   src/tools/tool_runner.py \
   src/tools/constants.py \
   src/tools/tool_name.py
```

- [ ] **Step 2: Run the test suite to verify nothing broke**

```bash
pytest tests/unit/ -v --tb=short 2>&1 | tail -30
```

Expected: same pass count as Task 1 Step 1.

- [ ] **Step 3: Commit**

```bash
git add -A
git commit -m "chore: remove Onyx-sampled dead-code top-level files from src/tools

interface.py, models.py, built_in_tools.py, utils.py, tool_constructor.py,
tool_runner.py, constants.py, and tool_name.py all imported from onyx.*
(a package that does not exist here) and were not referenced by any working code."
```

---

## Task 3: Delete the `tool_implementations/` sub-directory

**Files modified:** `src/tools/tool_implementations/` — entire directory deleted

- [ ] **Step 1: Delete the directory**

```bash
rm -rf src/tools/tool_implementations/
```

- [ ] **Step 2: Run the test suite**

```bash
pytest tests/unit/ -v --tb=short 2>&1 | tail -30
```

Expected: same pass count as before.

- [ ] **Step 3: Commit**

```bash
git add -A
git commit -m "chore: remove Onyx-sampled tool_implementations/ from src/tools

Every file imported from onyx.* and was unreferenced by any working code.
The OpenAPI-backed-tool use-case is now served by src/tools/api.py."
```

---

## Task 4: Delete the `fake_tools/` sub-directory

**Files modified:** `src/tools/fake_tools/` — entire directory deleted

- [ ] **Step 1: Delete the directory**

```bash
rm -rf src/tools/fake_tools/
```

- [ ] **Step 2: Run the test suite**

```bash
pytest tests/unit/ -v --tb=short 2>&1 | tail -30
```

Expected: same pass count as before.

- [ ] **Step 3: Commit**

```bash
git add -A
git commit -m "chore: remove Onyx-sampled fake_tools/ from src/tools

Both files imported onyx.* and were not used by any working code or tests."
```

---

## Task 5: Final verification

- [ ] **Step 1: Confirm `src/tools/` contains only the intended files**

```bash
find src/tools -name "*.py" | sort
```

Expected output (exactly):
```
src/tools/__init__.py
src/tools/api.py
src/tools/base.py
src/tools/parsers.py
src/tools/search.py
```

- [ ] **Step 2: Confirm no `onyx` imports remain anywhere in `src/tools/`**

```bash
grep -r "from onyx\." src/tools/ --include="*.py"
```

Expected: no output.

- [ ] **Step 3: Run the full test suite one final time**

```bash
pytest tests/unit/ -v 2>&1 | tail -40
```

Expected: all tests green.

- [ ] **Step 4: Final commit**

```bash
git add -A
git commit -m "chore: finalize tools consolidation

src/tools/ now contains only the repo-native abstractions:
  base.py      — Tool / ToolSchema / FunctionTool
  api.py       — ApiToolRegistry / ApiRequestTool (OpenAPI-backed tools)
  parsers.py   — ToolParser / FunctionCall
  search.py    — build_search_tool / SearchPage

All Onyx-sampled dead code (onyx.* imports, unreferenced by working code)
has been removed."
```
