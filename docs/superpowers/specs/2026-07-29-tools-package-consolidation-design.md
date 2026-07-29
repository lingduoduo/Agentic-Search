# Consolidate `src/tools` into `src/internal/tools`

**Date:** 2026-07-29
**Status:** Approved
**Baseline:** `ff091bc` (origin/main) — 2,732 tests collect; `pytest` run clean
(2,732 passed, 77s)

Related: [[project_tools_consolidation]], [[project_readme_feature_simplification_sweep]]

## Problem

The repo has two packages named `tools`:

- **`src/tools/`** — the real agent tool framework. 11 modules, ~2,800 LOC:
  `api`, `base`, `html_text`, `knowledge_base`, `openapi_schema`, `parsers`,
  `registry`, `routing_tools`, `search`, `semantic_router`, `validation`, plus a
  33-line re-export `__init__.py`. Consumed by 11 production files and 21 test
  files.
- **`src/internal/tools/`** — 37 LOC of Onyx-heritage chat scaffolding:
  `interface.py` (the `ChatTool` ABC, used only by
  `src/internal/chat/tool_call_args_streaming.py`) and `built_in_tools.py`
  (name-set constants, used by chat streaming and
  `src/internal/observability/admin_surface.py`).

Every other subsystem in this repo lives under `src/internal/`. `src/tools/`
being a sibling of `src/internal/` rather than a member of it is an accident of
history, not a boundary anyone relies on.

The split has already produced concrete tangle. `src/internal/tools/__init__.py`
re-exports `OpenAPISchema` *across packages* from `src.tools.openapi_schema`,
so the small package depends on the large one purely to re-export a name that
nobody imports from it.

## Goal

One tools package, at `src/internal/tools/`. `src/tools/` ceases to exist.

Behavior is unchanged. This is a relocation, plus removal of the redundancy the
relocation exposes — nothing more.

## Non-Goals

Explicitly out of scope, decided during design:

- **Pruning the export list.** The merged `__init__.py` carries the existing 33
  re-exports forward as-is, even where a name has no importer today.
- **Reconciling the two `OpenAPISchema` types.** `openapi_schema.py` defines a
  Pydantic model validating user-supplied tool specs;
  `api.py:33` defines a frozen dataclass holding a parsed subset used by
  `ApiRequestTool`. Same name, genuinely different roles. Confusing, but a
  separate change — folding it in would make a mechanical move impossible to
  review.
- **Relocating `interface.py` / `built_in_tools.py`** to `src/internal/chat/`.
  Considered and rejected: it buys separation of the two "tool" concepts at the
  cost of touching three extra consumer files, for 37 LOC.
- **Rewriting historical docs.** See "Documentation" below.
- **`src/internal/servers/tools/` and `src/internal/mcp_server/tools/`** are
  unrelated packages that merely share the word "tools". Untouched.

## Design

### 1. The move

`git mv` all 11 modules from `src/tools/` into `src/internal/tools/`, where they
join `interface.py` and `built_in_tools.py`. There are no filename collisions
between the two sets.

`git mv` (rather than delete + add) preserves rename detection, so the diff
reads as 11 moves instead of 2,800 added and 2,800 deleted lines.

The modules' *relative* imports travel unchanged — `registry.py` does
`from .validation import validate_arguments`, `search.py` does
`from .html_text import _html_to_text`, `knowledge_base.py` does
`from .routing_tools import ...`. Because the whole set moves together, these
keep resolving.

Two *absolute* self-references inside `registry.py` do need repointing:

- line 12 — a docstring usage example, `from src.tools.registry import tool, tool_registry`
- line 47 — a `TYPE_CHECKING` import, `from src.tools.openapi_schema import OpenAPISchema`

### 2. The merged `__init__.py`

`src/internal/tools/__init__.py` becomes:

- the 33 re-exports carried over verbatim from the old `src/tools/__init__.py`
  (`base`, `parsers`, `api`, `search`, `registry`, `openapi_schema`,
  `routing_tools`), and
- `from .interface import ChatTool`, retained from the current file.

Two things fall away as a direct consequence of the move:

- **The cross-package OpenAPI import.** `from src.tools.openapi_schema import ...`
  becomes an in-package `from .openapi_schema import ...`. The dependency edge
  from `src/internal/tools` to `src/tools` disappears because the target is now
  a sibling module.
- **Five dead chat-model re-exports.** `ChatFile`, `SearchToolUsage`,
  `ToolCallInfo`, `ToolCallKickoff`, `ToolResponse` are re-exported from
  `src.internal.chat.tool_models`. Nothing imports the `src.internal.tools`
  package root — every consumer imports a submodule directly
  (`src.internal.tools.built_in_tools`, `src.internal.tools.interface`). They
  are removed; they remain importable from their canonical home,
  `src.internal.chat.tool_models`.

`__all__` is updated to match the merged surface.

### 3. Repointing consumers

94 occurrences of `src.tools` across 11 production files and 21 test files.
They fall into two categories, and the second is the one that breaks silently.

**Ordinary imports** — mechanical. Production consumers:

```
src/__init__.py                                  (lines 64-71)
src/agents/tool/tool_calling.py
src/context/retrieval/search_runner.py
src/internal/memory/tools.py
src/internal/mcp_server/tools/dynamic.py
src/internal/mcp_server/tools/search.py
src/internal/servers/tools/api.py
src/internal/servers/web/app.py
src/internal/servers/web/debug_router.py
src/internal/servers/web/tool_agent_runner.py
src/internal/servers/web_search/api.py
```

`src/__init__.py` re-exports `from .tools.base import ...` and
`from .tools.parsers import ...`; these become `.internal.tools.*`.

**13 string-literal patch targets** in `patch(...)` / `monkeypatch.setattr(...)`,
for example:

```
"src.tools.search.retrieval_search"
"src.tools.search.search_for_tool_string"
"src.tools.api.aiohttp.ClientSession"
"src.tools.routing_tools.search_tool"
```

These are invisible to import-based tooling and to `ruff`. A stale target either
raises `ModuleNotFoundError` at patch time or — worse, where a module is still
importable — patches the wrong object and lets a test pass while asserting
nothing. They get an explicit dedicated grep pass, not just a find-and-replace
over import lines.

This is a known repeat failure in this repo: moving code has broken
`patch()` string targets in prior refactors (`src/agents` core move,
`index_builder` split).

### 4. Documentation

Two live doc lines describe the current layout and are updated:

- `docs/mcp.md:82`
- `docs/tool-engine.md:56`

Everything else matching `src/tools` under `docs/superpowers/` — plans, specs,
context-packs, and `archive/` — is a point-in-time record of work already
merged. Those documents describe what the tree looked like when that work
shipped. Rewriting them would falsify the record, so they are left untouched.

`README.md` and `.claude/CLAUDE.md` contain no `src/tools` references; no change
needed.

### 5. Packaging

`pyproject.toml` uses `[tool.setuptools.packages.find] include = ["src*"]` — a
glob, not an explicit package list. No packaging change is required.

`pytest` sets `pythonpath = ["src"]`, which makes `src/tools/` *also* importable
bare as `tools`. Verified: no file imports bare `tools` or bare `internal.tools`,
so this path is unused and creates no additional work.

## Verification

Baseline recorded at `ff091bc`: `pytest` → 2,732 passed in 77s.

After the change, all of the following must hold:

1. `python3 -c "import src"` succeeds.
2. `python3 -c "import src.internal.tools; print(src.internal.tools.Tool)"` succeeds.
3. `pytest` collects **2,732** tests and passes. A drop in the collected count
   means a test file failed to import — the count matters as much as the pass.
4. `grep -rn "src\.tools" --include="*.py" src/ tests/ examples/` returns zero hits.
5. `test ! -d src/tools` — the directory is gone, including its stale
   `__pycache__/`.
6. `ruff check . --fix && ruff format .` clean.
7. `cd web && npm run typecheck` — unaffected, but confirms no incidental breakage.

Checks 3 and 4 together are what make this safe: 4 proves no textual reference
survives, 3 proves the string patch targets actually resolve.

## Risks

**Concurrent work.** A parallel agent is live in
`.worktrees/mcp-document-extraction` on `feat/mcp-document-extraction`. This
change touches 47 files and will conflict with anything landing that touches
`src/tools`. Mitigation: implement on `chore/consolidate-tools-package`
and merge promptly rather than letting it sit. This checkout has a documented
history of parallel-agent collisions.

**Silent test weakening** via missed patch targets — addressed by the dedicated
grep pass and by asserting the exact collected-test count.

**Stale bytecode.** `src/tools/__pycache__/` currently holds `.pyc` files for
modules whose `.py` sources were already moved once. Deleting the directory
outright removes them.

## Files

- **Move (11):** `src/tools/{api,base,html_text,knowledge_base,openapi_schema,parsers,registry,routing_tools,search,semantic_router,validation}.py`
  → `src/internal/tools/`
- **Delete (1):** `src/tools/__init__.py` (contents merged into the destination `__init__.py`), then the empty `src/tools/` directory
- **Modify (36):** `src/internal/tools/__init__.py`, `src/internal/tools/registry.py`
  (post-move, for its two absolute self-references), 11 production consumers,
  21 test files, `docs/mcp.md`, `docs/tool-engine.md`

**47 files touched in total** (11 moved + 1 deleted + 36 modified, counting the
moved `registry.py` once under each heading it appears in).
