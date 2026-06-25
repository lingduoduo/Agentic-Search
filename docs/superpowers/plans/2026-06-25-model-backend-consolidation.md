# Model Backend Consolidation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Give the serving LLM boundary a formal `ServerManager` Protocol, move `OpenAIServerManager`/`LocalServerManager` out of `examples/` into `src/model/serving.py` (with a re-export shim), and add one `build_server_manager()` factory used by both the CLI and the web app.

**Architecture:** A new `src/model/serving.py` holds a `runtime_checkable` `ServerManager` Protocol (the `.generate()` contract every agent loop already calls), the two concrete managers (moved verbatim), and a `build_server_manager()` factory that selects between them. `examples/run_agentic_search.py` re-exports the two classes so existing `from examples.run_agentic_search import ...` call sites keep working. The agent loops are untouched (they already receive only an injected `server_manager`).

**Tech Stack:** Python 3, `typing.Protocol`/`runtime_checkable`, pytest. No new dependencies (aiohttp/transformers/torch stay lazily imported inside methods).

**Spec:** `docs/superpowers/specs/2026-06-25-model-backend-consolidation-design.md` (already merged to main via #335).

## Global Constraints

- **Behavior-preserving move.** The two manager classes move verbatim — no logic change. Their `.generate(request_id, prompt_ids, sampling_params) -> list[int]` contract is unchanged.
- **Backward-compatible imports.** `from examples.run_agentic_search import OpenAIServerManager, LocalServerManager` must still resolve (re-export shim). Importers today: `src/internal/servers/web/app.py`, `tests/unit/test_run_agentic_search.py`, `examples/run_bamboogle_eval.py`.
- **Training-side `LLMGenerationManager` (`src/model/generation.py`) is OUT of scope** — do not touch or unify it.
- **No new heavy module-level imports.** `aiohttp`/`transformers`/`torch` stay imported *inside* methods (as they are today), so `import src.model.serving` is cheap.
- **`_zeroed`/agent loops unchanged.** No agent loop edits; the factory is additive.

---

## File Structure

- **Create** `src/model/serving.py` — `ServerManager` Protocol + the two moved manager classes + `build_server_manager()`.
- **Modify** `examples/run_agentic_search.py` — remove the two class bodies; re-import them from `src.model.serving` (re-export shim); CLI uses `build_server_manager`.
- **Modify** `src/internal/servers/web/app.py` — the lifespan manager construction uses `build_server_manager` (imported from `src.model.serving`) instead of importing the classes from `examples`.
- **Test** `tests/unit/test_model_serving.py` (new — Protocol conformance + factory dispatch); existing `tests/unit/test_run_agentic_search.py` (back-compat imports).

---

### Task 1: `ServerManager` Protocol + serving module skeleton

**Files:**
- Create: `src/model/serving.py`
- Test: `tests/unit/test_model_serving.py`

**Interfaces:**
- Produces: `ServerManager` (`runtime_checkable` Protocol) with `async def generate(self, request_id: str, prompt_ids: list[int], sampling_params: dict[str, Any]) -> list[int]`.

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/test_model_serving.py
from src.model.serving import ServerManager

class _Conformer:
    async def generate(self, request_id, prompt_ids, sampling_params):
        return [1, 2, 3]

class _NonConformer:
    def something_else(self): ...

def test_protocol_is_runtime_checkable():
    assert isinstance(_Conformer(), ServerManager)
    assert not isinstance(_NonConformer(), ServerManager)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/unit/test_model_serving.py -v`
Expected: FAIL — module/`ServerManager` does not exist.

- [ ] **Step 3: Implement**

```python
# src/model/serving.py
"""Serving-side LLM backend: the ServerManager protocol, concrete managers, and
a factory selecting between them. The training-side LLMGenerationManager
(model/generation.py) is a separate concern and intentionally not here.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class ServerManager(Protocol):
    """The model boundary every agent loop calls: tokens in, tokens out."""

    async def generate(
        self,
        request_id: str,
        prompt_ids: list[int],
        sampling_params: dict[str, Any],
    ) -> list[int]: ...
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/unit/test_model_serving.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/model/serving.py tests/unit/test_model_serving.py
git commit -m "feat: ServerManager runtime_checkable Protocol in src/model/serving.py"
```

---

### Task 2: Move the two managers into `serving.py` + re-export shim

**Files:**
- Modify: `src/model/serving.py` (append the two classes)
- Modify: `examples/run_agentic_search.py` (remove class bodies `:465-757`; re-import from `src.model.serving`)
- Test: `tests/unit/test_model_serving.py`, `tests/unit/test_run_agentic_search.py`

**Interfaces:**
- Consumes: `ServerManager` (Task 1).
- Produces: `OpenAIServerManager`, `LocalServerManager` in `src.model.serving`; re-exported from `examples.run_agentic_search`.

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/test_model_serving.py  (add)
def test_managers_importable_from_serving():
    from src.model.serving import OpenAIServerManager, LocalServerManager
    assert OpenAIServerManager is not None and LocalServerManager is not None

def test_managers_still_importable_from_examples_shim():
    # Back-compat: existing call sites import from the examples module.
    from examples.run_agentic_search import OpenAIServerManager as A
    from src.model.serving import OpenAIServerManager as B
    assert A is B  # same class object, not a copy

def test_concrete_managers_conform_to_protocol():
    from src.model.serving import ServerManager, OpenAIServerManager, LocalServerManager
    # Instances need construction args; assert the method exists on the class via
    # the Protocol's structural check on a lightweight duck instance is covered in
    # Task 1 — here assert the classes expose an async generate attribute.
    assert callable(getattr(OpenAIServerManager, "generate", None))
    assert callable(getattr(LocalServerManager, "generate", None))
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/unit/test_model_serving.py -k "serving or shim or conform" -v`
Expected: FAIL — classes not yet in `src.model.serving`.

- [ ] **Step 3: Implement**

1. **Move** the `OpenAIServerManager` (`examples/run_agentic_search.py:465-534`) and `LocalServerManager` (`:535-757`) class bodies **verbatim** into `src/model/serving.py` (append after the Protocol). Keep all in-method imports (`aiohttp`, `transformers`, `torch`) exactly as they are. Add any module-level imports the class bodies reference at module scope in `serving.py` (e.g. `import asyncio`, `import logging`, `import os` — copy whatever the moved code uses that was previously module-level in `run_agentic_search.py`; check each name).
2. In `examples/run_agentic_search.py`, **delete** the two class bodies and add a re-export near the top imports:

```python
from src.model.serving import (  # re-export for back-compat
    OpenAIServerManager as OpenAIServerManager,
    LocalServerManager as LocalServerManager,
)
```

3. Verify nothing else in `run_agentic_search.py` referenced module-level names that only existed because of those classes.

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/unit/test_model_serving.py tests/unit/test_run_agentic_search.py -v`
Expected: PASS (new + all existing CLI tests — the shim keeps them green).

- [ ] **Step 5: Commit**

```bash
git add src/model/serving.py examples/run_agentic_search.py tests/unit/test_model_serving.py
git commit -m "refactor: move ServerManager classes into src/model/serving.py with re-export shim"
```

---

### Task 3: `build_server_manager` factory + wire CLI and web

**Files:**
- Modify: `src/model/serving.py` (add factory)
- Modify: `examples/run_agentic_search.py` (CLI uses factory)
- Modify: `src/internal/servers/web/app.py` (lifespan uses factory; import from `src.model.serving`, not `examples`)
- Test: `tests/unit/test_model_serving.py`

**Interfaces:**
- Produces: `build_server_manager(tokenizer, *, server_url: str | None = None, model: str | None = None, device: str | None = None, **kwargs) -> ServerManager`. Selection: `server_url` set → `OpenAIServerManager`; else `model` set → `LocalServerManager`; else `ValueError`.

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/test_model_serving.py  (add)
import pytest
from src.model.serving import build_server_manager, OpenAIServerManager

class _Tok:  # minimal tokenizer stand-in
    pad_token_id = 0
    eos_token_id = 0
    def encode(self, s): return [1]
    def decode(self, ids, **k): return "x"

def test_factory_selects_openai_when_server_url():
    mgr = build_server_manager(_Tok(), server_url="http://localhost:8080", model="m")
    assert isinstance(mgr, OpenAIServerManager)

def test_factory_raises_when_nothing_configured():
    with pytest.raises(ValueError):
        build_server_manager(_Tok())
```

(Do NOT instantiate `LocalServerManager` in tests — it would try to load a real HF model. Cover only the OpenAI branch and the error branch. State this in the report.)

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/unit/test_model_serving.py -k factory -v`
Expected: FAIL — `build_server_manager` not defined.

- [ ] **Step 3: Implement**

Add to `src/model/serving.py`:

```python
def build_server_manager(
    tokenizer: Any,
    *,
    server_url: str | None = None,
    model: str | None = None,
    device: str | None = None,
    **kwargs: Any,
) -> ServerManager:
    """Select the serving backend from resolved config.

    server_url set -> OpenAIServerManager (remote); else model set ->
    LocalServerManager (in-process); else ValueError.
    """
    if server_url:
        return OpenAIServerManager(tokenizer=tokenizer, base_url=server_url, model=model)
    if model:
        return LocalServerManager(model_path=model, device=device, **kwargs)
    raise ValueError("no model backend configured (set server_url or model)")
```

Match the **actual** constructor signatures of the two classes (read them after the move). If `OpenAIServerManager.__init__` uses different param names (e.g. `base_url`/`model`), use those exactly; same for `LocalServerManager` (`model_path`/`device`/`allow_unsafe_mps`/`local_files_only`). Thread the extra `LocalServerManager` kwargs (`allow_unsafe_mps`, `local_files_only`) through `**kwargs`.

Then in `src/internal/servers/web/app.py` lifespan (where it currently does `from examples.run_agentic_search import OpenAIServerManager` / `LocalServerManager` and the `if server_url / elif model` branching, ~`:566-612`), replace the duplicated branching with a single `build_server_manager(...)` call imported from `src.model.serving`. Preserve the surrounding try/except + the tokenizer setup + the `_app.state.search_agent_manager` assignment exactly.

In `examples/run_agentic_search.py`, replace the CLI's own manager construction (where it builds `OpenAIServerManager`/`LocalServerManager` from `--vllm_url`/`--local`/`--model` args) with `build_server_manager(...)`.

- [ ] **Step 4: Run tests**

Run: `pytest tests/unit/test_model_serving.py tests/unit/test_run_agentic_search.py tests/unit/servers/web/ -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/model/serving.py examples/run_agentic_search.py src/internal/servers/web/app.py tests/unit/test_model_serving.py
git commit -m "feat: build_server_manager factory; CLI and web select backend via it"
```

---

### Task 4: Full-suite verification + back-compat sweep

**Files:**
- Test: whole suite

- [ ] **Step 1: Grep all importers of the managers are satisfied**

Run: `grep -rn "OpenAIServerManager\|LocalServerManager" src/ tests/ examples/`
Confirm each importer resolves (either from `src.model.serving` or the `examples` shim). `examples/run_bamboogle_eval.py` imports from `examples.run_agentic_search` — confirm the shim covers it.

- [ ] **Step 2: Full unit suite**

Run: `pytest tests/unit -q`
Expected: PASS (no regressions). Then `ruff check . --fix && ruff format .` and re-run.

- [ ] **Step 3: Commit (only if lint/baseline changes were needed)**

```bash
git add -A
git commit -m "test: verify model-backend back-compat across importers"
```

(If nothing changed, skip the commit.)

---

## Self-Review

**Spec coverage:** `ServerManager` Protocol (Task 1) · move managers into `src/` + re-export shim (Task 2) · `build_server_manager` factory used by CLI + web (Task 3) · agent loops untouched (no task edits them) · training `LLMGenerationManager` out of scope (Global Constraints) · back-compat imports (Tasks 2,4). All spec sections map to a task.

**Placeholder scan:** Task 2 "move verbatim" references concrete line ranges (`465-534`, `535-757`) rather than reproducing ~290 lines — appropriate for a verbatim move; the implementer cuts and pastes, adjusting only module-level imports. Task 3 factory body is complete but explicitly says to match the real constructor param names after the move (the classes' exact `__init__` signatures live in the moved code).

**Type consistency:** `ServerManager` Protocol, `build_server_manager(tokenizer, *, server_url, model, device, **kwargs) -> ServerManager`, and the `.generate(request_id, prompt_ids, sampling_params) -> list[int]` contract are used identically across Tasks 1–3. Factory selection (server_url → OpenAI, model → Local, else ValueError) matches the spec.

**Scope deviation (noted):** the spec sketched `build_server_manager(cfg: ModelBackendConfig, tokenizer)`. This plan uses explicit keyword args instead of a new `ModelBackendConfig` dataclass — the three fields (server_url/model/device) are already locals at both call sites, so a wrapper type would be unused ceremony (YAGNI). Same selection logic; one fewer type.
