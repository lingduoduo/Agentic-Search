# Generated Context Pack

# Model Backend Consolidation

## Sources

- [Specification: 2026-06-25-model-backend-consolidation-design.md](../specs/2026-06-25-model-backend-consolidation-design.md)
- [Plan: 2026-06-25-model-backend-consolidation.md](../plans/2026-06-25-model-backend-consolidation.md)

## Specification Context

### Scope boundary: serving only

`LLMGenerationManager` (`model/generation.py:1765`) is the heavier **training-side**
manager. It is a separate world and is **out of scope** — this spec does not merge
or unify training and serving managers. Conflating them would balloon the change
and couple serving to training internals.

### Testing

- **Protocol conformance:** `isinstance(OpenAIServerManager(...), ServerManager)`
  and same for Local (via `runtime_checkable`).
- **Factory dispatch:** `server_url` → OpenAI; `model` only → Local; neither →
  clear error.
- **Parity:** web and CLI build the same manager (type + key params) as today for
  each config combination (golden-path).
- **Backward-compat imports:** `from examples.run_agentic_search import
  OpenAIServerManager, LocalServerManager` still resolves via the re-export shim.

### Non-goals

- Unifying training (`LLMGenerationManager`) and serving managers.
- Adding new providers (Anthropic, etc.) — the factory makes that a later one-line
  extension, but no new backend ships here.
- Streaming/token-by-token changes to `.generate()`.
- Any agent-loop behavior change, or the `LoopController` control-flow work
  (orthogonal).
- Folding model selection into agent invocation — kept separate, though both are
  "factory/selection" wiring (see Relationship).

## Implementation Plan Context

### Global Constraints

- **Behavior-preserving move.** The two manager classes move verbatim — no logic change. Their `.generate(request_id, prompt_ids, sampling_params) -> list[int]` contract is unchanged.
- **Backward-compatible imports.** `from examples.run_agentic_search import OpenAIServerManager, LocalServerManager` must still resolve (re-export shim). Importers today: `src/internal/servers/web/app.py`, `tests/unit/test_run_agentic_search.py`, `examples/run_bamboogle_eval.py`.
- **Training-side `LLMGenerationManager` (`src/model/generation.py`) is OUT of scope** — do not touch or unify it.
- **No new heavy module-level imports.** `aiohttp`/`transformers`/`torch` stay imported *inside* methods (as they are today), so `import src.model.serving` is cheap.
- **`_zeroed`/agent loops unchanged.** No agent loop edits; the factory is additive.

---

### Task 1: `ServerManager` Protocol + serving module skeleton

**Files:**
- Create: `src/model/serving.py`
- Test: `tests/unit/test_model_serving.py`

**Interfaces:**
- Produces: `ServerManager` (`runtime_checkable` Protocol) with `async def generate(self, request_id: str, prompt_ids: list[int], sampling_params: dict[str, Any]) -> list[int]`.

- [ ] **Step 1: Write the failing test**

```python

### tests/unit/test_model_serving.py

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

### tests/unit/test_model_serving.py  (add)

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

_[Section compacted.]_

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

### tests/unit/test_model_serving.py  (add)

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

_[Section compacted.]_

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

## Context Boundary

This pack summarizes its linked sources. Consult those documents for complete details; no implementation status is inferred here.
