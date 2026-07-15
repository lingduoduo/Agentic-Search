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

…

### Task 1: `ServerManager` Protocol + serving module skeleton

**Files:**
- Create: `src/model/serving.py`
- Test: `tests/unit/test_model_serving.py`

**Interfaces:**
- Produces: `ServerManager` (`runtime_checkable` Protocol) with `async def generate(self, request_id: str, prompt_ids: list[int], sampling_params: dict[str, Any]) -> list[int]`.

- [ ] **Step 1: Write the failing test**

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/unit/test_model_serving.py -v`
Expected: FAIL — module/`ServerManager` does not exist.

- [ ] **Step 3: Implement**

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/unit/test_model_serving.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

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

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/unit/test_model_serving.py -k "serving or shim or conform" -v`
Expected: FAIL — classes not yet in `src.model.serving`.

…

### Task 3: `build_server_manager` factory + wire CLI and web

**Files:**
- Modify: `src/model/serving.py` (add factory)
- Modify: `examples/run_agentic_search.py` (CLI uses factory)
- Modify: `src/internal/servers/web/app.py` (lifespan uses factory; import from `src.model.serving`, not `examples`)
- Test: `tests/unit/test_model_serving.py`

**Interfaces:**
- Produces: `build_server_manager(tokenizer, *, server_url: str | None = None, model: str | None = None, device: str | None = None, **kwargs) -> ServerManager`. Selection: `server_url` set → `OpenAIServerManager`; else `model` set → `LocalServerManager`; else `ValueError`.

- [ ] **Step 1: Write the failing test**

…

## Context Boundary

This pack summarizes its linked sources. Consult those documents for complete details; no implementation status is inferred here.
