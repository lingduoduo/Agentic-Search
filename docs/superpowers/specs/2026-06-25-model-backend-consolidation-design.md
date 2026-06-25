# Model backend consolidation — design

**Date:** 2026-06-25
**Status:** Approved scope; implementation plan pending.
**Scope:** Give the serving-side LLM backend a formal interface, a single
selection factory, and a proper home in `src/`. The agent loop's model boundary
stays exactly as-is (it already only sees an injected `server_manager`).

## Problem

The serving model boundary is real but unstructured:

- **Model classes live in `examples/` and production imports them.** The web app
  does `from examples.run_agentic_search import OpenAIServerManager`
  (`web/app.py:572`, `:597`). Production importing from an examples script is
  backwards.
- **No formal interface.** `server_manager` is duck-typed on `.generate()`. The
  codebase already uses `Protocol`/`runtime_checkable` for its other seams
  (`model/generation.py:13`, e.g. `Retriever`, `Fetcher`) — the manager has none.
- **Provider dispatch is duplicated and inline.** The `if server_url / elif model`
  selection is hand-written in the web lifespan (`web/app.py:566-612`) and again
  in the CLI (`--local` / `--vllm_url` / `--model`). No factory.

## The de-facto contract

Both serving managers already implement the same async signature, used by every
agent loop (`agents/base.py:155`):

```python
async def generate(
    self,
    request_id: str,
    prompt_ids: list[int],
    sampling_params: dict[str, Any],
) -> list[int]: ...
```

Concrete classes today (both in `examples/run_agentic_search.py`):

| Class | Backend |
|---|---|
| `OpenAIServerManager` (`:465`) | OpenAI-compatible HTTP server (vLLM/mlx) via `base_url` + `model` |
| `LocalServerManager` (`:535`) | in-process HF `AutoModelForCausalLM` on cpu/mps/cuda |

Selection today: `server_url` set → OpenAI; else `model` set → Local.

## Scope boundary: serving only

`LLMGenerationManager` (`model/generation.py:1765`) is the heavier **training-side**
manager. It is a separate world and is **out of scope** — this spec does not merge
or unify training and serving managers. Conflating them would balloon the change
and couple serving to training internals.

## Design

### 1. `ServerManager` Protocol (in `src/`)

A `runtime_checkable` Protocol capturing the `.generate()` contract, matching the
existing Protocol style in `model/generation.py`:

```python
@runtime_checkable
class ServerManager(Protocol):
    async def generate(
        self, request_id: str, prompt_ids: list[int], sampling_params: dict[str, Any]
    ) -> list[int]: ...
```

Lives in a new serving module, e.g. `src/model/serving.py` (alongside the existing
`model/` package).

### 2. Move the concrete managers into `src/`

`OpenAIServerManager` and `LocalServerManager` move from
`examples/run_agentic_search.py` into `src/model/serving.py`. `examples/` then
**imports from `src/`** (re-export shim kept in the example for backward compat so
existing CLI invocations and the `from examples.run_agentic_search import ...` call
sites keep working during transition).

### 3. One selection factory

```python
def build_server_manager(cfg: ModelBackendConfig, tokenizer) -> ServerManager:
    if cfg.server_url:
        return OpenAIServerManager(tokenizer, cfg.server_url, cfg.model)
    if cfg.model:
        return LocalServerManager(model_path=cfg.model, device=cfg.device, ...)
    raise ValueError("no model backend configured")
```

`ModelBackendConfig` is a typed view over the already-resolved config fields
(`search_agent_server_url`, `search_agent_model`, `search_agent_device`). Both the
web lifespan and the CLI call `build_server_manager` — the duplicated `if/elif`
disappears.

### 4. Agent loop unchanged

The loop already receives only the injected `server_manager` and calls
`.generate()`. No loop change; it now just receives something that *formally*
satisfies `ServerManager`.

### Data flow

```
resolved config ─► ModelBackendConfig ─► build_server_manager ─► ServerManager impl
                                                                     │
                              injected into AgentLoop; loop calls .generate()
```

## Testing

- **Protocol conformance:** `isinstance(OpenAIServerManager(...), ServerManager)`
  and same for Local (via `runtime_checkable`).
- **Factory dispatch:** `server_url` → OpenAI; `model` only → Local; neither →
  clear error.
- **Parity:** web and CLI build the same manager (type + key params) as today for
  each config combination (golden-path).
- **Backward-compat imports:** `from examples.run_agentic_search import
  OpenAIServerManager, LocalServerManager` still resolves via the re-export shim.

## Non-goals

- Unifying training (`LLMGenerationManager`) and serving managers.
- Adding new providers (Anthropic, etc.) — the factory makes that a later one-line
  extension, but no new backend ships here.
- Streaming/token-by-token changes to `.generate()`.
- Any agent-loop behavior change, or the `LoopController` control-flow work
  (orthogonal).
- Folding model selection into agent invocation — kept separate, though both are
  "factory/selection" wiring (see Relationship).

## Relationship to other specs

- Sibling to `2026-06-25-agent-invocation-consolidation-design.md`: that spec is a
  factory for *which agent loop*; this is a factory for *which model backend*.
  Same pattern, different axis; independent PRs.
- Independent of the `LoopController` control-flow and `ToolExecutionMode` specs.
