# Agentic Framework Full Stack Polish — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Wire `ToolAgentLoop` as a sixth web mode, add test coverage for all agent modes, surface `rounds_used` in the chat UI, and verify the full stack (retrieval, connectors, MCP, admin) works end-to-end.

**Architecture:** All agent modes share `POST /api/agent`. `tool_agent` reuses the `search_agent_manager` already loaded at startup; tools come from `build_search_tool(search_url)` + `tool_registry.list_tools()` so any tool registered via ToolPanel is automatically available. Areas 5–11 are verification tasks — no net-new code unless a breakage is found.

**Tech Stack:** Python 3.12, FastAPI, pytest, React 19 + TypeScript, Vite, aiohttp, transformers, `ToolAgentLoop`, `tool_registry`.

---

## Files Changed

| File | Change |
|---|---|
| `tests/unit/test_agent_loop.py` | Fix `DummyTokenizerWithTemplate` fixture; add `_build_prompt_ids_sync` test |
| `src/internal/configs/app_configs.py` | Add `tool_agent_parser: str = "json"` |
| `src/internal/servers/web/app.py` | Add `mode == "tool_agent"` branch; expose `rounds_used` in `ChatMessageView` |
| `web/src/types.ts` | Add `"tool_agent"` to `AgentMode`; add `metadata` to `ChatMessageView` |
| `web/src/components/SearchComposer.tsx` | Add Tool Agent to `MODE_OPTIONS` |
| `web/src/components/SessionTimeline.tsx` | Show `rounds_used` badge on `chat_loop` messages |
| `tests/unit/test_configs.py` | Test `tool_agent_parser` default and override |
| `tests/unit/servers/web/test_web_experience_app.py` | Add `tool_agent`, `search_tool`, `chat_once` mode smoke tests |

---

## Task 1: Fix broken test fixture for `tokenize=False` change

The `_build_prompt_ids_sync` change (this branch) calls `apply_chat_template(tokenize=False)` then `tokenizer.encode()`. The existing `DummyTokenizerWithTemplate` in `test_agent_loop.py` asserts `tokenize is True` — this will cause a test failure.

**Files:**
- Modify: `tests/unit/test_agent_loop.py:27-38`

- [ ] **Step 1: Run existing tests to confirm the failure**

```bash
pytest tests/unit/test_agent_loop.py -v 2>&1 | head -40
```

Expected: One or more tests fail with `AssertionError` on `assert tokenize is True`.

- [ ] **Step 2: Update `DummyTokenizerWithTemplate` to match new call signature**

In `tests/unit/test_agent_loop.py`, replace the existing `DummyTokenizerWithTemplate` class:

```python
class DummyTokenizerWithTemplate:
    chat_template = "dummy"  # must be non-None for apply_chat_template branch to trigger

    def apply_chat_template(self, messages, add_generation_prompt=True, tokenize=False):
        assert add_generation_prompt is True
        assert tokenize is False  # base.py now calls with tokenize=False
        return "dummy prompt text"  # returns str when tokenize=False

    def encode(self, text):
        return [ord(c) for c in text]

    def decode(self, token_ids, skip_special_tokens=True):
        del skip_special_tokens
        return "".join(chr(t) for t in token_ids if t < 128)
```

- [ ] **Step 3: Run tests to verify they pass**

```bash
pytest tests/unit/test_agent_loop.py -v 2>&1 | tail -20
```

Expected: All previously passing tests still pass.

- [ ] **Step 4: Commit**

```bash
git add tests/unit/test_agent_loop.py
git commit -m "fix(tests): update DummyTokenizerWithTemplate for tokenize=False change"
```

---

## Task 2: Add unit test for `_build_prompt_ids_sync` (search agent fix)

Verify that the `tokenize=False` + explicit `encode()` path in `_build_prompt_ids_sync` returns a list of integers for both tokenizer types.

**Files:**
- Modify: `tests/unit/test_agent_loop.py`

- [ ] **Step 1: Add two tests after the existing `DummyTokenizer*` classes**

```python
def test_build_prompt_ids_sync_with_chat_template_returns_int_list():
    """apply_chat_template(tokenize=False) path encodes result via tokenizer.encode()."""
    from src.agents.base import AgentLoopBase

    tokenizer = DummyTokenizerWithTemplate()
    manager = DummyServerManager([])
    loop = AgentLoopBase(tokenizer=tokenizer, server_manager=manager)

    result = loop._build_prompt_ids_sync([{"role": "user", "content": "hello"}])

    assert isinstance(result, list)
    assert len(result) > 0
    assert all(isinstance(x, int) for x in result)


def test_build_prompt_ids_sync_fallback_encode():
    """Falls back to tokenizer.encode() when no chat_template is present."""
    from src.agents.base import AgentLoopBase

    tokenizer = DummyTokenizerWithEncode()
    manager = DummyServerManager([])
    loop = AgentLoopBase(tokenizer=tokenizer, server_manager=manager)

    result = loop._build_prompt_ids_sync([{"role": "user", "content": "hi"}])

    assert isinstance(result, list)
    assert all(isinstance(x, int) for x in result)
```

- [ ] **Step 2: Run to confirm they pass**

```bash
pytest tests/unit/test_agent_loop.py::test_build_prompt_ids_sync_with_chat_template_returns_int_list \
       tests/unit/test_agent_loop.py::test_build_prompt_ids_sync_fallback_encode -v
```

Expected: Both PASS.

- [ ] **Step 3: Commit**

```bash
git add tests/unit/test_agent_loop.py
git commit -m "test(agents): verify _build_prompt_ids_sync tokenize=False path"
```

---

## Task 3: Add `tool_agent_parser` config field

**Files:**
- Modify: `src/internal/configs/app_configs.py`
- Modify: `tests/unit/test_configs.py`

- [ ] **Step 1: Write the failing test**

In `tests/unit/test_configs.py`, add:

```python
def test_tool_agent_parser_default():
    from src.internal.configs.app_configs import load_app_settings
    settings = load_app_settings({})
    assert settings.tool_agent_parser == "json"


def test_tool_agent_parser_override():
    from src.internal.configs.app_configs import load_app_settings
    settings = load_app_settings({"TOOL_AGENT_PARSER": "hermes"})
    assert settings.tool_agent_parser == "hermes"
```

- [ ] **Step 2: Run to confirm they fail**

```bash
pytest tests/unit/test_configs.py::test_tool_agent_parser_default \
       tests/unit/test_configs.py::test_tool_agent_parser_override -v
```

Expected: FAIL — `AttributeError: 'AppSettings' object has no attribute 'tool_agent_parser'`.

- [ ] **Step 3: Add the field to `AppSettings` and `load_app_settings`**

In `src/internal/configs/app_configs.py`, find the block with `search_agent_server_url` (around line 143) and add immediately after it:

```python
    tool_agent_parser: str = "json"  # "json" | "hermes" | "llama3"
```

In `load_app_settings`, find the block with `search_agent_server_url=get_env_str(...)` (around line 232) and add immediately after it:

```python
        tool_agent_parser=get_env_str(source, "TOOL_AGENT_PARSER", "json"),
```

- [ ] **Step 4: Run to confirm they pass**

```bash
pytest tests/unit/test_configs.py::test_tool_agent_parser_default \
       tests/unit/test_configs.py::test_tool_agent_parser_override -v
```

Expected: Both PASS.

- [ ] **Step 5: Commit**

```bash
git add src/internal/configs/app_configs.py tests/unit/test_configs.py
git commit -m "feat(config): add tool_agent_parser env var (default: json)"
```

---

## Task 4: Wire `tool_agent` mode into the web backend

**Files:**
- Modify: `src/internal/servers/web/app.py` — after the `search_agent` block (~line 622)

- [ ] **Step 1: Add the `tool_agent` branch in `run_agent()`**

In `src/internal/servers/web/app.py`, locate the line `if mode == "search_agent":` (around line 561). After the closing `return AgentExperienceResponse(...)` of that block (around line 622), add:

```python
            if mode == "tool_agent":
                from src.agents.tool_calling import ToolAgentLoop, ToolAgentLoopConfig
                from src.tools import build_search_tool
                from src.tools.registry import tool_registry

                manager = getattr(http_request.app.state, "search_agent_manager", None)
                tokenizer = getattr(
                    http_request.app.state, "search_agent_tokenizer", None
                )
                if manager is None or tokenizer is None:
                    raise HTTPException(
                        status_code=400,
                        detail=(
                            "tool_agent mode requires SEARCH_AGENT_MODEL or "
                            "SEARCH_AGENT_SERVER_URL to be set."
                        ),
                    )
                tools = [
                    build_search_tool(search_url=search_url)
                ] + tool_registry.list_tools()
                loop = ToolAgentLoop(
                    tokenizer=tokenizer,
                    server_manager=manager,
                    tools=tools,
                    config=ToolAgentLoopConfig(
                        tool_parser_format=resolved.tool_agent_parser
                    ),
                )
                output = await loop.run(
                    [{"role": "user", "content": query}],
                    sampling_params={"temperature": 0.0, "max_tokens": 512},
                )
                answer = output.final_answer or next(
                    (
                        m["content"]
                        for m in reversed(output.trajectory_messages)
                        if m.get("role") == "assistant"
                    ),
                    "",
                )
                db.add_chat_message(
                    session_id,
                    role="assistant",
                    content=answer,
                    metadata={
                        "mode": mode,
                        "hooks": hook_metadata,
                        "num_turns": output.num_turns,
                    },
                )
                messages = [
                    ChatMessageView(role=m.role, content=m.content)
                    for m in db.list_chat_messages(session_id)
                ]
                return AgentExperienceResponse(
                    session_id=session_id,
                    answer=answer,
                    citations=[],
                    documents=[],
                    messages=messages,
                    hook_metadata=hook_metadata,
                )
```

Also add `"tool_agent"` to the `_VALID_MODES` set near line 704:

```python
    "tool_agent",
```

- [ ] **Step 2: Verify the server starts without errors**

```bash
PYTHONPATH=src:. python3 -c "
from src.internal.servers.web.app import create_web_app, SearchExperienceSettings
app = create_web_app(SearchExperienceSettings())
print('OK — app created')
"
```

Expected: `OK — app created` with no import errors.

- [ ] **Step 3: Commit**

```bash
git add src/internal/servers/web/app.py
git commit -m "feat(web): add tool_agent mode backed by ToolAgentLoop + tool_registry"
```

---

## Task 5: Wire `tool_agent` mode into the frontend

**Files:**
- Modify: `web/src/types.ts`
- Modify: `web/src/components/SearchComposer.tsx`

- [ ] **Step 1: Add `"tool_agent"` to `AgentMode` in `web/src/types.ts`**

Find the line (line 25):
```ts
export type AgentMode = "search_tool" | "hybrid_search" | "chat_once" | "chat_loop" | "search_agent";
```

Replace with:
```ts
export type AgentMode = "search_tool" | "hybrid_search" | "chat_once" | "chat_loop" | "search_agent" | "tool_agent";
```

- [ ] **Step 2: Add Tool Agent to `MODE_OPTIONS` in `web/src/components/SearchComposer.tsx`**

Find the `MODE_OPTIONS` array (line 7). Add after the `search_agent` entry:

```ts
  { value: "tool_agent", label: "Tool Agent (Function Calling)" },
```

So the full array becomes:
```ts
const MODE_OPTIONS: Array<{ value: AgentMode; label: string }> = [
  { value: "search_tool", label: "Search: Direct Tool" },
  { value: "hybrid_search", label: "Search: Hybrid" },
  { value: "chat_once", label: "Chat: No Loop" },
  { value: "chat_loop", label: "Chat: Loop" },
  { value: "search_agent", label: "Search Agent (Local Model)" },
  { value: "tool_agent", label: "Tool Agent (Function Calling)" },
];
```

- [ ] **Step 3: Type-check the frontend**

```bash
cd web && npm run typecheck 2>&1 | tail -20
```

Expected: No type errors.

- [ ] **Step 4: Commit**

```bash
git add web/src/types.ts web/src/components/SearchComposer.tsx
git commit -m "feat(ui): add Tool Agent mode to mode selector"
```

---

## Task 6: Surface `rounds_used` in chat_loop SessionTimeline

**Files:**
- Modify: `src/internal/servers/web/app.py` — `ChatMessageView` model
- Modify: `web/src/types.ts` — `ChatMessageView` interface
- Modify: `web/src/components/SessionTimeline.tsx`

- [ ] **Step 1: Add `metadata` to `ChatMessageView` in `app.py`**

Find `class ChatMessageView(BaseModel):` (around line 121):

```python
class ChatMessageView(BaseModel):
    role: str
    content: str
```

Replace with:

```python
class ChatMessageView(BaseModel):
    role: str
    content: str
    metadata: dict[str, object] = Field(default_factory=dict)
```

- [ ] **Step 2: Pass `metadata` when building message list from DB**

There are several places in `app.py` that build `ChatMessageView` from `db.list_chat_messages()`. Find each occurrence of:

```python
messages = [
    ChatMessageView(role=m.role, content=m.content)
    for m in db.list_chat_messages(session_id)
]
```

Replace each with:

```python
messages = [
    ChatMessageView(role=m.role, content=m.content, metadata=getattr(m, "metadata", None) or {})
    for m in db.list_chat_messages(session_id)
]
```

- [ ] **Step 3: Add `metadata` to `ChatMessageView` in `web/src/types.ts`**

Find:
```ts
export interface ChatMessageView {
  role: ChatRole;
  content: string;
}
```

Replace with:
```ts
export interface ChatMessageView {
  role: ChatRole;
  content: string;
  metadata?: Record<string, unknown>;
}
```

- [ ] **Step 4: Show `rounds_used` badge in `SessionTimeline.tsx`**

Replace the full content of `web/src/components/SessionTimeline.tsx`:

```tsx
import { memo } from "react";
import type { ChatMessageView } from "../types";

interface SessionTimelineProps {
  messages: ChatMessageView[];
}

export const SessionTimeline = memo(function SessionTimeline({
  messages,
}: SessionTimelineProps) {
  if (messages.length === 0) {
    return <div className="empty-state compact">Start a query to create history.</div>;
  }

  return (
    <ol className="timeline">
      {messages.map((message, index) => {
        const rounds = message.metadata?.rounds_used as number | undefined;
        const turns = message.metadata?.num_turns as number | undefined;
        return (
          <li key={`${message.role}-${index}`}>
            <strong>{message.role}</strong>
            {rounds != null && (
              <span className="tool-badge" data-tone="active" style={{ marginLeft: 8 }}>
                {rounds} round{rounds !== 1 ? "s" : ""}
              </span>
            )}
            {turns != null && (
              <span className="tool-badge" data-tone="neutral" style={{ marginLeft: 8 }}>
                {turns} turn{turns !== 1 ? "s" : ""}
              </span>
            )}
            <p>{message.content}</p>
          </li>
        );
      })}
    </ol>
  );
});
```

- [ ] **Step 5: Type-check**

```bash
cd web && npm run typecheck 2>&1 | tail -10
```

Expected: No errors.

- [ ] **Step 6: Commit**

```bash
git add src/internal/servers/web/app.py web/src/types.ts web/src/components/SessionTimeline.tsx
git commit -m "feat(ui): show rounds_used and num_turns badges in SessionTimeline"
```

---

## Task 7: Add smoke tests for `search_tool`, `chat_once`, and `tool_agent` modes

**Files:**
- Modify: `tests/unit/servers/web/test_web_experience_app.py`

- [ ] **Step 1: Add imports and helper at the top of the test file**

After the existing imports, add:

```python
from unittest.mock import AsyncMock, MagicMock, patch
```

- [ ] **Step 2: Add `search_tool` smoke test**

```python
def test_run_agent_search_tool_mode(tmp_path):
    """search_tool mode returns documents and a formatted answer."""
    from src.context.models import ContextDocument, SearchContextBundle

    app = create_web_app(SearchExperienceSettings(db_path=tmp_path / "state.sqlite3"))
    client = TestClient(app)

    with patch("src.internal.servers.web.app._run_direct_search", new_callable=AsyncMock) as mock_search:
        mock_search.return_value = [
            ContextDocument(
                id="D1", title="FAISS", content="FAISS is a vector search library.",
                url="https://example.test/faiss", score=0.9,
            )
        ]
        response = client.post("/api/agent", json={"query": "What is FAISS?", "mode": "search_tool"})

    assert response.status_code == 200
    data = response.json()
    assert len(data["documents"]) == 1
    assert data["documents"][0]["title"] == "FAISS"
    assert "FAISS" in data["answer"]
```

- [ ] **Step 3: Add `chat_once` smoke test**

```python
def test_run_agent_chat_once_mode(tmp_path):
    """chat_once mode calls answer_with_retrieval and returns an answer."""
    app = create_web_app(SearchExperienceSettings(db_path=tmp_path / "state.sqlite3"))
    client = TestClient(app)

    with patch("src.internal.servers.web.app.answer_with_retrieval", new_callable=AsyncMock) as mock_answer:
        mock_answer.return_value = _answer_result("What is FAISS?")
        response = client.post("/api/agent", json={"query": "What is FAISS?", "mode": "chat_once"})

    assert response.status_code == 200
    data = response.json()
    assert data["answer"]
    assert data["session_id"]
```

- [ ] **Step 4: Add `tool_agent` smoke test**

```python
def test_run_agent_tool_agent_mode_returns_answer(tmp_path):
    """tool_agent mode runs ToolAgentLoop and returns final_answer."""
    from src.agents.base import AgentLoopOutput

    app = create_web_app(SearchExperienceSettings(db_path=tmp_path / "state.sqlite3"))

    mock_output = AgentLoopOutput(
        prompt_ids=[1, 2, 3],
        response_ids=[4, 5, 6],
        response_mask=[1, 1, 1],
        num_turns=2,
        final_answer="Tool answer here.",
        trajectory_messages=[{"role": "assistant", "content": "Tool answer here."}],
    )

    mock_manager = MagicMock()
    mock_tokenizer = MagicMock()
    mock_tokenizer.pad_token_id = 0

    app.state.search_agent_manager = mock_manager
    app.state.search_agent_tokenizer = mock_tokenizer

    with patch("src.agents.tool_calling.ToolAgentLoop") as MockLoop:
        mock_loop_instance = MagicMock()
        mock_loop_instance.run = AsyncMock(return_value=mock_output)
        MockLoop.return_value = mock_loop_instance

        client = TestClient(app)
        response = client.post("/api/agent", json={"query": "What tools do you have?", "mode": "tool_agent"})

    assert response.status_code == 200
    data = response.json()
    assert data["answer"] == "Tool answer here."


def test_run_agent_tool_agent_mode_returns_400_when_no_model(tmp_path):
    """tool_agent mode returns 400 when no model is configured."""
    app = create_web_app(SearchExperienceSettings(db_path=tmp_path / "state.sqlite3"))
    # app.state.search_agent_manager is None by default (no model configured)
    client = TestClient(app)
    response = client.post("/api/agent", json={"query": "test", "mode": "tool_agent"})
    assert response.status_code == 400
    assert "SEARCH_AGENT_MODEL" in response.json()["detail"]
```

- [ ] **Step 5: Run all new tests**

```bash
pytest tests/unit/servers/web/test_web_experience_app.py -v -k "search_tool or chat_once or tool_agent" 2>&1 | tail -20
```

Expected: All 4 tests PASS.

- [ ] **Step 6: Run full unit suite to check for regressions**

```bash
pytest tests/unit/ -x -q 2>&1 | tail -20
```

Expected: All pass.

- [ ] **Step 7: Commit**

```bash
git add tests/unit/servers/web/test_web_experience_app.py
git commit -m "test(web): add smoke tests for search_tool, chat_once, and tool_agent modes"
```

---

## Task 8: Push and open PR

- [ ] **Step 1: Push branch**

```bash
git push origin feat/mlx-lm-backend
```

- [ ] **Step 2: Update PR #242 description to reflect new scope**

```bash
gh pr edit 242 --title "feat(web): add tool_agent mode, fix transformers 5.x compat, test coverage" \
  --body "$(cat <<'EOF'
## Summary

- **Tool Agent mode** (`tool_agent`): wire `ToolAgentLoop` into `POST /api/agent`; tools come from `build_search_tool(search_url)` + `tool_registry.list_tools()` — any tool registered via ToolPanel is automatically available
- **Transformers 5.x compat**: `apply_chat_template(tokenize=False)` + explicit `encode()` in `_build_prompt_ids_sync`
- **Session cleanup**: close `aiohttp.ClientSession` after each bamboogle eval run
- **UI**: add Tool Agent to mode selector; show `rounds_used`/`num_turns` badges in SessionTimeline
- **Config**: `TOOL_AGENT_PARSER` env var (default `json`; options: `hermes`, `llama3`)
- **Tests**: fixture fix, `_build_prompt_ids_sync` unit tests, `search_tool`/`chat_once`/`tool_agent` smoke tests

## Test plan

**Terminal 1** — start mlx-lm inference server:
```bash
HF_HUB_DISABLE_XET=1 mlx_lm.server --model mlx-community/Qwen2.5-1.5B-Instruct-4bit --port 8081
```

**Terminal 2** — run bamboogle smoke test:
```bash
bin/run_bamboogle_eval.sh \
  --server_url http://localhost:8081 \
  --model mlx-community/Qwen2.5-1.5B-Instruct-4bit \
  --smoke
```

**Unit tests:**
```bash
pytest tests/unit/ -x -q
```

Expected: all pass, no `Unclosed client session` warnings.

🤖 Generated with [Claude Code](https://claude.com/claude-code)
EOF
)"
```

---

## Task 9: Verification Checklist (Areas 5–11, no new code)

These are run-and-observe tasks. Fix any breakage found inline before moving on.

### 9a — Web Search Servers

- [ ] Start SerpAPI server and confirm health: `python3 -m src.internal.servers.web_search.serp --port 8000` then `curl -s http://localhost:8000/health`
- [ ] Confirm `bin/run_web_stack.sh` reads `SERP_API_KEY` from `.env` and starts without error

### 9b — Demo Retrieval Server

- [ ] Start demo server: `python3 -m src.internal.servers.retrieval.demo --corpus_path data/corpus.jsonl`
- [ ] Query it: `curl -s -X POST http://localhost:8001/retrieve -H "Content-Type: application/json" -d '{"query":"FAISS","topk":3}'`
- [ ] Expected: JSON with non-empty `results` list

### 9c — Connectors & ToolPanel

- [ ] Start full web stack: `bin/run_web_stack.sh`
- [ ] Open `http://127.0.0.1:5173` → Connectors panel → create a WebConnector
- [ ] Open Tools panel → confirm tools list loads → test-invoke a tool

### 9d — MCP Server

- [ ] Install MCP extra if not done: `pip install -e ".[mcp]"`
- [ ] Start: `uvicorn src.internal.mcp_server.api:mcp_app --port 8090`
- [ ] Inspect: `npx @modelcontextprotocol/inspector http://localhost:8090/`
- [ ] Confirm `search_indexed_documents` and `ask_agentic_search` tools appear

### 9e — Admin & Observability

- [ ] Generate dev token: `export TOKEN=$(bin/gen_dev_token.sh)`
- [ ] `curl -s http://localhost:7860/health` → `{"status":"ok"}`
- [ ] `curl -s "http://localhost:7860/analytics/query?start=2024-01-01&end=2026-12-31" -H "Authorization: Bearer $TOKEN"` → JSON response

---

## Self-Review Notes

- `AgentLoopOutput` is imported from `src.agents.base` in Task 7 — this matches `from src.agents.base import AgentLoopOutput` which is exported via `src/__init__.py`.
- `_run_direct_search` is patched at `src.internal.servers.web.app._run_direct_search` — correct module path for `TestClient`.
- `getattr(m, "metadata", None) or {}` in Task 6 is safe: if the DB returns objects without `.metadata`, it silently falls back to `{}`.
- `ToolAgentLoop` is patched at `src.agents.tool_calling.ToolAgentLoop` — the import in `app.py` uses this path.
