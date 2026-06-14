# Tool Registry HTTP Tests + Chat Orchestration Fixes

**Goal:** Close two remaining gaps from the full-stack-polish spec: (1) add HTTP-level unit tests for all `/admin/tools/*` endpoints; (2) add unit tests for citation extraction through the web API and wire chat-history trimming into `run_agent` so long sessions do not overflow the LLM context window.

**Architecture:** All tool routes already exist in `src/internal/servers/tools/api.py` and are registered at `app.py:206`. The citation pipeline already works (`extract_citations` → `AnswerGenerationResult.citations` → `AgentExperienceResponse.citations`). Two things are genuinely missing: web-layer HTTP tests for the tool endpoints, and a history-trimming guard in `run_agent` (the enterprise `compress_chat_history` is only used in the full Postgres/Redis pipeline, not the lightweight web endpoint).

**Tech Stack:** Python 3.12, FastAPI, pytest, `TestClient`, `src/tools/registry.py`, `src/context/utils.py`.

---

## Gaps to fix

### Gap 1 — No HTTP tests for `/admin/tools/*` endpoints

`create_tools_router` wires five routes:

| Method | Path | Expected behaviour |
|--------|------|--------------------|
| `GET` | `/admin/tools` | 200 → `[]` when empty; list of `ToolView` when populated |
| `POST` | `/admin/tools/openapi` | 201 → `{provider_id, tool_names}` for a valid OpenAPI 3.x JSON |
| `GET` | `/admin/tools/{name}` | 200 → `ToolView`; 404 for unknown name |
| `POST` | `/admin/tools/{name}/invoke` | 200 → `{response, raw, errors}`; 404 for unknown name |
| `DELETE` | `/admin/tools/openapi/{provider_id}` | 204; 404 for unknown provider |

Auth: every route requires an admin JWT; unauthenticated requests must return 401.

### Gap 2 — Citation extraction not tested through the web API

`extract_citations` uses `_CITATION_RE = re.compile(r"\[(D\d+)\]")`. When the LLM answer contains `[D1]`, `AgentExperienceResponse.citations` should contain `"D1"`. There are no tests asserting this end-to-end through the `/api/agent` endpoint.

### Gap 3 — Chat history grows unbounded in `run_agent`

`run_agent` passes all stored messages as `history` to `answer_with_retrieval` (and to `AgenticRAGLoop.run`). After 50+ exchanges the prompt will exceed most LLM context windows. `compress_chat_history` is not usable here (requires Postgres, Redis, and the full enterprise LLM pipeline). The fix is a simple `_trim_history` guard: keep the most-recent `MAX_HISTORY_MESSAGES = 40` messages before passing to the pipeline.

---

## Files changed

| File | Change |
|------|--------|
| `tests/unit/servers/web/test_tool_admin_api.py` | New — HTTP tests for all `/admin/tools/*` routes |
| `tests/unit/servers/web/test_web_experience_app.py` | Add citation-extraction assertion to existing `chat_once` smoke test |
| `src/internal/servers/web/app.py` | Add `_trim_history` + call it before `answer_with_retrieval` and `AgenticRAGLoop.run` |

---

## Task 1: HTTP tests for `/admin/tools/*` endpoints

**Files:**
- Create: `tests/unit/servers/web/test_tool_admin_api.py`

### Step 1 — Minimal OpenAPI fixture

```python
import json

_OPENAPI_JSON = json.dumps({
    "openapi": "3.0.0",
    "info": {"title": "Echo API", "description": "Echo words back."},
    "servers": [{"url": "https://echo.test"}],
    "paths": {
        "/echo": {
            "post": {
                "operationId": "echo_word",
                "summary": "Echo a word.",
                "requestBody": {
                    "content": {
                        "application/json": {
                            "schema": {
                                "type": "object",
                                "properties": {
                                    "word": {"type": "string"}
                                },
                                "required": ["word"],
                            }
                        }
                    }
                },
                "responses": {"200": {"description": "OK"}},
            }
        }
    },
})
```

### Step 2 — Auth helper

Reuse the pattern from `test_admin_observability_surface.py`:

```python
from src.internal.auth import generate_user_jwt_token
from src.internal.configs import AppSettings, AuthSettings

_ADMIN = "admin"

def _admin_headers() -> dict[str, str]:
    token = generate_user_jwt_token(user_id=_ADMIN)
    return {"Authorization": f"Bearer {token}"}

def _settings() -> AppSettings:
    return AppSettings(auth=AuthSettings(super_users=(_ADMIN,)))
```

### Step 3 — Test: unauthenticated → 401

```python
from fastapi.testclient import TestClient
from src.internal.servers.web.app import create_web_app, SearchExperienceSettings
from src.tools.registry import ToolRegistry, tool_registry

def test_list_tools_requires_auth(tmp_path):
    app = create_web_app(
        SearchExperienceSettings(db_path=tmp_path / "state.sqlite3"),
        app_settings=_settings(),
    )
    client = TestClient(app)
    response = client.get("/admin/tools")
    assert response.status_code == 401
```

### Step 4 — Test: list tools empty

```python
def test_list_tools_empty(tmp_path):
    # Isolate registry from other tests
    from src.tools.registry import ToolRegistry
    original = tool_registry._entries.copy()
    tool_registry._entries.clear()
    try:
        app = create_web_app(
            SearchExperienceSettings(db_path=tmp_path / "state.sqlite3"),
            app_settings=_settings(),
        )
        client = TestClient(app)
        response = client.get("/admin/tools", headers=_admin_headers())
        assert response.status_code == 200
        assert response.json() == []
    finally:
        tool_registry._entries.update(original)
```

### Step 5 — Test: register OpenAPI tools

```python
def test_register_openapi_creates_tools(tmp_path):
    from src.tools.registry import ToolRegistry
    original = tool_registry._entries.copy()
    tool_registry._entries.clear()
    try:
        app = create_web_app(
            SearchExperienceSettings(db_path=tmp_path / "state.sqlite3"),
            app_settings=_settings(),
        )
        client = TestClient(app)
        response = client.post(
            "/admin/tools/openapi",
            json={"name": "Echo API", "openapi_json": _OPENAPI_JSON},
            headers=_admin_headers(),
        )
        assert response.status_code == 201
        data = response.json()
        assert data["provider_id"]
        assert "echo_word" in data["tool_names"]

        # Verify tool appears in list
        list_resp = client.get("/admin/tools", headers=_admin_headers())
        names = [t["name"] for t in list_resp.json()]
        assert "echo_word" in names
    finally:
        tool_registry._entries.clear()
        tool_registry._entries.update(original)
```

### Step 6 — Test: get tool by name, 404 on unknown

```python
def test_get_tool_returns_view(tmp_path):
    from src.tools.registry import ToolRegistry
    original = tool_registry._entries.copy()
    tool_registry._entries.clear()
    try:
        app = create_web_app(
            SearchExperienceSettings(db_path=tmp_path / "state.sqlite3"),
            app_settings=_settings(),
        )
        client = TestClient(app)
        client.post(
            "/admin/tools/openapi",
            json={"name": "Echo API", "openapi_json": _OPENAPI_JSON},
            headers=_admin_headers(),
        )
        response = client.get("/admin/tools/echo_word", headers=_admin_headers())
        assert response.status_code == 200
        assert response.json()["name"] == "echo_word"

        missing = client.get("/admin/tools/nonexistent", headers=_admin_headers())
        assert missing.status_code == 404
    finally:
        tool_registry._entries.clear()
        tool_registry._entries.update(original)
```

### Step 7 — Test: invoke a built-in function tool

```python
def test_invoke_function_tool_via_http(tmp_path):
    from src.tools.registry import ToolRegistry
    import asyncio

    original = tool_registry._entries.copy()
    tool_registry._entries.clear()
    try:
        def double(n: int) -> int:
            """Double a number."""
            return n * 2

        tool_registry.register(double)
        app = create_web_app(
            SearchExperienceSettings(db_path=tmp_path / "state.sqlite3"),
            app_settings=_settings(),
        )
        client = TestClient(app)
        response = client.post(
            "/admin/tools/double/invoke",
            json={"arguments": {"n": 7}},
            headers=_admin_headers(),
        )
        assert response.status_code == 200
        data = response.json()
        assert data["errors"] == []
        assert "14" in data["response"]
    finally:
        tool_registry._entries.clear()
        tool_registry._entries.update(original)
```

### Step 8 — Test: invoke unknown tool → 404

```python
def test_invoke_unknown_tool_returns_404(tmp_path):
    app = create_web_app(
        SearchExperienceSettings(db_path=tmp_path / "state.sqlite3"),
        app_settings=_settings(),
    )
    client = TestClient(app)
    response = client.post(
        "/admin/tools/ghost/invoke",
        json={"arguments": {}},
        headers=_admin_headers(),
    )
    assert response.status_code == 404
```

### Step 9 — Test: delete OpenAPI provider

```python
def test_delete_openapi_provider(tmp_path):
    from src.tools.registry import ToolRegistry
    original = tool_registry._entries.copy()
    tool_registry._entries.clear()
    try:
        app = create_web_app(
            SearchExperienceSettings(db_path=tmp_path / "state.sqlite3"),
            app_settings=_settings(),
        )
        client = TestClient(app)
        reg = client.post(
            "/admin/tools/openapi",
            json={"name": "Echo API", "openapi_json": _OPENAPI_JSON},
            headers=_admin_headers(),
        )
        provider_id = reg.json()["provider_id"]

        delete_resp = client.delete(
            f"/admin/tools/openapi/{provider_id}", headers=_admin_headers()
        )
        assert delete_resp.status_code == 204

        # Tool is gone
        assert client.get("/admin/tools/echo_word", headers=_admin_headers()).status_code == 404

        # Deleting again → 404
        assert client.delete(
            f"/admin/tools/openapi/{provider_id}", headers=_admin_headers()
        ).status_code == 404
    finally:
        tool_registry._entries.clear()
        tool_registry._entries.update(original)
```

### Step 10 — Run tests

```bash
cd /Users/linghuang/Git/Agentic-Search
PYTHONPATH=src:. pytest tests/unit/servers/web/test_tool_admin_api.py -v 2>&1 | tail -20
```

Expected: all 7 tests PASS.

### Step 11 — Run full suite

```bash
PYTHONPATH=src:. pytest tests/unit/ -x -q 2>&1 | tail -5
```

Expected: all pass (1496+).

### Step 12 — Commit

```bash
git add tests/unit/servers/web/test_tool_admin_api.py
git commit -m "test(tools): add HTTP-level tests for /admin/tools/* endpoints"
```

---

## Task 2: Citation extraction test through the web API

**Files:**
- Modify: `tests/unit/servers/web/test_web_experience_app.py`

The citation regex is `[(D\d+)]`. When `answer_with_retrieval` produces an answer containing `[D1]`, `AgentExperienceResponse.citations` must contain `"D1"`.

### Step 1 — Add the test

In `tests/unit/servers/web/test_web_experience_app.py`, after existing `chat_once` tests, add:

```python
def test_run_agent_chat_once_citations_extracted(tmp_path):
    """AgentExperienceResponse.citations contains markers found in the answer text."""
    from src.context.models import AnswerGenerationResult, SearchContextBundle

    app = create_web_app(SearchExperienceSettings(db_path=tmp_path / "state.sqlite3"))
    client = TestClient(app)

    fake_result = AnswerGenerationResult(
        answer="See [D1] and [D2] for details.",
        citations=["D1", "D2"],
        context=SearchContextBundle(documents=[], sections=[]),
    )

    with patch("src.internal.servers.web.app.answer_with_retrieval", new_callable=AsyncMock) as mock_answer:
        mock_answer.return_value = fake_result
        response = client.post(
            "/api/agent", json={"query": "What is FAISS?", "mode": "chat_once"}
        )

    assert response.status_code == 200
    data = response.json()
    assert data["citations"] == ["D1", "D2"]
    assert "[D1]" in data["answer"]
```

### Step 2 — Run

```bash
PYTHONPATH=src:. pytest tests/unit/servers/web/test_web_experience_app.py -v -k "citation" 2>&1 | tail -10
```

Expected: PASS.

### Step 3 — Commit

```bash
git add tests/unit/servers/web/test_web_experience_app.py
git commit -m "test(web): assert citations are extracted from answer text in chat_once mode"
```

---

## Task 3: Wire history trimming into `run_agent`

**Files:**
- Modify: `src/internal/servers/web/app.py`

### Step 1 — Write the failing test

In `tests/unit/servers/web/test_web_experience_app.py`:

```python
def test_run_agent_trims_long_history(tmp_path):
    """When history exceeds MAX_HISTORY_MESSAGES, only the tail is passed to the LLM."""
    from src.internal.servers.web.app import MAX_HISTORY_MESSAGES

    app = create_web_app(SearchExperienceSettings(db_path=tmp_path / "state.sqlite3"))

    # Pre-fill a session with 60 messages
    from src.internal.db import AgenticSearchStore
    store = AgenticSearchStore(tmp_path / "state.sqlite3")
    session = store.create_chat_session(title="long")
    for i in range(60):
        role = "user" if i % 2 == 0 else "assistant"
        store.add_chat_message(session.id, role=role, content=f"msg {i}")
    store.close()

    captured: list = []

    async def _capture_history(question, *, llm=None, chat_history=None, **kw):
        captured.append(chat_history or [])
        return _answer_result(question)

    with patch("src.internal.servers.web.app.answer_with_retrieval", new=_capture_history):
        client = TestClient(app)
        client.post(
            "/api/agent",
            json={"query": "follow up", "mode": "chat_once", "session_id": session.id},
        )

    assert len(captured) == 1
    assert len(captured[0]) <= MAX_HISTORY_MESSAGES
```

### Step 2 — Run to verify it fails

```bash
PYTHONPATH=src:. pytest tests/unit/servers/web/test_web_experience_app.py -v -k "trims_long_history" 2>&1 | tail -10
```

Expected: FAIL — `ImportError: cannot import name 'MAX_HISTORY_MESSAGES'` (constant doesn't exist yet).

### Step 3 — Add `MAX_HISTORY_MESSAGES` constant and `_trim_history` helper

In `src/internal/servers/web/app.py`, near the other module-level constants (around the `_VALID_AGENT_MODES` block), add:

```python
MAX_HISTORY_MESSAGES = 40


def _trim_history(history: list, max_messages: int = MAX_HISTORY_MESSAGES) -> list:
    """Keep only the tail of chat history to avoid overflowing the LLM context."""
    if len(history) <= max_messages:
        return history
    return history[-max_messages:]
```

### Step 4 — Call `_trim_history` before LLM calls

In `run_agent`, locate the line that reads history from DB:

```python
history = [
    ChatMessage(role=message.role, content=message.content)
    for message in db.list_chat_messages(session_id)
]
```

Replace with:

```python
history = _trim_history([
    ChatMessage(role=message.role, content=message.content)
    for message in db.list_chat_messages(session_id)
])
```

### Step 5 — Run to verify it passes

```bash
PYTHONPATH=src:. pytest tests/unit/servers/web/test_web_experience_app.py -v -k "trims_long_history" 2>&1 | tail -10
```

Expected: PASS.

### Step 6 — Run full unit suite

```bash
PYTHONPATH=src:. pytest tests/unit/ -x -q 2>&1 | tail -5
```

Expected: all pass.

### Step 7 — Commit

```bash
git add src/internal/servers/web/app.py tests/unit/servers/web/test_web_experience_app.py
git commit -m "feat(web): trim chat history to last 40 messages before LLM call"
```

---

## Task 4: Push branch and open PR

### Step 1 — Create feature branch

```bash
git checkout -b feat/tool-registry-tests-chat-history-trim
git cherry-pick HEAD~3..HEAD   # if committing on main — skip if already on branch
```

### Step 2 — Push and open PR

```bash
git push -u origin feat/tool-registry-tests-chat-history-trim
gh pr create \
  --title "feat(tools+web): HTTP tests for /admin/tools, citation test, history trimming" \
  --body "$(cat <<'EOF'
## Summary

- **Area 8 — Tool registry HTTP tests**: 7 new TestClient tests cover all `/admin/tools/*` endpoints (list, register OpenAPI, get by name, invoke, delete provider, auth guard, 404 handling)
- **Area 9a — Citation extraction test**: asserts `AgentExperienceResponse.citations` contains marker strings extracted from the answer text in `chat_once` mode
- **Area 9b — History trimming**: adds `MAX_HISTORY_MESSAGES = 40` constant and `_trim_history` guard; long sessions no longer pass 60+ messages to the LLM context window

## Test plan

\`\`\`bash
PYTHONPATH=src:. pytest tests/unit/ -x -q
\`\`\`

Expected: all pass.

🤖 Generated with [Claude Code](https://claude.com/claude-code)
EOF
)"
```

---

## Self-Review Notes

- **Registry isolation**: Tests that modify `tool_registry._entries` must save/restore the original dict in a `try/finally` block. The registry is a module-level singleton shared across tests in the same process.
- **`_trim_history` placement**: Applied once, immediately after building `history` from the DB, before any mode branch is entered. This ensures `chat_once`, `hybrid_search`, `chat_loop`, `search_agent`, and `tool_agent` all receive a trimmed history.
- **`MAX_HISTORY_MESSAGES = 40`**: Keeps ~20 exchange rounds. This is a simple safeguard — not full summarization. The enterprise `compress_chat_history` (in `src/internal/chat/compression.py`) belongs in the full Postgres/Redis pipeline, not here.
- **Citation test mock**: `AnswerGenerationResult` must be imported from `src.context.models` (not `src.context` directly) and `SearchContextBundle` requires `documents=[]` and `sections=[]`.
