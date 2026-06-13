# Search Agent UI Integration — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Wire `SearchAgentLoop` into the existing web UI as a fifth agent mode (`search_agent`) so users can type a question in the browser and receive a grounded answer from the local MPS model.

**Architecture:** Add `SEARCH_AGENT_MODEL` / `SEARCH_AGENT_DEVICE` env vars to `AppSettings`. At app startup, load the tokenizer and create a `LocalServerManager` stored on `app.state`. The `/api/agent` endpoint gains a `search_agent` branch that awaits `SearchAgentLoop.run()` directly (it is already async) and maps `AgentLoopOutput` → `AgentExperienceResponse` using the existing `ContextDocument.from_search_result()` and `_document_view()` helpers. The frontend adds `"search_agent"` to the mode dropdown.

**Tech Stack:** FastAPI, `SearchAgentLoop` (`src/agents/search.py`), `LocalServerManager` (`examples/run_agentic_search.py`), React/TypeScript

**Spec:** `docs/superpowers/specs/2026-06-13-search-agent-ui-integration-design.md`

---

## File Map

| File | Change |
|------|--------|
| `src/internal/configs/app_configs.py` | Add `search_agent_model` + `search_agent_device` to `AppSettings` and `load_app_settings` |
| `src/internal/servers/web/app.py` | Extend lifespan; add `"search_agent"` to `_VALID_AGENT_MODES`; add `search_agent` branch in `run_agent()` |
| `web/src/types.ts` | Add `"search_agent"` to `AgentMode` union |
| `web/src/components/SearchComposer.tsx` | Add "Search Agent" to `MODE_OPTIONS` |
| `bin/run_web_stack.sh` | New script: starts all three processes in one command |
| `tests/unit/test_configs.py` | Test new env vars are read correctly |
| `tests/unit/servers/web/test_web_experience_app.py` | Test `search_agent` mode validation and 400 when not configured |

---

## Task 1: Add search_agent settings to AppSettings

**Files:**
- Modify: `src/internal/configs/app_configs.py`
- Test: `tests/unit/test_configs.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/unit/test_configs.py`:

```python
def test_load_app_settings_reads_search_agent_config():
    settings = load_app_settings(
        {
            "SEARCH_AGENT_MODEL": "Qwen/Qwen2.5-1.5B-Instruct",
            "SEARCH_AGENT_DEVICE": "mps",
        }
    )
    assert settings.search_agent_model == "Qwen/Qwen2.5-1.5B-Instruct"
    assert settings.search_agent_device == "mps"


def test_load_app_settings_search_agent_defaults_to_none_and_mps():
    settings = load_app_settings({})
    assert settings.search_agent_model is None
    assert settings.search_agent_device == "mps"
```

- [ ] **Step 2: Run tests to confirm they fail**

```bash
pytest tests/unit/test_configs.py::test_load_app_settings_reads_search_agent_config \
       tests/unit/test_configs.py::test_load_app_settings_search_agent_defaults_to_none_and_mps -v
```

Expected: `AttributeError: 'AppSettings' object has no attribute 'search_agent_model'`

- [ ] **Step 3: Add fields to AppSettings**

In `src/internal/configs/app_configs.py`, add two fields to the `AppSettings` dataclass (after `dev_mode: bool = False` at line 140):

```python
    search_agent_model: str | None = None
    search_agent_device: str = "mps"
```

- [ ] **Step 4: Add loading in load_app_settings**

In `load_app_settings()`, add two entries before the closing `)` of the `return AppSettings(...)` call (after `dev_mode=get_env_bool(source, "DEV_MODE", False),`):

```python
        search_agent_model=get_env_str(source, "SEARCH_AGENT_MODEL", None),
        search_agent_device=get_env_str(source, "SEARCH_AGENT_DEVICE", "mps"),
```

- [ ] **Step 5: Run tests to confirm they pass**

```bash
pytest tests/unit/test_configs.py -v
```

Expected: all tests pass.

- [ ] **Step 6: Commit**

```bash
git add src/internal/configs/app_configs.py tests/unit/test_configs.py
git commit -m "feat(config): add SEARCH_AGENT_MODEL and SEARCH_AGENT_DEVICE settings"
```

---

## Task 2: Wire search_agent mode into the web backend

**Files:**
- Modify: `src/internal/servers/web/app.py`
- Test: `tests/unit/servers/web/test_web_experience_app.py`

- [ ] **Step 1: Write the failing tests**

Add to `tests/unit/servers/web/test_web_experience_app.py`:

```python
def test_search_agent_mode_is_valid():
    from src.internal.servers.web.app import _normalize_agent_mode
    # Should not raise — "search_agent" must be in _VALID_AGENT_MODES
    assert _normalize_agent_mode("search_agent") == "search_agent"


def test_search_agent_returns_400_when_not_configured(tmp_path):
    app = create_web_app(SearchExperienceSettings(db_path=tmp_path / "state.sqlite3"))
    client = TestClient(app)
    response = client.post(
        "/api/agent",
        json={"query": "What is FAISS?", "mode": "search_agent"},
    )
    assert response.status_code == 400
    assert "SEARCH_AGENT_MODEL" in response.json()["detail"]
```

- [ ] **Step 2: Run tests to confirm they fail**

```bash
pytest tests/unit/servers/web/test_web_experience_app.py::test_search_agent_mode_is_valid \
       tests/unit/servers/web/test_web_experience_app.py::test_search_agent_returns_400_when_not_configured -v
```

Expected: first test raises `HTTPException` (mode not in valid set); second test returns 422 not 400.

- [ ] **Step 3: Add "search_agent" to _VALID_AGENT_MODES**

In `src/internal/servers/web/app.py`, update the `_VALID_AGENT_MODES` set (at line 582):

```python
_VALID_AGENT_MODES = {
    "search_tool",
    "hybrid_search",
    "chat_once",
    "chat_loop",
    "search_agent",
}
```

- [ ] **Step 4: Add imports at the top of app.py**

Add after the existing imports block (after `from src.tools import search_tool`):

```python
from src.agents.search import SearchAgentLoop, SearchAgentLoopConfig
from src.context.models import ContextDocument
```

- [ ] **Step 5: Extend the lifespan to init the search agent**

Replace the existing `lifespan` context manager (lines 272–280) with:

```python
    @asynccontextmanager
    async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
        seed_db(db)
        check_router_auth(_app, PUBLIC_ENDPOINT_SPECS)
        _app.state.search_agent_manager = None
        _app.state.search_agent_tokenizer = None
        if resolved.search_agent_model:
            try:
                from transformers import AutoTokenizer
                from examples.run_agentic_search import LocalServerManager

                tokenizer = AutoTokenizer.from_pretrained(
                    resolved.search_agent_model,
                    trust_remote_code=True,
                    local_files_only=True,
                )
                if tokenizer.pad_token_id is None:
                    tokenizer.pad_token_id = tokenizer.eos_token_id
                manager = LocalServerManager(
                    model_path=resolved.search_agent_model,
                    device=resolved.search_agent_device,
                    allow_unsafe_mps=True,
                    local_files_only=True,
                )
                _app.state.search_agent_tokenizer = tokenizer
                _app.state.search_agent_manager = manager
                logger.info(
                    "search_agent: loaded tokenizer for %s on %s",
                    resolved.search_agent_model,
                    resolved.search_agent_device,
                )
            except Exception:
                logger.exception("search_agent: failed to load model — mode will return 400")
        try:
            yield
        finally:
            if owns_store:
                db.close()
```

Note: `resolved` is the variable name for the loaded `AppSettings` in `create_web_app`. Check that the variable in scope is `resolved` (it is, from line 260 context). The imports are inside the lifespan to keep them lazy — torch/transformers only load if `SEARCH_AGENT_MODEL` is set.

- [ ] **Step 6: Add the search_agent branch in run_agent()**

In the `run_agent()` handler, add the following block **before** the `if mode == "search_tool":` block (i.e., after the `filters = ...` assignment block ends, before `try:`):

```python
        if mode == "search_agent":
            manager = http_request.app.state.search_agent_manager
            tokenizer = http_request.app.state.search_agent_tokenizer
            if manager is None or tokenizer is None:
                raise HTTPException(
                    status_code=400,
                    detail=(
                        "search_agent mode is not configured. "
                        "Set SEARCH_AGENT_MODEL in .env and restart the server."
                    ),
                )
            loop = SearchAgentLoop(
                tokenizer=tokenizer,
                server_manager=manager,
                search_config=SearchAgentLoopConfig(
                    search_url=search_url,
                    topk=top_k,
                ),
            )
            output = await loop.run(
                [{"role": "user", "content": query}],
                sampling_params={"temperature": 0.0, "max_tokens": 512},
            )
            answer = output.final_answer or ""
            documents: list[ContextDocument] = []
            if output.context is not None:
                for sc in output.context.turns:
                    for result in sc.results:
                        documents.append(
                            ContextDocument.from_search_result(
                                result, index=len(documents) + 1
                            )
                        )
            documents = _dedupe_documents(documents)
            db.add_chat_message(
                session_id,
                role="assistant",
                content=answer,
                metadata={
                    "citations": [doc.citation for doc in documents],
                    "document_ids": [doc.id for doc in documents],
                    "hooks": hook_metadata,
                    "mode": mode,
                },
            )
            messages = [
                ChatMessageView(role=m.role, content=m.content)
                for m in db.list_chat_messages(session_id)
            ]
            return AgentExperienceResponse(
                session_id=session_id,
                answer=answer,
                citations=[doc.citation for doc in documents],
                documents=[_document_view(doc) for doc in documents],
                messages=messages,
                hook_metadata=hook_metadata,
            )
```

- [ ] **Step 7: Run the tests**

```bash
pytest tests/unit/servers/web/test_web_experience_app.py -v
```

Expected: all existing tests pass + 2 new tests pass.

- [ ] **Step 8: Commit**

```bash
git add src/internal/servers/web/app.py tests/unit/servers/web/test_web_experience_app.py
git commit -m "feat(web): add search_agent mode to /api/agent endpoint"
```

---

## Task 3: Frontend — add Search Agent mode

**Files:**
- Modify: `web/src/types.ts`
- Modify: `web/src/components/SearchComposer.tsx`

No unit tests for these — `npm run typecheck` is the verification.

- [ ] **Step 1: Add "search_agent" to AgentMode**

In `web/src/types.ts`, update the `AgentMode` type (line 25):

```typescript
export type AgentMode = "search_tool" | "hybrid_search" | "chat_once" | "chat_loop" | "search_agent";
```

- [ ] **Step 2: Add Search Agent to MODE_OPTIONS in SearchComposer.tsx**

In `web/src/components/SearchComposer.tsx`, update `MODE_OPTIONS` (lines 7–12):

```typescript
const MODE_OPTIONS: Array<{ value: AgentMode; label: string }> = [
  { value: "search_tool", label: "Search: Direct Tool" },
  { value: "hybrid_search", label: "Search: Hybrid" },
  { value: "chat_once", label: "Chat: No Loop" },
  { value: "chat_loop", label: "Chat: Loop" },
  { value: "search_agent", label: "Search Agent (Local Model)" },
];
```

- [ ] **Step 3: Show retrieval URL for search_agent mode**

In `SearchComposer.tsx`, update the `isSearchMode` and `usesRetrievalUrl` logic (lines 55–60):

```typescript
  const isSearchMode = mode === "search_tool" || mode === "hybrid_search";
  const isSearchAgentMode = mode === "search_agent";
  const usesRetrievalUrl =
    isSearchAgentMode ||
    !isSearchMode ||
    sourceProvider === "retrieval" ||
    sourceProvider === "browser" ||
    sourceProvider === "all";
  const urlLabel =
    sourceProvider === "browser" ? "Browser Retrieval URL" : "Retrieval URL";
```

- [ ] **Step 4: Run type check**

```bash
cd /Users/linghuang/Git/Agentic-Search/web && npm run typecheck
```

Expected: no type errors.

- [ ] **Step 5: Commit**

```bash
git add web/src/types.ts web/src/components/SearchComposer.tsx
git commit -m "feat(ui): add Search Agent mode to mode selector"
```

---

## Task 4: Dev startup script

**Files:**
- Create: `bin/run_web_stack.sh`

- [ ] **Step 1: Create the script**

```bash
cat > /Users/linghuang/Git/Agentic-Search/bin/run_web_stack.sh << 'SCRIPT'
#!/usr/bin/env bash
# Start the full local dev stack: SerpAPI retrieval server + web backend + Vite frontend.
#
# Usage:
#   bin/run_web_stack.sh
#
# Required: SERP_API_KEY in .env
# Optional: SEARCH_AGENT_MODEL and SEARCH_AGENT_DEVICE in .env
#           (enables the "Search Agent" mode in the UI)
#
# Defaults:
#   SEARCH_AGENT_MODEL  (not set — Search Agent mode disabled unless you set it)
#   SEARCH_AGENT_DEVICE mps
#   SERP_PORT           8000
#   WEB_PORT            7860

set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# Load .env
if [ -f "$ROOT/.env" ]; then
  set -a
  # shellcheck disable=SC1091
  . "$ROOT/.env"
  set +a
fi

if [ -z "${SERP_API_KEY:-}" ]; then
  echo "ERROR: SERP_API_KEY not set (add it to .env or export it)" >&2
  exit 1
fi

SERP_PORT="${SERP_PORT:-8000}"
WEB_PORT="${WEB_PORT:-7860}"
SEARCH_AGENT_MODEL="${SEARCH_AGENT_MODEL:-}"
SEARCH_AGENT_DEVICE="${SEARCH_AGENT_DEVICE:-mps}"

echo ">> Starting SerpAPI retrieval server on port ${SERP_PORT}..."
PYTHONPATH="$ROOT" python3 -m src.internal.servers.web_search.serp --port "$SERP_PORT" &
SERP_PID=$!
trap 'echo ">> Stopping processes..."; kill $SERP_PID $WEB_PID 2>/dev/null; wait 2>/dev/null || true' EXIT

# Wait for retrieval server (up to 15s)
for i in $(seq 1 15); do
  if curl -sf "http://localhost:${SERP_PORT}/health" >/dev/null 2>&1; then
    echo ">> Retrieval server ready."
    break
  fi
  [ "$i" -eq 15 ] && { echo "ERROR: retrieval server did not start" >&2; exit 1; }
  sleep 1
done

echo ">> Starting web backend on port ${WEB_PORT}..."
if [ -n "$SEARCH_AGENT_MODEL" ]; then
  echo ">> Search Agent mode enabled: model=${SEARCH_AGENT_MODEL} device=${SEARCH_AGENT_DEVICE}"
else
  echo ">> Search Agent mode disabled (set SEARCH_AGENT_MODEL in .env to enable)"
fi

PYTHONPATH="$ROOT" \
SEARCH_AGENT_MODEL="$SEARCH_AGENT_MODEL" \
SEARCH_AGENT_DEVICE="$SEARCH_AGENT_DEVICE" \
AGENTIC_SEARCH_RETRIEVAL_URL="http://localhost:${SERP_PORT}/retrieve" \
  uvicorn src.internal.servers.web.app:app \
    --host 127.0.0.1 --port "$WEB_PORT" &
WEB_PID=$!

echo ">> Starting frontend dev server..."
echo ">> Open http://127.0.0.1:5173 when ready."
cd "$ROOT/web" && npm run dev
SCRIPT
chmod +x /Users/linghuang/Git/Agentic-Search/bin/run_web_stack.sh
```

- [ ] **Step 2: Verify it parses without errors**

```bash
bash -n /Users/linghuang/Git/Agentic-Search/bin/run_web_stack.sh && echo "Syntax OK"
```

Expected: `Syntax OK`

- [ ] **Step 3: Update CLAUDE.md dev startup section**

In `.claude/CLAUDE.md`, replace the "Running the 3-process local stack" section:

```markdown
### Running the local stack

```bash
# One command — reads SERP_API_KEY from .env, starts all three processes:
bin/run_web_stack.sh

# To enable Search Agent mode, add to .env first:
# SEARCH_AGENT_MODEL=Qwen/Qwen2.5-1.5B-Instruct
# SEARCH_AGENT_DEVICE=mps

# Manual (3 terminals):
# Terminal 1 — retrieval server (demo, port 8001)
python3 -m src.internal.servers.retrieval.demo --corpus_path data/corpus.jsonl

# Terminal 2 — web backend (port 7860)
PYTHONPATH=. uvicorn src.internal.servers.web.app:app --host 127.0.0.1 --port 7860

# Terminal 3 — frontend dev server (port 5173)
cd web && npm install && npm run dev
```

Open `http://127.0.0.1:5173`.
```

- [ ] **Step 4: Commit**

```bash
git add bin/run_web_stack.sh .claude/CLAUDE.md
git commit -m "feat(dev): add run_web_stack.sh to start all three processes in one command"
```

---

## Task 5: Push and open PR

- [ ] **Step 1: Run full test suite**

```bash
pytest tests/unit/ -v --tb=short 2>&1 | tail -20
```

Expected: all tests pass.

- [ ] **Step 2: Run frontend type check**

```bash
cd /Users/linghuang/Git/Agentic-Search/web && npm run typecheck
```

Expected: no errors.

- [ ] **Step 3: Push and create PR**

```bash
git push -u origin docs/bamboogle-eval-speedup-flags
gh pr create \
  --title "feat(web): wire SearchAgentLoop into UI as search_agent mode" \
  --body "$(cat <<'EOF'
## Summary

- Adds \`search_agent\` as a fifth mode to \`POST /api/agent\`
- At startup, loads tokenizer + \`LocalServerManager\` (if \`SEARCH_AGENT_MODEL\` is set); stored on \`app.state\`
- Per request: creates \`SearchAgentLoop\`, awaits \`run()\`, maps \`AgentLoopOutput\` → \`AgentExperienceResponse\` using existing \`ContextDocument.from_search_result()\` + \`_document_view()\`
- Frontend: adds "Search Agent (Local Model)" option to mode dropdown
- Adds \`bin/run_web_stack.sh\` to replace the 3-terminal startup workflow

## Usage

Add to \`.env\`:
\`\`\`
SEARCH_AGENT_MODEL=Qwen/Qwen2.5-1.5B-Instruct
SEARCH_AGENT_DEVICE=mps
\`\`\`

Then:
\`\`\`bash
bin/run_web_stack.sh
# open http://127.0.0.1:5173, select "Search Agent (Local Model)"
\`\`\`

## Test plan

- [ ] \`pytest tests/unit/ -v\` — all pass
- [ ] \`cd web && npm run typecheck\` — no errors
- [ ] With \`SEARCH_AGENT_MODEL\` unset: \`mode=search_agent\` returns HTTP 400 with clear message
- [ ] With \`SEARCH_AGENT_MODEL\` set: ask a question in the UI, get answer + source cards
- [ ] Other modes (chat_once, search_tool, etc.) unaffected

🤖 Generated with [Claude Code](https://claude.com/claude-code)
EOF
)"
```
