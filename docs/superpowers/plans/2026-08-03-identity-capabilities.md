# Identity Capabilities Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make a request's identity decide what it may see, what tools it is offered, and whether its memory is injected — from one resolver instead of three independent derivations.

**Architecture:** A new `src/internal/access/capabilities.py` maps the resolved user to a frozen `RequestCapabilities` record. Anonymous becomes an identity carrying `["public"]` rather than the absence of one, so `access_acl` is never empty. Tools stay owned by the registry: `ToolEntry` gains `user_scoped`, set at MCP registration from configuration, mirroring the existing `agent_callable` / `agent_exclude` pattern.

**Tech Stack:** Python 3.12, FastAPI, pytest, `mcp` client SDK.

**Spec:** `docs/superpowers/specs/2026-08-03-identity-capabilities-design.md`

## Global Constraints

- Work on branch `feat/identity-capabilities`, which is branched from `fix/auth-same-route` (PR #487, unmerged). Never commit to `main`.
- `access_acl` is never empty. `["public"]` for anonymous; public plus `user:` / `email:` / `group:` entries when signed in.
- A user id that no longer resolves in the store degrades to anonymous. It never raises.
- Anonymous gets no memory preamble and no user-scoped tools.
- Documents that declare no ACL remain public — do not change `SearchFilters.matches`.
- `python3 -m pytest` and `ruff check . && ruff format .` pass before every commit.
- Run all commands from the repo root with `PYTHONPATH=src:.` where a script needs it.

**One deviation from the spec, deliberate:** the spec names the registry accessor `tools_for(user_present=...)`. The registry already has `agent_tools()` filtering on `agent_callable`; a second method would duplicate that logic and let the two drift. This plan extends the existing accessor to `agent_tools(*, user_present: bool = True)` instead. Same behaviour, one method.

## File Structure

| File | Status | Responsibility |
| --- | --- | --- |
| `src/internal/access/capabilities.py` | Create | `RequestCapabilities` + `resolve_capabilities`. The only place identity becomes entitlement. |
| `tests/unit/access/test_capabilities.py` | Create | Resolver behaviour: anonymous, signed-in, unresolvable. |
| `src/internal/tools/registry.py` | Modify | `ToolEntry.user_scoped`; `register(user_scoped=)`; `agent_tools(user_present=)`; summaries. |
| `src/internal/tools/mcp_client.py` | Modify | `DEFAULT_USER_SCOPED`; `McpServerSpec.user_scoped`; parse + register. |
| `src/internal/configs/app_configs.py` | Modify | `mcp_user_scoped` setting. |
| `src/internal/servers/web/app.py` | Modify | Use the resolver for filters and memory; drop `memory_injection`; pass `user_present` to the tool runner. |
| `src/internal/servers/web/tool_agent_runner.py` | Modify | Accept `user_present` and ask the registry for the matching tools. |
| `src/internal/servers/query_and_chat/tool_backend.py` | Modify | Pass `user_present` from the resolved user. |
| `tests/unit/test_agent_callable_tools.py` | Modify | Extend for `user_scoped`. |
| `tests/unit/test_search_route_access_filters.py` | Modify | Anonymous must not see a restricted document. |

---

### Task 1: The capability resolver

**Files:**
- Create: `src/internal/access/capabilities.py`
- Create: `tests/unit/access/test_capabilities.py`

**Interfaces:**
- Consumes: `src.internal.access.access.PUBLIC_ACL` (`"public"`), `src.context.preprocessing.access_filters.build_access_filter`, `src.internal.memory.service.memory_preamble`, `src.internal.auth.users.AuthenticatedUser` (fields `id`, `email`, `group_ids`, `is_anonymous`).
- Produces: `RequestCapabilities(user_id: str | None, access_acl: list[str], memory_preamble: str)` and `resolve_capabilities(user, store) -> RequestCapabilities`.

- [ ] **Step 1: Write the failing test**

Create `tests/unit/access/test_capabilities.py`:

```python
"""Identity decides entitlement, in exactly one place."""

from __future__ import annotations

from src.internal.access.capabilities import RequestCapabilities, resolve_capabilities
from src.internal.auth.users import AuthenticatedUser


class _Store:
    """Minimal stand-in for AgenticSearchStore's memory reads."""

    def __init__(self, memories=None):
        self._memories = memories or {}

    def get_user_memories(self, user_id):
        return self._memories.get(user_id, [])


def test_anonymous_is_public_only():
    caps = resolve_capabilities(None, _Store())
    assert caps == RequestCapabilities(
        user_id=None, access_acl=["public"], memory_preamble=""
    )


def test_anonymous_user_object_is_treated_as_anonymous():
    user = AuthenticatedUser(id="anon", is_anonymous=True)
    assert resolve_capabilities(user, _Store()).user_id is None


def test_signed_in_user_gets_public_plus_their_own_entries():
    user = AuthenticatedUser(
        id="u1", email="a@b.c", group_ids=frozenset({"g1"})
    )
    caps = resolve_capabilities(user, _Store())
    assert caps.user_id == "u1"
    assert set(caps.access_acl) == {"public", "user:u1", "email:a@b.c", "group:g1"}


def test_access_acl_is_never_empty():
    # An empty list would read as "no filter" downstream, which is the hole
    # this resolver exists to close.
    for user in (None, AuthenticatedUser(id="u1")):
        assert resolve_capabilities(user, _Store()).access_acl


def test_signed_in_user_gets_their_memory():
    store = _Store({"u1": ["prefers hybrid retrieval"]})
    caps = resolve_capabilities(AuthenticatedUser(id="u1"), store)
    assert "prefers hybrid retrieval" in caps.memory_preamble


def test_anonymous_gets_no_memory_even_if_the_store_has_some():
    store = _Store({"u1": ["prefers hybrid retrieval"]})
    assert resolve_capabilities(None, store).memory_preamble == ""


def test_a_store_failure_degrades_to_no_memory():
    # The store is the source of truth (#476); a user whose row is gone must
    # not take the request down with it.
    class _Broken:
        def get_user_memories(self, user_id):
            raise RuntimeError("row is gone")

    caps = resolve_capabilities(AuthenticatedUser(id="u1"), _Broken())
    assert caps.memory_preamble == ""
    assert caps.user_id == "u1"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/unit/access/test_capabilities.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.internal.access.capabilities'`

- [ ] **Step 3: Write minimal implementation**

Create `src/internal/access/capabilities.py`:

```python
"""What a request's identity is entitled to.

Identity used to be re-derived at each site that cared — filters here, memory
there, tools somewhere else — so a path could enforce the ACL while its
neighbour did not. This is the one place that mapping happens.

Anonymous is an identity, not the absence of one: it carries ``["public"]``.
``access_acl`` is therefore never empty, so no caller can accidentally express
"unfiltered" by passing nothing.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from src.context.preprocessing.access_filters import build_access_filter
from src.internal.access.access import PUBLIC_ACL
from src.internal.memory.service import memory_preamble

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RequestCapabilities:
    """What this caller may see, and what it brings with it."""

    user_id: str | None
    access_acl: list[str]
    memory_preamble: str

    @property
    def user_present(self) -> bool:
        return self.user_id is not None


ANONYMOUS = RequestCapabilities(
    user_id=None, access_acl=[PUBLIC_ACL], memory_preamble=""
)


def resolve_capabilities(user, store) -> RequestCapabilities:
    """Map a resolved user (or ``None``) to its capabilities.

    ``store`` is required rather than optional because the memory preamble is
    read from it. Keeping it an argument leaves this a plain function with no
    global state, so the agent loops and MCP paths can call it too — something a
    FastAPI dependency could not reach.
    """
    if user is None or getattr(user, "is_anonymous", False):
        return ANONYMOUS

    user_id = user.id
    acl = build_access_filter(
        user_id,
        email=getattr(user, "email", None),
        group_ids=getattr(user, "group_ids", None),
    )
    try:
        preamble = memory_preamble(store, user_id)
    except Exception as exc:  # noqa: BLE001 — memory must never fail a request
        logger.warning("memory preamble unavailable for %s: %s", user_id, exc)
        preamble = ""
    return RequestCapabilities(
        user_id=user_id, access_acl=acl, memory_preamble=preamble
    )


__all__ = ["ANONYMOUS", "RequestCapabilities", "resolve_capabilities"]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m pytest tests/unit/access/test_capabilities.py -q`
Expected: PASS (7 tests)

- [ ] **Step 5: Commit**

```bash
ruff check . --fix && ruff format .
git add src/internal/access/capabilities.py tests/unit/access/test_capabilities.py
git commit -m "feat(access): resolve a request's identity into capabilities

Anonymous becomes an identity carrying [\"public\"] rather than the absence of
one, so access_acl is never empty and no caller can express \"unfiltered\" by
passing nothing. Memory failures degrade to no preamble instead of failing the
request."
```

---

### Task 2: `user_scoped` on the registry

**Files:**
- Modify: `src/internal/tools/registry.py`
- Modify: `tests/unit/test_agent_callable_tools.py`

**Interfaces:**
- Consumes: nothing from Task 1.
- Produces: `ToolEntry.user_scoped: bool`, `ToolRegistry.register(..., user_scoped: bool = False)`, `ToolRegistry.agent_tools(*, user_present: bool = True) -> list[Tool]`. Summaries gain a `user_scoped` key.

- [ ] **Step 1: Write the failing test**

Append to `tests/unit/test_agent_callable_tools.py`:

```python
# ---------------------------------------------------------------------------
# User-scoped tools. Memory tools write per user; offering them with no user
# pools everyone's memories into one shared bucket.
# ---------------------------------------------------------------------------


def test_tools_are_not_user_scoped_by_default():
    registry = ToolRegistry()
    registry.register(_tool("weather"))
    assert registry.list()[0].user_scoped is False


def test_user_scoped_tools_are_withheld_when_there_is_no_user():
    registry = ToolRegistry()
    registry.register(_tool("weather"))
    registry.register(_tool("save_memory"), user_scoped=True)

    assert [t.name for t in registry.agent_tools(user_present=False)] == ["weather"]
    assert {t.name for t in registry.agent_tools(user_present=True)} == {
        "weather",
        "save_memory",
    }


def test_user_scoped_tools_stay_registered_and_invocable():
    registry = ToolRegistry()
    registry.register(_tool("save_memory"), user_scoped=True)
    # Withheld from the agent, still reachable through /admin/tools.
    assert registry.get("save_memory") is not None
    assert registry.tool_summary("save_memory")["user_scoped"] is True


def test_agent_callable_still_wins_over_user_presence():
    registry = ToolRegistry()
    registry.register(_tool("runs_an_agent"), agent_callable=False, user_scoped=True)
    assert registry.agent_tools(user_present=True) == []
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/unit/test_agent_callable_tools.py -q`
Expected: FAIL — `TypeError: register() got an unexpected keyword argument 'user_scoped'`

- [ ] **Step 3: Write minimal implementation**

In `src/internal/tools/registry.py`, add the field to `ToolEntry` beneath `agent_callable`:

```python
    # Scoped to a specific user (per-user storage such as memory). Offered only
    # when a user is present; with none, an anonymous write would land in a
    # shared bucket and pool unrelated people's data.
    user_scoped: bool = False
```

Extend `register`:

```python
    def register(
        self,
        tool: Tool,
        *,
        source: str = "function",
        provider_id: str | None = None,
        agent_callable: bool = True,
        user_scoped: bool = False,
    ) -> None:
        """Add a tool to the registry (replaces any existing tool with the same name)."""
        self._entries[tool.name] = ToolEntry(
            tool=tool,
            source=source,
            provider_id=provider_id,
            agent_callable=agent_callable,
            user_scoped=user_scoped,
        )
        logger.debug("Tool registered: %s (source=%s)", tool.name, source)
```

Replace `agent_tools`:

```python
    def agent_tools(self, *, user_present: bool = True) -> list[Tool]:
        """Tools an agent loop may be offered for this caller.

        ``user_present=False`` also withholds user-scoped tools; see
        ``ToolEntry.user_scoped``.
        """
        return [
            e.tool
            for e in self._entries.values()
            if e.agent_callable and (user_present or not e.user_scoped)
        ]
```

Add `"user_scoped": entry.user_scoped,` to the dict returned by `tool_summary`, and `"user_scoped": e.user_scoped,` to each dict in `all_summaries`.

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m pytest tests/unit/test_agent_callable_tools.py tests/unit/test_tool_registry.py -q`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
ruff check . --fix && ruff format .
git add src/internal/tools/registry.py tests/unit/test_agent_callable_tools.py
git commit -m "feat(tools): mark tools user-scoped so they are withheld without a user"
```

---

### Task 3: MCP registers its memory tools as user-scoped

**Files:**
- Modify: `src/internal/tools/mcp_client.py`
- Modify: `src/internal/configs/app_configs.py`
- Modify: `tests/unit/test_agent_callable_tools.py`

**Interfaces:**
- Consumes: `ToolRegistry.register(..., user_scoped=...)` from Task 2.
- Produces: `mcp_client.DEFAULT_USER_SCOPED: frozenset[str]`, `McpServerSpec.user_scoped: frozenset[str]`, `parse_mcp_servers(raw, *, token=None, agent_exclude=None, user_scoped=None)`, `AppSettings.mcp_user_scoped: str | None`.

- [ ] **Step 1: Write the failing test**

Append to `tests/unit/test_agent_callable_tools.py`:

```python
@pytest.mark.asyncio
async def test_mcp_memory_tools_register_as_user_scoped(fake_mcp):
    from src.internal.tools.mcp_client import DEFAULT_USER_SCOPED

    registry = ToolRegistry()
    await register_mcp_tools(
        registry,
        [
            McpServerSpec(
                name="agentic", url="http://x/", user_scoped=DEFAULT_USER_SCOPED
            )
        ],
    )

    assert registry._entries["save_memory"].user_scoped is True
    assert [t.name for t in registry.agent_tools(user_present=False)] == [
        "ask_agentic_search"
    ]


def test_parse_mcp_servers_applies_the_default_user_scope():
    from src.internal.tools.mcp_client import parse_mcp_servers

    spec = parse_mcp_servers("agentic=http://x/")[0]
    assert "save_memory" in spec.user_scoped
    assert "search_memories" in spec.user_scoped


def test_parse_mcp_servers_accepts_an_explicit_user_scope():
    from src.internal.tools.mcp_client import parse_mcp_servers

    spec = parse_mcp_servers("a=http://x/", user_scoped="foo, bar")[0]
    assert spec.user_scoped == frozenset({"foo", "bar"})
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/unit/test_agent_callable_tools.py -q -k "user_scope"`
Expected: FAIL — `ImportError: cannot import name 'DEFAULT_USER_SCOPED'`

- [ ] **Step 3: Write minimal implementation**

In `src/internal/tools/mcp_client.py`, beneath `DEFAULT_AGENT_EXCLUDE`:

```python
# Remote tools backed by per-user storage. Offered only when a caller has an
# identity to scope them to; anonymously they would write to a shared bucket.
DEFAULT_USER_SCOPED: frozenset[str] = frozenset(
    {
        "save_memory",
        "update_memory_from_conversation",
        "generate_user_profile",
        "get_user_profile",
        "search_memories",
        "consolidate_memories",
    }
)
```

Add to `McpServerSpec`, beneath `agent_exclude`:

```python
    # Tool names this server exposes that are backed by per-user storage.
    user_scoped: frozenset[str] = frozenset()
```

Extend `parse_mcp_servers` with a `user_scoped: str | None = None` keyword, resolving it exactly as `agent_exclude` is resolved:

```python
    scoped = (
        frozenset(n.strip() for n in user_scoped.split(",") if n.strip())
        if user_scoped is not None
        else DEFAULT_USER_SCOPED
    )
```

and pass `user_scoped=scoped` into each `McpServerSpec(...)` it builds.

In `register_mcp_tools`, extend the registration call:

```python
            registry.register(
                _build_tool(spec, remote),
                source=MCP_SOURCE,
                provider_id=spec.name,
                agent_callable=remote.name not in spec.agent_exclude,
                user_scoped=remote.name in spec.user_scoped,
            )
```

In `src/internal/configs/app_configs.py`, add beneath `mcp_agent_exclude`:

```python
    # Remote tool names backed by per-user storage. Unset uses
    # mcp_client.DEFAULT_USER_SCOPED.
    mcp_user_scoped: str | None = None
```

and in `load_app_settings`, beneath the `mcp_agent_exclude` line:

```python
        mcp_user_scoped=get_env_str(source, "AGENTIC_SEARCH_MCP_USER_SCOPED", None),
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m pytest tests/unit/test_agent_callable_tools.py tests/unit/test_mcp_client.py -q`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
ruff check . --fix && ruff format .
git add src/internal/tools/mcp_client.py src/internal/configs/app_configs.py tests/unit/test_agent_callable_tools.py
git commit -m "feat(tools): register MCP memory tools as user-scoped"
```

---

### Task 4: The tool runner asks for the caller's tools

**Files:**
- Modify: `src/internal/servers/web/tool_agent_runner.py`
- Modify: `src/internal/servers/query_and_chat/tool_backend.py:86-96`
- Modify: `src/internal/servers/web/app.py:1082`, `src/internal/servers/web/app.py:1680`
- Modify: `tests/unit/servers/web/test_loop_runners.py`

**Interfaces:**
- Consumes: `ToolRegistry.agent_tools(user_present=...)` from Task 2.
- Produces: `_run_tool_agent(..., user_present: bool = True)`.

- [ ] **Step 1: Write the failing test**

Append to `tests/unit/servers/web/test_loop_runners.py`:

```python
@pytest.mark.asyncio
async def test_tool_agent_withholds_user_scoped_tools_without_a_user(monkeypatch):
    from src.internal.tools.base import FunctionTool
    from src.internal.tools.registry import ToolRegistry

    async def _noop() -> str:
        return ""

    registry = ToolRegistry()
    registry.register(
        FunctionTool(
            _noop,
            name="save_memory",
            description="Save a memory.",
            parameters={"type": "object"},
        ),
        user_scoped=True,
    )
    monkeypatch.setattr("src.internal.tools.tool_registry", registry)

    captured = _capture_tool_agent_loop(monkeypatch)
    await web_app._run_tool_agent(
        "q",
        manager=MagicMock(),
        tokenizer=MagicMock(),
        search_url="http://x/retrieve",
        history=[],
        resolved=types.SimpleNamespace(tool_agent_parser="json"),
        on_turn=None,
        with_search_tool=False,
        user_present=False,
    )
    assert "save_memory" not in [t.name for t in captured["tools"]]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/unit/servers/web/test_loop_runners.py -q -k withholds`
Expected: FAIL — `TypeError: _run_tool_agent() got an unexpected keyword argument 'user_present'`

- [ ] **Step 3: Write minimal implementation**

In `src/internal/servers/web/tool_agent_runner.py`, add the parameter to `_run_tool_agent`'s signature after `with_search_tool: bool`:

```python
    user_present: bool = True,
```

and replace the tool lookup:

```python
    # agent_tools() excludes anything registered as not agent-callable, and —
    # without a user — anything backed by per-user storage.
    tools = list(tool_registry.agent_tools(user_present=user_present))
```

In `src/internal/servers/query_and_chat/tool_backend.py`, inside `_run`, add to the `_run_tool_agent(...)` call:

```python
                user_present=user is not None and not user.is_anonymous,
```

In `src/internal/servers/web/app.py` at both call sites (lines ~1082 and ~1680), add:

```python
                    user_present=user_id is not None,
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m pytest tests/unit/servers/web/test_loop_runners.py tests/unit/test_tool_backend.py -q`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
ruff check . --fix && ruff format .
git add src/internal/servers/web/tool_agent_runner.py src/internal/servers/query_and_chat/tool_backend.py src/internal/servers/web/app.py tests/unit/servers/web/test_loop_runners.py
git commit -m "feat(tools): withhold user-scoped tools from anonymous callers"
```

---

### Task 5: Wire capabilities into the request path

**Files:**
- Modify: `src/internal/servers/web/app.py:150`, `:170`, `:1409-1417`, `:1452-1458`
- Modify: `tests/unit/test_search_route_access_filters.py`

**Interfaces:**
- Consumes: `resolve_capabilities` from Task 1.
- Produces: no new public names. `AppSettings.memory_injection` is removed.

- [ ] **Step 1: Write the failing test**

Append to `tests/unit/test_search_route_access_filters.py`:

```python
def test_anonymous_callers_are_public_only():
    # Anonymous used to carry no ACL at all, so a document restricted to
    # another user was readable by anyone logged out.
    from src.internal.access.capabilities import resolve_capabilities

    class _Store:
        def get_user_memories(self, user_id):
            return []

    caps = resolve_capabilities(None, _Store())
    assert caps.access_acl == ["public"]


def test_a_restricted_document_is_hidden_from_anonymous(monkeypatch):
    from src.context.models import SearchFilters

    restricted = _doc("theirs", ["user:someone_else"])
    public = _doc("public", ["public"])
    _seen, (_a, _c, documents, _i, _e) = _call(
        monkeypatch,
        filters=SearchFilters(access_acl=["public"]),
        direct_documents=[restricted, public],
    )
    assert {d.id for d in documents} == {"public"}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/unit/test_search_route_access_filters.py -q -k "anonymous or restricted"`
Expected: FAIL — `ModuleNotFoundError` if Task 1 is not yet merged into this branch; otherwise the second test passes already (enforcement landed in #487) and the first fails only if the resolver is missing. Both must pass before moving on.

- [ ] **Step 3: Write minimal implementation**

In `src/internal/servers/web/app.py`:

Delete the `memory_injection: bool = False` field (line ~150) and the `memory_injection=_flag("AGENTIC_SEARCH_MEMORY_INJECTION"),` line (~170).

Replace the memory block (lines ~1409-1417) with:

```python
        auth_user = _optional_user_from_request(http_request, db)
        capabilities = resolve_capabilities(auth_user, db)
        user_id = request.user_id or capabilities.user_id
        # Memory-augmented generation: a signed-in caller's stored memories are
        # injected because they are signed in. Anonymous callers have none.
        user_memory = capabilities.memory_preamble or None
```

Replace the filter block (lines ~1452-1458) with:

```python
        # Always filtered: anonymous means ["public"], not "unfiltered".
        filters = SearchFilters(access_acl=capabilities.access_acl)
```

Add the imports at the top of the file, beside the existing access imports:

```python
from src.internal.access.capabilities import resolve_capabilities
from src.context.models import SearchFilters
```

(If `SearchFilters` is already imported, do not import it twice.)

- [ ] **Step 4: Run the full suite**

Run: `python3 -m pytest -q`
Expected: PASS. If a test asserts `filters is None` for an anonymous request, that assertion encodes the old hole — update it to expect `["public"]` and say so in the commit.

- [ ] **Step 5: Commit**

```bash
ruff check . --fix && ruff format .
git add src/internal/servers/web/app.py tests/unit/test_search_route_access_filters.py
git commit -m "feat(access): anonymous means public-only; memory follows sign-in

Anonymous requests carried no ACL, so a document restricted to another user was
readable by anyone logged out. They now carry [\"public\"].

AGENTIC_SEARCH_MEMORY_INJECTION is gone: the preamble is built whenever a user
resolves, which is what signing in is for."
```

---

### Task 6: Prove it end to end

**Files:**
- Create: `examples/verify_identity_capabilities.sh`

**Interfaces:**
- Consumes: everything above.
- Produces: a runnable check.

- [ ] **Step 1: Write the verification script**

Create `examples/verify_identity_capabilities.sh`:

```bash
#!/usr/bin/env bash
# Proves identity shapes results against a retrieval server that ignores
# filters (demo.py does), so enforcement is the web layer's, not the server's.
#
# Usage: examples/verify_identity_capabilities.sh
set -euo pipefail

WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"; kill $(jobs -p) 2>/dev/null || true' EXIT

cat > "$WORK/corpus.jsonl" <<'JSON'
{"id": "pub_1", "title": "Zebra Handbook", "contents": "Zebra migration patterns.", "metadata": {"acl": ["public"]}}
{"id": "sec_1", "title": "Zebra Handbook", "contents": "Zebra confidential notes.", "metadata": {"acl": ["user:someone_else"]}}
JSON

PYTHONPATH=src:. python3 -m src.internal.servers.retrieval.demo \
  --corpus_path "$WORK/corpus.jsonl" --port 8001 >"$WORK/retrieval.log" 2>&1 &
env -u SEARCH_AGENT_MODEL PYTHONPATH=src:. \
  AGENTIC_SEARCH_WEB_DB_PATH="$WORK/web.db" \
  python3 -m uvicorn src.internal.servers.web.app:app \
  --host 127.0.0.1 --port 7860 >"$WORK/web.log" 2>&1 &

for _ in $(seq 1 60); do
  curl -sf -m 2 http://127.0.0.1:7860/admin/tools >/dev/null 2>&1 && break
  sleep 1
done

ask() {  # $1 = output file, $2... = extra curl args
  local out="$1"; shift
  curl -s -m 120 "$@" -X POST http://127.0.0.1:7860/api/agent \
    -H 'Content-Type: application/json' \
    -d '{"query":"Zebra Handbook"}' -o "$out"
}

ask "$WORK/anon.json"
curl -s -X POST http://127.0.0.1:7860/auth/register -H 'Content-Type: application/json' \
  -d '{"email":"dev@localhost","username":"dev","password":"devpass"}' >/dev/null
curl -s -c "$WORK/ck.txt" -X POST http://127.0.0.1:7860/auth/login \
  -H 'Content-Type: application/x-www-form-urlencoded' \
  -d 'username=dev@localhost&password=devpass' >/dev/null
ask "$WORK/auth.json" -b "$WORK/ck.txt"

python3 - "$WORK/anon.json" "$WORK/auth.json" <<'PY'
import json, sys

def leaked(path):
    docs = json.load(open(path)).get("documents") or []
    return any("confidential" in (d.get("content") or "").lower() for d in docs)

anon, auth = (leaked(p) for p in sys.argv[1:3])
print(f"anonymous sees the restricted document: {anon}")
print(f"signed-in  sees the restricted document: {auth}")
if anon or auth:
    raise SystemExit("FAIL: a restricted document leaked")
print("PASS: neither identity can read another user's document")
PY
```

- [ ] **Step 2: Make it executable and run it**

Run: `chmod +x examples/verify_identity_capabilities.sh && examples/verify_identity_capabilities.sh`
Expected: `PASS: neither identity can read another user's document`

- [ ] **Step 3: Commit**

```bash
git add examples/verify_identity_capabilities.sh
git commit -m "test: end-to-end check that identity shapes what is readable"
```

---

## Self-Review

**Spec coverage.** Anonymous means `["public"]` → Tasks 1, 5. Memory without a flag → Tasks 1, 5. User-scoped tools withheld → Tasks 2, 3, 4. One place decides entitlement → Task 1. Degrade-to-anonymous on failure → Task 1 (`test_a_store_failure_degrades_to_no_memory`, and `resolve_active_user` already returns `None` for a token whose row is gone). Verification section → Tasks 1, 5, 6.

**Placeholders.** None: every code step carries the code, and every test step carries the assertions.

**Type consistency.** `resolve_capabilities(user, store)` is called with `(auth_user, db)` in Task 5, matching Task 1's signature. `agent_tools(*, user_present: bool = True)` is defined in Task 2 and called in Task 4 with the same keyword. `user_scoped` is spelled identically in `ToolEntry`, `register`, `McpServerSpec`, `parse_mcp_servers`, and `AppSettings.mcp_user_scoped`.

**Known deviation.** The spec's `tools_for(...)` is implemented as `agent_tools(*, user_present=...)`; recorded under Global Constraints with the reason.
