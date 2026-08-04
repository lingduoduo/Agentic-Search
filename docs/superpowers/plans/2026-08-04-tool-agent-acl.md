# Tool-Agent ACL Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** The corpus `search` tool offered to the tool agent returns only documents the caller may read, and no unfiltered corpus-search tool can reach an agent loop by construction.

**Architecture:** `build_search_routing_tool` gains a `filters` argument and both *sends* it to retrieval and *enforces* it on the results, because the bundled retrieval servers ignore the field. `_run_tool_agent` stops handing out the globally-seeded instance, and that instance is seeded `agent_callable=False` so no loop can receive it even if a call site forgets. Finally the bundled servers learn to honour `access_acl` as a second layer.

**Tech Stack:** Python 3.12, FastAPI, pytest.

**Spec:** `docs/superpowers/specs/2026-08-04-tool-agent-acl-design.md`

## Global Constraints

- Work on branch `feat/tool-agent-acl`. Never commit to `main`.
- **THE INVARIANT:** anywhere filters are made serializable for a retrieval call (`to_payload()`), the same change must enforce them on that call's results. An unpaired serialization turns a fail-closed crash into a silent cross-user read.
- `filters=None` must preserve today's unfiltered behaviour — training scripts and evals call these helpers without an identity.
- Documents that declare no ACL are public. Do not change `SearchFilters.matches`.
- Tasks 1-3 must pass their tests **against `demo.py` unchanged** — the web layer enforces without server cooperation. Task 4 is a second layer, added after.
- `python3 -m pytest` and `ruff check . && ruff format .` pass before every commit.
- Run commands from the repo root.

## File Structure

| File | Status | Responsibility |
| --- | --- | --- |
| `src/internal/tools/routing_tools.py` | Modify | `build_search_routing_tool` takes `filters`; sends and enforces them. |
| `tests/unit/test_tool_search_acl.py` | Create | Builder-level: sends the payload, drops inaccessible pages, `None` is unfiltered. |
| `src/internal/servers/web/tool_agent_runner.py` | Modify | Never offer the seeded instance; accept and forward `filters`. |
| `src/internal/servers/web/app.py` | Modify | Pass `filters` at both `_run_tool_agent` call sites. |
| `src/internal/servers/query_and_chat/tool_backend.py` | Modify | Pass `filters` from the resolved capabilities. |
| `tests/unit/servers/web/test_loop_runners.py` | Modify | The agent's tool list never contains an unfiltered corpus search. |
| `src/internal/tools/knowledge_base.py` | Modify | Seed the corpus search `agent_callable=False`. |
| `src/internal/servers/retrieval/demo.py` | Modify | `RetrieveRequest.filters`; drop documents outside the ACL. |
| `tests/unit/test_retrieval_server_acl.py` | Create | Both bundled servers honour `access_acl`. |

---

### Task 1: The corpus search tool enforces the caller's ACL

**Files:**
- Modify: `src/internal/tools/routing_tools.py`
- Create: `tests/unit/test_tool_search_acl.py`

**Interfaces:**
- Consumes: `SearchFilters` from `src.context.models` (has `.to_payload() -> dict` and `.matches(metadata) -> bool`); `SearchPage` from `src.internal.tools.search` (fields `title`, `summary`, `url`, `error`, `score`, `metadata`).
- Produces: `build_search_routing_tool(*, search_url, top_k, name="search", filters=None) -> FunctionTool`.

- [ ] **Step 1: Write the failing test**

Create `tests/unit/test_tool_search_acl.py`:

```python
"""The corpus search tool returns only what the caller may read.

demo.py and hybrid.py accept the `filters` field and ignore it, so sending the
payload is not enforcement — the tool drops inaccessible pages itself.
"""

from __future__ import annotations

import asyncio
import json
from unittest.mock import AsyncMock, patch

from src.context.models import SearchFilters
from src.internal.tools.routing_tools import build_search_routing_tool
from src.internal.tools.search import SearchPage


def _page(title: str, acl: list[str] | None) -> SearchPage:
    return SearchPage(
        title=title,
        summary="body",
        url=f"http://x/{title}",
        metadata={"acl": acl} if acl is not None else {},
    )


PAGES = [
    _page("mine", ["user:userA"]),
    _page("theirs", ["user:userB"]),
    _page("open", ["public"]),
    _page("undeclared", None),
]


def _run(filters):
    tool = build_search_routing_tool(
        search_url="http://x/retrieve", top_k=5, filters=filters
    )
    with patch(
        "src.internal.tools.routing_tools.search_tool",
        new=AsyncMock(return_value=list(PAGES)),
    ) as mock:
        raw, _out, _meta = asyncio.run(tool.execute("id", {"query": "q"}))
    return json.loads(raw), mock


def test_documents_outside_the_acl_are_dropped():
    results, _ = _run(SearchFilters(access_acl=["public", "user:userA"]))
    titles = {r["title"] for r in results}
    assert "theirs" not in titles
    assert {"mine", "open"} <= titles


def test_documents_with_no_declared_acl_stay_public():
    results, _ = _run(SearchFilters(access_acl=["public"]))
    assert "undeclared" in {r["title"] for r in results}


def test_no_filters_returns_everything():
    # Training scripts and evals build this tool without an identity.
    results, _ = _run(None)
    assert len(results) == len(PAGES)


def test_the_payload_is_sent_to_retrieval():
    # Backends that honour access_acl should get the chance to.
    _results, mock = _run(SearchFilters(access_acl=["public", "user:userA"]))
    sent = mock.await_args.kwargs["filters"]
    assert sent == {"access_acl": ["public", "user:userA"]}


def test_no_filters_sends_no_payload():
    _results, mock = _run(None)
    assert mock.await_args.kwargs["filters"] is None


def test_everything_filtered_out_is_an_empty_list_not_an_error():
    # An empty result is a legitimate answer; it must not read as a failure.
    results, _ = _run(SearchFilters(access_acl=["user:nobody"]))
    assert results == []
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/unit/test_tool_search_acl.py -q`
Expected: FAIL — `TypeError: build_search_routing_tool() got an unexpected keyword argument 'filters'`

- [ ] **Step 3: Write minimal implementation**

In `src/internal/tools/routing_tools.py`, replace `build_search_routing_tool` with:

```python
def build_search_routing_tool(
    *, search_url: str, top_k: int, name: str = "search", filters=None
) -> FunctionTool:
    """FunctionTool that retrieves documents from the corpus.

    Named ``search`` by default: it *is* the corpus search a model should reach
    for, and an opaque name costs tool-selection accuracy on small models.

    ``filters`` are both sent to retrieval and enforced on what comes back.
    Sending alone is not enforcement: ``demo.py`` and ``hybrid.py`` accept the
    field and ignore it, so an unpaired send would hand the model documents the
    caller may not read. ``None`` keeps the tool unfiltered for callers that
    have no identity (training scripts, evals).
    """

    async def _execute(query: str) -> str:
        pages = await search_tool(
            query,
            provider="retrieval",
            search_url=search_url,
            page_size=top_k,
            filters=filters.to_payload() if filters is not None else None,
        )
        if filters is not None:
            pages = [
                p for p in pages if p.error or filters.matches(p.metadata or {})
            ]
        results = [
            {"title": p.title or "", "content": p.summary or "", "url": p.url}
            for p in pages
            if not p.error
        ]
        if not results and any(p.error for p in pages):
            errors = [p.error for p in pages if p.error]
            return json.dumps({"error": errors[0]})
        return json.dumps(results)

    return FunctionTool(
        fn=_execute,
        name=name,
        description="Retrieve relevant documents from the corpus given a search query.",
        parameters=_SEARCH_TOOL_PARAMS,
        effect=ToolEffect.READ_ONLY,
        citeable=True,
    )
```

Note two deliberate changes beyond adding the argument:
- error pages survive the ACL filter (`p.error or ...`) so a transport failure is still reported rather than silently filtered away;
- the "no results" branch now triggers only when there *was* an error, so a result set legitimately emptied by the ACL returns `[]` instead of a misleading error.

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m pytest tests/unit/test_tool_search_acl.py -q`
Expected: PASS (6 tests)

- [ ] **Step 5: Run the existing suites that use this builder**

Run: `python3 -m pytest tests/unit/test_intent_routing.py tests/unit/test_knowledge_base.py -q`
Expected: PASS — the new argument defaults to `None`, so existing callers are unchanged.

- [ ] **Step 6: Commit**

```bash
ruff check . --fix && ruff format .
git add src/internal/tools/routing_tools.py tests/unit/test_tool_search_acl.py
git commit -m "feat(tools): the corpus search tool enforces the caller's ACL

Sends filters to retrieval and drops inaccessible pages from what it returns.
Both, because demo.py and hybrid.py accept the field and ignore it."
```

---

### Task 2: The runner never hands out the unfiltered instance

**Files:**
- Modify: `src/internal/servers/web/tool_agent_runner.py`
- Modify: `src/internal/servers/web/app.py` (both `_run_tool_agent` call sites)
- Modify: `src/internal/servers/query_and_chat/tool_backend.py`
- Modify: `tests/unit/servers/web/test_loop_runners.py`

**Interfaces:**
- Consumes: `build_search_routing_tool(*, search_url, top_k, name, filters)` from Task 1.
- Produces: `_run_tool_agent(..., filters=None)`.

- [ ] **Step 1: Write the failing test**

Append to `tests/unit/servers/web/test_loop_runners.py`:

```python
@pytest.mark.asyncio
async def test_the_seeded_corpus_search_never_reaches_the_agent(monkeypatch):
    # The globally-seeded instance is built where no request identity exists,
    # so it is unfiltered. With with_search_tool=False the agent must get no
    # corpus search at all rather than that one.
    from src.internal.tools.registry import ToolRegistry
    from src.internal.tools.routing_tools import build_search_routing_tool

    registry = ToolRegistry()
    registry.register(build_search_routing_tool(search_url="http://seeded/", top_k=5))
    monkeypatch.setattr("src.internal.tools.tool_registry", registry)

    captured = _capture_tool_agent_loop(monkeypatch)
    await web_app._run_tool_agent(
        "q",
        manager=MagicMock(),
        tokenizer=MagicMock(),
        search_url="http://request/retrieve",
        history=[],
        resolved=types.SimpleNamespace(tool_agent_parser="json"),
        on_turn=None,
        with_search_tool=False,
    )
    assert "search" not in [t.name for t in captured["tools"]]


@pytest.mark.asyncio
async def test_the_request_bound_corpus_search_carries_the_filters(monkeypatch):
    from src.context.models import SearchFilters

    built = {}
    real_builder = web_app_routing_tools.build_search_routing_tool

    def _spy(**kwargs):
        built.update(kwargs)
        return real_builder(**kwargs)

    monkeypatch.setattr(web_app_routing_tools, "build_search_routing_tool", _spy)

    filters = SearchFilters(access_acl=["public", "user:userA"])
    captured = _capture_tool_agent_loop(monkeypatch)
    await web_app._run_tool_agent(
        "q",
        manager=MagicMock(),
        tokenizer=MagicMock(),
        search_url="http://request/retrieve",
        history=[],
        resolved=types.SimpleNamespace(tool_agent_parser="json"),
        on_turn=None,
        with_search_tool=True,
        filters=filters,
    )
    assert built["filters"] is filters
    assert built["search_url"] == "http://request/retrieve"
    assert "search" in [t.name for t in captured["tools"]]
```

Add this import near the top of the file, beside the existing `web_app` import:

```python
from src.internal.tools import routing_tools as web_app_routing_tools
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/unit/servers/web/test_loop_runners.py -q -k "seeded_corpus or request_bound"`
Expected: FAIL — the first because `search` is still present from the registry, the second with `TypeError: _run_tool_agent() got an unexpected keyword argument 'filters'`

- [ ] **Step 3: Write minimal implementation**

In `src/internal/servers/web/tool_agent_runner.py`, add `filters=None` to `_run_tool_agent`'s keyword arguments (after `user_present: bool = True`), and replace the tool-list construction with:

```python
    tools = [
        t
        for t in tool_registry.agent_tools(user_present=user_present)
        # Never the globally-seeded corpus search: it is built at process start
        # where no request identity exists, so it is unfiltered. The only
        # acceptable corpus search is the request-bound one built below.
        if t.name != _CORPUS_SEARCH_NAME
    ]
    if with_search_tool:
        corpus_search = build_search_routing_tool(
            search_url=search_url,
            top_k=_CORPUS_SEARCH_TOP_K,
            name=_CORPUS_SEARCH_NAME,
            filters=filters,
        )
        tools = [corpus_search] + tools
```

In `src/internal/servers/web/app.py`, at both `_run_tool_agent(` call sites, add:

```python
                    filters=filters,
```

`filters` is already in scope at both (built from `capabilities.access_acl`). Locate the sites by searching for `_run_tool_agent(` rather than trusting line numbers.

In `src/internal/servers/query_and_chat/tool_backend.py`, inside `_run`, add the same argument to its `_run_tool_agent(...)` call:

```python
                filters=SearchFilters(access_acl=capabilities.access_acl),
```

and import `SearchFilters` from `src.context.models` at the top of that file if it is not already imported. `capabilities` is already in scope there.

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m pytest tests/unit/servers/web/test_loop_runners.py tests/unit/test_tool_backend.py -q`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
ruff check . --fix && ruff format .
git add src/internal/servers/web/tool_agent_runner.py src/internal/servers/web/app.py src/internal/servers/query_and_chat/tool_backend.py tests/unit/servers/web/test_loop_runners.py
git commit -m "feat(tools): the agent only ever gets a request-bound corpus search"
```

---

### Task 3: The seeded instance is not agent-callable

**Files:**
- Modify: `src/internal/tools/knowledge_base.py`
- Modify: `tests/unit/test_knowledge_base.py`

**Interfaces:**
- Consumes: `ToolRegistry.register(..., agent_callable=...)` and `NOT_AGENT_CALLABLE` (already in `knowledge_base.py`).
- Produces: nothing new.

- [ ] **Step 1: Write the failing test**

Append to `tests/unit/test_knowledge_base.py`:

```python
def test_the_seeded_corpus_search_is_not_agent_callable():
    # It is built at process start with no request identity, so it cannot carry
    # an ACL. Keeping it out of every agent's tool list makes that structural
    # rather than something each call site has to remember.
    from src.internal.tools.knowledge_base import seed_tools, tool_knowledge_base
    from src.internal.tools.registry import ToolRegistry

    reg = ToolRegistry()
    seed_tools(reg, tools=tool_knowledge_base())

    assert reg.get("search") is not None  # still listed and invocable
    assert "search" not in [t.name for t in reg.agent_tools()]
    assert "web_search" in [t.name for t in reg.agent_tools()]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/unit/test_knowledge_base.py -q -k not_agent_callable`
Expected: FAIL — `assert 'search' not in ['web_search', 'search']`

- [ ] **Step 3: Write minimal implementation**

In `src/internal/tools/knowledge_base.py`, extend the existing constant:

```python
# Tools no agent loop should be offered, keyed by name at seed time so the
# decision travels with registration instead of being re-derived downstream.
# ``rag_routing_tool`` generates a whole answer rather than returning evidence.
# ``search`` is seeded at process start, where no request identity exists, so
# this instance can carry no ACL; the tool agent builds its own request-bound
# one. This instance stays listed and invocable through /admin/tools.
NOT_AGENT_CALLABLE: frozenset[str] = frozenset({"rag_routing_tool", "search"})
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m pytest tests/unit/test_knowledge_base.py tests/unit/test_agent_callable_tools.py -q`
Expected: PASS

- [ ] **Step 5: Run the full suite**

Run: `python3 -m pytest -q`
Expected: PASS. A test asserting the agent sees `search` from the *registry* encodes the behaviour being removed — update it to build the request-bound tool instead, and say so in the commit.

- [ ] **Step 6: Commit**

```bash
ruff check . --fix && ruff format .
git add src/internal/tools/knowledge_base.py tests/unit/test_knowledge_base.py
git commit -m "feat(tools): the seeded corpus search is never agent-callable"
```

---

### Task 4: The bundled retrieval servers honour `access_acl`

**Files:**
- Modify: `src/internal/servers/retrieval/demo.py`
- Create: `tests/unit/test_retrieval_server_acl.py`

**Interfaces:**
- Consumes: nothing from earlier tasks. `hybrid.py` imports `RetrieveRequest` from `demo.py`, so both servers gain the field from one edit.
- Produces: `RetrieveRequest.filters: dict | None`.

- [ ] **Step 1: Write the failing test**

Create `tests/unit/test_retrieval_server_acl.py`:

```python
"""The bundled retrieval servers honour the access_acl they are sent.

Defense in depth. The web layer enforces regardless, because a third-party
backend is free to ignore the field — but the servers we ship should not.
"""

from __future__ import annotations

from fastapi.testclient import TestClient

from src.internal.servers.retrieval.demo import TfidfRetriever, create_app

DOCS = [
    {"id": "open", "title": "Zebra Handbook", "contents": "zebra migration",
     "metadata": {"acl": ["public"]}},
    {"id": "theirs", "title": "Zebra Handbook", "contents": "zebra migration secrets",
     "metadata": {"acl": ["user:someone_else"]}},
    {"id": "undeclared", "title": "Zebra Notes", "contents": "zebra migration notes"},
]


def _client():
    return TestClient(create_app(TfidfRetriever.from_docs(list(DOCS))))


def _ids(payload):
    rows = payload["results"]
    row = rows[0] if rows and isinstance(rows[0], list) else rows
    return {d.get("id") for d in row}


def test_documents_outside_the_acl_are_withheld():
    r = _client().post(
        "/retrieve",
        json={"queries": ["zebra migration"], "topk": 5,
              "filters": {"access_acl": ["public"]}},
    )
    assert r.status_code == 200
    ids = _ids(r.json())
    assert "theirs" not in ids
    assert "open" in ids


def test_documents_with_no_declared_acl_stay_public():
    r = _client().post(
        "/retrieve",
        json={"queries": ["zebra migration"], "topk": 5,
              "filters": {"access_acl": ["public"]}},
    )
    assert "undeclared" in _ids(r.json())


def test_no_filters_returns_everything():
    r = _client().post("/retrieve", json={"queries": ["zebra migration"], "topk": 5})
    assert {"open", "theirs", "undeclared"} <= _ids(r.json())


def test_the_matching_user_sees_their_own_document():
    r = _client().post(
        "/retrieve",
        json={"queries": ["zebra migration"], "topk": 5,
              "filters": {"access_acl": ["public", "user:someone_else"]}},
    )
    assert "theirs" in _ids(r.json())
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/unit/test_retrieval_server_acl.py -q`
Expected: FAIL — `assert 'theirs' not in {'open', 'theirs', 'undeclared'}`; the server ignores the field today.

- [ ] **Step 3: Write minimal implementation**

In `src/internal/servers/retrieval/demo.py`, add the field to `RetrieveRequest`:

```python
    filters: dict | None = None
```

Add this helper above `create_app`:

```python
def _allowed_by_acl(document: dict, filters: dict | None) -> bool:
    """Whether *document* is readable under *filters*.

    A document that declares no ACL is public, matching
    ``SearchFilters.matches``. Kept local so the retrieval servers stay free of
    web-layer imports.
    """
    if not filters:
        return True
    allowed = filters.get("access_acl")
    if not allowed:
        return True
    declared = (document.get("metadata") or {}).get("acl")
    if not declared:
        return True
    if isinstance(declared, str):
        declared = [declared]
    return bool(set(declared) & set(allowed))
```

Then filter inside `retrieve_endpoint`, before the `return_scores` unwrapping:

```python
    @app.post("/retrieve")
    def retrieve_endpoint(body: RetrieveRequest):
        queries = body.resolved_queries()
        rows = retriever.retrieve(queries, topk=body.topk)
        rows = [
            [item for item in row if _allowed_by_acl(item["document"], body.filters)]
            for row in rows
        ]
        if not body.return_scores:
            rows = [[item["document"] for item in row] for row in rows]
        if body.query is not None:
            return {"results": rows[0] if rows else []}
        return {"results": rows}
```

Apply the identical filtering to `hybrid.py`'s `retrieve_endpoint`, immediately after `rows = _fuse_rows(...)`, importing `_allowed_by_acl` from `demo` alongside the `RetrieveRequest` import it already has:

```python
        rows = [
            [item for item in row if _allowed_by_acl(item["document"], body.filters)]
            for row in rows
        ]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m pytest tests/unit/test_retrieval_server_acl.py -q`
Expected: PASS (4 tests)

- [ ] **Step 5: Run the full suite**

Run: `python3 -m pytest -q`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
ruff check . --fix && ruff format .
git add src/internal/servers/retrieval/demo.py src/internal/servers/retrieval/hybrid.py tests/unit/test_retrieval_server_acl.py
git commit -m "feat(retrieval): the bundled servers honour the access_acl they are sent"
```

---

### Task 5: Prove it end to end

**Files:**
- Modify: `examples/verify_identity_capabilities.sh`

**Interfaces:**
- Consumes: everything above.

- [ ] **Step 1: Extend the existing script**

The script already starts `demo.py` and the web backend with a two-document corpus (`pub_1` public, `sec_1` ACL'd to `user:someone_else`) and asserts neither identity can read the restricted one on the default route. Add a tool-agent leg.

After the existing checks, add a request that exercises the tool surface and assert the confidential string is absent:

```bash
tool_leg() {  # $1 = output file, $2... = extra curl args
  local out="$1"; shift
  curl -s -m 180 "$@" -X POST http://127.0.0.1:7860/tool/send-tool-message \
    -H 'Content-Type: application/json' \
    -d '{"message":"Search the corpus for Zebra Handbook","stream":false}' \
    -o "$out"
}
```

Call it anonymously and with the cookie jar, then assert with the same
`leaked()`-style check the script already uses, reading `answer` and any
`tool_calls[].result_summary` for the confidential string.

**If no local model is configured** the tool endpoint returns HTTP 400 with
`NO_LOCAL_MODEL_MESSAGE`. Treat that as SKIP for this leg — print
`SKIP: tool-agent leg needs SEARCH_AGENT_MODEL` and continue — rather than
failing the script. Do not treat a 400 as a pass.

- [ ] **Step 2: Run it**

Run: `examples/verify_identity_capabilities.sh`
Expected: the existing PASS line, plus either a tool-agent PASS or the SKIP line.

- [ ] **Step 3: Prove the new leg can fail**

Temporarily revert Task 1's enforcement (drop the `if filters is not None:` block in `routing_tools.py`), re-run with a model configured, and confirm the tool leg reports FAIL. Restore, confirm `git diff` is clean, and record what you saw in the report.

If you cannot configure a local model, say so plainly instead of claiming the leg was verified.

- [ ] **Step 4: Commit**

```bash
git add examples/verify_identity_capabilities.sh
git commit -m "test: extend the identity check to the tool-agent surface"
```

---

## Self-Review

**Spec coverage.** Builder takes filters and enforces → Task 1. Runner never hands out the shared instance → Task 2. Seeded instance not agent-callable → Task 3. Bundled servers honour `access_acl` → Task 4. "Tasks 1-3 pass against `demo.py` unchanged" → the constraint is in Global Constraints and Task 4 comes last, so the ordering is enforced by the plan itself. Verification bullets → Tasks 1, 2, 4, 5.

**Placeholders.** None: every code step carries its code, every test step its assertions. Task 5's script edit gives the helper verbatim and names the exact skip condition rather than saying "handle the no-model case".

**Type consistency.** `build_search_routing_tool(*, search_url, top_k, name, filters)` is defined in Task 1 and called with those exact keywords in Task 2. `_run_tool_agent(..., filters=None)` is defined in Task 2 and its three call sites updated in the same task. `_allowed_by_acl(document, filters)` is defined in Task 4 and used by both servers there. `NOT_AGENT_CALLABLE` already exists in `knowledge_base.py`; Task 3 extends it rather than introducing a new name.

**One risk the plan carries deliberately.** Task 3 changes what the registry offers, so a test asserting the agent sees `search` from the registry will fail. Task 3 Step 5 names that explicitly rather than letting an implementer discover it and quietly weaken the assertion.
