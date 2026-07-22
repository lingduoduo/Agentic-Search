# Spec: Memory CLI + backend HTTP memory endpoints

## Origin

Deliverable 2 of the conversation-memory work (deliverable 1 = MCP tools, merged
PR #456). A CLI to manage user memory, decided during that brainstorming to be a
**Go subcommand-style binary over new backend HTTP endpoints** (one source of
truth — the same `AgenticSearchStore`), NOT the sampled standalone Go tool with
its own `data/memories/*.json` store (which would fork memory into a second
silo).

## Goals

Two coupled pieces, one deliverable, built backend-first:

1. **Backend** `create_memory_router(db, llm)` (`prefix="/api/memory"`) exposing
   the memory service over HTTP, registered in `create_web_app`.
2. **CLI** a new `cli/cmd/memory` binary with `flag.NewFlagSet` subcommands over
   those endpoints, reusing `cli/api` + `cli/config`.

Scope (v1): **core + curate** — `save`, `list`, `search`, `consolidate`,
`profile` (get + generate), `curate`.

## Non-goals

- Manual update/delete-by-id endpoints/subcommands (curation reconciles;
  deferred).
- A `--json` output mode (plain text for v1).
- Touching the `query` binary's default behavior or TUI (only a shared
  token-helper extraction, below).

## Architecture

```
cli/cmd/memory/main.go        cli/api/client.go            src/internal/servers/web/
  flag.NewFlagSet subcmds  →    typed methods +        →     memory_router.py
  add/list/search/               ClientAPI iface              create_memory_router(db, llm)
  consolidate/profile/curate     POST/GET /api/memory/*       (prefix /api/memory)
                                                                    │ wraps
                                                              src/internal/memory/service.py
```

The router **closes over the web app's own `db`** (the `AgenticSearchStore`
holding chat sessions), so `curate` reconciles from real conversations with no
shared-file requirement. Both the router and the existing MCP tools stay thin
adapters over the single `service.py` — no logic duplication.

## Backend — `src/internal/memory/router.py`

`create_memory_router(db: AgenticSearchStore, llm) -> APIRouter`, mirroring
`src/internal/servers/web/debug_router.py` (factory closing over `db`/`llm`,
Pydantic request models, no `Depends` for `db`).

| Method | Path | Wraps (`service.*`) | Response |
|---|---|---|---|
| POST | `/api/memory/save` | `save_memory(db, uid, text)` | `{"memory_id": str \| null}` |
| GET | `/api/memory/list` | `db.get_user_memory_records(uid)` | `{"memories": [{"id","text","updated_at"}]}` |
| POST | `/api/memory/search` | `search_memories(db, uid, query, max_results, encoder)` | `{"results": [{"id","text","score"}]}` |
| POST | `/api/memory/consolidate` | `consolidate_memories(db, uid, resolve_conflicts)` | `{"report": {...}}` |
| GET | `/api/memory/profile` | `get_user_profile(db, uid)` | `{"profile": [asdict(entry)]}` |
| POST | `/api/memory/profile/generate` | `generate_user_profile(db, uid, llm)` | `{"profile": [...]}` |
| POST | `/api/memory/curate` | `await curate_from_conversation(db, uid, llm, session_id)` | `{"status","trajectory_id","counts","memory_count"}` |

- **Identity:** `user_id = auth_user.id if auth_user else DEFAULT_MEMORY_USER_ID`,
  where `auth_user = _optional_user_from_request(request)` (works unauthenticated
  against a local dev backend). Router takes an optional `default_user_id` arg so
  tests can pin it.
- **LLM-dependent** endpoints (`profile/generate`, `curate`): if `llm is None`,
  return HTTP 503 `{"detail": "LLM not configured"}`.
- **`/curate`** is `async def` (the service fn is async); the rest are `def`.
- **Search encoder:** reuse the shared `maybe_build_encoder()` helper (see DRY
  #2) — lexical by default, e5 when `AGENTIC_SEARCH_MEMORY_SEMANTIC` is set.
- Registered via `app.include_router(create_memory_router(db, llm))` in the
  `app.py:354-410` router block.

Serialize profile records with `dataclasses.asdict` (they are `slots` dataclasses
with no `__dict__`).

## CLI — `cli/cmd/memory`

New binary `cli/cmd/memory/main.go`. Dispatch on `os.Args[1]` to a per-subcommand
`flag.FlagSet`:

- `add <text...>` → `POST /memory/save`
- `list` → `GET /memory/list`
- `search <query...> [--top-k N]` → `POST /memory/search`
- `consolidate [--no-conflict]` → `POST /memory/consolidate`
- `profile [--generate]` → `GET /memory/profile` or `POST /memory/profile/generate`
- `curate [--session-id S]` → `POST /memory/curate`

Reuses `cli/config.Load()` (`AGENTIC_SEARCH_URL`, `AGENTIC_SEARCH_PAT`) and
`cli/api.NewClient`. Plain-text output (counts, ids, table-ish lines). New typed
methods on `api.Client` via `doJSON`, added to the `ClientAPI` interface (with
the compile-time `var _ ClientAPI` assertion), and request/response structs in
`cli/models` (json-tagged, `,omitempty` on optional request fields).

Common flags per subcommand: `--url`, `--token`, `--user-id` (mint), resolved via
the shared helper (DRY #1). No token → request is sent without `Authorization`;
the backend then uses `DEFAULT_MEMORY_USER_ID`.

## DRY refactors (approved)

1. **Extract token resolution** — move `resolveToken` / `mintJWT` (currently in
   `cli/cmd/query/main.go` `package main`) into a new shared package
   `cli/clientauth/` (e.g. `ResolveToken(...)`, `MintJWT(...)`). Repoint the
   `query` binary to it (import-only change; behavior unchanged) and use it from
   `cli/cmd/memory`.
2. **Move `_maybe_encoder`** out of `mcp_server/tools/memory.py` into
   `src/internal/memory/service.py` as public
   `maybe_build_encoder() -> Encoder | None`; repoint the MCP tool to it and use
   it in the router. Avoids the web layer importing from `mcp_server`.

## Auth secret gotcha

The `query` binary mints HS256 JWTs signed with the `AUTH_SECRET` env var, while
the backend validates with `AGENTIC_SEARCH_AUTH_SECRET`
(`get_auth_secret`). The shared `clientauth` package MUST read
`AGENTIC_SEARCH_AUTH_SECRET` (falling back to `AUTH_SECRET` for back-compat) so
CLI-minted tokens validate. Document this in the CLI help/README.

## Error handling

- Backend: each endpoint wraps its body; validation via Pydantic models
  (`Field(min_length=1)` on `text`/`query`); `503` when `llm is None` for LLM
  endpoints; unknown errors → `500` with the message.
- CLI: non-2xx → the existing `api.APIError` is surfaced as a non-zero exit with
  a readable stderr message; unknown subcommand → usage + exit 2.

## Testing

- **Backend** (`tests/unit/servers/web/test_memory_router.py`, offline): seed a
  store, drive each endpoint via `TestClient`; `save`+`list`+`search`+
  `consolidate`+`profile GET` with no LLM; `profile/generate` + `curate` with a
  fake LLM (reuse the FakeLLM pattern from `tests/unit/memory/`); assert the 503
  path when `llm is None`; assert identity falls back to `DEFAULT_MEMORY_USER_ID`.
- **CLI** (`cli/api/memory_test.go`, `cli/cmd/memory/main_test.go`): `httptest`
  per client method (mirror `client_test.go` — external `api_test` package,
  `testutil.NewClient`); a dispatch test that an unknown subcommand exits
  non-zero and `add`/`list` hit the right method+path.
- `maybe_build_encoder` default (no `AGENTIC_SEARCH_MEMORY_SEMANTIC`) returns
  `None` (lexical), with no import of `sentence_transformers`.

## Acceptance criteria

- `create_memory_router` registered; the 7 endpoints behave per the table;
  identity + 503 paths covered by tests.
- `cli/cmd/memory` builds (`go build ./...` / `go vet`) and its subcommands hit
  the right endpoints; Go tests pass.
- `resolveToken`/`mintJWT` live in `cli/clientauth`, used by both binaries;
  `query` behavior unchanged.
- `maybe_build_encoder` shared by router + MCP tool.
- `pytest` green; `ruff` clean; `go test ./...` green.
