# Command-line tools (`cli/`)

The `cli/` directory holds the project's **Go command-line clients**. They are
the *terminal / scriptable* front door to the running backend — for humans at a
keyboard and for shell scripts / CI. They are **not** driven by the MCP server,
and they hold no logic or data of their own: each command parses flags, calls
the backend's HTTP `/api/*` endpoints, and prints the result.

There are exactly **two** binaries today:

| Binary | Purpose | Source |
|---|---|---|
| `query` | One-shot search + agentic answer (sources + answer + `session_id`) | [cli/cmd/query](../cli/cmd/query/main.go) |
| `memory` | Manage user memory (add / list / search / consolidate / profile / curate) | [cli/cmd/memory](../cli/cmd/memory/main.go) |

> Note: things like **connectors** and **evals** are *not* CLI tools — they live
> in the Web UI and the backend REST API. See [Extending the CLI](#extending-the-cli).

## Three doors to the same backend

The CLI is one of three independent front ends over the same backend and
services — you pick by *who is driving*:

```
   You, in a terminal / a script / CI   ──►  Go CLI (query, memory)   ─┐
   You, in a browser                    ──►  Web UI                   ─┼──►  backend :7860  ──►  services
   An LLM agent                         ──►  MCP tools (:8090)        ─┘
```

They are alternative front doors, not layers that call each other: the CLI is
*your* door, MCP is the *agent's* door ([docs/mcp.md](mcp.md)), the Web UI is the
*browser* door. All three ultimately reach the same routers and services (e.g.
memory converges on `src/internal/memory/service.py`).

## Prerequisites

The CLI talks to a **running backend** (it is a client, not a standalone tool):

```bash
# terminal 1 — retrieval server (see the main README for options)
python3 -m src.internal.servers.retrieval.demo --corpus demo

# terminal 2 — web backend on :7860 (the CLI's default target)
PYTHONPATH=src:. uvicorn src.internal.servers.web.app:app --host 127.0.0.1 --port 7860
```

## Build

```bash
cd cli
go build -o ../bin/query  ./cmd/query
go build -o ../bin/memory ./cmd/memory
# then run ../bin/query and ../bin/memory, or `go run ./cmd/memory -- <args>`
```

Go tooling: `go test ./...`, `go vet ./...`, `go build ./...` (run from `cli/`).

## `query` — search + answer

Positional args are the question; flags tune it.

```bash
query "What is FAISS?"
query -top-k 8 "Compare dense and sparse retrieval"
query -session-id <id> "follow-up question"        # resume a chat session
```

Flags: `-url`, `-token`, `-user-id`, `-email`, `-secret` (auth — see below),
`-top-k N` (default 5), `-session-id S`, `-width N` (markdown wrap; 0 = auto).
It hits `POST /api/agent` and renders the source cards, the answer, and the
`session_id`.

## `memory` — manage user memory

```bash
memory add "User prefers window seats"
memory list
memory search "seating" -top-k 5
memory consolidate                 # dedup + resolve tagged conflicts (-no-conflict to dedup only)
memory profile                     # show the stored profile
memory profile -generate           # (re)build the profile via the LLM
memory curate                      # LLM reconciles memories from the conversation
memory curate -session-id <id>     # ...restricted to one session
```

Each subcommand maps to an `/api/memory/*` endpoint (see
[docs/api-reference.md](api-reference.md)). `profile -generate` and `curate`
require the backend to have an LLM configured (else they return a 503, surfaced
as a non-zero exit).

`-session-id` may only name a session you can already read: one you own, or one
with no owner. A session belonging to someone else is skipped rather than
refused, so `curate` reports `empty` — the same answer it gives for a session id
that does not exist.

### JSON output (scripting)

Add `-json` to any `memory` subcommand to print the raw response as JSON instead
of the human-formatted lines — for pipelines and CI:

```bash
memory list -json | jq -r '.memories[].text'
memory search "seat" -json | jq '.results[] | {id, score}'
```

## Authentication

Config comes from a file (`~/.config/agentic-search/config.json`) overridden by
environment variables, overridden by flags:

| Setting | Env var | Flag | Default |
|---|---|---|---|
| Backend URL | `AGENTIC_SEARCH_URL` | `-url` | `http://localhost:7860` |
| Bearer token (PAT/JWT) | `AGENTIC_SEARCH_PAT` | `-token` | — |
| Mint a JWT for a user | — | `-user-id` (+ `-email`) | — |
| JWT signing secret | `AGENTIC_SEARCH_AUTH_SECRET` (then `AUTH_SECRET`) | `-secret` | — |

- **Unauthenticated is fine for local/research use** — with no token the backend
  treats the caller as the default user (`default_user`). That bucket is
  **shared**: every anonymous caller reads and writes the same memories, which is
  what makes the single-operator local flow work and what makes it wrong for a
  deployment with more than one person. Set
  `AGENTIC_SEARCH_MEMORY_REQUIRE_AUTH=1` on the backend to refuse anonymous
  memory callers instead; the CLI then needs `-token` or `-user-id`.
- To act as a specific user without a pre-issued token, pass `-user-id alice`;
  the CLI mints an HS256 JWT signed with `AGENTIC_SEARCH_AUTH_SECRET` (falling
  back to `AUTH_SECRET`). That secret **must match** the backend's
  `AGENTIC_SEARCH_AUTH_SECRET` for the token to validate.

## Exit codes

The `memory` binary maps failures to the semantic codes in
[cli/exitcodes](../cli/exitcodes/codes.go), so scripts can branch on them:

| Code | Meaning | Code | Meaning |
|---|---|---|---|
| 0 | success | 5 | backend unreachable |
| 1 | general error | 6 | rate limited (429) |
| 2 | bad request / bad args (400/422) | 7 | timeout (408/504) |
| 3 | not configured | 8 | server error (5xx) |
| 4 | auth failure (401/403) | 9 | not available (404) |

## Extending the CLI

The CLI covers only `query` + `memory` today. Because it is a thin client over
the REST API, adding a command for another capability (connectors, evals,
health, workers, …) is mechanical and self-contained:

1. add a typed method to [cli/api/client.go](../cli/api/client.go) that calls the
   endpoint (and to the `ClientAPI` interface),
2. add request/response structs to [cli/models](../cli/models/models.go),
3. add a subcommand (a new `cli/cmd/<name>` binary, following `cli/cmd/memory`).

Keep every command a **thin translator** (parse flags → call `/api` → print) —
never let logic or local state live in the CLI; that belongs in the backend
service layer so all three front doors share one source of truth.
