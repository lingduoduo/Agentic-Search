# CLI simplification — design

**Date:** 2026-08-06
**Scope:** `cli/` (Go) and `src/cli/` (Python)

## Problem

`cli/` holds 6,676 LOC of Go but exposes exactly two binaries, `cmd/query` and
`cmd/memory`. An import-reachability pass from those two `main` packages shows
**4,358 LOC (65%) is unreachable from any binary**. It is upstream heritage: it
references a binary name that does not exist here (`agentic-search`), a
"star us on GitHub" prompt, and backend endpoints that were never built.

Separately, `src/cli/` (228 LOC of Python) is a behavioral duplicate of
`cli/cmd/query` — same flags, same endpoint, same output — referenced by nothing
and documented nowhere.

Two latent bugs surfaced during the audit; both are fixed here.

### Unreachable packages

Nothing imports these, transitively or otherwise:

| Package | LOC | What it is |
|---|---|---|
| `tui/` (13 files) | ~2,140 | A full bubbletea chat TUI (splash, sshauth, statusbar, viewport, configure). No binary launches it. |
| `parser/` + `api/stream.go` + `models/events.go` | 976 | SSE stream parsing, consumed only by `tui/stream_adapter.go` |
| `overflow/` | 263 | Pager / temp-file writer |
| `version/` | 168 | Semver gate requiring "backend >= 3.0.0" — no `/version` endpoint exists |
| `fsutil/` | 166 | Skill-file install helper |
| `starprompt/` | 83 | One-time GitHub star prompt, documented as "shown before the TUI" |
| `embedded/` + `SKILL.md` | 7 + doc | Embeds an agent skill for a binary named `agentic-search` |
| `browser/` | 29 | Called only from `tui/commands.go` |
| `config/experiments.go` | 46 | Flag registry whose only entry (`stream_markdown`) is read only by `tui/` |

### Dead client surface

10 of 18 `ClientAPI` methods have no caller outside `tui/`: `TestConnection`,
`ListAgents`, `ListChatSessions`, `GetChatSession`, `RenameChatSession`,
`UploadFile`, `GetBackendVersion`, `StopChatSession`, `SendMessageStream`,
`Search`. Three of their endpoints do not exist on the backend at all —
`/chat/stop-chat-session/`, `/user/projects/file/upload`, `/version`.
`GetBackendVersion` and `Search` have no caller anywhere, including `tui/`.

### Bug 1 — a fresh clone cannot build `cli/`

`.gitignore:5` is a bare `models/`, which matches *any* directory named `models`
at any depth. It swallows `cli/models/events.go`. `cli/models/models.go` is
tracked (tracked files are unaffected by gitignore) but `events.go` never was, so:

```
git archive HEAD cli | tar -x -C /tmp/fresh && cd /tmp/fresh/cli && go build ./...
# parser/parser.go:16:42: undefined: models.StreamEvent   (+10 more)
```

Neither CI workflow (`ci.yml`, `eval-gate.yml`) has a `setup-go` step, which is
why this went unnoticed.

### Bug 2 — `src/cli/` duplicates `cli/cmd/query`

Same flags (`--url/--token/--user-id/--email/--secret/--top-k/--session-id`),
same `POST /api/agent`, same output sequence (source cards → "Answer" rule →
progressive answer → `session_id`). Two implementations of one command that can
drift apart. `docs/cli.md` documents only the Go one. The
[2026-07-29 consolidate-scripts spec](2026-07-29-consolidate-runnable-scripts-design.md)
explicitly deferred `cli/` and `src/cli/` as "separate questions"; this is that
question.

## Design

### 1. Delete the unreachable packages

Whole packages: `tui/`, `parser/`, `starprompt/`, `browser/`, `overflow/`,
`version/`, `fsutil/`, `embedded/` (including `SKILL.md`).
Files: `api/stream.go`, `models/events.go`, `config/experiments.go`.

### 2. Prune the surface inside kept files

`tui/` was the only consumer of most of the client. Leaving that API in place
would invite the next TUI to grow against it, so it goes too.

- **`api/client.go`** — remove the 10 dead methods and their `ClientAPI`
  entries. Leaves 8: `QueryAgent` plus the 7 memory calls — exactly what the two
  binaries use and what `docs/cli.md` documents.
- **`api/client_test.go`** — the `ListAgents` / `TestConnection` / `Search`
  cases go with their methods. `QueryAgent` cases stay.
- **`models/models.go`** — drop the chat / persona / file / search types that
  only the removed methods reached.
- **`config/config.go`** — drop `Features`, `StreamMarkdownEnabled`,
  `EnvStreamMarkdown`, `AgentID` / `EnvAgentID`, and
  `Save` / `ConfigExists` / `IsConfigured` / `DefaultConfig`; `tui/configure.go`
  was their only caller. The live surface is `Load`, `APIURL`, and
  `Config{ServerURL, APIKey}`. The three `config_test.go` feature-flag tests go
  with the flag.
- **`testutil/testutil.go`** — drop the `/api/version` and `/api/me` handlers in
  `AgentSearchServer` if no surviving test hits them.
- **`go mod tidy`** — drops `bubbletea`, `bubbles`, `lipgloss`, `logrus`, and
  `x/text` plus ~25 indirect deps. Kept: `glamour` (render), `golang-jwt`
  (clientauth), `x/term`.

**Result:** `cli/` goes 6,676 → ~2,300 LOC; eight packages behind two binaries,
every one reachable.

#### Tradeoff: the `stream_markdown` config key

`stream_markdown` is a documented config-file key, but once `tui/` is gone
nothing reads it — `cmd/query` streams markdown unconditionally via
`render.Progressive`. The alternative is wiring `-no-stream-markdown` into
`cmd/query`, which would be adding a feature rather than simplifying. Decision:
remove the flag.

### 3. Fix the `.gitignore` rule

The bare `models/` rule is genuinely protecting `src/model/models/intent_classifier.pt`,
a real checkpoint, so it stays. One line is added after it:

```
models/
!cli/models/
```

This changes nothing today (`models.go` is already tracked). It stops the *next*
file added under `cli/models/` from silently vanishing — precisely how
`events.go` was lost.

### 4. Delete `src/cli/`

Remove `src/cli/` (5 files, 228 LOC) and its 4 unit tests (252 LOC:
`tests/unit/test_cli_{auth,client,query,render}.py`). No `pyproject.toml` change
is needed — `include = ["src*"]` is a glob and there is no console-script entry.
`cli/cmd/query` becomes the single query client.

### 5. Rewrite `docs/cli.md`

The doc's current shape is accurate for what survives. Three changes:

1. Correct "Extending the CLI" to describe the pruned 8-method `ClientAPI`.
2. Add a **"What `cli/` is not"** section recording that a bubbletea TUI, star
   prompt, embedded skill, and version gate were removed as unreachable
   heritage — so the next reader neither re-adds them nor hunts for a streaming
   client that was never wired up.
3. Note that `src/cli/` was a Python duplicate and is gone.

## Non-goals

- No behavior change to either binary. Flags, output, and exit codes stay
  byte-identical.
- No new features (no `-no-stream-markdown`, no TUI, no new subcommands).
- No changes to the backend, or to any `cli/` package that is reachable.

## Verification

1. `cd cli && go build ./... && go vet ./... && go test ./...` → green
2. `git archive HEAD cli | tar -x -C <tmp> && cd <tmp>/cli && go build ./...` →
   green. **This is the check that fails today** and is the regression test for
   Bug 1.
3. `pytest tests/unit tests/regression` → green, 4 fewer files collected
4. `grep -rn "src\.cli\|src/cli" --include='*.py' --include='*.md' .` → only
   historical spec/plan mentions remain
5. Smoke: build `query` and `memory` against a live backend on :7860 and confirm
   output identical to pre-change
