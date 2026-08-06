# CLI simplification — implementation plan

Spec: [2026-08-06-cli-simplification-design.md](../specs/2026-08-06-cli-simplification-design.md)

Baseline before starting: `cd cli && go vet ./... && go test ./...` is green
(17 packages, 11 with tests). `pytest tests/unit tests/regression` is green.

## Task 1 — capture the failing clone build (regression check for Bug 1)

Confirm the bug before fixing it:

```bash
rm -rf /tmp/cli-fresh && mkdir -p /tmp/cli-fresh
git archive HEAD cli | tar -x -C /tmp/cli-fresh
cd /tmp/cli-fresh/cli && go build ./...
```

→ verify: fails with `undefined: models.StreamEvent` (RED).

## Task 2 — delete the unreachable packages

```bash
cd cli
git rm -r tui parser starprompt browser overflow version fsutil embedded
git rm api/stream.go config/experiments.go
rm -f models/events.go        # untracked — see Task 5
```

→ verify: `go build ./...` fails only on references from files still to be
pruned in Task 3 (`api/client.go` streaming method, `config/config.go` features).
No other package should break.

## Task 3 — prune the surface inside kept files

Do these together; each leaves the tree non-compiling on its own.

1. `api/client.go` — delete `TestConnection`, `ListAgents`, `ListChatSessions`,
   `GetChatSession`, `RenameChatSession`, `UploadFile`, `GetBackendVersion`,
   `StopChatSession`, `SendMessageStream`, `Search` and their `ClientAPI`
   entries. Remove imports orphaned by the deletions.
2. `api/client_test.go` — delete the `ListAgents`, `TestConnection`, and
   `Search` cases.
3. `models/models.go` — delete the chat / persona / file / search types now
   unreachable. Drive this off the compiler plus a `grep -rn "models\.<Type>"`
   sweep; keep any type still referenced as a nested field of a surviving type.
4. `config/config.go` — delete `Features`, `StreamMarkdownEnabled`,
   `EnvStreamMarkdown`, `AgentID`, `EnvAgentID`, `Save`, `ConfigExists`,
   `IsConfigured`, `DefaultConfig`, and the `Features`/`AgentID` fields on
   `Config`.
5. `config/config_test.go` — delete `TestDefaultFeaturesStreamMarkdownNil`,
   `TestEnvOverrideStreamMarkdownFalse`, `TestLoadFeaturesFromFile`, and drop
   `EnvStreamMarkdown` from the env-clearing loop. Delete any test covering a
   function removed in step 4.
6. `testutil/testutil.go` — remove the `/api/version` and `/api/me` handlers
   from `AgentSearchServer` if no surviving test exercises them.

→ verify: `go build ./... && go vet ./... && go test ./...` all green.

## Task 4 — prune `go.mod`

```bash
cd cli && go mod tidy
```

→ verify: `bubbletea`, `bubbles`, `lipgloss`, `logrus`, and `x/text` are gone
from the direct require block; `glamour`, `golang-jwt/jwt/v5`, and `x/term`
remain. `go build ./... && go test ./...` still green.

## Task 5 — fix the `.gitignore` rule (Bug 1)

Add `!cli/models/` immediately after the `models/` rule at `.gitignore:5`.

→ verify two things:
- `git check-ignore -v cli/models/models.go` → no match
- `git status --porcelain --ignored src/model/models` → still `!!` (the
  checkpoint stays ignored)

Then re-run Task 1's clone build → green (GREEN for Bug 1).

## Task 6 — delete `src/cli/` (Bug 2)

```bash
git rm -r src/cli
git rm tests/unit/test_cli_auth.py tests/unit/test_cli_client.py \
       tests/unit/test_cli_query.py tests/unit/test_cli_render.py
rm -rf src/cli/__pycache__
```

→ verify: `pytest tests/unit tests/regression` green with 4 fewer files
collected; `grep -rn "src\.cli\|src/cli" --include='*.py' .` returns nothing.

## Task 7 — rewrite `docs/cli.md`

Per spec §5: correct "Extending the CLI" for the 8-method `ClientAPI`, add the
"What `cli/` is not" section, and record the `src/cli/` removal. Keep the
Prerequisites / Build / `query` / `memory` / Authentication / Exit codes sections
— they describe surviving behavior and stay accurate.

→ verify: every relative link in the file resolves (`cli/cmd/query`,
`cli/cmd/memory`, `cli/exitcodes/codes.go`, `cli/api/client.go`, `cli/models`),
and no sentence describes a deleted package.

## Task 8 — full verification

1. `cd cli && go build ./... && go vet ./... && go test ./...`
2. clone build from Task 1 → green
3. `pytest tests/unit tests/regression`
4. `ruff check . && ruff format --check .`
5. Smoke against a live backend on :7860:
   ```bash
   cd cli && go build -o ../bin/query ./cmd/query && go build -o ../bin/memory ./cmd/memory
   ../bin/query "What is FAISS?"
   ../bin/memory list
   ```
   → verify: same output shape as before the change (source cards, Answer rule,
   `session_id`; memory lines).
6. `git status` clean except intended deletions; `main` unmoved.

## Task 9 — PR

Commit, push `refactor/simplify-cli`, open a PR summarizing the reachability
findings and both bug fixes.
