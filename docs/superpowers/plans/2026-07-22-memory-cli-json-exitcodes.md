# memory CLI `--json` + exit codes — Plan

Spec: `docs/superpowers/specs/2026-07-22-memory-cli-json-exitcodes-design.md`

Single trivial change to `cli/cmd/memory/main.go` + tests. Implemented directly
(not via subagents) given the scope. All Go commands run from `cli/`.

## Step 1 — Tests (TDD)
Add to `cli/cmd/memory/main_test.go`:
- `TestRunListJSONOutput` — stdout-capture; `list --json` vs an httptest server
  returning a memories body; `json.Unmarshal` output into
  `models.MemoryListResponse`, assert the id round-trips.
- `TestRunExitCodeAuthFailure` — `list` vs a 401 server → `run` returns
  `int(exitcodes.AuthFailure)`.
- `TestRunExitCodeBadRequestOnMissingText` — `run([]string{"add"})` →
  `int(exitcodes.BadRequest)` (no network).
Run `cd cli && go test ./cmd/memory/` → the new tests FAIL to compile (`--json`
flag / exit-code behavior absent).

## Step 2 — Implement in `cli/cmd/memory/main.go`
- Imports: add `encoding/json`, `errors`, `github.com/lingduoduo/Agentic-Search/cli/exitcodes`.
- `usage()`: append `--json` to the common-flags line.
- In `run`: add `jsonFlag := fs.Bool("json", false, "output raw JSON")`; change
  the `return 2`/`return 0` literals to `int(exitcodes.BadRequest)` /
  `int(exitcodes.Success)`; change the final error handling to:
  ```go
  if err := dispatch(ctx, client, cmd, fs, *jsonFlag, *topK, *noConflict, *generate, *sessionID); err != nil {
      fmt.Fprintln(os.Stderr, "error:", err)
      var exitErr *exitcodes.ExitError
      if errors.As(err, &exitErr) {
          return int(exitErr.Code)
      }
      var apiErr *api.APIError
      if errors.As(err, &apiErr) {
          return int(exitcodes.ForHTTPStatus(apiErr.StatusCode))
      }
      return int(exitcodes.General)
  }
  return int(exitcodes.Success)
  ```
- Add helper:
  ```go
  func printJSON(v any) error {
      b, err := json.MarshalIndent(v, "", "  ")
      if err != nil {
          return err
      }
      fmt.Println(string(b))
      return nil
  }
  ```
- `dispatch` signature gains `jsonOut bool` (after `cmd`/`fs`). In each branch,
  after the client call returns without error, insert
  `if jsonOut { return printJSON(resp) }` before the formatted prints. (For
  `add` the value is `resp`; for `profile` the value is `resp`; etc.)
- Change the two arg-validation errors to
  `return exitcodes.Newf(exitcodes.BadRequest, "add requires memory text")` and
  `... "search requires a query"`.

Run `cd cli && go test ./cmd/memory/` → GREEN; then `go build ./...` + `go vet ./...`.

## Step 3 — Commit + PR
Commit `cli/cmd/memory/`, spec, plan. Push; open PR against main.
