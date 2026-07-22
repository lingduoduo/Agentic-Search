# Spec: `--json` output + semantic exit codes for the memory CLI

## Origin

Follow-on to the memory CLI (#457). To leverage the CLI in scripts/CI it needs
machine-readable output and meaningful exit codes; today `cli/cmd/memory` prints
human-only lines and returns a flat `0/1/2`.

## Goal

- `--json` flag: print the raw decoded response as indented JSON instead of the
  formatted human lines, for every subcommand.
- Semantic exit codes via the existing `cli/exitcodes` package: map API failures
  to `exitcodes.ForHTTPStatus` and argument-validation failures to `BadRequest`.

## Non-goals

- Touching the `query` binary (it's an interactive TUI/streaming client; `--json`
  doesn't fit it). Scope is `cli/cmd/memory` only.
- Any backend / MCP / service change.
- A unified `agentic` binary (separate, larger idea).

## Design

Single file: `cli/cmd/memory/main.go`.

- Add `--json` bool flag to the per-subcommand `flag.FlagSet`; thread a
  `jsonOut bool` param into `dispatch`.
- Add a `printJSON(v any) error` helper (`json.MarshalIndent(v, "", "  ")` →
  stdout). In each `dispatch` branch, after the client call succeeds, when
  `jsonOut` is set, `return printJSON(resp)` before the formatted output.
- Exit codes (reuse `cli/exitcodes`, already in the repo):
  - `run` maps a returned error: `*exitcodes.ExitError` → its `.Code`;
    `*api.APIError` → `exitcodes.ForHTTPStatus(status)`; else `General` (1).
  - The two argument-validation errors in `dispatch` (`add` with no text,
    `search` with no query) become `exitcodes.Newf(exitcodes.BadRequest, ...)`.
  - `run`'s existing literal returns become the named constants
    (`Success` = 0, `BadRequest` = 2) for readability. Usage/no-args/unknown-cmd
    stay `BadRequest` (2).
- Add `--json` to the `usage()` common-flags line.

## Testing (`cli/cmd/memory/main_test.go`, Go)

- `--json`: capture stdout, run `list --json` against an `httptest` server, and
  `json.Unmarshal` the output into `models.MemoryListResponse` — asserts it emits
  valid, parseable JSON.
- Exit code mapping: `list` against a 401 server → `run` returns
  `int(exitcodes.AuthFailure)` (4); `add` with no text → `int(exitcodes.BadRequest)`
  (2), with no network call.
- Existing tests (unknown-cmd, no-args, add-hits-save, curate-empty-message) keep
  passing.

## Acceptance criteria

- `memory <cmd> --json` prints valid JSON of the response for all 6 subcommands.
- Exit codes: success 0; bad args 2; auth 4; server 5xx→8; etc. via `exitcodes`.
- `go build ./...`, `go vet ./...`, `go test ./cmd/memory/` all pass; `query`
  binary unchanged.
