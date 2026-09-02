# Run the server import guard in a subprocess

Date: 2026-09-02

## Problem

The import guard added in #566 imports every module under
`src/internal/servers/` inside the pytest process. That has two costs the
original design did not account for.

**Memory.** The imports add **164 MB of RSS** and stay resident for the rest of
the session: measured 147 `src.internal.servers*` entries in `sys.modules` after
the guard runs, against 0 without it.

**Import side effects.** Importing 145 server modules registers routers, builds
module-level singletons and reads environment, for every test that follows in
the same session. A guard that only asserts "these files import" has no business
imposing that on the rest of the suite.

The memory cost surfaced first. On an 8 GB machine with roughly 200 MB free,
`tests/unit/test_mcp_document_tools.py::test_parser_watchdog_terminates_a_process_over_the_rss_limit`
began failing intermittently in full-suite runs after #566 merged — twice in the
first two runs, then not again in five.

That test spawns a child that allocates 384 MB and requires the parent watchdog
to *sample* the child's RSS above a 256 MB limit before a timeout. Add 164 MB to
the parent on a machine near capacity and the child's allocation slows enough
for the sample to miss. The evidence lines up: the test passes 20/20 standalone
and 5/5 for its own file, and only ever failed inside a full-suite run.

## Design

Run the probe in a fresh interpreter via `subprocess.run`, and have it print the
module count followed by one line per failure. The pytest process parses that
output and asserts there are no failures.

The probe still catches `BaseException` rather than `Exception`: a module that
calls `sys.exit()` at import time is one of the two shapes the guard exists to
catch, and a bare `except Exception` would let it through.

Cost is unchanged in wall clock — one interpreter start plus 145 imports, about
1.5 s per probe — and the modules are freed when the subprocess exits.

### Reporting

The in-process version was parametrised, one case per module, so a failure named
itself in the test id. The subprocess version is a single test whose assertion
message lists every failing module and its exception. That keeps the diagnostic
and drops 145 test ids from the suite.

## Testing

- `test_every_server_module_imports` — the guard itself, with a `count > 100`
  assertion so a broken path cannot make it vacuous.
- `test_the_guard_does_not_import_the_tree_into_this_process` — snapshots
  `sys.modules` around the probe and asserts the guard added nothing. This is the
  regression: it fails against the #566 implementation and passes here, and it is
  order-independent because it compares before against after rather than against
  zero.

Both mutation shapes remain caught: a module that raises on import, and one that
calls `sys.exit(1)`.

## Out of scope

The watchdog test. It is legitimately memory-sensitive, and making it robust to
an arbitrarily large parent process is a separate question from whether this
guard should be inflating that parent in the first place.
