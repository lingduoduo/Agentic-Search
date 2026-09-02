# Extract the parser and routing out of the agentic-search CLI

Date: 2026-09-02

## Problem

`examples/run_agentic_search.py` is 889 lines: 16 top-level definitions, 29
argparse flags, three mode runners, intent routing and model routing, all in one
file. It is the entry point 27 documents point at.

## Design

### Not a full split

The obvious refactor — the `index_builder` treatment from #363, a thin facade
over `bootstrap` / `routing` / `runners` / `cli` — is the wrong shape here.
This file's job is to be read. Someone following the quickstart opens it to see
how a mode goes from question to answer, and a four-way split turns one readable
pass into a hunt across five files. #368/#369 also recorded that internal
cross-imports through a slimmed facade break in non-obvious ways.

So extract only what a reader does not need in order to follow the flow, and
leave the flow itself in place.

### What moves

`examples/agentic_search/parser.py` — `_build_parser`, 135 lines of flag
declarations that sit between the reader and the code they came for. Nothing in
it decides behaviour.

`examples/agentic_search/routing.py` — `IntentPrediction`,
`_load_intent_prediction`, `resolve_search_settings`, `ModelRouteDecision`,
`_resolve_model_route`. Both behaviours are off unless `--intent_index` or
`--model_routing` turns them on, so neither is part of the three modes the
example demonstrates.

Both groups are self-contained: the routing functions reference only their own
classes plus `argparse`, `dataclass`, `Path` and `Any`; `_build_parser`
references only `argparse`. No module-level constant moves with them.

### What stays

`main`, `run_single_turn`, `run_search_agent`, `run_tool_agent`,
`_print_result`, `_build_sampling_params`, and the model-bootstrap helpers —
about 580 lines that read top to bottom, with the existing section banners
still accurate.

### Back-compat

`run_agentic_search.py` re-exports the five moved routing names, beside the
existing `src.model.serving` re-exports. `tests/unit/test_run_agentic_search.py`
imports 13 names from the module and keeps working unchanged; so does anything
else importing from the documented entry point.

## Testing

`test_run_agentic_search_has_no_minisweagent_dependency` reads the entry file's
text. After the split, that check would silently stop covering the moved code —
the dependency could return through a submodule with the test still green. It
now scans `run_agentic_search.py` plus every module in `examples/agentic_search/`,
and asserts it found at least four files so a bad glob cannot make it vacuous.

Everything else is behaviour-preserving and covered by the existing 26 tests
plus `--help` under direct `python3 -m` invocation.

## Out of scope

The three mode runners and `main`. Splitting those is what would make the
example unreadable, which is the point of stopping here.
