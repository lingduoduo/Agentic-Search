# Auto-register converted BEIR corpora — plan

## Task 1: `register_corpus` in the registry
Write side of `load_manifest`. Create-if-absent, merge, replace-same-name, refuse
unparseable.
**Verify:** tests red before, green after; a round-trip test registers then calls
`resolve_corpus_docs(name)` — a manifest writing fields the reader ignores would
pass every other test.

## Task 2: Structured `SUPPORTED`
`name -> {size, domain, source}` so registration has real fields and `--help` still
renders.
**Verify:** `_build_parser().format_help()` contains every dataset, size and domain.

## Task 3: Wire the converter
Register after converting; report a manifest error without failing the run.
**Verify:** stubbed-BEIR tests cover default path, `--out`, unlisted dataset,
pre-existing entries, corrupt manifest.

## Task 4: Fix the invocation the `src` import broke
Switch docstring + README to `python3 -m examples.beir_to_corpus`.
**Verify:** `-m` runs; no `examples/beir_to_corpus.py` invocation left in any doc.

## Task 5: Gates
**Verify:** full suite, ruff, torch-blocked collection, and a mutation check per
guard (drop registration / clobber manifest / overwrite corrupt / ignore `--out` /
break epilog) — each must turn its test red.
