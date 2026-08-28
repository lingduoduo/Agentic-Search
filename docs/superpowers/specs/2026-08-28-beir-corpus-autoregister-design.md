# Auto-register converted BEIR corpora — design

## Problem

`examples/beir_to_corpus.py` writes `data/corpus_<dataset>.jsonl` and stops there.
The corpus registry (`data/corpora.json`, #429) is what maps a *name* to a path, so
until it is hand-edited:

    python3 -m examples.beir_to_corpus --dataset nfcorpus   # writes the file
    ... --corpus nfcorpus                                   # ValueError: Unknown corpus spec

The registry already assumes the converter closes this loop — two of its own
messages read *"regenerate via beir_to_corpus.py"* — but nothing ever wrote the
entry. The two halves were built one PR apart (#322, #429) and never joined.

## Approach

The manifest **write** belongs next to the manifest **read**, in
`corpus_registry.py`, because that module owns the entry format. The converter
becomes a caller.

`register_corpus(name, *, corpus_path, doc_count, domain, source, manifest_path)`:

- creates the manifest when absent (a fresh checkout has none — `data/` is gitignored)
- merges rather than replaces, so registering one corpus cannot cost the user others
- replaces an entry of the same name, so re-running without `--limit` corrects the count
- **refuses a manifest it cannot parse** rather than overwriting it. A hand-edited
  file with a syntax error is a typo to fix, not a file to discard.

## Decisions

**`SUPPORTED` gains structure.** Registration needs a `domain` and a `source` (the
label that reaches the citation card), and both existed only inside prose strings
like `"~3.6K docs (medical/nutrition)"`. Parsing that back out would be worse than
storing it properly. The `--help` epilog now renders from the structured form.

**An unlisted dataset still registers**, with `domain=None` and the slug as its
source. BEIR has more datasets than this script lists; refusing them would be a
regression, and inventing a domain for them would be a lie.

**Registration failure must not look like conversion failure.** The `.jsonl` is
written first and is usable on its own, so a broken manifest prints the error plus
the `--corpus_path` fallback and exits normally.

**Invocation moves to `python3 -m examples.beir_to_corpus`.** Importing `src` breaks
direct script execution, and every other example in the repo is already invoked with
`-m`. This script was the lone holdout, and only got away with it by importing
nothing from `src`.

## Out of scope

BEIR queries and qrels are still discarded. Exporting them would enable real
retrieval evaluation (recall@k, nDCG@k) but is a separate, larger piece of work.
