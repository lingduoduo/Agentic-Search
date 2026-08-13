# Intent model: pretrained wordpiece embeddings — design

**Date:** 2026-08-12
**Status:** Approved
**Follows:** #506 (training/promotion pipeline), #507 (out-of-scope probes, calibration, telemetry), #508 (route clarification), #509 (frame generation, unknown-token embedding, realistic-accuracy instrument)

## Problem

#509 built the instrument that measures the learned route honestly, and the
instrument returned a verdict: the model cannot read the language it is asked to
classify.

Frame-based generation grew the learned vocabulary from 126 to 147 tokens. The
30 hand-authored evaluation queries are **47% in-vocabulary**. The other half is
ordinary English no synthetic frame set contains — *dashboard, postmortem,
checklist, spreadsheet, standup, recall, latency, whether, anything*. Realistic
accuracy came to `0.567`, against the `3/5 = 0.60` hand-scored baseline that
motivated the work. No improvement.

The cause is structural, not a data shortage. A bag-of-embeddings model over a
vocabulary built from its own training set cannot represent a word it never saw;
adding synthetic examples grows that vocabulary a few tokens at a time and never
closes a gap made of general English. #509's own conclusion was that the next
step is a representation change.

## Goal

Eliminate out-of-vocabulary entirely, so the model reads every query, at the
latency and dependency footprint the learned route exists to provide.

## Why this representation

The learned route's promotion gates require *reduced LLM-classifier usage* and
*lower latency than LLM classification*. Its purpose is cheap local routing, so
the representation is chosen on latency first. Measured on the development
machine, single query, after warm-up:

| option | model load | per-query p50 | p95 |
|---|---|---|---|
| current bag-of-embeddings | none | 0.21ms | 0.52ms |
| MiniLM-L6-v2 encoder, CPU | 1.1s | 14.4ms | 19.3ms |
| MiniLM-L6-v2 encoder, MPS | 1.1s | 16.0ms | 27.7ms |
| e5-base-v2 encoder, MPS | 1.6s | 30.6ms | 68.4ms |
| e5-base-v2 encoder, CPU | 0.8s | 86.3ms | 120.8ms |

Every option clears the latency gate against an LLM call. But running a
transformer per request is not required to fix out-of-vocabulary. Taking only
the **wordpiece tokenizer and the input embedding table** — a dictionary lookup
and a mean — keeps today's sub-millisecond cost while removing the defect,
because wordpiece decomposes any word into known subwords: *postmortem* becomes
`post ##mo ##rte ##m`, each with a pretrained vector behind it.

Two consequences worth naming. MiniLM on MPS is *slower* than on CPU at batch 1,
where kernel-launch overhead dominates a 6-layer model, so an encoder-based
follow-up would run on CPU and sidestep the MPS instability this repo already
documents. And e5-base is the retrieval encoder, 2–6× slower than MiniLM, and
the wrong tool for short-text classification.

## Scope

### In scope

- A dependency-free WordPiece tokenizer over MiniLM's vocabulary.
- An offline command that extracts the tokenizer vocabulary and input embedding
  matrix into a local artifact.
- `IntentPipeline` reading wordpiece ids against a frozen pretrained embedding
  matrix.
- Checkpoint format version 4, self-contained.
- Deletion of the configuration that pretrained embeddings make meaningless.

### Out of scope

- Running a transformer at serving time. That is the documented follow-up if
  this design's accuracy disappoints, and it reuses this tokenizer unchanged.
- Changing `INTENT_LABELS`, the runtime cascade, the clarification path, or any
  dispatcher.
- Changing the generated dataset, the splits, the promotion gates, the
  calibration report, or the realistic-accuracy instrument.
- Fine-tuning the pretrained embeddings.
- Promoting a checkpoint. This design makes promotion possible; activating an
  artifact stays a separate operator decision.

## Architecture

```text
query text
  -> normalize_text            (existing: lowercase, strip accents, drop punctuation)
  -> WordPiece                 (new: greedy longest-match over 30522 tokens)
  -> ids                       ([PAD]=0, [UNK]=100 — BERT's own layout)
  -> frozen pretrained matrix  (30522 x 384, stored fp16, loaded fp32)
  -> masked mean pool          (unchanged)
  -> trained MLP head          (384 -> hidden -> hidden/2 -> 3)
  -> softmax -> IntentPrediction
```

Only the head trains. With 341 training examples, fine-tuning the embeddings
would overwrite the pretrained semantics and recreate the defect being fixed, so
the matrix is frozen.

BERT's `[PAD]` is id `0`, the same padding id the pooling already masks, so the
padding contract carries over untouched and the batching-invariance guarantee
keeps its meaning. `[UNK]` moves from #509's synthetic `1` to BERT's `100` and
becomes near-unreachable: it fires only on characters WordPiece cannot decompose
at all, never on an ordinary unseen word.

## Components

### WordPiece tokenizer

`src/model/wordpiece.py`, roughly 80 lines of pure Python with no torch import:
`WordPieceVocabulary.from_file(path)` and `.encode(text) -> list[int]`,
implementing greedy longest-match-first with `##` continuations, a per-word
`[UNK]` fallback, and the reference 100-character-per-word guard (HuggingFace's
`max_input_chars_per_word`, confirmed against the loaded tokenizer).

Dependency-free is a deliberate requirement, not an aesthetic one. The unit-test
CI job installs neither torch nor transformers, and this repo has twice shipped
collection failures caused by unguarded imports in that job. A pure-Python
tokenizer means these tests actually execute in CI rather than being skipped.

Input normalization reuses the existing `normalize_text`, so the contract is
"normalize, then wordpiece". That differs from HuggingFace's BasicTokenizer in
punctuation handling, which is why parity is asserted over *normalized* text —
a precise, testable claim rather than an approximate one.

### Pretrained artifact

A new `embeddings` subcommand on `src/model/intent_training.py`, run once
offline, pulls MiniLM's tokenizer vocabulary and input embedding matrix through
`sentence-transformers` — already a project dependency, used by the hybrid
retrieval server — and writes `data/intent_pretrained/vocab.txt` and
`data/intent_pretrained/embeddings.fp16.npy`. `train` gains `--pretrained`.

`data/` is gitignored, so these are regenerable local artifacts on the same
footing as the corpus files. The matrix is stored fp16: 23MB rather than 47MB,
with precision loss far below what a three-class head over mean-pooled vectors
can resolve. Pruning the matrix to observed tokens is rejected — it would
quietly reintroduce out-of-vocabulary, the defect this design exists to remove.

### Pipeline and checkpoint

`IntentPipeline` replaces its word-level `Vocabulary` with the wordpiece
vocabulary and builds its embedding layer frozen, with `padding_idx=0`. The
shared `Vocabulary` class in `src/internal/document_index/text.py` is **not**
modified: `document_index/cli.py` depends on it, exactly as in #509.

Checkpoints carry the matrix and vocabulary so an artifact stays self-contained,
growing from about 10MB to about 23MB. The format goes to version `4`; versions
1, 2, and 3 are rejected with a retraining message and never reinterpreted,
because each was trained under a different encoding and reusing its weights
would silently change what every index means.

### Configuration removed

Pretrained embeddings make three settings meaningless, and all three are
deleted rather than left as inert knobs:

- `min_freq`, because no vocabulary is built from the training data. This also
  removes the trap #509 measured and documented, where raising it to manufacture
  unknown-word training signal cost more vocabulary than it bought.
- `vocab_size` and `embedding_dim`, now derived from the matrix at 30522 and
  384.

## Error handling

- A missing or malformed pretrained bundle fails before training begins, naming
  the `embeddings` command that produces it.
- A `vocab.txt` line count that disagrees with the matrix row count is rejected.
  This is the silent-corruption case: a shifted vocabulary gives every token the
  wrong vector while training proceeds happily.
- Version 1, 2, or 3 checkpoints raise a retraining error.
- A query that decomposes to nothing encodes as a single `[UNK]`, preserving the
  rule that reading no tokens is a fact about the input rather than padding.
- A failed load disables the model route and records a diagnostic; the request
  continues through the existing fallback.

## Verification

### Unit tests, running in CI

- Known decompositions, `##` continuation, `[UNK]` fallback for undecomposable
  characters, the 100-character guard, and empty input.
- Vocabulary/matrix size disagreement is rejected. The bundle loader reads the
  matrix with numpy, which the unit-test job already installs through
  scikit-learn, so this check stays torch-free.

### Tests requiring optional dependencies

- A version 3 checkpoint raises the migration error. This one needs torch,
  because loading a checkpoint is a torch operation.

- **Parity:** the tokenizer agrees with HuggingFace's over the evaluation set,
  the out-of-scope probes, and the generated dataset, on normalized text.
- **No unknowns:** every evaluation query and every out-of-scope probe
  decomposes with zero `[UNK]`. This is checkable before training and is the
  direct refutation of the 47%-unread measurement.
- **Frozen:** embedding weights are bit-identical before and after training.
- Batching invariance and checkpoint round-trip carry over from #509.

### No new serving dependency

The routing path imports with `transformers` blocked through a `sys.meta_path`
finder — the technique this repo already uses to reproduce its torch-free CI job
locally.

### Measurement

Realistic accuracy and the out-of-scope separation margin are re-measured and
re-pinned. Because frozen embeddings change the optimization problem, the run
includes a small `epochs`/`lr` sweep and records the chosen values; that sweep is
what exposed the full-batch `epochs` defect in #509.

## Success criteria

The decision rule is fixed in advance, because #509 set a bar and missed it:

| realistic accuracy | verdict |
|---|---|
| ≥ 0.75 | worth promoting; proceed to activate behind the existing gates |
| 0.60 – 0.75 | real improvement, artifact stays dark, report and stop |
| ≤ 0.60 | the representation change failed; report and stop, do not iterate blindly |

Alongside: 100% token coverage on the evaluation set, an out-of-scope margin
still positive, p95 routing latency under 2ms, and no new serving dependency.

## Risks and mitigations

- **Mean-pooling discards word order.** "explain how to send an email" (`chat`)
  and "send the email" (`tool`) differ by the presence of the *how* and *to*
  tokens, not their arrangement. Those function words do carry signal and the
  dataset's hard-negative pairs test exactly this boundary; if it fails, that is
  the diagnosis, and the fix is the encoder follow-up behind the same interface.
- **Static embeddings are uncontextualized,** so a word's vector is the same in
  every sentence. Accepted: the alternative costs 14ms per request and a model
  in the serving path, and is the documented next step if this falls short.
- **A hand-written tokenizer can diverge from the reference.** Mitigated by
  parity testing against HuggingFace over every corpus the model actually sees,
  and by keeping the algorithm to the specified greedy longest-match rather than
  inventing behavior.
- **A 23MB artifact is larger than operators may expect.** Stated in the
  operator documentation, alongside the fact that it loads once at startup and
  costs nothing per request.
