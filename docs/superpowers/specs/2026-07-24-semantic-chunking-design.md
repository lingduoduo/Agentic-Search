# Semantic chunking (opt-in embedding-similarity chunker)

**Date:** 2026-07-24
**Branch:** `feat/semantic-chunking` (off `main`)
**Status:** design approved, pending spec review

## Context

The live chunker (`src/internal/document_index/chunking.py`) is a structure-aware
(paragraph/section → sentence → token-window) splitter with fixed-size overlap
fallback. It covers fixed-size and recursive/structure-aware chunking, but the
third canonical strategy — **semantic chunking** (place boundaries where adjacent
sentences become semantically dissimilar) — is absent.

This adds semantic chunking as an **opt-in** strategy behind a `ChunkingConfig`
flag. With the flag off, behavior is byte-identical to today (matches the repo's
"all optimizations opt-in, default-off" convention).

## Goal

Add an embedding-similarity semantic chunker that splits a document where the
adjacent-sentence cosine distance exceeds a per-document percentile breakpoint,
with a hard `chunk_size` cap and safe fallbacks, wired through the existing
indexing pipeline with zero new module dependency in `chunking.py`.

## Key enabler

`run_indexing_pipeline` (`pipeline.py:37`) already receives
`embedding_fn: EmbeddingFn` (`Callable[[list[str]], np.ndarray]`, `embedding.py:44`)
and calls `chunk_documents(...)` (`pipeline.py:51`) **before** the embedding stage.
The embedder is therefore already in scope at chunk time — semantic chunking threads
it into `chunk_documents`. Tests use the existing `deterministic_embedding_fn(dim=8)`
(`embedding.py:392`).

## Design

### Config (`ChunkingConfig`, `models.py`)

Add three fields (all default to the no-op / off state):

- `semantic_chunking: bool = False` — master switch.
- `semantic_breakpoint_percentile: float = 95.0` — the Nth percentile of a
  document's adjacent-sentence distances above which a boundary is placed.
- `semantic_buffer_size: int = 1` — embed each sentence combined with ±(buffer-1)
  neighbors to denoise the similarity signal; `1` = embed each sentence alone.

`ChunkingConfig.validate()` gains: `0 < semantic_breakpoint_percentile < 100` and
`semantic_buffer_size >= 1`.

### Algorithm — `_split_text_semantic(text, chunk_size, chunk_overlap, embedding_fn, percentile, buffer_size)`

1. Flatten the document into an ordered sentence list by reusing the existing
   `_split_paragraphs` then `_split_sentences_in_paragraph` on each paragraph.
2. If `buffer_size > 1`, build a "combined" string per sentence = the sentence
   plus its `buffer_size-1` neighbors on each side (sliding window over the
   sentence list); otherwise the combined string is the sentence itself.
3. Embed all combined strings in **one** `embedding_fn(list[str])` call.
4. Compute cosine distance `1 - cos(e_i, e_{i+1})` for each adjacent pair
   (L2-normalize defensively; guard zero-norm vectors → distance 0).
5. `threshold = numpy.percentile(distances, percentile)`. Place a boundary after
   sentence `i` wherever `distances[i] > threshold`. (Strict `>` so a flat/uniform
   document with all-equal distances yields no interior boundary → one chunk.)
6. Group the sentences between boundaries; join each group with a space into a
   candidate chunk (same join convention as `_split_text_paragraphs`).
7. **Size cap:** for any candidate group whose `_token_count` exceeds `chunk_size`,
   re-split that group's text with the existing `_split_text_paragraphs(...,
   chunk_size, chunk_overlap)` and emit those sub-chunks in place. (Chosen over a
   raw token window so oversized semantic groups still respect paragraph/sentence
   structure and get overlap.)
8. Return the flat chunk-text list. No cross-chunk overlap is added *between*
   distinct semantic chunks (the whole point is that they are topically distinct);
   overlap applies only within the size-cap re-split path.

### Fallbacks (never crash indexing)

`_split_text_semantic` returns `_split_text_paragraphs(text, chunk_size,
chunk_overlap)` unchanged when any of: `embedding_fn is None`; fewer than 2
sentences; `embedding_fn` raises; or the embedding output shape is unusable
(wrong length / empty). A logged warning accompanies the error fallback.

### Wiring

- `chunk_document(document, config, *, embedding_fn=None)` and
  `chunk_documents(documents, config, *, callback=None, embedding_fn=None)` gain
  the optional keyword.
- Inside `chunk_document`, replace the `_split_text(...)` call so that when
  `config.semantic_chunking and embedding_fn is not None`, it routes to
  `_split_text_semantic(...)`; otherwise unchanged.
- `pipeline.py:51` passes its existing `embedding_fn` into `chunk_documents(...)`.
- **Architectural boundary preserved:** `chunking.py` does NOT import from
  `embedding.py`. The parameter is typed structurally
  (`Callable[[list[str]], "np.ndarray"] | None`); `chunking.py` imports `numpy`
  directly (numpy is already a hard dep of the indexing stack) for the distance /
  percentile math. This keeps chunking independent of the embedding model, as its
  module docstring states.

## Verification / success criteria

1. `semantic_chunking=False` ⇒ `chunk_document` output is byte-identical to today
   (a regression test over a sample doc asserts equality with the flag off vs. the
   pre-change behavior).
2. With the deterministic embedder and a document of two clearly distinct topic
   blocks, the boundary lands at the topic shift (one chunk per topic).
3. Size-cap: a semantic group exceeding `chunk_size` is re-split; no emitted chunk
   exceeds `chunk_size` tokens.
4. Every fallback path returns the paragraph-splitter result and never raises:
   `embedding_fn=None`, `<2` sentences, embedder raises, malformed embedding shape.
5. `validate()` rejects out-of-range percentile / buffer.
6. `ruff check .` + `pytest` green; the pipeline still builds an index with
   `semantic_chunking=True` end-to-end using the deterministic embedder.

## Risks / tradeoffs

- **Cost:** semantic chunking embeds every sentence at index time — one extra
  embedding pass over the corpus. This is the documented, inherent tradeoff, gated
  behind the opt-in flag.
- **Quality is embedder-dependent:** with the deterministic (hash) embedder used in
  tests the boundaries are not meaningful — tests assert *mechanics* (boundary at a
  controlled distance spike), not semantic quality; real quality needs e5.
- **Boundedness:** the size cap guarantees no oversized chunk regardless of how
  large a single semantic region is, so a low-contrast document degrades to the
  existing splitter rather than emitting a giant chunk.

## Out of scope

- Code-block / table structure awareness (the other identified gap) — separate.
- Exposing semantic chunking via any HTTP/env surface beyond `ChunkingConfig` (the
  build tool / pipeline callers set it programmatically).
- Retuning default `chunk_size`/`overlap`.
