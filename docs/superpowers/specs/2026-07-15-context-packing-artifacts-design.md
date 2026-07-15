# Context Packing Artifacts Design

## Goal

Create focused, navigable context packs for every design specification and implementation plan under `docs/superpowers/`. The packs should let an agent load the context for one topic without flooding its context window with the complete documentation corpus.

## Scope

- Index every Markdown file directly under `docs/superpowers/specs/` and `docs/superpowers/plans/`.
- Match specifications and plans by their date-independent topic slug, removing the specification's trailing `-design` suffix.
- Create one compact context pack for each matched topic.
- Create a standalone pack when a specification or plan has no counterpart.
- Preserve all source specifications and plans unchanged.

## Output Structure

Generated artifacts live under `docs/superpowers/context-packs/`:

- `INDEX.md` is the master hierarchical index, grouped chronologically and then by topic. Each entry links to its source specification, source plan, and generated pack when present.
- `<topic>-context-pack.md` is the focused pack for a matched pair or standalone source. Topic slugs omit the date so related documents with differing dates can still be paired.

If two distinct source topics normalize to the same slug, the pack filename includes the source date to remain unique.

## Context Pack Contents

Each pack contains only information supported by its source documents:

1. Topic and source links.
2. Purpose and intended outcome.
3. Key decisions and constraints.
4. Architecture, components, or affected areas explicitly named by the sources.
5. Implementation sequence or task summary when a plan exists.
6. Verification commands and acceptance criteria when specified.
7. Open questions, risks, or unresolved items when specified.

The pack summarizes rather than reproduces entire documents. It does not infer implementation status from the existence of source files or from the current codebase.

## Generation and Repeatability

Generation is deterministic for a given documentation corpus:

- Source discovery is lexicographically sorted.
- Pairing uses normalized topic slugs.
- Links are relative Markdown links.
- Existing generated packs are replaced only when they correspond to discovered sources.
- The index records unmatched sources explicitly.

## Validation

After generation:

- Every source spec and plan appears exactly once in `INDEX.md`.
- Every index link resolves to an existing file.
- Every generated pack links back to all of its sources.
- No pack is empty.
- Original specs and plans have no content changes.
- Markdown contains no placeholder markers introduced by generation.

## Non-Goals

- Changing or correcting existing specifications and plans.
- Determining whether planned work has been implemented.
- Creating implementation plans for the context-packing machinery itself beyond what is needed to generate the requested artifacts.
- Loading every pack simultaneously into an agent session.
