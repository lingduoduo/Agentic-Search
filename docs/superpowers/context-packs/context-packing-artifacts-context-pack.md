# Generated Context Pack

# Context Packing Artifacts

## Sources

- [Specification: 2026-07-15-context-packing-artifacts-design.md](../specs/2026-07-15-context-packing-artifacts-design.md)
- [Plan: 2026-07-15-context-packing-artifacts.md](../plans/2026-07-15-context-packing-artifacts.md)

## Specification Context

### Goal

Create focused, navigable context packs for every design specification and implementation plan under `docs/superpowers/`. The packs should let an agent load the context for one topic without flooding its context window with the complete documentation corpus.

### Scope

- Index every Markdown file directly under `docs/superpowers/specs/` and `docs/superpowers/plans/`.
- Match specifications and plans by their date-independent topic slug, removing the specification's trailing `-design` suffix.
- Create one compact context pack for each matched topic.
- Create a standalone pack when a specification or plan has no counterpart.
- Preserve all source specifications and plans unchanged.

## Implementation Plan Context

### Global Constraints

- Preserve all source specifications and plans unchanged.
- Summaries must contain only information supported by source documents.
- Do not infer implementation status from the current codebase.
- Generated links must be relative Markdown links.
- Source discovery and output ordering must be deterministic.

---

### Task 1: Deterministic Context-Pack Generator

**Files:**
- Create: `scripts/generate_context_packs.py`
- Create: `tests/unit/test_generate_context_packs.py`

**Interfaces:**
- Consumes: Markdown files in `<root>/specs/*.md` and `<root>/plans/*.md`.
- Produces: `normalize_source(path: Path, kind: str) -> SourceDoc`, `pair_sources(specs: list[SourceDoc], plans: list[SourceDoc]) -> list[TopicBundle]`, `render_pack(bundle: TopicBundle) -> str`, `render_index(bundles: list[TopicBundle]) -> str`, and `generate(source_root: Path, output_dir: Path) -> list[Path]`.

- [ ] **Step 1: Write failing unit tests**

…

### Task 2: Generate and Validate Repository Context Packs

**Files:**
- Create: `docs/superpowers/context-packs/INDEX.md`
- Create: `docs/superpowers/context-packs/*-context-pack.md`

**Interfaces:**
- Consumes: `generate(source_root: Path, output_dir: Path) -> list[Path]` from Task 1 and every source file under `docs/superpowers/specs/` and `docs/superpowers/plans/`.
- Produces: a complete navigable context-pack corpus whose source coverage can be checked mechanically.

- [ ] **Step 1: Record source checksums**

Run:

Expected: one checksum line per source document.

- [ ] **Step 2: Generate all packs**

Run: `python scripts/generate_context_packs.py`

Expected: output reports the generated pack count and `INDEX.md` exists.

…

## Context Boundary

This pack summarizes its linked sources. Consult those documents for complete details; no implementation status is inferred here.
