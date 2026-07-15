# Generated Context Pack

# Archive Context Sources

## Sources

- [Specification: 2026-07-15-archive-context-sources-design.md](../archive/specs/2026-07-15-archive-context-sources-design.md)
- [Plan: 2026-07-15-archive-context-sources.md](../archive/plans/2026-07-15-archive-context-sources.md)

## Specification Context

### Goal

Move the current specification and implementation-plan corpus out of the active documentation directories without losing the source material required by generated context packs.

## Implementation Plan Context

### Task 1: Active and Archived Source Discovery

**Files:**
- Modify: `scripts/generate_context_packs.py`
- Modify: `tests/unit/test_generate_context_packs.py`

**Interfaces:**
- Consumes: Markdown sources under active `{specs,plans}/` and archived `archive/{specs,plans}/` directories.
- Produces: `SourceDoc.relative_path: Path`, combined `discover_sources(source_root)`, location-aware `_source_link(source)`, and explicit duplicate-filename validation.

- [ ] **Step 1: Write failing discovery and link tests**

Add tests that create one active spec and one archived plan, run `generate`, and assert both are indexed once with links to their actual locations.

- [ ] **Step 2: Write a failing duplicate-source test**

…

### Task 2: Archive Migration and Regeneration

**Files:**
- Create: `docs/superpowers/archive/specs/`
- Create: `docs/superpowers/archive/plans/`
- Create: `docs/superpowers/specs/.gitkeep`
- Create: `docs/superpowers/plans/.gitkeep`
- Move: `docs/superpowers/specs/*.md` → `docs/superpowers/archive/specs/*.md`
- Move: `docs/superpowers/plans/*.md` → `docs/superpowers/archive/plans/*.md`
- Modify: `docs/superpowers/context-packs/INDEX.md`
- Modify: `docs/superpowers/context-packs/*-context-pack.md`

**Interfaces:**
- Consumes: the combined discovery behavior from Task 1 and all current source documents.
- Produces: an archived source corpus with unchanged content and regenerated location-correct context packs.

…

## Context Boundary

This pack summarizes its linked sources. Consult those documents for complete details; no implementation status is inferred here.
