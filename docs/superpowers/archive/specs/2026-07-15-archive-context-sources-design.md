# Archive Context Sources Design

## Goal

Move the current specification and implementation-plan corpus out of the active documentation directories without losing the source material required by generated context packs.

## Archive Layout

All current Markdown sources move as Git renames:

- `docs/superpowers/specs/*.md` → `docs/superpowers/archive/specs/*.md`
- `docs/superpowers/plans/*.md` → `docs/superpowers/archive/plans/*.md`

The active `docs/superpowers/specs/` and `docs/superpowers/plans/` directories remain present with `.gitkeep` files so future design and planning workflows retain their established output locations.

## Discovery Behavior

The context-pack generator discovers sources in four locations:

1. `docs/superpowers/specs/*.md`
2. `docs/superpowers/plans/*.md`
3. `docs/superpowers/archive/specs/*.md`
4. `docs/superpowers/archive/plans/*.md`

Active and archived sources use the same filename normalization, pairing, rendering, and validation rules. Discovery order is deterministic and does not give active files different semantic priority.

If two discovered sources of the same kind have the same filename, generation fails with a clear duplicate-source error. This prevents an active file from silently shadowing an archived file.

## Link Behavior

Each generated source link reflects the source's actual location. Archived sources use `../archive/specs/<name>` or `../archive/plans/<name>`; future active sources continue using `../specs/<name>` or `../plans/<name>`.

The index and every pack are regenerated after the move. Existing validation continues to require exactly one index link and one pack backlink per discovered source.

## Migration

- Move every current specification and plan with `git mv` semantics so history remains traceable.
- Add `.gitkeep` to both active directories.
- Regenerate all context packs and the index from the combined active/archive corpus.
- Do not alter the contents of archived source documents.

## Validation

- The active directories contain only `.gitkeep` after migration.
- The archive contains every source that existed before migration, with identical content checksums.
- The generator discovers active and archived fixtures in tests.
- Duplicate same-kind filenames across active and archive locations fail explicitly.
- Every generated relative link resolves.
- `--check` validates the complete archived corpus without drift, stale owned files, or missing backlinks.
- The focused generator tests, Ruff checks, and repository unit suite pass subject to the two previously documented environment-dependent baseline exclusions.

## Non-Goals

- Deleting specifications or plans.
- Inferring completion state from document age or filename.
- Defining an automatic age-based archival policy.
- Changing the contents or meaning of existing sources.
