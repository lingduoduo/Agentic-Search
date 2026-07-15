# Archive Context Sources Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Archive the current spec/plan corpus while keeping future active sources and all generated context packs fully supported.

**Architecture:** Extend the existing generator's source metadata and discovery layer to combine active and archived directories while preserving each source's actual relative link. Then migrate the current corpus with Git renames, retain empty active directories, regenerate packs, and validate checksum-preserving coverage.

**Tech Stack:** Python 3 standard library, pytest, Git, Markdown.

## Global Constraints

- Do not delete or alter the contents of any specification or plan.
- Preserve source history with Git rename detection.
- Keep `docs/superpowers/specs/` and `docs/superpowers/plans/` available for future active work.
- Fail explicitly when active and archived directories contain the same filename for the same source kind.
- Keep generation and validation deterministic.

---

### Task 1: Active and Archived Source Discovery

**Files:**
- Modify: `scripts/generate_context_packs.py`
- Modify: `tests/unit/test_generate_context_packs.py`

**Interfaces:**
- Consumes: Markdown sources under active `{specs,plans}/` and archived `archive/{specs,plans}/` directories.
- Produces: `SourceDoc.relative_path: Path`, combined `discover_sources(source_root)`, location-aware `_source_link(source)`, and explicit duplicate-filename validation.

- [ ] **Step 1: Write failing discovery and link tests**

Add tests that create one active spec and one archived plan, run `generate`, and assert both are indexed once with links to their actual locations.

```python
def test_discover_sources_combines_active_and_archive(tmp_path: Path) -> None:
    root = tmp_path / "superpowers"
    write(root / "specs/2026-07-01-active-design.md", "# Active\n")
    write(root / "archive/plans/2026-06-01-old.md", "# Old\n")

    generate(root, root / "context-packs")

    index = (root / "context-packs/INDEX.md").read_text()
    assert "../specs/2026-07-01-active-design.md" in index
    assert "../archive/plans/2026-06-01-old.md" in index
```

- [ ] **Step 2: Write a failing duplicate-source test**

Create the same spec filename under active and archive directories and assert `discover_sources` raises `ValueError` containing the filename and both paths.

- [ ] **Step 3: Run focused tests to confirm RED**

Run: `pytest tests/unit/test_generate_context_packs.py -q`

Expected: new archive discovery and duplicate tests fail because only active directories are scanned.

- [ ] **Step 4: Implement combined discovery and location-aware links**

Store each source path relative to `source_root` during discovery. Scan active before archive for deterministic diagnostics, sort the combined result by relative path, and reject duplicate basenames within each kind before normalization.

```python
@dataclass(frozen=True)
class SourceDoc:
    path: Path
    relative_path: Path
    kind: Literal["spec", "plan"]
    # existing fields unchanged


def _source_link(source: SourceDoc) -> str:
    return f"../{source.relative_path.as_posix()}"
```

- [ ] **Step 5: Run focused tests to confirm GREEN**

Run: `pytest tests/unit/test_generate_context_packs.py -q`

Expected: all generator tests pass.

- [ ] **Step 6: Run formatting and lint checks**

Run:

```bash
ruff format scripts/generate_context_packs.py tests/unit/test_generate_context_packs.py
ruff check scripts/generate_context_packs.py tests/unit/test_generate_context_packs.py
```

Expected: both commands exit 0.

- [ ] **Step 7: Commit discovery support**

```bash
git add scripts/generate_context_packs.py tests/unit/test_generate_context_packs.py
git commit -m "feat: discover archived context sources"
```

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

- [ ] **Step 1: Record pre-migration source checksums**

Run:

```bash
find docs/superpowers/specs docs/superpowers/plans -type f -name '*.md' -print0 | sort -z | xargs -0 shasum -a 256 | sed 's#docs/superpowers/\(specs\|plans\)/##' > /tmp/context-source-archive.before
```

Expected: one checksum line for every current source.

- [ ] **Step 2: Move the corpus and retain active directories**

Create the archive directories, move every Markdown source with `git mv`, and add `.gitkeep` files to both now-empty active directories.

- [ ] **Step 3: Verify archived checksums**

Run:

```bash
find docs/superpowers/archive/specs docs/superpowers/archive/plans -type f -name '*.md' -print0 | sort -z | xargs -0 shasum -a 256 | sed 's#docs/superpowers/archive/\(specs\|plans\)/##' > /tmp/context-source-archive.after
diff -u /tmp/context-source-archive.before /tmp/context-source-archive.after
```

Expected: `diff` exits 0 with no output.

- [ ] **Step 4: Regenerate and validate all packs**

Run:

```bash
python scripts/generate_context_packs.py
python scripts/generate_context_packs.py --check
```

Expected: generation reports the same source and pack coverage as before migration plus this design/plan pair; validation exits 0 and all links resolve.

- [ ] **Step 5: Verify archive layout**

Run:

```bash
find docs/superpowers/specs docs/superpowers/plans -mindepth 1 -maxdepth 1 -type f | sort
find docs/superpowers/archive/specs docs/superpowers/archive/plans -type f -name '*.md' | wc -l
```

Expected: active directories list only their `.gitkeep` files; archive count equals the pre-migration source count.

- [ ] **Step 6: Run final tests and hygiene checks**

Run:

```bash
pytest tests/unit/test_generate_context_packs.py -q
ruff check scripts/generate_context_packs.py tests/unit/test_generate_context_packs.py
ruff format --check scripts/generate_context_packs.py tests/unit/test_generate_context_packs.py
git diff --check
```

Expected: all commands exit 0.

- [ ] **Step 7: Commit the migration**

```bash
git add docs/superpowers scripts/generate_context_packs.py tests/unit/test_generate_context_packs.py
git commit -m "docs: archive context source documents"
```

- [ ] **Step 8: Push the updated PR branch**

Run: `git push origin context-packing-artifacts`

Expected: draft PR #414 updates to the migration commit.
