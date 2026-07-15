# Generated Context Pack

# Readme Optimization

## Sources

- [Specification: 2026-07-13-readme-optimization-design.md](../specs/2026-07-13-readme-optimization-design.md)
- [Plan: 2026-07-13-readme-optimization.md](../plans/2026-07-13-readme-optimization.md)

## Specification Context

### Goal

Make the top-level README the fastest path for a developer to install and run Agentic Search locally. Preserve the existing deep technical material by moving it into focused documents under `docs/`.

### Scope boundaries

This change reorganizes and edits documentation only. It does not alter application behavior, dependencies, APIs, commands, configuration semantics, or generated architecture assets. Any incorrect existing instruction discovered during migration may be corrected only when the repository provides clear evidence for the correction.

## Implementation Plan Context

### Global Constraints

- Optimize the top-level README for a developer encountering the repository for the first time and trying to start the local stack.
- Preserve useful commands, warnings, and behavior descriptions from the existing README.
- Use relative Markdown links that work in repository browsers and local editors.
- Keep essential setup values and first-run troubleshooting in the top-level README.
- Do not alter application behavior, dependencies, APIs, commands, configuration semantics, or generated architecture assets.
- Correct an existing instruction only when repository files provide clear evidence for the correction.

### Task 1: Migrate architecture and frontend guidance

**Files:**
- Create: `docs/architecture.md`
- Create: `docs/frontend.md`

**Interfaces:**
- Consumes: the current `README.md` sections `Repository Structure`, `Frontend`, `Intent Routing`, `Agent Framework & Control Flow`, and `Agentic RAG`.
- Produces: stable relative targets `docs/architecture.md` and `docs/frontend.md` for the new README documentation index.

- [ ] **Step 1: Inventory the source headings and boundaries**

Run:

```bash
rg -n '^## (Repository Structure|Frontend|Intent Routing|Agent Framework & Control Flow|Agentic RAG|Examples|Retrieval Setup)$|^### ' README.md
```

Expected: every named source section has one identifiable start line; subordinate headings show the material that belongs with it.

- [ ] **Step 2: Create the architecture guide**

Create `docs/architecture.md` with this order:

```markdown

### Architecture

← Back to README

This guide explains the repository layout, agent families, request routing, and retrieval-grounded agent flow.

### Task 2: Migrate retrieval and API reference material

**Files:**
- Create: `docs/retrieval.md`
- Create: `docs/api-reference.md`

**Interfaces:**
- Consumes: `Retrieval Setup`, `Neural Reranking`, `Retrieval Optimization`, `Query Transformation Optimization`, `Routing & Query Construction`, `Retrieval Server API`, `Web Backend API`, `Chat & Session API`, and `API Health Checks` from `README.md`.
- Produces: `docs/retrieval.md` and `docs/api-reference.md` targets for onboarding and cross-guide links.

- [ ] **Step 1: Create the retrieval guide**

Create `docs/retrieval.md` with this structure:

```markdown

### Task 3: Migrate training, MCP, configuration, and testing guidance

**Files:**
- Create: `docs/training-and-evaluation.md`
- Create: `docs/mcp.md`
- Create: `docs/configuration.md`
- Create: `docs/testing.md`

**Interfaces:**
- Consumes: the advanced portions of `Examples`, plus `Training`, `MCP Server`, `Evaluation`, `Configuration`, `Tests`, and specialized debug notes from `README.md`.
- Produces: four focused references linked from the top-level README.

- [ ] **Step 1: Create the training and evaluation guide**

Create `docs/training-and-evaluation.md` with:

```markdown

### Task 4: Rewrite the top-level README for local onboarding

**Files:**
- Modify: `README.md`

**Interfaces:**
- Consumes: all eight focused guides from Tasks 1–3 and verified repository commands from `pyproject.toml`, `package.json`, and `web/package.json`.
- Produces: the canonical developer onboarding path and documentation index.

- [ ] **Step 1: Replace the monolithic structure with the approved outline**

Rewrite `README.md` in this order:

```markdown

### Task 5: Verify the complete documentation reorganization

**Files:**
- Verify: `README.md`
- Verify: `docs/architecture.md`
- Verify: `docs/retrieval.md`
- Verify: `docs/api-reference.md`
- Verify: `docs/training-and-evaluation.md`
- Verify: `docs/frontend.md`
- Verify: `docs/mcp.md`
- Verify: `docs/configuration.md`
- Verify: `docs/testing.md`

**Interfaces:**
- Consumes: the complete documentation set produced by Tasks 1–4.
- Produces: evidence that links resolve, key guidance remains documented, Markdown is structurally sound, and no application files changed.

- [ ] **Step 1: Verify all expected files exist**

Run:

```bash
for file in README.md docs/architecture.md docs/retrieval.md docs/api-reference.md docs/training-and-evaluation.md docs/frontend.md docs/mcp.md docs/configuration.md docs/testing.md; do test -f "$file" || exit 1; done
```

Expected: exit status 0 with no output.

- [ ] **Step 2: Verify local Markdown links and heading fragments resolve**

Run this read-only Python check. It validates both relative paths and GitHub-style
heading fragments, including the `-1`, `-2`, ... suffixes assigned to duplicate
headings:

```bash
python3 - <<'PY'
import collections
import pathlib
import re
import sys
from urllib.parse import unquote

files = [pathlib.Path("README.md"), *pathlib.Path("docs").glob("*.md")]
link_re = re.compile(r"\[[^]]+\]\(([^)]+)\)")
heading_re = re.compile(r"^ {0,3}#{1,6}\s+(.+?)\s*#*\s*$")

def anchors(path):
    counts = collections.Counter()
    result = set()
    fenced = False
    for line in path.read_text().splitlines():
        if re.match(r"^ {0,3}(```|~~~)", line):

_[Section compacted.]_

## Context Boundary

This pack summarizes its linked sources. Consult those documents for complete details; no implementation status is inferred here.
