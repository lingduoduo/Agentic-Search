# README Optimization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Turn the top-level README into a fast local-development guide while preserving deep technical guidance in focused documents under `docs/`.

**Architecture:** `README.md` becomes the canonical onboarding entry point and links to eight focused guides organized by responsibility. Existing content is migrated, consolidated, and lightly edited; application code, commands, configuration semantics, and generated architecture assets remain unchanged.

**Tech Stack:** GitHub-flavored Markdown, relative repository links, shell-based documentation integrity checks, Git.

## Global Constraints

- Optimize the top-level README for a developer encountering the repository for the first time and trying to start the local stack.
- Preserve useful commands, warnings, and behavior descriptions from the existing README.
- Use relative Markdown links that work in repository browsers and local editors.
- Keep essential setup values and first-run troubleshooting in the top-level README.
- Do not alter application behavior, dependencies, APIs, commands, configuration semantics, or generated architecture assets.
- Correct an existing instruction only when repository files provide clear evidence for the correction.

## File map

- Modify `README.md`: concise project overview, local setup path, smoke test, common commands, troubleshooting, and documentation index.
- Create `docs/architecture.md`: repository structure, agent taxonomy and control flow, intent routing, and Agentic RAG.
- Create `docs/retrieval.md`: retrieval servers and modes, reranking, retrieval optimization, query transformations, and query routing.
- Create `docs/api-reference.md`: retrieval, web, chat/session, and health HTTP APIs.
- Create `docs/training-and-evaluation.md`: example workflows, dataset preparation, SFT/GRPO/PPO, and Bamboogle evaluation.
- Create `docs/frontend.md`: frontend workflow, local admin dashboard, dev console, and UI behavior.
- Create `docs/mcp.md`: MCP installation, configuration, tools, resources, authentication, and client setup.
- Create `docs/configuration.md`: environment variables and operational configuration.
- Create `docs/testing.md`: backend, integration, and frontend verification commands.

---

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
# Architecture

[← Back to README](../README.md)

This guide explains the repository layout, agent families, request routing, and retrieval-grounded agent flow.

## Repository structure
## Agent framework and control flow
## Intent routing
## Agentic RAG
```

Move the matching README material beneath those headings. Retain module paths, flow diagrams, code samples, capability notes, and routing behavior. Remove any duplicate feature-summary prose already covered by the guide introduction.

- [ ] **Step 3: Create the frontend guide**

Create `docs/frontend.md` with this order:

```markdown
# Frontend development

[← Back to README](../README.md)

This guide covers the React/Vite development workflow and local administration and observability surfaces.

## Development workflow
## Admin dashboard
## Dev console
## UI features
```

Move the matching README content into these sections. Preserve the `npm install`, `npm run dev`, `npm run build`, `npm run typecheck`, and `npm run test -- --run` commands and the warning that port 5173 is the live Vite server while port 7860 serves the last production build.

- [ ] **Step 4: Validate the guides**

Run:

```bash
rg -n '^# |^## ' docs/architecture.md docs/frontend.md
rg -n '8001|7860|5173|npm run (dev|build|typecheck|test)|SearchAgentLoop|AgenticRAGLoop' docs/architecture.md docs/frontend.md
```

Expected: each file has exactly one level-one heading, ordered level-two headings, and the named commands and concepts remain present.

- [ ] **Step 5: Commit the guides**

```bash
git add docs/architecture.md docs/frontend.md
git commit -m "docs: split architecture and frontend guides"
```

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
# Retrieval

[← Back to README](../README.md)

This guide covers retrieval services, ranking modes, reranking, and query optimization.

## Retrieval setup
## Neural reranking
## Retrieval optimization
## Query transformation optimization
## Routing and query construction
```

Move the complete matching sections, preserving launch commands, request examples, tuning outputs, environment-variable names, warnings, and benchmark commands. Add a single contextual link to `[HTTP API reference](api-reference.md)` where the retrieval endpoint payloads are discussed.

- [ ] **Step 2: Create the API reference**

Create `docs/api-reference.md` with this structure:

```markdown
# HTTP API reference

[← Back to README](../README.md)

This guide documents the local retrieval, web, chat/session, and health endpoints.

## Retrieval server API
## Web backend API
## Chat and session API
## API health checks
```

Move the complete endpoint examples and dispatch notes. Preserve HTTP methods, paths, ports, JSON bodies, response shapes, SSE event examples, authentication notes, and known-gap notes.

- [ ] **Step 3: Validate endpoint and retrieval coverage**

Run:

```bash
rg -n '/retrieve|/api/|/chat|health|text/event-stream|curl ' docs/api-reference.md
rg -n 'demo|hybrid|rerank|RRF|BM25|dense|sparse|graph|query' docs/retrieval.md
```

Expected: the API guide includes retrieval, web, chat/session, SSE, and health references; the retrieval guide includes all major modes and optimization areas.

- [ ] **Step 4: Commit the guides**

```bash
git add docs/retrieval.md docs/api-reference.md
git commit -m "docs: split retrieval and API references"
```

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
# Training and evaluation

[← Back to README](../README.md)

This guide covers dataset preparation, supervised and reinforcement-learning workflows, and benchmark evaluation.

## Runnable examples
## Dataset preparation
## Training
## Evaluation
### Bamboogle
```

Move advanced example, dataset, SFT, GRPO, PPO, reward, local-device, and Bamboogle content. Preserve safety flags, hardware notes, prerequisites, and exact commands.

- [ ] **Step 2: Create the MCP guide**

Create `docs/mcp.md` with:

```markdown
# MCP server

[← Back to README](../README.md)

This guide explains how to install, run, configure, and connect to the Agentic Search MCP server.
```

Use descriptive level-two headings matching the source material. Preserve the `pip install -e ".[mcp]"` instruction, launch commands, tool/resource descriptions, authentication and transport details, and client configuration examples.

- [ ] **Step 3: Create configuration and testing guides**

Create `docs/configuration.md` beginning with `# Configuration` and `docs/testing.md` beginning with `# Testing`. Both must include `[← Back to README](../README.md)` immediately after the title and a one-sentence purpose statement.

Move the full configuration reference into `docs/configuration.md`, grouped by the existing areas and retaining exact variable names and defaults. Move backend unit/regression, opt-in integration, frontend typecheck/build/test, and relevant debugging commands into `docs/testing.md`.

- [ ] **Step 4: Validate command and variable preservation**

Run:

```bash
rg -n 'GRPO|PPO|SFT|Bamboogle|--device|--allow_unsafe_mps' docs/training-and-evaluation.md
rg -n 'pip install -e .*mcp|MCP|transport|auth' docs/mcp.md
rg -n 'GEN_AI_|SEARCH_AGENT_MODEL|SERP_API_KEY|GOOGLE_' docs/configuration.md
rg -n 'pytest|npm run (typecheck|build|test)|integration' docs/testing.md
```

Expected: every search returns the migrated terms and representative commands in its designated guide.

- [ ] **Step 5: Commit the guides**

```bash
git add docs/training-and-evaluation.md docs/mcp.md docs/configuration.md docs/testing.md
git commit -m "docs: split operations and training guides"
```

### Task 4: Rewrite the top-level README for local onboarding

**Files:**
- Modify: `README.md`

**Interfaces:**
- Consumes: all eight focused guides from Tasks 1–3 and verified repository commands from `pyproject.toml`, `package.json`, and `web/package.json`.
- Produces: the canonical developer onboarding path and documentation index.

- [ ] **Step 1: Replace the monolithic structure with the approved outline**

Rewrite `README.md` in this order:

```markdown
# Agentic Search

Agentic Search is a retrieval-backed platform for building multi-turn search, RAG, and tool-using agents. It combines a FastAPI backend, interchangeable retrieval services, a React development UI, and training and evaluation workflows in one repository.

## What it provides
## Architecture
## Prerequisites
## Install
## Configure
## Run locally
### 1. Start retrieval
### 2. Start the API
### 3. Start the frontend
## Verify the stack
## Common development commands
## Troubleshooting
## Documentation
```

Keep the architecture image/link under `Architecture`. Limit `What it provides` to a compact list covering agentic RAG, conversation/tool agents, hybrid retrieval and reranking, connectors/indexing, web search, chat UI/observability, training/evaluation, and MCP.

- [ ] **Step 2: Preserve the canonical setup path**

Under `Prerequisites`, list Python 3.10+, Node.js/npm, an LLM provider key for agent loops, and Java only for BM25/pyserini. Under `Install`, retain editable installation, requirements installation, optional MCP installation, and frontend dependency installation. Under `Configure`, show the minimal `.env` variables and link to `docs/configuration.md`.

Under `Run locally`, retain these exact service commands and ports:

```bash
python3 -m src.internal.servers.retrieval.demo --corpus_path data/corpus.jsonl
PYTHONPATH=src:. uvicorn src.internal.servers.web.app:app --host 127.0.0.1 --port 7860
cd web && npm run dev
```

State that retrieval uses port 8001, the API uses port 7860, and the live frontend uses port 5173.

- [ ] **Step 3: Add verification and troubleshooting**

Add one retrieval `curl` request using the existing `/retrieve` payload and one API health request using the existing health endpoint. Include concise troubleshooting for:

- opening port 5173 for live Vite changes instead of the production bundle on 7860;
- setting `JAVA_HOME` when BM25/pyserini cannot find Java;
- weak local policy models failing to emit `<search>`/`<answer>` tags, with a link to `docs/architecture.md` or the relevant advanced guide;
- required live services and API keys.

- [ ] **Step 4: Add the documentation index**

Add a one-line description and relative link for each guide:

```markdown
- [Architecture](docs/architecture.md)
- [Retrieval](docs/retrieval.md)
- [HTTP API reference](docs/api-reference.md)
- [Training and evaluation](docs/training-and-evaluation.md)
- [Frontend development](docs/frontend.md)
- [MCP server](docs/mcp.md)
- [Configuration](docs/configuration.md)
- [Testing](docs/testing.md)
```

- [ ] **Step 5: Check readability and size**

Run:

```bash
wc -l README.md
rg -n '^# |^## |^### ' README.md
rg -n '^## Features$' README.md
```

Expected: the README is materially shorter than 1,718 lines, has one level-one heading, follows the approved heading order, and has no duplicated `Features` heading (the final command exits with no matches).

- [ ] **Step 6: Commit the onboarding README**

```bash
git add README.md
git commit -m "docs: streamline local development README"
```

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
            fenced = not fenced
            continue
        match = None if fenced else heading_re.match(line)
        if not match:
            continue
        text = re.sub(r"<[^>]+>", "", match.group(1)).lower()
        slug = re.sub(r"[^\w\- ]", "", text, flags=re.UNICODE)
        slug = re.sub(r"\s", "-", slug)
        number = counts[slug]
        counts[slug] += 1
        result.add(slug if number == 0 else f"{slug}-{number}")
    return result

bad = []
for source in files:
    for target in link_re.findall(source.read_text()):
        if not target or target.startswith(("http://", "https://", "mailto:")):
            continue
        path_text, separator, fragment = target.partition("#")
        destination = source if not path_text else source.parent / unquote(path_text)
        if not destination.exists():
            bad.append((source, target, "missing path"))
        elif separator and unquote(fragment) not in anchors(destination):
            bad.append((source, target, "missing fragment"))

for source, target, reason in bad:
    print(f"{source}: {target} ({reason})")
sys.exit(bool(bad))
PY
```

Expected: exit status 0 with no output.

- [ ] **Step 3: Verify representative content coverage**

Run:

```bash
rg -l 'AgenticRAGLoop|SearchAgentLoop' README.md docs/*.md
rg -l '/retrieve|text/event-stream' README.md docs/*.md
rg -l 'GRPO|Bamboogle' README.md docs/*.md
rg -l 'MCP' README.md docs/*.md
rg -l 'pytest|npm run test' README.md docs/*.md
```

Expected: each command lists at least one focused guide, and onboarding-relevant concepts may additionally list `README.md`.

- [ ] **Step 4: Check Markdown and diff hygiene**

Run:

```bash
git diff --check HEAD~4..HEAD
git diff --name-only HEAD~4..HEAD
git status --short
```

Expected: `git diff --check` has no output; changed paths are limited to `README.md` and Markdown files under `docs/`; working tree is clean.

- [ ] **Step 5: Perform the final review**

Read the rendered source order in `README.md` and confirm a new developer encounters prerequisites, install, configuration, startup, and verification before advanced references. Compare the original README from `git show 6f96fee^:README.md` against the focused guide headings to confirm no substantive section was silently omitted.

- [ ] **Step 6: Record any verification-only correction**

If verification required a documentation correction, commit only the affected Markdown files:

```bash
git add README.md docs/*.md
git commit -m "docs: fix documentation navigation"
```

If no correction was needed, do not create an empty commit.
