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

### Task 1: Migrate architecture and frontend guidance

**Files:**
- Create: `docs/architecture.md`
- Create: `docs/frontend.md`

**Interfaces:**
- Consumes: the current `README.md` sections `Repository Structure`, `Frontend`, `Intent Routing`, `Agent Framework & Control Flow`, and `Agentic RAG`.
- Produces: stable relative targets `docs/architecture.md` and `docs/frontend.md` for the new README documentation index.

- [ ] **Step 1: Inventory the source headings and boundaries**

Run:

Expected: every named source section has one identifiable start line; subordinate headings show the material that belongs with it.

- [ ] **Step 2: Create the architecture guide**

Create `docs/architecture.md` with this order:

…

### Task 2: Migrate retrieval and API reference material

**Files:**
- Create: `docs/retrieval.md`
- Create: `docs/api-reference.md`

**Interfaces:**
- Consumes: `Retrieval Setup`, `Neural Reranking`, `Retrieval Optimization`, `Query Transformation Optimization`, `Routing & Query Construction`, `Retrieval Server API`, `Web Backend API`, `Chat & Session API`, and `API Health Checks` from `README.md`.
- Produces: `docs/retrieval.md` and `docs/api-reference.md` targets for onboarding and cross-guide links.

- [ ] **Step 1: Create the retrieval guide**

Create `docs/retrieval.md` with this structure:

…

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

Move advanced example, dataset, SFT, GRPO, PPO, reward, local-device, and Bamboogle content. Preserve safety flags, hardware notes, prerequisites, and exact commands.

…

### Task 4: Rewrite the top-level README for local onboarding

**Files:**
- Modify: `README.md`

**Interfaces:**
- Consumes: all eight focused guides from Tasks 1–3 and verified repository commands from `pyproject.toml`, `package.json`, and `web/package.json`.
- Produces: the canonical developer onboarding path and documentation index.

- [ ] **Step 1: Replace the monolithic structure with the approved outline**

Rewrite `README.md` in this order:

Keep the architecture image/link under `Architecture`. Limit `What it provides` to a compact list covering agentic RAG, conversation/tool agents, hybrid retrieval and reranking, connectors/indexing, web search, chat UI/observability, training/evaluation, and MCP.

…

## Context Boundary

This pack summarizes its linked sources. Consult those documents for complete details; no implementation status is inferred here.
