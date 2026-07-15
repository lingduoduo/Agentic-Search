# README Optimization Design

## Goal

Make the top-level README the fastest path for a developer to install and run Agentic Search locally. Preserve the existing deep technical material by moving it into focused documents under `docs/`.

## Audience and success criteria

The primary audience is a developer encountering the repository for the first time and trying to start the local stack.

The optimized documentation should:

- explain the project and its main capabilities without an exhaustive feature catalog;
- surface prerequisites and required configuration before startup commands;
- provide one canonical local-development path for the retrieval service, API, and frontend;
- include a short smoke test and common troubleshooting guidance;
- make advanced architecture, API, retrieval, training, frontend, MCP, configuration, and testing material easy to discover;
- preserve useful commands and technical guidance from the existing README;
- avoid duplicated sections and keep heading hierarchy consistent.

## Chosen approach

Use a task-oriented top-level README with focused reference documents.

This approach favors onboarding speed while retaining depth. It avoids both extremes: a minimal landing page that forces immediate navigation and a monolithic README that remains difficult to scan and maintain.

## Top-level README structure

The revised `README.md` will contain:

1. Project name and concise value proposition.
2. A compact list of core capabilities.
3. The existing architecture image and link to its interactive form.
4. Prerequisites.
5. Installation and environment configuration.
6. A canonical local quick start covering:
   - retrieval service on port 8001;
   - FastAPI web API on port 7860;
   - Vite frontend on port 5173.
7. A smoke test that confirms the stack is reachable.
8. Common developer commands.
9. High-value troubleshooting notes, including the live Vite port and local-model capability limitations.
10. A documentation index linking to every focused guide.

The README will not duplicate detailed reference content that belongs in the focused documents.

## Focused documents

The existing README content will be reorganized into:

- `docs/architecture.md`: repository map, agent families, control flow, intent routing, and Agentic RAG.
- `docs/retrieval.md`: retrieval setup and modes, neural reranking, optimization, query transformation, and query routing.
- `docs/api-reference.md`: retrieval, web backend, chat/session, and health endpoints.
- `docs/training-and-evaluation.md`: dataset preparation, SFT, GRPO, PPO, Bamboogle, and evaluation workflows.
- `docs/frontend.md`: frontend development, admin dashboard, dev console, and UI behavior.
- `docs/mcp.md`: MCP installation, configuration, tools, resources, and client setup.
- `docs/configuration.md`: environment variables and operational settings.
- `docs/testing.md`: backend unit and regression tests, opt-in integration tests, and frontend checks.

Each guide will start with its purpose and link back to the top-level README. Related guides may link to one another where that helps navigation.

## Content migration rules

- Move content without silently dropping commands, warnings, or behavior descriptions that remain accurate.
- Consolidate repeated material into a single canonical location.
- Keep essential setup values in the README even when the full configuration reference lives elsewhere.
- Keep essential troubleshooting in the README when it directly blocks a first local run; move specialized diagnostics to the relevant guide.
- Preserve code examples in executable form and retain their required context.
- Use relative Markdown links so documentation works in repository browsers and local editors.

## Validation

Because this is a documentation-only change, validation will focus on documentation integrity:

- confirm the README has one coherent heading hierarchy and no duplicate major sections;
- verify all relative documentation links resolve to existing files;
- compare old and new headings to ensure substantive sections were migrated;
- search for key startup, testing, training, retrieval, and MCP commands to confirm they remain documented;
- run a Markdown link checker if the repository already provides one; otherwise use a local script or shell checks for relative links;
- inspect the final diff for accidental code or configuration changes.

## Scope boundaries

This change reorganizes and edits documentation only. It does not alter application behavior, dependencies, APIs, commands, configuration semantics, or generated architecture assets. Any incorrect existing instruction discovered during migration may be corrected only when the repository provides clear evidence for the correction.
