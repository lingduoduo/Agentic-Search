# Generated Context Pack

# Dev Admin Bypass

## Sources

- [Specification: 2026-07-11-dev-admin-bypass-design.md](../specs/2026-07-11-dev-admin-bypass-design.md)
- [Plan: 2026-07-11-dev-admin-bypass.md](../plans/2026-07-11-dev-admin-bypass.md)

## Specification Context

### Non-goals (YAGNI)

- No frontend change, no `/auth/dev-login` endpoint, no cookie minting.
- No host/secret interlock (refuse-if-prod). The explicit default-off flag +
  startup warning is the agreed guardrail.
- Does not change the real JWT/super-user path in any way.

## Implementation Plan Context

### Overview

Spec: 2026-07-11-dev-admin-bypass-design.md

## Context Boundary

This pack summarizes its linked sources. Consult those documents for complete details; no implementation status is inferred here.
