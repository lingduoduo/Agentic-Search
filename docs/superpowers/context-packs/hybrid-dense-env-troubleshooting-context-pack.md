# Generated Context Pack

# Hybrid Dense Env Troubleshooting

## Sources

- [Specification: 2026-06-21-hybrid-dense-env-troubleshooting-design.md](../specs/2026-06-21-hybrid-dense-env-troubleshooting-design.md)
- [Plan: 2026-06-21-hybrid-dense-env-troubleshooting.md](../plans/2026-06-21-hybrid-dense-env-troubleshooting.md)

## Specification Context

### Goal

Help users get the dense leg working — document the clean-environment setup and the failure
symptoms, and make the dense-init failure message actionable. No requirements version
changes (a clean install already works).

### Scope

- New: `docs/hybrid-dense-setup.md`
- Modify: `src/internal/servers/retrieval/hybrid.py` (`_build_dense` warning message)
- Modify: `.claude/CLAUDE.md` (link the troubleshooting doc from the hybrid command)
- Test: `tests/unit/servers/retrieval/test_hybrid_retrieval.py` (dense-init failure logs a hint, returns None)

Out of scope: changing `requirements.txt` versions; fixing the user's base env (a PR can't);
any change to the fusion/contract behavior.

### Testing

- `_build_dense` with `build_e5_encoder` monkeypatched to raise returns `None` and logs a
  warning that mentions the setup doc (via `caplog`).

## Implementation Plan Context

### Risk / rollback

- Docs + a log-string change + one test; no behavior change to retrieval. Rollback = revert
  the branch.

## Context Boundary

This pack summarizes its linked sources. Consult those documents for complete details; no implementation status is inferred here.
