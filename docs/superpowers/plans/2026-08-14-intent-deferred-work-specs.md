# Plan: capture the intent arc's deferred work as specs, then delete the branches

Specs written by this plan:

- `docs/superpowers/specs/2026-08-14-intent-min-module-score-design.md`
- `docs/superpowers/specs/2026-08-14-intent-top-k-selection-design.md`
- `docs/superpowers/specs/2026-08-14-intent-routing-telemetry-design.md`
- `docs/superpowers/specs/2026-08-14-intent-canonical-coverage-design.md`

## Why

The intent arc ran across eight branches (#509–#518) and two worktrees. All of it
merged, so the branches hold no unlanded code — but they were the working context
for four items the PRs explicitly deferred. Deleting them without writing the
deferred items down loses the reasoning, not the code.

## 1. Establish that no branch holds unlanded work

```bash
gh pr list --head "$b" --state all --json number,state   # per branch
```

→ verify: every local branch maps to a **MERGED** PR. `ahead`/`--merged` are both
unreliable here because this repo squash-merges: the branch SHA never becomes an
ancestor of `main`. Confirm by PR state, and spot-check content for any branch
without a PR.

→ result: 16 branches, 15 with MERGED PRs; `intent-next` has no PR and a zero
diff against `main` (a stale pointer at #516).

## 2. Verify each deferred item against the code before writing it up

Specs asserting things that are no longer true are worse than no specs.

- `min_module_score`: score every module for every test-slice query and compare
  against the `0.45` threshold.
  → verify: **0 of 1554** scores fall below it; range `0.7428`–`0.8943`. Dead, confirmed.
- Telemetry: read `intent_routing.py` around the `telemetry.update(...)` call.
  → verify: `modules`/`composite` go to `record_stage` only; `telemetry.update`
  carries five `route_*` fields and neither of them.
- `top_k`: read the sweep from the freshly built `evaluation_report.json`.
  → verify: tuning accuracy plateaus at `0.8714` from `k=8`; leave-one-out climbs
  monotonically to `0.8464` at `k=25`; separation falls at every step.
- Canonical coverage: the 47% concentration and 13-of-16 abstention figures are
  #511's, carried forward as cited history rather than re-measured here.

## 3. Write the four specs

Each states status, the measured evidence, the approach, and acceptance criteria.
Acceptance criteria name the numbers that must not move — for the module and
telemetry specs, route accuracy `0.7928` is the invariant, because neither change
may touch routing.

## 4. Delete the branches and worktrees

Worktrees first (both already verified clean and merged, with the one unique
untracked planning record preserved), then the local branches.

`git branch -d` will refuse every one of them — squash-merge again — so `-D` is
required. That is safe **only because** step 1 established merged status by PR
state rather than by ancestry.

→ verify: `git branch` lists `main` and this branch only; `git worktree list`
lists the primary checkout only.

## 5. Ship

→ verify: `git status` clean apart from the untracked `.planning/`; PR opened;
`Intent Routing Gate` green.

## Not covered

One further open thread was identified and deliberately **not** specced: the
per-caller anonymous identity fix that #500's revert left unbuilt. Its current
behavior could not be established cheaply enough to write acceptance criteria
against, and a spec built on an unverified premise is exactly what this plan's
step 2 exists to prevent. It needs its own investigation first.
