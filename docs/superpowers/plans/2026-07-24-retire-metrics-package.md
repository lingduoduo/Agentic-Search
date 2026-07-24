# Retire the metrics Package Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans (or execute inline). Steps use checkbox (`- [ ]`) syntax.

**Goal:** Delete the dead `src/internal/metrics/` package (2,391 LOC) and drop the now-unused `prometheus-client` dependency, zero live-behavior change.

**Architecture:** Whole-package deletion + two requirements edits. Verified: zero external importers, no entrypoint, no tests, `prometheus-client` used only by this package, no `src/__init__` re-export. Spec: `docs/superpowers/specs/2026-07-24-retire-metrics-package-design.md`.

**Tech Stack:** Python, pytest, ruff.

## Global Constraints

- Branch: `chore/retire-metrics-package` (off `chore/retire-nlp-package`). PR opens against `main` after #468 merges (rebase onto main). Never commit to `main`.
- Zero live-behavior change (the package is unwired dead code).
- `ruff check .` + `pytest` green after deletion.

---

### Task 1: Delete the metrics package + drop the prometheus dep

**Files:**
- Delete: `src/internal/metrics/` (whole dir)
- Modify: `requirements.txt`, `requirements-unit-test.txt`

- [ ] **Step 1: Re-confirm zero importers**

Run: `grep -rn "internal.metrics" src/ tests/ examples/ docker/ --include="*.py" --include="*.yml" | grep -v "src/internal/metrics/"`
Expected: no output. If anything references it, STOP.

- [ ] **Step 2: Delete the package**

```bash
git rm -r src/internal/metrics
```

- [ ] **Step 3: Drop `prometheus-client` from both requirements files**

Remove the `prometheus-client>=0.20.0` line from `requirements.txt` and `requirements-unit-test.txt`.

- [ ] **Step 4: Verify**

Run: `python -c "import src" && grep -rn "prometheus_client\|prometheus_fastapi_instrumentator" src/ --include="*.py" && echo "STRAY PROM IMPORT" || echo "no prom imports"`
Expected: import OK; no stray prometheus import (grep empty → the `&&` short-circuits to the `||` echo "no prom imports").
Run: `grep -rniE "prometheus" requirements.txt requirements-unit-test.txt || echo "prometheus dropped"`
Expected: `prometheus dropped`.
Run: `ruff check . && ruff format --check . && pytest -q`
Expected: all green.

- [ ] **Step 5: Commit**

```bash
git add -A
git commit -m "chore: delete the dead metrics package (~2,391 LOC) + drop prometheus-client

src/internal/metrics/ was unwired onyx observability heritage: zero external
importers, no entrypoint, no tests; the web app does no Prometheus instrumentation.
Its campaign-orphaned metrics monitored machinery already deleted (#461-464/#468);
the generic HTTP-metrics scaffolding was never wired in. prometheus-client had no
other user.

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 2: Gate + PR (after #468 merges)

- [ ] **Step 1: Once #468 is merged, rebase onto main**

```bash
git fetch origin && git rebase --onto origin/main chore/retire-nlp-package chore/retire-metrics-package
```
(Rebases only this branch's own commits — spec, plan, deletion — onto fresh main; the NLP-branch commits are excluded.)

- [ ] **Step 2: Full gate on the rebased branch**

Run: `python -c "import src" && ruff check . && ruff format --check . && pytest -q`
Expected: all green.

- [ ] **Step 3: Confirm diff shape**

Run: `git diff --stat main...HEAD`
Expected: `src/internal/metrics/` deleted; `requirements.txt` + `requirements-unit-test.txt` trimmed; spec + plan added. No other file touched.

- [ ] **Step 4: Push + open PR**

```bash
git push -u origin chore/retire-metrics-package
gh pr create --base main --title "chore: retire the dead metrics package" --body "$(cat <<'EOF'
Deletes the entire `src/internal/metrics/` package (~2,391 LOC, 17 modules) and drops the now-unused `prometheus-client` dependency.

The package is dead, unwired Onyx observability heritage: **zero external importers** anywhere (src/tests/examples/docker), no `python -m` entrypoint, no startup wiring, no tests, and the web app performs no Prometheus instrumentation. Two dead groups — campaign-orphaned metrics that monitored machinery removed in #461–#464/#468 (celery/indexing/connector/pruning/perm_sync/deletion/embedding/image), and generic HTTP-metrics scaffolding (metrics_server/prometheus_setup/metrics_auth/per_tenant/slow_requests/postgres_pool) that was never wired into the app.

`prometheus-client` had no user outside this package, so it's dropped from both requirements files. No `src/__init__.py` re-export trap. Full suite green; ruff+format clean; `import src` works.

Follow-up (separate, from #468): the `document_index/utils.setup_logger` generic-helper relocation.

Spec: `docs/superpowers/specs/2026-07-24-retire-metrics-package-design.md`
Plan: `docs/superpowers/plans/2026-07-24-retire-metrics-package.md`

🤖 Generated with [Claude Code](https://claude.com/claude-code)
EOF
)"
```

---

## Self-Review

**Spec coverage:** delete package + drop dep (T1); rebase-after-#468 + gate + PR (T2). Every spec criterion maps: zero-importer re-confirm (T1 S1 + T2 S3), no stray prometheus import / dep dropped (T1 S4), ruff+pytest green (T1 S4 / T2 S2), no re-export trap (verified in spec, unchanged).

**Placeholder scan:** no vague steps — exact `git rm`, exact requirement lines, exact grep/gate commands and the rebase invocation.

**Type consistency:** deletion-only + manifest edits; no symbols introduced.
