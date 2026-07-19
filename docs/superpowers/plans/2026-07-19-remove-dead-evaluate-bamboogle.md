# Remove dead evaluate_bamboogle.py — Implementation Plan

**Goal:** Delete the non-functional `examples/evaluate_bamboogle.py`.

## Global Constraints
- Never commit to `main`; branch `chore/remove-dead-evaluate-bamboogle`.
- No change to `src/training/eval/bamboogle.py` or `examples/run_bamboogle_eval.py`.

### Task 1: Delete the dead script

- [ ] **Step 1: Re-verify no importers**

Run: `grep -rn "examples.evaluate_bamboogle\|examples/evaluate_bamboogle\|from examples import evaluate_bamboogle" --include="*.py" --include="*.sh" .`
Expected: only matches inside `examples/evaluate_bamboogle.py`'s own docstring (self-reference). If any real importer appears, STOP.

- [ ] **Step 2: Delete the file**

```bash
git rm examples/evaluate_bamboogle.py
```

- [ ] **Step 3: Verify the bamboogle eval suite still passes**

Run: `python -m pytest tests/unit/test_bamboogle_eval.py -q`
Expected: PASS (it imports `evaluate_bamboogle` from `src.training.eval.bamboogle`, unaffected).

- [ ] **Step 4: Commit**

```bash
git add -A
git commit -m "chore: remove dead examples/evaluate_bamboogle.py (always raised NotImplementedError)"
```
