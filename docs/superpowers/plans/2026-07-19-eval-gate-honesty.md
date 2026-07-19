# Eval-Gate Honesty — Implementation Plan

**Goal:** Make the two eval CI gates visibly report when they are inactive, instead of showing a misleading green.

## Global Constraints
- Never commit to `main`; branch `fix/eval-gate-honesty`.
- No fabricated baselines, no hard-fail on missing/placeholder baseline, no eval-code changes.

### Task 1: Honest gates + note + docs

- [ ] **Step 1: retrieval-eval-gate — flag the zero-placeholder baseline**

In `.github/workflows/eval-gate.yml`, in the retrieval job's "Check for regressions" inline Python, after loading `baseline`, detect the placeholder (all gated metrics zero) and emit a warning + step summary, before the comparison loop:

```python
gated = ["recall@10", "ndcg@10"]
if all((baseline.get(k) or 0) == 0 for k in gated):
    msg = "Retrieval eval gate INACTIVE: baseline is a zero placeholder; not enforcing regressions."
    print(f"::warning title=Eval gate inactive::{msg}")
    import os
    summary = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary:
        open(summary, "a").write(f"⚠️ {msg}\n")
```

Keep the existing comparison loop unchanged (it still enforces once a real baseline lands).

- [ ] **Step 2: ragas-eval-gate — make the missing-baseline skip visible**

Add an always-running step (no `if:` guard) right after the "Check baseline exists" step:

```yaml
      - name: Report gate status
        run: |
          if [ "${{ steps.check.outputs.exists }}" != "true" ]; then
            msg="RAGAS eval gate INACTIVE: no data/eval/ragas_baseline.json committed (see docs/training-and-evaluation.md)."
            echo "::warning title=Eval gate inactive::$msg"
            echo "⚠️ $msg" >> "$GITHUB_STEP_SUMMARY"
          fi
```

Leave the existing eval/compare steps gated on `exists == 'true'` as-is.

- [ ] **Step 3: baseline note**

In `data/eval/baseline_metrics.json`, change the `_note` value to:
`"Non-enforcing placeholder (all zeros). The retrieval eval gate stays inactive until a real baseline is committed — see docs/training-and-evaluation.md."`

- [ ] **Step 4: docs — activation instructions**

In `docs/training-and-evaluation.md`, add a short "Activating the eval gates" subsection: the gates are informational placeholders; to enforce, run the eval harness against the canonical stack and commit the resulting metrics as `data/eval/baseline_metrics.json` (retrieval) and `data/eval/ragas_baseline.json` (RAGAS). Show the two commands:

```bash
python -m src.internal.retrieval.eval_runner --dataset data/eval/qa_pairs.jsonl --top_k 10 --output data/eval/baseline_metrics.json
python -m src.internal.retrieval.ragas_eval --dataset data/eval/ragas_qa.jsonl --metrics faithfulness answer_relevancy --output data/eval/ragas_baseline.json
```

- [ ] **Step 5: Validate YAML + placeholder logic**

Run:
```bash
python -c "import yaml; yaml.safe_load(open('.github/workflows/eval-gate.yml')); print('yaml ok')"
python -c "import json; b=json.load(open('data/eval/baseline_metrics.json')); print('placeholder' if all((b.get(k) or 0)==0 for k in ['recall@10','ndcg@10']) else 'real')"
```
Expected: `yaml ok`, then `placeholder`.

- [ ] **Step 6: Commit**

```bash
git add .github/workflows/eval-gate.yml data/eval/baseline_metrics.json docs/training-and-evaluation.md
git commit -m "ci: make eval gates report when inactive instead of silent green"
```
