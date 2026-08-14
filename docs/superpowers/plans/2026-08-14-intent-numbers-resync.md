# Plan: re-sync the intent routing numbers after #516

Spec: `docs/superpowers/specs/2026-08-14-intent-numbers-resync-design.md`

Documentation only. Each step's verification is a measurement, because the whole
defect is that prose and tables disagreed about measurements.

## 1. Rebuild the index from the committed canonical set

```bash
python -m src.model.intent_index_cli build \
  --canonical data/intent_canonical.json --output data/intent_index
```

→ verify: `built index of 280 examples`, `low support modules (0): none`.

## 2. Re-run the evaluation CLI

```bash
python -m src.model.intent_index_cli evaluate \
  --index data/intent_index \
  --eval-queries data/intent_eval_queries.json \
  --hard-queries data/intent_eval_hard.json \
  --out-of-scope data/intent_out_of_scope.json \
  --canonical data/intent_canonical.json \
  --output data/intent_index/evaluation_report.json
```

→ verify: headline prints `test_slice_accuracy 0.7928`, `hard 0.675`,
`auc 0.8551` (the three the document says are unchanged) and
`leave_one_out 0.7321`, `cohens_d 1.4852` (the two it prints stale).

## 3. Re-measure the MiniLM comparison column

No CLI emits it. Encode the canonical set under each encoder, then call
`separability_report` and `leave_one_out_route_accuracy` directly.

→ verify: MiniLM argmax `0.6216` and AUC `0.8848` reproduce the values already in
the table, which is what makes the two that move — `d 1.6365`, LOO `0.6643` —
trustworthy rather than a setup error.

## 4. Re-run the operating point

```bash
python -m examples.measure_intent_operating_point
```

→ verify: e5 column unchanged (`0.4505`, `0.9600`, 2 wrong); MiniLM moves to
`0.6757` (75/111) at `0.7200`; wrong-route counts still `21` and `7`.

## 5. Break out per-route served counts on the test slice

Not emitted by any report; computed directly from the index over the 111 test
queries at the shipped thresholds.

→ verify: served totals sum to 50 with 48 correct, matching the report's
`test_slice.served` / `served_accuracy`. Any other sum means the thresholds or the
split were applied differently than serving applies them.

## 6. Apply the corrections to `docs/training-and-evaluation.md`

Every edit in the spec's correction table, plus the per-route finding and the
structural fix (intro to the top, back-link to the bottom).

→ verify: `grep` finds no surviving instance of the superseded values except
inside the explicitly-historical MiniLM section and the before→after prose that
records the change.

## 7. Confirm nothing shipped changed

```bash
python -m pytest tests/unit/test_intent_index_eval.py tests/unit/test_intent_canonical_data.py -q
git diff --stat
```

→ verify: 22 passed; the diff touches `docs/` only.
