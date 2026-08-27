# GRPO Optimization Benchmarks

Evidence for the reward and online-training optimizations in
`docs/superpowers/specs/2026-08-27-grpo-minimal-architecture-design.md`.

Regenerate a section with identical flags before and after a change:

```bash
python -m examples.benchmark_grpo_optimization \
  --warmup 5 --iterations 25 \
  --output docs/benchmarks/grpo-optimization-baseline.md \
  --heading "<section name>"
```

Numbers are medians over repeated samples on one machine. They are comparable
**within** this file (same host, same fixtures, same flags) and meaningless
across machines. `peak alloc` is the Python-level peak from `tracemalloc`, not
RSS, so it tracks list/dict/tensor-object churn rather than the allocator.

Timing and allocation are measured in **separate** passes. An earlier version of
this harness timed the samples with `tracemalloc` running, which inflated every
median roughly three-fold and made the reward paths look far more expensive than
they are. Those first numbers were discarded rather than reported.

**Read the verdict tables, not the raw ones.** The three `## ` sections below
are single sequential harness runs, recorded so the raw numbers exist. They
drift: the baseline section was captured minutes before the others, and cases
this branch never touched (`token_f1`, `policy_update_loss`) move by 3-14%
between them purely from machine load. Comparing two of those tables column-wise
will overstate every change.

Every **claim** in this file instead comes from a `###` verdict table, each of
which is the median of three independent runs of the same case measured
back-to-back with only the code under test swapped. That discipline is not
optional here: one pair of single runs showed `response_log_probs` "regressing"
29% on code that was never touched, and three runs put it back at parity.

## Baseline (commit 7265517)

- Python 3.12.2, torch 2.11.0
- Platform: macOS-26.6.2-arm64-arm-64bit (cpu)
- Warmup 5, iterations 25
- Fixtures: num_groups=16, rollouts_per_group=8, num_rollouts=128, prompt_len=32, response_len=96, retrieved_docs=24, seed=20260827

| case | median (µs) | min (µs) | max (µs) | peak alloc (KiB) |
| --- | ---: | ---: | ---: | ---: |
| `group_advantages` | 19.9 | 19.0 | 122.5 | 3.4 |
| `group_advantages_normalized` | 33.1 | 32.2 | 57.9 | 3.5 |
| `group_advantages_tensor` | 465.4 | 413.7 | 988.2 | 1.0 |
| `token_f1` | 957.1 | 934.2 | 1709.8 | 4.4 |
| `reward_components_shaped` | 6509.8 | 5908.0 | 11353.8 | 193.2 |
| `reward_components_sparse` | 7176.8 | 5889.4 | 8915.3 | 193.1 |
| `reward_batch` | 1087.4 | 1030.2 | 1862.5 | 106.7 |
| `reward_token_advantages` | 1137.7 | 1064.2 | 1420.8 | 111.9 |
| `score_prompt_group` | 483.9 | 404.0 | 1367.2 | 12.5 |
| `response_log_probs` | 114.1 | 88.5 | 293.0 | 3.5 |
| `left_pad_prompt_rows` | 216.5 | 212.7 | 449.7 | 36.1 |
| `log_prob_row_collection` | 751.2 | 711.0 | 1178.4 | 14.9 |
| `policy_update_loss` | 386.1 | 331.8 | 1738.7 | 6.4 |

## After reward optimization

- Python 3.12.2, torch 2.11.0
- Platform: macOS-26.6.2-arm64-arm-64bit (cpu)
- Warmup 5, iterations 25
- Fixtures: num_groups=16, rollouts_per_group=8, num_rollouts=128, prompt_len=32, response_len=96, retrieved_docs=24, seed=20260827

| case | median (µs) | min (µs) | max (µs) | peak alloc (KiB) |
| --- | ---: | ---: | ---: | ---: |
| `group_advantages` | 13.9 | 13.4 | 22.1 | 3.4 |
| `group_advantages_normalized` | 25.2 | 25.0 | 25.8 | 3.6 |
| `group_advantages_tensor` | 391.4 | 384.8 | 640.1 | 1.0 |
| `token_f1` | 902.9 | 890.1 | 929.9 | 4.4 |
| `reward_components_shaped` | 5621.4 | 5535.3 | 7337.6 | 194.6 |
| `reward_components_sparse` | 1412.5 | 1372.6 | 2022.3 | 155.3 |
| `reward_batch` | 1010.3 | 993.5 | 1047.3 | 106.7 |
| `reward_token_advantages` | 1024.2 | 1006.1 | 1062.0 | 111.9 |
| `score_prompt_group` | 370.3 | 365.8 | 870.4 | 13.7 |
| `response_log_probs` | 91.9 | 83.8 | 125.0 | 3.5 |
| `left_pad_prompt_rows` | 205.0 | 204.3 | 215.0 | 36.1 |
| `log_prob_row_collection` | 687.9 | 679.4 | 717.6 | 14.9 |
| `policy_update_loss` | 345.9 | 251.0 | 696.4 | 6.4 |
### What changed, and what the numbers say

| case | change | verdict |
| --- | ---: | --- |
| `group_advantages` | **−29%** | One grouping pass instead of building `(index, reward)` tuple lists and re-zipping them. |
| `group_advantages_normalized` | **−17%** | Same kernel; the duplicate second implementation is gone. |
| `reward_components_sparse` | **−75%** | A fully zeroed preset no longer resolves citations or evaluates shaping arithmetic. |
| `reward_components_shaped` | ±1% | Flat. Citations are resolved once instead of twice, but the shaped path's cost is dominated by terms that still have to run. |
| everything else | ±3% | Noise. No case regressed repeatably. |

**The advantage kernel must keep using `sum()`.** An earlier version of it
replaced the builtin with a hand-rolled `total += ...` accumulator, which
benchmarked better — **−45% / −39%** instead of −29% / −17%. It was reverted,
because on CPython 3.12+ `sum()` applies Neumaier compensation over floats and a
manual loop does not: against the pre-refactor code, **8,745 / 20,000** random
groups differed on centering and **10,634 / 20,000** on the normalized path
(max relative error ~1e-12). The divergence is also Python-version dependent —
3.11 agrees, 3.12 does not — so "identical" would not even have been a stable
property. The extra 16 points of speed were not the kernel's to sell.

The remaining win is real and comes from dropping the `(index, reward)` tuple
lists, not from the summation: a manual mean loop is itself *slower* than
`sum()` on an 8-element group (122 ns vs 74 ns).

`sparse_final_only` is the preset the docs recommend for a first agent-RL
training phase, so the 4× speedup lands on the configuration most likely to be
running.

The shaped path being flat is the honest result: sharing the citation
resolution removes a duplicated traversal without moving the median, because
`token_f1_score` and the surviving shaping terms dominate it. It is kept for the
duplication it removes, not for a speedup it does not deliver.

An intermediate version of the zero-weight guards used
`all(getattr(cfg, name) == 0.0 for name in WEIGHTS)`, which cost the shaped path
a repeatable **+1.9%** — the guard ran on every rollout that could never take
the fast path. Rewriting it as a short-circuiting `and` chain ordered by the
default config's first non-zero weight returned the shaped path to parity.

## After online-path optimization

- Python 3.12.2, torch 2.11.0
- Platform: macOS-26.6.2-arm64-arm-64bit (cpu)
- Warmup 5, iterations 25
- Fixtures: num_groups=16, rollouts_per_group=8, num_rollouts=128, prompt_len=32, response_len=96, retrieved_docs=24, seed=20260827

| case | median (µs) | min (µs) | max (µs) | peak alloc (KiB) |
| --- | ---: | ---: | ---: | ---: |
| `group_advantages` | 13.8 | 13.5 | 14.7 | 3.4 |
| `group_advantages_normalized` | 25.9 | 25.5 | 28.5 | 3.6 |
| `group_advantages_tensor` | 400.4 | 375.0 | 1116.4 | 1.0 |
| `token_f1` | 932.8 | 907.0 | 1696.5 | 4.4 |
| `reward_components_shaped` | 5649.3 | 5573.7 | 6083.3 | 195.0 |
| `reward_components_sparse` | 1426.5 | 1384.3 | 1636.8 | 155.3 |
| `reward_batch` | 1032.0 | 1008.0 | 1093.5 | 106.7 |
| `reward_token_advantages` | 1054.5 | 1026.2 | 1083.8 | 111.9 |
| `score_prompt_group` | 369.1 | 365.5 | 386.6 | 14.1 |
| `response_log_probs` | 94.0 | 81.2 | 133.8 | 3.5 |
| `left_pad_prompt_rows` | 189.3 | 185.2 | 193.3 | 0.4 |
| `log_prob_row_collection` | 148.8 | 143.0 | 155.2 | 10.1 |
| `policy_update_loss` | 333.6 | 291.0 | 497.2 | 6.4 |
### Online-path results

Measured against the same fixtures with only the online-path changes reverted,
median of three runs each:

| case | time | peak alloc | verdict |
| --- | ---: | ---: | --- |
| `left_pad_prompt_rows` | **−13.5%** | **−99%** (36.1 → 0.4 KiB) | Kept. Left-padding writes into a preallocated tensor instead of building a nested Python list. |
| `log_prob_row_collection` | **−81%** (694.6 → 132.4 µs) | — | Kept. Rows already stored as tensors are sliced out instead of rebuilt from `.tolist()`. |
| `policy_update_loss` | +3.0% | −2.3% | See below. |

The two cases are measured separately on purpose. An earlier version of this
file reported a single `training_batch_assembly` row that only ever exercised
the padding helper, which made the table read as if the whole collate path had
been measured while the log-prob change carried no evidence at all.

**`torch.inference_mode()` for the frozen reference was measured and rejected.**
The plan proposed it as a candidate optimization. It is correct — every
supported KL type still backpropagates through the policy — but it is not
faster. On a toy model it wins 18%; at realistic sizes it loses, repeatably:

| hidden | vocab | seq | batch | `no_grad` | `inference_mode` | change |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 64 | 512 | 64 | 8 | 250.6 µs | 205.4 µs | −18.0% |
| 256 | 4096 | 128 | 8 | 6250.0 µs | 6713.4 µs | **+7.4%** |
| 512 | 8192 | 256 | 16 | 81680.0 µs | 91501.8 µs | **+12.0%** |

Training runs at the bottom two rows, not the top one, so the reference stays
under `torch.no_grad()`. The contract test asserts the reference runs with grad
*disabled* — the invariant that actually matters — rather than pinning the
specific mechanism.

Reference forward count is unchanged at exactly one per optimization step; the
`policy_update_loss` case exists to keep that measurable.

**Left-padding via `pad_sequence` was also measured and rejected**: reversing
each row, right-padding, and reversing back ran 18% *slower* than the nested
list it replaced, and 25% slower than the preallocated copy that shipped.
