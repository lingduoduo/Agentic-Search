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

Each figure below is the median of three independent runs of the harness, so a
one-off scheduling artifact cannot be mistaken for a change: a single pair of
runs showed `response_log_probs` "regressing" 29% on code that was never
touched, and three runs put it back at parity.

## Baseline (commit 7265517)

- Python 3.12.2, torch 2.11.0
- Platform: macOS-26.6.2-arm64-arm-64bit (cpu)
- Warmup 5, iterations 25
- Fixtures: num_groups=16, rollouts_per_group=8, num_rollouts=128, prompt_len=32, response_len=96, retrieved_docs=24, seed=20260827

| case | median (µs) | min (µs) | max (µs) | peak alloc (KiB) |
| --- | ---: | ---: | ---: | ---: |
| `group_advantages` | 18.2 | 17.5 | 28.0 | 3.4 |
| `group_advantages_normalized` | 29.8 | 28.9 | 32.7 | 3.5 |
| `group_advantages_tensor` | 388.0 | 370.1 | 421.4 | 1.0 |
| `token_f1` | 901.6 | 889.3 | 1012.4 | 4.4 |
| `reward_components_shaped` | 5624.5 | 5606.1 | 5764.5 | 193.5 |
| `reward_components_sparse` | 5693.6 | 5583.0 | 5795.4 | 193.4 |
| `reward_batch` | 975.7 | 964.9 | 1000.5 | 106.7 |
| `reward_token_advantages` | 1011.0 | 997.9 | 1034.5 | 111.9 |
| `score_prompt_group` | 366.1 | 362.6 | 374.7 | 12.6 |
| `response_log_probs` | 70.2 | 65.5 | 114.3 | 3.6 |
| `training_batch_assembly` | 482.0 | 477.3 | 486.3 | 50.4 |

## After reward optimization

- Python 3.12.2, torch 2.11.0
- Platform: macOS-26.6.2-arm64-arm-64bit (cpu)
- Warmup 5, iterations 25
- Fixtures: num_groups=16, rollouts_per_group=8, num_rollouts=128, prompt_len=32, response_len=96, retrieved_docs=24, seed=20260827

| case | median (µs) | min (µs) | max (µs) | peak alloc (KiB) |
| --- | ---: | ---: | ---: | ---: |
| `group_advantages` | 10.8 | 10.6 | 11.4 | 3.2 |
| `group_advantages_normalized` | 18.9 | 18.8 | 19.3 | 3.3 |
| `group_advantages_tensor` | 386.0 | 378.5 | 401.5 | 1.0 |
| `token_f1` | 883.4 | 881.2 | 899.7 | 4.4 |
| `reward_components_shaped` | 5509.3 | 5468.6 | 8185.8 | 194.7 |
| `reward_components_sparse` | 1344.1 | 1339.1 | 1348.1 | 155.3 |
| `reward_batch` | 970.7 | 956.3 | 980.2 | 106.7 |
| `reward_token_advantages` | 994.4 | 987.6 | 1017.4 | 111.9 |
| `score_prompt_group` | 361.3 | 357.4 | 370.4 | 14.0 |
| `response_log_probs` | 91.5 | 87.8 | 111.6 | 3.6 |
| `training_batch_assembly` | 481.0 | 474.7 | 493.8 | 50.3 |

### What changed, and what the numbers say

| case | change | verdict |
| --- | ---: | --- |
| `group_advantages` | **−45%** | One grouping pass instead of building `(index, reward)` tuple lists and re-zipping them. |
| `group_advantages_normalized` | **−39%** | Same kernel; the duplicate second implementation is gone. |
| `reward_components_sparse` | **−75%** | A fully zeroed preset no longer resolves citations or evaluates shaping arithmetic. |
| `reward_components_shaped` | −1% | Flat. Citations are resolved once instead of twice, but the shaped path's cost is dominated by terms that still have to run. |
| everything else | ±3% | Noise. No case regressed repeatably. |

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
| `group_advantages` | 10.9 | 10.7 | 11.2 | 3.2 |
| `group_advantages_normalized` | 19.3 | 19.1 | 19.6 | 3.3 |
| `group_advantages_tensor` | 393.8 | 382.4 | 410.2 | 1.0 |
| `token_f1` | 918.4 | 903.1 | 949.4 | 4.4 |
| `reward_components_shaped` | 5585.8 | 5532.0 | 5839.9 | 194.4 |
| `reward_components_sparse` | 1374.0 | 1372.0 | 1393.7 | 155.3 |
| `reward_batch` | 989.7 | 967.0 | 1019.5 | 106.7 |
| `reward_token_advantages` | 1008.0 | 992.5 | 1042.4 | 111.9 |
| `score_prompt_group` | 366.9 | 364.2 | 374.1 | 14.1 |
| `response_log_probs` | 90.4 | 86.0 | 136.1 | 3.5 |
| `training_batch_assembly` | 182.0 | 181.1 | 184.4 | 0.4 |
| `policy_update_loss` | 299.2 | 281.1 | 410.8 | 6.4 |

### Online-path results

Measured against the same fixtures with only the online-path changes reverted,
median of three runs each:

| case | time | peak alloc | verdict |
| --- | ---: | ---: | --- |
| `training_batch_assembly` | **−13.5%** | **−99%** (36.1 → 0.4 KiB) | Kept. Left-padding writes into a preallocated tensor instead of building a nested Python list. |
| `policy_update_loss` | +3.0% | −2.3% | See below. |

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
