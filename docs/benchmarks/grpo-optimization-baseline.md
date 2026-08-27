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

## Baseline (before optimization, commit 5ec0191)

- Python 3.12.2, torch 2.11.0
- Platform: macOS-26.6.2-arm64-arm-64bit (cpu)
- Warmup 5, iterations 25
- Fixtures: num_groups=16, rollouts_per_group=8, num_rollouts=128, prompt_len=32, response_len=96, retrieved_docs=24, seed=20260827

| case | median (µs) | min (µs) | max (µs) | peak alloc (KiB) |
| --- | ---: | ---: | ---: | ---: |
| `group_advantages` | 201.2 | 169.5 | 219.1 | 6.7 |
| `group_advantages_normalized` | 369.4 | 253.9 | 409.0 | 6.8 |
| `group_advantages_tensor` | 582.5 | 569.8 | 603.4 | 1.9 |
| `token_f1` | 5681.2 | 4189.2 | 65837.0 | 7.6 |
| `reward_components_shaped` | 35728.5 | 35172.4 | 60514.2 | 202.1 |
| `reward_components_sparse` | 36083.8 | 35290.2 | 53085.0 | 201.8 |
| `reward_batch` | 5045.4 | 4870.2 | 5199.7 | 114.3 |
| `reward_token_advantages` | 6826.0 | 5287.0 | 12809.2 | 119.5 |
| `score_prompt_group` | 2300.2 | 2176.6 | 2346.9 | 16.0 |
| `response_log_probs` | 104.3 | 89.7 | 184.5 | 58.0 |
| `training_batch_assembly` | 833.8 | 785.0 | 859.6 | 61.9 |
