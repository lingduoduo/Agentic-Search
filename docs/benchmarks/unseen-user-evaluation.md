# Unseen-user evaluation

Provenance: simulated cohort (60 users x 12 sessions, seed 0) -- NOT real users; planted effect sizes: alignment=2, behavior_shift=1.5, instruction_gap=0.25
Measured on 12 held-out users (unit of independence: user, not session).

## Conversion alignment
AUC 0.860 [0.786, 0.926] over 12 users (0 excluded: no outcome variation)

## Behavioral separation
| name | n users | trained | baseline | mean paired diff [CI] | effect (Cliff's d) | p | p (BH) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `search_rounds` | 12 | 2.271 | 3.333 | -1.062 [-1.344, -0.812] | -0.410 | 0.0010 | 0.0045 |
| `web_searches` | 12 | 0.702 | 1.174 | -0.472 [-0.971, -0.074] | -0.236 | 0.0700 | 0.0900 |
| `vdb_searches` | 12 | 2.275 | 3.646 | -1.371 [-2.031, -0.766] | -0.361 | 0.0040 | 0.0060 |
| `rerank_calls` | 12 | 0.476 | 0.569 | -0.094 [-0.218, +0.037] | -0.083 | 0.1994 | 0.2243 |
| `repeated_search_queries` | 12 | 0.437 | 0.431 | +0.006 [-0.187, +0.225] | -0.111 | 0.9575 | 0.9575 |

## Instruction following (vs the baseline arm of the same simulated cohort)
| name | n users | trained | baseline | mean paired diff [CI] | effect (Cliff's d) | p | p (BH) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `answer_tag_present` | 12 | 0.910 | 0.674 | +0.236 [+0.090, +0.375] | +0.701 | 0.0030 | 0.0054 |
| `citations_wellformed` | 12 | 0.840 | 0.688 | +0.153 [+0.083, +0.222] | +0.472 | 0.0020 | 0.0045 |
| `tool_calls_parseable` | 12 | 0.896 | 0.646 | +0.250 [+0.181, +0.326] | +0.764 | 0.0005 | 0.0045 |
| `round_budget_respected` | 12 | 0.826 | 0.674 | +0.153 [+0.083, +0.229] | +0.431 | 0.0020 | 0.0045 |

## Achieved power

Rejection rate against the effect sizes above, over 200 freshly seeded cohorts.

| measurement | power |
| --- | ---: |
| `alignment` | 0.97 |
| `answer_tag_present` | 1.00 |
| `citations_wellformed` | 0.99 |
| `repeated_search_queries` | 0.31 |
| `rerank_calls` | 0.24 |
| `round_budget_respected` | 0.99 |
| `search_rounds` | 1.00 |
| `tool_calls_parseable` | 1.00 |
| `vdb_searches` | 0.97 |
| `web_searches` | 0.73 |

## Reproducing this report

```bash
python -m examples.run_unseen_user_eval \
  --users 60 \
  --sessions 12 \
  --holdout 0.3 \
  --seed 0 \
  --resamples 2000 \
  --power-replications 200 \
  --allowed-tools search,fetch \
  --max-search-rounds 5 \
  --baseline-label 'the baseline arm of the same simulated cohort' \
  --output docs/benchmarks/unseen-user-evaluation.md
```
