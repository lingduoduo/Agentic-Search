# Unseen-user evaluation

Provenance: simulated cohort (60 users x 12 sessions, seed 0) -- NOT real users
Measured on 12 held-out users (unit of independence: user, not session).

## Conversion alignment
AUC 0.810 [0.751, 0.868] over 12 users (0 excluded: no outcome variation)

## Behavioral separation
| component | trained | baseline | effect (Cliff's d) | p | p (BH) |
| --- | ---: | ---: | ---: | ---: | ---: |
| `search_rounds` | 2.028 | 3.403 | -0.653 | 0.0020 | 0.0036 |
| `web_searches` | 0.829 | 1.049 | -0.507 | 0.0845 | 0.0950 |
| `vdb_searches` | 1.269 | 1.972 | -0.903 | 0.0010 | 0.0030 |
| `rerank_calls` | 0.343 | 0.569 | -0.417 | 0.0255 | 0.0382 |
| `repeated_search_queries` | 0.451 | 0.465 | +0.056 | 0.8146 | 0.8146 |

## Instruction following (vs larger baseline)
| constraint | trained | baseline | effect (Cliff's d) | p | p (BH) |
| --- | ---: | ---: | ---: | ---: | ---: |
| `answer_tag_present` | 0.826 | 0.542 | +0.882 | 0.0010 | 0.0030 |
| `citations_wellformed` | 0.826 | 0.542 | +0.882 | 0.0015 | 0.0034 |
| `tool_calls_parseable` | 0.826 | 0.542 | +0.882 | 0.0010 | 0.0030 |
| `round_budget_respected` | 0.910 | 0.819 | +0.292 | 0.0415 | 0.0533 |

## Achieved power

Rejection rate over 200 freshly seeded cohorts.

| measurement | power |
| --- | ---: |
| `alignment` | 1.00 |
| `answer_tag_present` | 1.00 |
| `citations_wellformed` | 1.00 |
| `repeated_search_queries` | 0.14 |
| `rerank_calls` | 0.15 |
| `round_budget_respected` | 0.78 |
| `search_rounds` | 0.99 |
| `tool_calls_parseable` | 1.00 |
| `vdb_searches` | 0.98 |
| `web_searches` | 0.65 |
