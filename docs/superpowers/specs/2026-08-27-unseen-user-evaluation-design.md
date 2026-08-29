# Unseen-User Evaluation Design

## Goal

Measure, on users never seen during training, whether a GRPO-trained search
policy:

1. optimizes a reward that **aligns with conversion**;
2. exhibits **behavioral separation** from a baseline policy that survives a
   significance test;
3. **follows instructions better than a larger baseline model**.

The deliverable is the evaluation harness. It consumes a flat record format, so
the population it measures is swappable: a simulated cohort today, real users
and real model rollouts when they exist.

## Why a simulated cohort

The claims cannot be made against this repository's data today, and the design
must not pretend otherwise.

| Source | Rows |
| --- | ---: |
| `data/web.db` users / sessions / feedback | 1 / 21 / **0** |
| `data/agentic_search.db` users / sessions / feedback | 2 / 15 / **0** |
| Generative models in the local HF cache | **0** (embedding models only) |

Two users and zero conversion labels cannot support a statement about unseen
users, and no cached generative model can support a policy-versus-larger-baseline
comparison without ~18 GB of downloads and minutes-per-query local inference.

So the harness ships with a generator that produces a user population from a
conversion rule it keeps hidden. The harness never sees the rule. What that
supports is:

> The pipeline detects an effect of this size, on held-out users, at this power.

and explicitly **not**:

> The model converts real users.

`cohort.py` is the seam. Replacing it with a reader over `chat_sessions` +
`retrieval_feedback` changes the claim without changing the analysis.

## The Statistical Spine

**The unit of independence is the user, not the session.** Every other decision
follows from this.

A user contributes many sessions and those sessions are correlated — same
person, same habits, same phrasing. Treating them as independent samples
inflates *n* and manufactures significance: 2 users and 20 sessions analysed
per-session look like `n = 20` and will report `p < 0.05` on pure noise.

Therefore:

- Every metric aggregates to **one value per user** before any test runs.
- Every resample draws **users**, carrying all their sessions with them
  (a cluster/block bootstrap), never individual sessions.
- Reported `n` is always the number of users, never the number of sessions.

A test in this design asserts that ignoring clustering visibly inflates
significance, so the failure mode this rules out stays ruled out.

## Record Format

One record per rollout. This is the contract between population and analysis;
nothing downstream knows whether a record came from a simulation or a database.

```python
@dataclass(frozen=True)
class EvalRecord:
    user_id: str
    prompt_id: str          # identifies the task, shared across policies
    policy: str             # "trained" | "baseline"
    reward: float           # what the policy optimizes
    converted: bool         # the outcome reward is supposed to track
    response: str           # for instruction-following checks
    metrics: dict[str, float]   # AgentLoopOutput.metrics shape
    cited_ids: frozenset[str]   # citation labels resolved against retrieval
    tool_calls: tuple[str, ...] # raw tool-call payloads, unparsed
```

`prompt_id` is what makes the instruction-following comparison **paired**: both
policies answer the same task, so per-prompt differences cancel task difficulty.

For real data, `converted` derives from signals already in the schema:

```text
converted := thumbs_up
          OR (final_answer non-empty AND cited_ids non-empty
              AND NOT search_budget_exhausted_without_answer)
```

This definition is the contract for a future DB adapter — **not built here**,
see Out of Scope. It is written down now so the simulated cohort is shaped like
the real thing rather than around it. The cohort produces `converted` from its
hidden rule instead, and nothing in the analysis knows the difference.

## Held-Out User Split

Deterministic and by user, never by session:

```python
def split_users(user_ids, *, holdout_fraction, seed) -> tuple[set[str], set[str]]
```

Assignment is `sha256(f"{seed}:{user_id}")` thresholded against the fraction, so
the split is stable across runs and machines and independent of iteration order.
A user's every session lands on one side.

The harness asserts the two sets are disjoint and that the evaluation frame
contains no training user. This is a structural guarantee, checked, not a
statistic.

## Measurements

### Conversion alignment

Does the reward the policy optimizes actually track the outcome, on users it has
never seen?

- **Statistic:** AUC of `reward` as a ranker of `converted`, computed per user
  and averaged across users.
- **Interval:** cluster bootstrap percentile CI over users.
- **Criterion:** the CI's **lower bound exceeds 0.5** — the claim is directional,
  so a two-sided exclusion of 0.5 would also be satisfied by a reward that
  ranks conversion backwards.

AUC is undefined for a user whose sessions all share one outcome, so the
generator must give most users both converted and unconverted sessions. Such
users are excluded and **the exclusion count is reported**: silently dropping
them would let a cohort that is mostly degenerate masquerade as a clean result.

AUC is itself the effect size; it is threshold-free and insensitive to reward
scale, which matters because reward scale is a free parameter of the config.

### Behavioral separation

Do the trained and baseline policies actually behave differently — not just
score differently?

- **Statistic:** a per-user behavior vector from `metrics`
  (`search_rounds`, `web_searches`, `vdb_searches`, `rerank_calls`,
  `repeated_search_queries`), compared between policies.
- **Test:** one permutation test **per component**, shuffling the policy label
  within each user so user-level structure is preserved under the null. Testing
  each component separately rather than the vector jointly keeps the result
  interpretable — "it searches less" is actionable, "the vectors differ" is not
  — at the cost of more comparisons, which the correction below absorbs.
- **Effect size:** Cliff's delta per component — non-parametric and ordinal,
  which suits small user counts and skewed action counts better than a
  standardized mean difference.

### Instruction following

Objectively checkable constraints, no judge. Each is a predicate over one
record, and each is grounded in a contract this repo already enforces:

| Constraint | Check |
| --- | --- |
| `answer_tag_present` | a well-formed `<answer>…</answer>` in the response |
| `citations_wellformed` | every `[RxQyDz]` label parses **and** resolves to a retrieved doc |
| `tool_calls_parseable` | each tool-call payload is valid JSON naming a registered tool |
| `round_budget_respected` | `metrics["rounds_used"] <= max_search_rounds` |

- **Statistic:** per-user compliance rate, paired by `prompt_id` between the
  trained policy and the larger baseline.
- **Test:** paired permutation over users, one-sided (the claim is directional:
  trained ≥ baseline).
- **Effect size:** mean paired difference in compliance rate with a cluster
  bootstrap CI.

A judge is deliberately excluded. These predicates are reproducible and cost
nothing; an LLM judge would make the headline number depend on a third model.

### Multiple comparisons

Every p-value the harness produces enters one Benjamini–Hochberg family at
q = 0.05: five behavioral components plus four instruction-following
constraints. Conversion alignment is reported as an interval, not a p-value, and
so does not enter the correction. Both raw and adjusted values are printed. Every p-value is reported beside its effect size and CI — an effect
that is significant and negligible must be legible as such.

### Achieved power

Because the cohort generator knows ground truth, the harness re-runs the whole
analysis over *K* freshly seeded cohorts and reports the rejection rate: the
power actually achieved at the configured user count and effect size. This is
what makes "statistically significant" a claim about the pipeline rather than a
single lucky draw.

Defaults are chosen to run in CI: 2000 bootstrap resamples, 2000 permutations,
K = 200 power replications, all configurable.

## Module Layout

Pure Python and numpy — no torch, no transformers. `reward.py` and the eval
package are the torch-free half of post-training, and the CI unit-test job
installs no torch; a torch import here would silently drop these tests from that
gate.

```text
src/model/post_training/eval/
  cohort.py                 simulated users, hidden conversion rule, EvalRecord
  stats.py                  cluster bootstrap, permutation, BH, Cliff's delta
  instruction_following.py  the four constraint predicates
  unseen_users.py           split, orchestration, report
examples/run_unseen_user_eval.py
```

`stats.py` knows nothing about rewards or agents; it takes grouped numbers. That
is what makes it testable against distributions with known answers.

## Testing

Two tests carry the credibility of everything else:

1. **A null cohort must not produce significance.** Generated with no real
   effect, the harness must fail to reject at the configured rate. Without this
   the harness is a significance-manufacturing machine, and every number it
   prints is worthless.
2. **Ignoring clustering must visibly inflate significance.** Analysing the same
   correlated data per-session rather than per-user must produce materially more
   rejections, pinning the bug this design exists to avoid.

Alongside those:

- `stats.py` against closed-form answers: a permutation test on exchangeable
  data yields approximately uniform p-values; bootstrap CIs cover a known mean
  at roughly the nominal rate; BH is checked against a hand-worked example.
- Cliff's delta at its known boundaries (disjoint distributions → ±1, identical
  → 0).
- Split determinism: same seed and IDs give the same partition, and no user
  crosses sides.
- Constraint predicates against adversarial strings — unclosed tags, citation
  labels that parse but reference nothing retrieved, tool calls that are valid
  JSON but name no registered tool.
- Report provenance: the rendered report always states that the cohort is
  simulated, with its user count, effect size and achieved power.

Every test is deterministic under a fixed seed. Every new test must fail when
the behavior it describes is removed.

## Out of Scope

- Any claim about real users or real conversion. The DB adapter is a follow-up.
- Running actual model inference. The harness consumes records; producing them
  from live models is a separate task gated on model availability.
- An LLM judge for instruction following.
- Changing reward functions, training, or the agent loops. This is evaluation
  only; it reads `metrics` and responses and writes nothing back.

## Acceptance Criteria

- Held-out users share no ID with training users, asserted structurally.
- Every statistic aggregates per user; every resample draws users.
- Conversion alignment reports AUC with a cluster bootstrap CI.
- Behavioral separation reports a permutation p-value with Cliff's delta.
- Instruction following reports a paired, one-sided comparison against the
  larger baseline, per constraint and in aggregate.
- p-values are BH-corrected and printed raw and adjusted, each beside its
  effect size.
- A null cohort does not produce significance; per-session analysis of
  clustered data demonstrably does.
- The report states its own provenance and achieved power.
- The eval package imports and runs with torch blocked.
- `ruff check`, `ruff format --check`, and the full suite pass.
