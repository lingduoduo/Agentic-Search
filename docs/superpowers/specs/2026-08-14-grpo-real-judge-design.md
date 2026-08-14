# A judge that reads the gold answer

## Status

Not started. The largest correctness gap in the training stack, and the one that
makes every reward number downstream of it hard to interpret.

## Problem

`SimulatedPreferenceJudge` in `src/training/judge.py` is the judge used by
`run_bamboogle_grpo_train` and `run_bamboogle_synthetic_grpo` — the two examples
that actually update a policy. Its scoring signature is:

```python
def score(self, answer: str) -> float:
    """Return a quality score in [0, 1] from the answer text alone."""
```

**It never sees the gold answer.** It is a deterministic reference-free
heuristic — length credit up to `max_words`, lexical diversity, a hedging
penalty, plus a deterministic jitter term for tie-breaking.

So a policy trained against it is optimised toward *answers that look good*:
roughly 40 words, varied vocabulary, no hedging. Correctness contributes nothing.
A confidently-worded wrong answer outscores a correct hedged one, by
construction.

The class docstring is honest about this and the training doc labels it a
stand-in. This spec is about replacing it, not about discovering it.

## Why it matters more than "the placeholder is a placeholder"

Two things make this worse than an obvious TODO:

1. **The reward plumbing around it is real and well-tested.** `reward_components`,
   the four-dimension rollup, `score_prompt_group`, GRPO advantage
   normalisation — all of that is exercised and correct. It is easy to read a
   green reward test suite as evidence that the training signal is sound, when
   the one component that decides *what good means* is a text-shape heuristic.
2. **`simple_sparse_correctness_reward` already exists** and does compare against
   gold (exact-normalised match → 1.0, gold contained in prediction → 0.7). So
   the repo has two judges with incompatible contracts, and which one a run uses
   depends on which example script was invoked. That is a trap for anyone
   comparing results across scripts.

## Approach

`BatchJudgeFn` is already the interface, and the training doc states a real judge
"would slot in behind" it. So the work is a new implementation, not a refactor:

1. A judge that takes `(answer, gold)` and returns a score — LLM-as-judge behind
   the existing provider config, with a deterministic offline fallback so the
   test suite and the no-network smoke path keep working.
2. Prompt and parsing that fail closed: an unparseable judge response must not
   silently become a middling score, because a constant score across a rollout
   group makes every GRPO advantage zero and training becomes a no-op that looks
   like it is running.
3. Cache by `(answer, gold)` — GRPO scores G rollouts per prompt and re-judges
   the same gold repeatedly.

## Acceptance

- A judge whose score depends on `gold`: swapping the gold answer must change the
  score for a fixed prediction. Asserting that is the whole point — it is exactly
  what the current judge cannot do.
- A wrong-but-fluent answer scores below a correct-but-terse one on a small
  hand-built set. This is the failure mode the heuristic has today.
- Deterministic offline path with no network, so `run_grpo_training_pipeline`
  stays a no-GPU no-network smoke test.
- An unparseable judge response is surfaced, not scored — with a test that a
  degenerate group (all scores equal) is detectable rather than silently
  producing zero advantages.
- The two-judge trap is resolved: one documented default, with the other
  reachable only by explicit choice.

## Out of scope

A trained reward model. The repo is critic-free GRPO with a judge function, and
this spec keeps that shape — it replaces the judge, not the algorithm.

Also out of scope: the unwired PPO-critic helpers, which have their own spec.
