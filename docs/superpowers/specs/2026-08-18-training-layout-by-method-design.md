# Group `src/training/` by post-training method

## Problem

`src/training/` held 6,831 LOC in a shape that did not say what any of it was:
ten flat modules at the top level, plus a `ppo/` package and an `eval/` package.
Reading the directory gave no answer to "which post-training methods does this
repo implement?" — the question the directory exists to answer.

Three specific problems:

**1. Methods were not separable.** SFT was one flat file (`sft.py`) sitting
beside `grpo.py`, `rl_agent.py`, `search_environment.py`, `reward.py`,
`judge.py`, and `data.py`. Nothing distinguished a training method from a shared
building block from a toy demo.

**2. `ppo/` was misnamed.** It holds three GRPO trainers, a GRPO controller, and
a GRPO train loop. Only `core_algos.py` contains genuine PPO math — and that is
live, used by `llm_grpo_trainer` and by `src/model/generation.py`. The name
advertised a method the package mostly does not implement.

**3. `evaluation.py` was not training code.** `SearchResultEvaluator` is a
runtime sufficiency heuristic — "are these search results good enough to stop?"
— imported by `src/agents/search/search.py` and
`src/agents/components/evidence_judge.py` on the request path. It had no
training caller at all. `src/training/` was holding live serving code.

## Investigation findings that shaped this

**There is no DPO.** No preference-pair loss, no reference-model KL over
chosen/rejected, no preference dataset builder. `SimulatedPreferenceJudge`
scores *single* answers as a GRPO reward (RLAIF); that is a different method.
No `dpo/` package is created — an empty folder would advertise a capability the
repo does not have. The top-level docstring records where one would go.

**The two `compute_grpo_outcome_advantage` functions are not duplicates.**
`grpo.py`'s takes `list[float]` and returns scalar per-rollout advantages;
`ppo/core_algos.py`'s takes token-level reward tensors with an eos mask and
returns tensors. Different layers of the same idea, correctly separate. Left
alone.

## Design

Method-named packages, with genuinely shared building blocks at the top level:

```
src/training/
  data.py                  shared: prompt + dataset construction (SFT and RL)
  reward.py                shared: reward functions (RL and eval)
  judge.py                 shared: RLAIF judges (RL and eval)
  train_query_router.py    offline sklearn trainer — not LLM post-training
  sft/trainer.py           supervised fine-tuning
  rl/                      the GRPO stack (was ppo/) + grpo.py + the Q-learning demo
  eval/                    benchmark harnesses (unchanged)
```

`rl/` absorbs `grpo.py` as `rollouts.py` (grouped sampling and scoring — the
name `grpo.py` next to `grpo_trainer.py` said nothing about the difference), and
the tabular Q-learning demo as `qlearning.py` + `search_environment.py`. Those
three are deliberately **not** re-exported from `rl/__init__.py`: `rollouts`
pulls in the agent loop, and the Q-learning pair is a self-contained demo with
no connection to the LLM stack. This preserves the existing import weight of
`from src.training.rl import ...` exactly.

`evaluation.py` moves to `src/agents/components/result_evaluation.py`, beside
`evidence_judge.py`, which wraps it.

## What this does not change

Behaviour. Every function, signature, threshold, and training script is
untouched; this is a move-and-rename with import updates. `src/training/__init__`
keeps re-exporting the same names (minus `SearchEvaluationConfig`, which is no
longer training), and the `src/__init__` lazy table keeps every entry, repointed.
