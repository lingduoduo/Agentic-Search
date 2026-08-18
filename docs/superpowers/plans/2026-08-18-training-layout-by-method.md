# Plan — group `src/training/` by post-training method

Design: [`2026-08-18-training-layout-by-method-design.md`](../specs/2026-08-18-training-layout-by-method-design.md)

## 1. Establish what post-training methods actually exist

- Read every module's docstring and public surface; classify each as a method, a
  shared building block, a benchmark harness, or something mis-filed.
- Check specifically for DPO: a preference-pair loss, a reference-model KL, or a
  chosen/rejected dataset builder. Do not infer DPO from the presence of a
  *preference judge* — `SimulatedPreferenceJudge` scores single answers for GRPO
  reward, which is RLAIF, not DPO.
- Diff the two same-named `compute_grpo_outcome_advantage` functions before
  assuming duplication. They differ (list-scalar vs token-tensor); leave both.
- **Verify:** every one of the 22 modules has a stated category and a named
  consumer.

## 2. Confirm the mis-filings with call sites, not names

- `evaluation.py`: grep `SearchResultEvaluator` outside `src/training/`. It is
  imported by `src/agents/search/search.py` and
  `src/agents/components/evidence_judge.py` — request-path code.
- `ppo/`: confirm `compute_ppo_policy_loss_core` and the KL controllers are
  *live* (`llm_grpo_trainer`, `src/model/generation.py`) before renaming the
  package, so the rename does not imply the PPO math is dead.
- **Verify:** both claims backed by a file:line, not by the module name.

## 3. Move

- `git mv` the eight `ppo/*` modules into `rl/`; `grpo.py` → `rl/rollouts.py`;
  `rl_agent.py` → `rl/qlearning.py`; `search_environment.py` → `rl/`;
  `sft.py` → `sft/trainer.py`; `evaluation.py` →
  `src/agents/components/result_evaluation.py`.
- Fix relative-import depth for every module that moved a level deeper:
  `..agents` → `...agents`, `.data` → `..data`, `.reward` → `..reward`.
- Write `sft/__init__.py`; rewrite the `rl/` and `src/training/` docstrings to
  say what each package is, including the absence of DPO.
- **Verify:** `ast.parse` clean; `git status` shows renames.

## 4. Repoint every consumer

- Rewrite module paths across `src/`, `examples/`, `tests/`, live `docs/`,
  `.github/`, and `.claude/CLAUDE.md`. Order the replacement map longest-first so
  `src.training.ppo` does not eat `src.training.ppo.core_algos`, and guard
  against double-application (`sft` → `sft.trainer` → `sft.trainer.trainer`).
- Revert `docs/superpowers/archive/`, `context-packs/`, `plans/` and `specs/` —
  they are records of past work, not runnable references, and a blanket rewrite
  will touch them.
- Update the `src/__init__` lazy table **including its section comments**, which
  name modules and will otherwise go stale silently.
- **Verify:** `grep -rn "training\.ppo\|training\.grpo\b\|training\.rl_agent\|training\.evaluation"` returns nothing in code.

## 5. Verify

- Import smoke test across every new package path plus both re-export surfaces
  (`src.training.*` and the `src.__init__` lazy table). Note `QLearningAgent` is
  re-exported from `src/training/__init__` but was never in the top-level lazy
  table — do not "fix" its absence.
- `ruff check . --fix`, `ruff format`.
- Full unit suite. **Expected: 10 pre-existing failures**, all from the absent
  HuggingFace download of `intfloat/e5-small-v2`. All 332 training-related tests
  must pass.
- **Verify:** no failure outside that known set.
