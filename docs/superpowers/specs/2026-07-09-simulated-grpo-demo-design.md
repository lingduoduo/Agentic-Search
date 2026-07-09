# Simulated-Judge GRPO Demo — Design

Date: 2026-07-09
Status: Approved (brainstorming)
Related: [Simulated Preference Judge](2026-07-09-simulated-preference-judge-design.md)

## Problem

We have a reference-free `SimulatedPreferenceJudge` (`src/training/judge.py`, PR #387)
and a working online GRPO trainer (`src/training/ppo/llm_grpo_trainer.py`), but no
runnable entrypoint that closes the loop: **sample a prompt → generate an output →
score it with the judge → update the policy**. The existing
`examples/run_bamboogle_synthetic_grpo.py` scores rollouts and computes group
advantages, then *dumps JSONL* — it never performs a policy update.

Goal: a minimal, runnable demo that optimizes a policy with GRPO against the
simulated judge and makes the learning signal visible.

## Non-goals

- No trained reward model / pairwise comparison data (the judge is pointwise and
  reference-free; that stays as-is).
- No classic PPO with a value critic — GRPO only (critic-free group baseline).
- No retrieval server, no `SearchAgentLoop` rollouts.
- No durability: no checkpointing, resume, or `train_loop.py` wiring. (Possible
  future follow-up, explicitly out of scope here.)
- No changes to `reward.py`, `judge.py`, or the trainers.

## Approach (chosen: plain-generation GRPO)

Drive the existing `LLMGRPOTrainer` (plain prompt→response, no search) rather than
`SearchAgentGRPOTrainer`. Both run the identical GRPO update
(rollout → judge → group advantage → PPO-clip + KL to frozen reference →
`optimizer.step`); they differ only in how the rollout is produced. Plain
generation needs no retrieval server and a step is `model.generate × G + one
backward`, fast enough to watch reward move on CPU/MPS. Because the judge is
reference-free and only inspects the answer text, search grounding adds nothing to
the reward signal here.

Rejected alternatives:
- **Search-grounded (`SearchAgentGRPOTrainer`)**: faithful to the real agent but
  needs a live retrieval server and full search rollouts (minutes per rollout on
  CPU) — too slow for a "watch it learn" demo whose reward ignores retrieved docs.
- **`--train` flag on the synthetic-data demo**: overloads a script whose job is
  data generation + agreement reporting, and inherits the slow search rollouts.

## Component: `examples/run_bamboogle_simulated_grpo.py`

Single new script. Uses only existing machinery.

Flow:
1. `load_bamboogle(limit=args.limit)` → list of prompts (from
   `src/training/eval/bamboogle.py`).
2. `judge = SimulatedPreferenceJudge()`; adapt to the pointwise `JudgeFn` seam:
   `judge_fn = lambda pred, _gt: judge.score(pred)`.
3. `trainer = LLMGRPOTrainer.from_pretrained(args.model, judge_fn=judge_fn,
   config=LLMGRPOConfig(num_rollouts=args.num_rollouts,
   max_new_tokens=args.max_new_tokens, temperature=args.temperature),
   lr=args.lr, device=args.device, local_files_only=not
   args.allow_remote_model_downloads)`.
4. For `args.steps` iterations: take the next `args.batch_prompts` prompts (cycling
   the loaded list), call `trainer.step(prompts, ground_truths=[""] * len(prompts))`
   (ground truth ignored by the judge).
5. Print `step | mean_reward | mean_advantage | mean_kl | clip_fraction | loss`
   plus a rolling mean of `mean_reward`.

### Two pure helpers (unit-tested)

- `make_judge_fn(judge) -> JudgeFn`: returns `lambda pred, _gt: judge.score(pred)`.
- `cycle_prompt_batches(prompts, steps, batch_size) -> list[list[str]]` (or a
  generator): yields `steps` batches of size `batch_size`, cycling `prompts` when
  exhausted. Keeps the demo runnable with a small `--limit`.

### Lazy imports

Import `torch` / `transformers` / trainer modules *inside* functions, not at module
top level, so `python3 -m examples.run_bamboogle_simulated_grpo --help` stays light
(matches commit `0328b12`).

## CLI & defaults

| Flag | Default | Purpose |
|---|---|---|
| `--model` | required | HF model id/path (e.g. `Qwen/Qwen2.5-0.5B-Instruct`) |
| `--steps` | `10` | GRPO update steps |
| `--num_rollouts` | `4` | G completions per prompt (group size) |
| `--batch_prompts` | `2` | prompts per step |
| `--limit` | `8` | bamboogle prompts to load (cycled) |
| `--max_new_tokens` | `64` | short rollouts → fast CPU steps |
| `--lr` | `1e-5` | AdamW learning rate |
| `--temperature` | `0.8` | rollout sampling temperature |
| `--device` | `cpu` | `cpu` / `mps` / `cuda` |
| `--allow_remote_model_downloads` | off | else `local_files_only=True` |
| `--seed` | `0` | `torch.manual_seed` for reproducible rollouts |

## Success criteria

1. Script runs `--steps N` to completion, printing finite metrics every step.
2. `mean_reward ∈ [0, 1]`; `mean_kl` rises from ~0 (policy diverging from the frozen
   reference ⇒ gradients are landing); `clip_fraction` is reported.
3. Policy weights differ from a pre-training snapshot (proves `optimizer.step`
   mutated the policy — not a no-op).
4. `--help` runs without importing torch/transformers.
5. A rolling-mean reward line is printed so an upward trend is visible **when it
   occurs** — presented as a trend, not a guarantee (RL on a 0.5B model over ~10
   steps is noisy; monotonic improvement is not promised).

## Testing

Automated (fast, **no model load** — respects the web-test model-load gotcha):
- `make_judge_fn`: returned fn ignores the ground-truth arg and returns
  `judge.score(pred)` for a few sample answers.
- `cycle_prompt_batches`: yields exactly `steps` batches of `batch_size`, cycling
  correctly when `steps * batch_size > len(prompts)`; handles `len(prompts) == 1`.

Manual (`/verify`, not default pytest): one real `--steps 2` run on a tiny model
(`Qwen/Qwen2.5-0.5B-Instruct`, `--device cpu`) asserting criteria 1–4 by
observation.

## Risks

- **Noisy signal**: reward may not rise in 10 steps. Mitigated by framing success
  as "real updates + sane diagnostics + weights changed", not monotonic reward.
- **Reward hacks form, not correctness**: the judge rewards length/uniqueness/no-
  hedging, so the policy may learn to pad answers. This is expected and is exactly
  the known limitation of the reference-free judge — the demo illustrates the
  mechanism, not a production reward. Documented in the script docstring.
