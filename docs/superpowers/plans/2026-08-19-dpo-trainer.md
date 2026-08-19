# Plan — a DPO trainer for `src/training/dpo/`

Design: [`2026-08-19-dpo-trainer-design.md`](../specs/2026-08-19-dpo-trainer-design.md)

TDD throughout: each step writes the failing test first, then the code that
passes it.

## 1. `dpo/data.py` — the preference pair and its loader

- `PreferenceExample` (frozen dataclass: `prompt`, `chosen`, `rejected`) and
  `load_preference_pairs(path)`.
- Tests first, covering every rejection: malformed JSON, a non-object line, a
  missing key, a blank value, an empty file, and `chosen == rejected`.
- **Verify:** each rejection raises with a message naming the offending line
  number — a loader that says only "invalid file" is useless against a
  thousand-line dataset.

## 2. `dpo/trainer.py` — sequence log probabilities

- A helper that tokenizes `(prompt, response)` into `input_ids`, `prompt_len`
  and a response mask, then returns the summed response log prob via
  `get_response_log_probs` from `src.training.rl.llm_grpo_trainer`.
- Do **not** reimplement the logits shift. Import it. The
  `logits[:, prompt_len - 1 : -1]` alignment is silent when wrong.
- **Verify:** a test that changes only prompt tokens leaves the sum unchanged,
  proving prompt tokens are excluded.

## 3. `dpo/trainer.py` — the loss

- `dpo_loss(policy_chosen, policy_rejected, ref_chosen, ref_rejected, beta)`
  returning loss, margin, and accuracy — a pure function of four scalars per
  pair, so it is testable with no model at all.
- **Verify (load-bearing):** with policy log probs equal to reference log probs,
  loss is exactly `log 2` (≈ 0.6931). Assert to ~1e-6. This pins the formula; a
  sign error or dropped `β` still yields a plausible number and only a known
  value catches it.
- **Verify:** β monotonicity on a fixed positive margin; margin sign matches
  which response is preferred.

## 4. `DPOConfig` and `DPOTrainer`

- Mirror `SFTTrainer`'s shape exactly: frozen config, injectable optimizer,
  `train()` → per-step records, `save()` → HF format.
- Reference = `deepcopy(policy)`, `.eval()`, `requires_grad_(False)` at init;
  overridable by injection.
- **Verify:** after `train()`, reference parameters are bit-identical
  (`torch.equal`) and none carry grads; policy parameters *have* changed.
- **Verify:** one training step raises the chosen response's log prob relative
  to the rejected one — the gradient actually points the right way.

## 5. Correct the two now-false statements

- `src/training/__init__.py` and `.claude/CLAUDE.md` both assert that no DPO
  trainer exists and name `dpo/` as where one would go. Rewrite both to describe
  the package that now exists.
- **Verify:** `grep -rn "no DPO"` returns nothing outside archived docs.

## 6. Full verification

- `pytest tests/unit/dpo/ -v`, then the full unit suite.
- `ruff check . && ruff format`.
- Run the torch-blocked collection sweep
  (`PYTHONPATH=/tmp pytest tests/unit/ --collect-only -q -p block_torch_plugin`):
  the new package imports torch at module scope, so its tests must guard with
  `pytest.importorskip("torch")` and must not break collection in a CI job that
  installs no torch. This repo has shipped that failure four times.
- **Verify:** 0 collection errors; full suite shows no failure outside the 10
  known HuggingFace-download failures.
