# Generated Context Pack

# Sft Grpo Warmstart

## Sources

- [Specification: 2026-06-17-sft-grpo-warmstart-design.md](../specs/2026-06-17-sft-grpo-warmstart-design.md)
- [Plan: 2026-06-17-sft-grpo-warmstart.md](../plans/2026-06-17-sft-grpo-warmstart.md)

## Specification Context

### Out of Scope

- LoRA / PEFT adapters
- DPO or preference-pair training
- Auto-triggering from the beat worker
- New dataclasses — `SFTExample` (already in `src/training/sft.py`) is reused as-is

---

### 2. Architecture

```
Data sources
  AgenticSearchStore (thumbs-up sessions)     data/sft_pairs.jsonl (optional)
       │  list_chat_messages → assistant turns      │  {"question": ..., "response": ...}
       └──────────────────────────┬─────────────────┘
                                  ▼
                 load_sft_examples(db_path, jsonl_path=None, min_ratings=1)
                 → list[SFTExample(prompt_messages, completion, trajectory_messages)]
                                  │
                    ──────── Phase 1: SFT ────────
                                  ▼
                           SFTTrainer(policy, tokenizer, optimizer, config)
                    .train(examples) — cross-entropy on assistant tokens only
                    (system / user / tool-result tokens masked to −100)
                                  │
                    data/checkpoints/sft_warmstart/   ← intermediate checkpoint
                                  │
                    ──────── Phase 2: GRPO ────────
                                  ▼
                    SearchAgentGRPOTrainer.from_pretrained(
                        "data/checkpoints/sft_warmstart/",
                        judge_fn=simple_sparse_correctness_reward,
                        reward_fn=SearchRewardFunction(
                            SearchRewardConfig(human_feedback_weight=0.5)
                        ),
                    )
                    .step_async(prompts, ground_truths, metadata=metadata)
                                  │
                    data/checkpoints/sft_grpo/   ← final checkpoint
```

_[Section compacted.]_

### 5. Testing

All unit tests — no GPU, no live server, no model downloads.

### `test_sft_examples.py`

- Thumbs-up session with assistant turn → `SFTExample` with correct `prompt_messages` and `completion`
- Session with no assistant turn is skipped
- JSONL row `{"question": "Q", "response": "R"}` → correct `SFTExample`
- JSONL row missing `response` key is skipped
- Both sources merged: total count = DB examples + JSONL examples
- `ValueError` when total < `min_ratings`

### `test_sft_trainer.py`

- Non-assistant tokens are masked to `−100` in labels (check label tensor directly)
- Loss decreases after one gradient step on a synthetic example (tiny vocab, cpu)
- `save()` calls `policy.save_pretrained(output_dir)` and `tokenizer.save_pretrained(output_dir)`
- `SFTConfig(epochs=0)` → `train()` returns empty loss history without calling forward pass

---

## Implementation Plan Context

### Task 1: `load_sft_examples` in `src/training/data.py`

**Files:**
- Modify: `src/training/data.py` (append after the `load_feedback_examples` function)
- Test: `tests/unit/test_sft_examples.py`

- [x] **Step 1: Write the failing tests**

Create `tests/unit/test_sft_examples.py`:

```python
"""Tests for load_sft_examples."""
from __future__ import annotations

import json
import pytest

from src.internal.db import AgenticSearchStore
from src.training.data import load_sft_examples
from src.training.sft import SFTExample


def _seed_store(db_path: str, rows: list[dict]) -> None:
    with AgenticSearchStore(db_path) as store:
        for r in rows:
            sid = r["session_id"]
            store.create_chat_session(session_id=sid, user_id=None)
            if r.get("user_msg"):
                store.add_chat_message(sid, "user", r["user_msg"])
            if r.get("assistant_msg"):
                store.add_chat_message(sid, "assistant", r["assistant_msg"])
            if r.get("signal"):
                store.save_retrieval_feedback(sid, r["signal"])


def test_thumbs_up_session_becomes_sft_example(tmp_path):
    db = str(tmp_path / "t.sqlite3")
    _seed_store(db, [{"session_id": "s1", "signal": "thumbs_up",
                       "user_msg": "Q?", "assistant_msg": "A."}])
    examples = load_sft_examples(db, min_ratings=1)
    assert len(examples) == 1
    ex = examples[0]
    assert ex.prompt_messages == [{"role": "user", "content": "Q?"}]
    assert ex.completion == "A."


def test_thumbs_down_session_excluded(tmp_path):
    db = str(tmp_path / "t.sqlite3")

_[Section compacted.]_

### Task 2: `SFTConfig` and `SFTTrainer` in `src/training/sft.py`

**Files:**
- Modify: `src/training/sft.py`
- Test: `tests/unit/test_sft_trainer.py`

### Task 3: CLI script `examples/run_sft_grpo.py`

**Files:**
- Create: `examples/run_sft_grpo.py`

### Task 4: Final integration check + commit spec and plan

- [x] **Step 1: Run all new tests**

```bash
pytest tests/unit/test_sft_examples.py tests/unit/test_sft_trainer.py -v
```
Expected: 12 passed (8 + 4)

- [x] **Step 2: Run full unit suite**

```bash
pytest tests/unit/ -q
```
Expected: 1840+ passed, 0 failed

- [x] **Step 3: Run linter**

```bash
ruff check . --fix && ruff format .
```
Expected: no errors

- [x] **Step 4: Commit spec and plan**

```bash
git add docs/superpowers/specs/2026-06-17-sft-grpo-warmstart-design.md \
        docs/superpowers/plans/2026-06-17-sft-grpo-warmstart.md
git commit -m "docs: add spec and plan for SFT warm-start before GRPO"
```

- [x] **Step 5: Push and open PR**

```bash
git push -u origin <branch>
gh pr create \
  --title "feat(training): SFT warm-start before GRPO" \
  --body "Two-phase training: SFT on thumbs-up sessions + optional JSONL, then GRPO with human feedback. 12 new unit tests, 0 regressions."
```

## Context Boundary

This pack summarizes its linked sources. Consult those documents for complete details; no implementation status is inferred here.
