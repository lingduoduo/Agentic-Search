# SFT Warm-Start Before GRPO — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a two-phase training script that runs supervised fine-tuning (SFT) on thumbs-up sessions + optional JSONL before handing off to the existing GRPO loop, so the policy starts from demonstrated behaviour rather than random exploration.

**Architecture:** `load_sft_examples()` merges thumbs-up chat sessions from SQLite with an optional `{question, response}` JSONL file into `list[SFTExample]`. `SFTTrainer` trains cross-entropy loss on assistant tokens only (prompt masked to `-100`) for N epochs and saves an HF checkpoint. `run_sft_grpo.py` loads that checkpoint into `SearchAgentGRPOTrainer.from_pretrained()` and runs Phase 2 GRPO — or skips Phase 1 entirely when `--sft_epochs 0`.

**Tech Stack:** Python 3.12, PyTorch, HuggingFace Transformers, SQLite via `AgenticSearchStore`, pytest (no GPU required for tests).

---

## File Map

| Action | File | Responsibility |
|--------|------|----------------|
| Modify | `src/training/data.py` | Add `load_sft_examples()` |
| Modify | `src/training/sft.py` | Add `SFTConfig` + `SFTTrainer` |
| Create | `examples/run_sft_grpo.py` | Two-phase CLI entry point |
| Create | `tests/unit/test_sft_examples.py` | Unit tests for `load_sft_examples` |
| Create | `tests/unit/test_sft_trainer.py` | Unit tests for `SFTTrainer` |

---

## Background: key types to know

**`SFTExample`** — already defined in `src/training/sft.py`:
```python
@dataclass(frozen=True)
class SFTExample:
    prompt_messages: list[dict[str, Any]]   # e.g. [{"role":"user","content":"..."}]
    completion: str                          # full assistant output (trace + answer)
    trajectory_messages: list[dict[str, Any]]  # empty list is fine for our use case
```

**`AgenticSearchStore`** — SQLite wrapper at `src/internal/db/store.py`. Relevant API:
```python
store.create_chat_session(session_id=..., user_id=None)
store.add_chat_message(session_id, role, content)
store.save_retrieval_feedback(session_id, signal)  # signal = "thumbs_up" | "thumbs_down"
store.list_chat_messages(session_id) -> list[ChatMessageRecord]
# ChatMessageRecord has .role (str) and .content (str)
# Raw query for feedback: store._conn.execute("SELECT session_id, signal FROM retrieval_feedback")
```

**`SearchAgentGRPOTrainer.from_pretrained`** — factory in `src/training/ppo/search_agent_grpo_trainer.py`:
```python
SearchAgentGRPOTrainer.from_pretrained(
    model_name_or_path,   # str path to HF checkpoint
    judge_fn,
    loop_factory,
    reward_fn=...,
    config=LLMGRPOConfig(num_rollouts=4),
    device="cpu",
)
```
This handles `reference_policy = copy.deepcopy(policy)` and `optimizer` internally.

---

## Task 1: `load_sft_examples` in `src/training/data.py`

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
    _seed_store(db, [{"session_id": "s1", "signal": "thumbs_down",
                       "user_msg": "Q?", "assistant_msg": "A."}])
    with pytest.raises(ValueError, match="Only 0 SFT examples"):
        load_sft_examples(db, min_ratings=1)


def test_session_without_assistant_turn_skipped(tmp_path):
    db = str(tmp_path / "t.sqlite3")
    _seed_store(db, [{"session_id": "s1", "signal": "thumbs_up", "user_msg": "Q?"}])
    with pytest.raises(ValueError, match="Only 0 SFT examples"):
        load_sft_examples(db, min_ratings=1)


def test_jsonl_row_becomes_sft_example(tmp_path):
    db = str(tmp_path / "t.sqlite3")
    AgenticSearchStore(db).__exit__(None, None, None)  # init schema only
    jsonl = tmp_path / "pairs.jsonl"
    jsonl.write_text(json.dumps({"question": "Q?", "response": "R."}) + "\n")
    examples = load_sft_examples(db, jsonl_path=str(jsonl), min_ratings=1)
    assert len(examples) == 1
    assert examples[0].prompt_messages == [{"role": "user", "content": "Q?"}]
    assert examples[0].completion == "R."


def test_jsonl_row_missing_response_skipped(tmp_path):
    db = str(tmp_path / "t.sqlite3")
    AgenticSearchStore(db).__exit__(None, None, None)
    jsonl = tmp_path / "pairs.jsonl"
    jsonl.write_text(json.dumps({"question": "Q?"}) + "\n")
    with pytest.raises(ValueError, match="Only 0 SFT examples"):
        load_sft_examples(db, jsonl_path=str(jsonl), min_ratings=1)


def test_db_and_jsonl_merged(tmp_path):
    db = str(tmp_path / "t.sqlite3")
    _seed_store(db, [{"session_id": "s1", "signal": "thumbs_up",
                       "user_msg": "Q1?", "assistant_msg": "A1."}])
    jsonl = tmp_path / "pairs.jsonl"
    jsonl.write_text(json.dumps({"question": "Q2?", "response": "R2."}) + "\n")
    examples = load_sft_examples(db, jsonl_path=str(jsonl), min_ratings=2)
    assert len(examples) == 2


def test_raises_when_below_min_ratings(tmp_path):
    db = str(tmp_path / "t.sqlite3")
    _seed_store(db, [{"session_id": "s1", "signal": "thumbs_up",
                       "user_msg": "Q?", "assistant_msg": "A."}])
    with pytest.raises(ValueError, match="Only 1 SFT examples found; need at least 3"):
        load_sft_examples(db, min_ratings=3)


def test_returns_sft_example_instances(tmp_path):
    db = str(tmp_path / "t.sqlite3")
    _seed_store(db, [{"session_id": "s1", "signal": "thumbs_up",
                       "user_msg": "Q?", "assistant_msg": "A."}])
    examples = load_sft_examples(db, min_ratings=1)
    assert all(isinstance(ex, SFTExample) for ex in examples)
```

- [x] **Step 2: Run tests — verify they fail**

```bash
pytest tests/unit/test_sft_examples.py -v
```
Expected: all 8 fail with `ImportError: cannot import name 'load_sft_examples'`

- [x] **Step 3: Add `load_sft_examples` to `src/training/data.py`**

Add `import json` and `import logging` to the imports block (after `from pathlib import Path`), then append after `load_feedback_examples`:

```python
_logger = logging.getLogger(__name__)


def load_sft_examples(
    db_path: str | Path,
    jsonl_path: str | Path | None = None,
    *,
    min_ratings: int = 1,
) -> list["SFTExample"]:
    """Load SFT training examples from thumbs-up sessions and/or a JSONL file.

    DB source: thumbs-up sessions only. Prompt = first user message.
    Completion = first assistant message content. Sessions with no assistant
    turn are skipped silently.

    JSONL source: each line must be {"question": "...", "response": "..."}.
    Rows missing either key are skipped with a warning.

    Raises ValueError if total count < min_ratings.
    """
    from src.internal.db import AgenticSearchStore
    from src.training.sft import SFTExample

    examples: list[SFTExample] = []

    # --- DB source: thumbs-up sessions ---
    with AgenticSearchStore(str(db_path)) as store:
        rows = store._conn.execute(
            "SELECT session_id, signal FROM retrieval_feedback WHERE signal = 'thumbs_up'"
        ).fetchall()
        for row in rows:
            session_id = row["session_id"]
            if not session_id:
                continue
            messages = store.list_chat_messages(session_id)
            first_user = next((m.content for m in messages if m.role == "user"), None)
            first_assistant = next(
                (m.content for m in messages if m.role == "assistant"), None
            )
            if not first_user or not first_assistant:
                continue
            examples.append(
                SFTExample(
                    prompt_messages=[{"role": "user", "content": first_user}],
                    completion=first_assistant,
                    trajectory_messages=[],
                )
            )

    # --- JSONL source ---
    if jsonl_path is not None:
        with open(jsonl_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    _logger.warning("Skipping malformed JSONL line: %r", line[:80])
                    continue
                question = row.get("question")
                response = row.get("response")
                if not question or not response:
                    _logger.warning("Skipping JSONL row missing question/response: %r", row)
                    continue
                examples.append(
                    SFTExample(
                        prompt_messages=[{"role": "user", "content": question}],
                        completion=response,
                        trajectory_messages=[],
                    )
                )

    if len(examples) < min_ratings:
        raise ValueError(
            f"Only {len(examples)} SFT examples found; need at least {min_ratings}"
        )
    return examples
```

- [x] **Step 4: Run tests — verify they pass**

```bash
pytest tests/unit/test_sft_examples.py -v
```
Expected: 8 passed

- [x] **Step 5: Commit**

```bash
git add src/training/data.py tests/unit/test_sft_examples.py
git commit -m "feat(training): add load_sft_examples for SFT warm-start"
```

---

## Task 2: `SFTConfig` and `SFTTrainer` in `src/training/sft.py`

**Files:**
- Modify: `src/training/sft.py`
- Test: `tests/unit/test_sft_trainer.py`

### Background

The tokenization strategy:
1. Build full message list: `example.prompt_messages + [{"role": "assistant", "content": example.completion}]`
2. Tokenize the prompt-only messages to get `prompt_len`
3. Tokenize the full sequence (both parts together)
4. `labels = input_ids.clone(); labels[:, :prompt_len] = -100`
5. Feed `(input_ids, attention_mask, labels)` to the model — HF causal LMs compute cross-entropy over `labels` when `labels` is passed

If the tokenizer has `apply_chat_template`, use it. Otherwise fall back to joining message contents with newlines.

- [x] **Step 1: Write the failing tests**

Create `tests/unit/test_sft_trainer.py`:

```python
"""Tests for SFTTrainer."""
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, call, patch
import pytest

pytest.importorskip("torch")
import torch

from src.training.sft import SFTConfig, SFTExample, SFTTrainer


def _make_example(prompt: str = "Q?", completion: str = "A.") -> SFTExample:
    return SFTExample(
        prompt_messages=[{"role": "user", "content": prompt}],
        completion=completion,
        trajectory_messages=[],
    )


def _make_tokenizer(prompt_ids: list[int], full_ids: list[int]):
    """Minimal tokenizer mock that returns different ids for prompt vs full."""
    tok = MagicMock()
    tok.pad_token_id = 0
    tok.eos_token_id = 2

    def side_effect(text_or_messages, **kwargs):
        # Detect prompt-only call by checking for assistant content absence
        if isinstance(text_or_messages, str) and "A." not in text_or_messages:
            ids = prompt_ids
        elif isinstance(text_or_messages, list) and not any(
            m.get("role") == "assistant" for m in text_or_messages
        ):
            ids = prompt_ids
        else:
            ids = full_ids
        t = torch.tensor([ids])
        result = MagicMock()
        result.input_ids = t
        result.attention_mask = torch.ones_like(t)
        result.__getitem__ = lambda self, k: t if k == "input_ids" else torch.ones_like(t)
        return result

    tok.side_effect = side_effect
    tok.__call__ = side_effect
    tok.apply_chat_template = None  # force fallback path
    return tok


def test_non_assistant_tokens_masked_to_minus_100(tmp_path):
    """Labels for prompt tokens must be -100; only assistant tokens get loss."""
    prompt_ids = [10, 11, 12]       # 3 prompt tokens
    full_ids = [10, 11, 12, 20, 21] # same prompt + 2 assistant tokens

    captured_labels = []

    class _FakePolicy(torch.nn.Module):
        def forward(self, input_ids=None, attention_mask=None, labels=None, **kw):
            captured_labels.append(labels.clone())
            loss = torch.tensor(1.0, requires_grad=True)
            out = MagicMock()
            out.loss = loss
            return out

    tokenizer = MagicMock()
    tokenizer.pad_token_id = 0
    tokenizer.apply_chat_template = None

    call_count = [0]

    def tok_call(msgs_or_text, **kwargs):
        call_count[0] += 1
        # First call = prompt only (no assistant role), second = full sequence
        if call_count[0] % 2 == 1:
            ids = prompt_ids
        else:
            ids = full_ids
        t = torch.tensor([ids])
        r = MagicMock()
        r.input_ids = t
        r.attention_mask = torch.ones_like(t)
        return r

    tokenizer.__call__ = tok_call

    policy = _FakePolicy()
    optimizer = torch.optim.SGD(policy.parameters() if list(policy.parameters()) else [torch.nn.Parameter(torch.zeros(1))], lr=1e-3)
    trainer = SFTTrainer(policy, tokenizer, optimizer, SFTConfig(epochs=1, batch_size=1))
    trainer.train([_make_example()])

    assert len(captured_labels) == 1
    labels = captured_labels[0][0]  # shape (seq_len,)
    # First 3 positions (prompt) must be -100
    assert all(labels[i].item() == -100 for i in range(3))
    # Last 2 positions (assistant) must NOT be -100
    assert all(labels[i].item() != -100 for i in range(3, 5))


def test_loss_history_length_matches_steps(tmp_path):
    """train() returns one loss value per gradient step."""

    class _FakePolicy(torch.nn.Module):
        def forward(self, **kw):
            out = MagicMock()
            out.loss = torch.tensor(1.0, requires_grad=True)
            return out

    tokenizer = MagicMock()
    tokenizer.pad_token_id = 0
    tokenizer.apply_chat_template = None
    ids = [1, 2, 3, 4]
    call_count = [0]

    def tok_call(x, **kw):
        call_count[0] += 1
        t = torch.tensor([[1, 2]] if call_count[0] % 2 == 1 else [ids])
        r = MagicMock()
        r.input_ids = t
        r.attention_mask = torch.ones_like(t)
        return r

    tokenizer.__call__ = tok_call

    policy = _FakePolicy()
    param = torch.nn.Parameter(torch.zeros(1))
    optimizer = torch.optim.SGD([param], lr=1e-3)
    trainer = SFTTrainer(policy, tokenizer, optimizer, SFTConfig(epochs=2, batch_size=1))
    history = trainer.train([_make_example(), _make_example()])
    # 2 examples, batch_size=1, epochs=2 → 4 steps
    assert len(history) == 4


def test_sft_epochs_zero_returns_empty_history():
    """SFTConfig(epochs=0) must skip training entirely."""
    policy = MagicMock()
    tokenizer = MagicMock()
    tokenizer.pad_token_id = 0
    optimizer = MagicMock()
    trainer = SFTTrainer(policy, tokenizer, optimizer, SFTConfig(epochs=0))
    history = trainer.train([_make_example()])
    assert history == []
    policy.forward.assert_not_called()


def test_save_calls_save_pretrained(tmp_path):
    policy = MagicMock()
    tokenizer = MagicMock()
    optimizer = MagicMock()
    trainer = SFTTrainer(policy, tokenizer, optimizer)
    trainer.save(tmp_path)
    policy.save_pretrained.assert_called_once_with(tmp_path)
    tokenizer.save_pretrained.assert_called_once_with(tmp_path)
```

- [x] **Step 2: Run tests — verify they fail**

```bash
pytest tests/unit/test_sft_trainer.py -v
```
Expected: fail with `ImportError: cannot import name 'SFTConfig'` or `'SFTTrainer'`

- [x] **Step 3: Add `SFTConfig` and `SFTTrainer` to `src/training/sft.py`**

Add these imports at the top of `src/training/sft.py`:

```python
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

_logger = logging.getLogger(__name__)
```

Then append after `build_search_sft_example`:

```python
@dataclass(frozen=True)
class SFTConfig:
    epochs: int = 3
    lr: float = 2e-5
    batch_size: int = 4
    max_length: int = 2048
    grad_clip: float = 1.0


class SFTTrainer:
    """Supervised fine-tuning trainer for search-agent trajectories.

    Applies cross-entropy loss on assistant tokens only. Prompt tokens
    (system + user + tool-result) are masked to -100 so they do not
    contribute to the loss.
    """

    def __init__(
        self,
        policy: nn.Module,
        tokenizer: Any,
        optimizer: torch.optim.Optimizer,
        config: SFTConfig | None = None,
        device: str | torch.device = "cpu",
    ) -> None:
        self.policy = policy
        self.tokenizer = tokenizer
        self.optimizer = optimizer
        self.config = config or SFTConfig()
        self.device = torch.device(device)

    def _tokenize_example(self, example: SFTExample) -> dict[str, torch.Tensor]:
        """Return input_ids, attention_mask, and labels for one example."""
        cfg = self.config

        full_messages = list(example.prompt_messages) + [
            {"role": "assistant", "content": example.completion}
        ]

        # Tokenize prompt only to find where assistant tokens begin.
        if callable(getattr(self.tokenizer, "apply_chat_template", None)):
            prompt_ids = self.tokenizer.apply_chat_template(
                example.prompt_messages,
                tokenize=True,
                add_generation_prompt=True,
            )
            prompt_len = len(prompt_ids)
            full_enc = self.tokenizer.apply_chat_template(
                full_messages,
                tokenize=True,
                return_tensors="pt",
                max_length=cfg.max_length,
                truncation=True,
            )
            input_ids = full_enc if isinstance(full_enc, torch.Tensor) else full_enc["input_ids"]
            attention_mask = torch.ones_like(input_ids)
        else:
            # Fallback: join content strings, track prompt length by char count.
            prompt_text = "\n".join(m["content"] for m in example.prompt_messages)
            full_text = prompt_text + "\n" + example.completion
            prompt_enc = self.tokenizer(prompt_text, return_tensors="pt")
            full_enc = self.tokenizer(
                full_text,
                return_tensors="pt",
                max_length=cfg.max_length,
                truncation=True,
            )
            prompt_len = prompt_enc.input_ids.shape[1]
            input_ids = full_enc.input_ids
            attention_mask = full_enc.attention_mask

        labels = input_ids.clone()
        labels[:, :prompt_len] = -100  # mask prompt tokens
        return {
            "input_ids": input_ids.to(self.device),
            "attention_mask": attention_mask.to(self.device),
            "labels": labels.to(self.device),
        }

    def train(self, examples: list[SFTExample]) -> list[float]:
        """Train for config.epochs on examples. Returns per-step loss history."""
        if self.config.epochs == 0 or not examples:
            return []

        self.policy.train()
        history: list[float] = []

        for _epoch in range(self.config.epochs):
            for i in range(0, len(examples), self.config.batch_size):
                batch = examples[i : i + self.config.batch_size]
                self.optimizer.zero_grad()
                batch_loss = torch.tensor(0.0, device=self.device)
                for example in batch:
                    enc = self._tokenize_example(example)
                    output = self.policy(**enc)
                    batch_loss = batch_loss + output.loss
                batch_loss = batch_loss / len(batch)
                batch_loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.policy.parameters(), self.config.grad_clip
                )
                self.optimizer.step()
                history.append(float(batch_loss))
        return history

    def save(self, output_dir: str | Path) -> None:
        """Save policy and tokenizer in HuggingFace format."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        self.policy.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)
```

- [x] **Step 4: Run SFT trainer tests**

```bash
pytest tests/unit/test_sft_trainer.py -v
```
Expected: 4 passed

- [x] **Step 5: Run full unit suite — no regressions**

```bash
pytest tests/unit/ -q
```
Expected: 1830+ passed

- [x] **Step 6: Commit**

```bash
git add src/training/sft.py tests/unit/test_sft_trainer.py
git commit -m "feat(training): add SFTConfig and SFTTrainer for warm-start"
```

---

## Task 3: CLI script `examples/run_sft_grpo.py`

**Files:**
- Create: `examples/run_sft_grpo.py`

### Background

`SearchAgentGRPOTrainer.from_pretrained` is defined in `src/training/ppo/search_agent_grpo_trainer.py`. It handles `reference_policy` and `optimizer` internally. When Phase 1 completes, we reload the policy from the SFT checkpoint using the same factory instead of passing the in-memory policy object — this ensures the reference policy is a clean deep copy of the SFT weights, not the base model.

`load_feedback_examples` (from `src/training/data.py`) is reused for Phase 2 to get `prompts`, `ground_truths`, and `metadata` with `human_signal`.

- [x] **Step 1: Create the script**

```python
"""Two-phase training: SFT warm-start followed by GRPO with human feedback.

Phase 1 (SFT): imitate thumbs-up sessions + optional JSONL pairs, save checkpoint.
Phase 2 (GRPO): load SFT checkpoint, run on-policy rollouts with human feedback signal.

Skip Phase 1 with --sft_epochs 0 to run pure GRPO from a base model.

Usage::

    python3 -m examples.run_sft_grpo \\
      --db_path data/feedback.sqlite3 \\
      --model Qwen/Qwen2.5-0.5B-Instruct \\
      --sft_epochs 1 --min_ratings 1 --device cpu
"""
from __future__ import annotations

import argparse
import asyncio
import os
from pathlib import Path


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="SFT warm-start + GRPO fine-tuning")
    p.add_argument(
        "--db_path",
        default=os.environ.get("AGENTIC_SEARCH_WEB_DB_PATH", ":memory:"),
        help="SQLite DB path",
    )
    p.add_argument("--jsonl_path", default=None, help="Optional JSONL SFT pairs file")
    p.add_argument("--model", required=True, help="HuggingFace model id or local path")
    p.add_argument("--sft_epochs", type=int, default=3, help="0 to skip SFT phase")
    p.add_argument(
        "--sft_output_dir",
        default="data/checkpoints/sft_warmstart/",
        help="Intermediate SFT checkpoint directory",
    )
    p.add_argument(
        "--grpo_output_dir",
        default="data/checkpoints/sft_grpo/",
        help="Final GRPO checkpoint directory",
    )
    p.add_argument("--min_ratings", type=int, default=1)
    p.add_argument("--human_feedback_weight", type=float, default=0.5)
    p.add_argument("--num_rollouts", type=int, default=4)
    p.add_argument("--search_url", default="http://localhost:8001/retrieve")
    p.add_argument("--device", default="mps", choices=["cpu", "mps", "cuda"])
    return p.parse_args()


async def _train(args: argparse.Namespace) -> None:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from src.agents.search import SearchAgentLoop
    from src.training.data import load_feedback_examples, load_sft_examples
    from src.training.grpo import GRPOAdvantageConfig
    from src.training.ppo.llm_grpo_trainer import LLMGRPOConfig
    from src.training.ppo.search_agent_grpo_trainer import SearchAgentGRPOTrainer
    from src.training.reward import SearchRewardConfig, SearchRewardFunction
    from src.training.reward import simple_sparse_correctness_reward
    from src.training.sft import SFTConfig, SFTTrainer

    device = torch.device(args.device)

    # ── Phase 1: SFT ─────────────────────────────────────────────────────────
    grpo_model_path = args.model  # default: start GRPO from base model

    if args.sft_epochs > 0:
        print(f"[Phase 1] Loading SFT examples …")
        sft_examples = load_sft_examples(
            args.db_path, args.jsonl_path, min_ratings=args.min_ratings
        )
        print(f"  {len(sft_examples)} SFT examples loaded")

        tokenizer = AutoTokenizer.from_pretrained(args.model)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = tokenizer.eos_token_id
        policy = AutoModelForCausalLM.from_pretrained(args.model).to(device)
        optimizer = torch.optim.AdamW(policy.parameters(), lr=2e-5)

        trainer = SFTTrainer(
            policy, tokenizer, optimizer,
            SFTConfig(epochs=args.sft_epochs),
            device=device,
        )
        history = trainer.train(sft_examples)
        print(f"  SFT complete. Final loss: {history[-1]:.4f}" if history else "  SFT complete.")
        trainer.save(args.sft_output_dir)
        print(f"  SFT checkpoint saved to {args.sft_output_dir}")
        grpo_model_path = args.sft_output_dir  # GRPO starts from SFT checkpoint

    # ── Phase 2: GRPO ────────────────────────────────────────────────────────
    print(f"[Phase 2] Loading feedback examples for GRPO …")
    feedback_examples = load_feedback_examples(
        args.db_path, min_ratings=args.min_ratings
    )
    print(f"  {len(feedback_examples)} rated sessions loaded")

    prompts = [ex.question for ex in feedback_examples]
    ground_truths = [ex.ground_truth for ex in feedback_examples]
    metadata = [dict(ex.metadata) for ex in feedback_examples]

    reward_fn = SearchRewardFunction(
        SearchRewardConfig(
            human_feedback_weight=args.human_feedback_weight,
            correctness_weight=0.0,
        )
    )

    def loop_factory():
        return SearchAgentLoop(search_url=args.search_url)

    grpo_trainer = SearchAgentGRPOTrainer.from_pretrained(
        grpo_model_path,
        judge_fn=simple_sparse_correctness_reward,
        loop_factory=loop_factory,
        reward_fn=reward_fn,
        config=LLMGRPOConfig(num_rollouts=args.num_rollouts),
        advantage_config=GRPOAdvantageConfig(),
        device=args.device,
    )

    print("  Running GRPO step …")
    metrics = await grpo_trainer.step_async(prompts, ground_truths, metadata=metadata)
    print(f"  GRPO metrics: {metrics}")

    grpo_output_dir = Path(args.grpo_output_dir)
    grpo_output_dir.mkdir(parents=True, exist_ok=True)
    grpo_trainer.policy.save_pretrained(grpo_output_dir)
    grpo_trainer.tokenizer.save_pretrained(grpo_output_dir)
    print(f"  GRPO checkpoint saved to {grpo_output_dir}")


def main() -> None:
    args = _parse_args()
    asyncio.run(_train(args))


if __name__ == "__main__":
    main()
```

- [x] **Step 2: Verify import works**

```bash
python3 -c "from examples.run_sft_grpo import _parse_args; print('import ok')"
```
Expected: `import ok`

- [x] **Step 3: Commit**

```bash
git add examples/run_sft_grpo.py
git commit -m "feat(training): add run_sft_grpo two-phase CLI script"
```

---

## Task 4: Final integration check + commit spec and plan

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
