# SFT Warm-Start Before GRPO — Design Spec

**Date:** 2026-06-17
**Status:** Approved

---

## 1. Goals & Success Criteria

### Problem

The GRPO training loop (`examples/run_feedback_grpo.py`) starts from a pre-trained base model with no prior exposure to search-agent behaviour. Cold-starting GRPO from a base model is sample-inefficient — the policy must explore randomly before reward signals can shape it. Users have already rated good sessions (thumbs-up), which contain exactly the behaviour we want the model to learn first.

### Success Criteria

- `load_sft_examples(db_path, jsonl_path=None, min_ratings=1)` merges thumbs-up sessions from `AgenticSearchStore` with optional JSONL rows into `list[SFTExample]`
- `SFTTrainer` in `src/training/sft.py` trains a causal LM with cross-entropy loss on assistant tokens only (non-assistant tokens masked to `−100`)
- `examples/run_sft_grpo.py` runs Phase 1 (SFT) then Phase 2 (GRPO) as a single command, saving an intermediate SFT checkpoint between phases
- Setting `--sft_epochs 0` skips Phase 1 and goes straight to GRPO — no code-path change needed for pure-GRPO runs
- All new unit tests run with no GPU, no live server, no model downloads

### Out of Scope

- LoRA / PEFT adapters
- DPO or preference-pair training
- Auto-triggering from the beat worker
- New dataclasses — `SFTExample` (already in `src/training/sft.py`) is reused as-is

---

## 2. Architecture

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

Signal is preserved across phases: thumbs-up sessions are used in Phase 1 as imitation targets and in Phase 2 as `human_signal=+1.0` via `load_feedback_examples`.

---

## 3. Components

### 3.1 `load_sft_examples` — `src/training/data.py`

New function, added alongside `load_feedback_examples`.

```python
def load_sft_examples(
    db_path: str | Path,
    jsonl_path: str | Path | None = None,
    *,
    min_ratings: int = 1,
) -> list[SFTExample]:
    """Load SFT training examples from thumbs-up sessions and/or a JSONL file.

    DB source: loads thumbs-up sessions from AgenticSearchStore. For each
    session, reconstructs prompt_messages from the first user turn and
    completion from concatenated assistant turns. Sessions with no assistant
    turn are skipped silently.

    JSONL source: each line must be {"question": "...", "response": "..."}.
    Rows missing either key are skipped with a warning.

    Both sources are merged. Raises ValueError if total < min_ratings.
    """
```

**Implementation steps:**
1. Open `AgenticSearchStore(db_path)`, query `retrieval_feedback` for `signal='thumbs_up'`
2. For each thumbs-up session: call `list_chat_messages(session_id)`, extract first `role=user` message as prompt, concatenate all `role=assistant` messages as completion
3. Skip sessions with no assistant turn
4. If `jsonl_path` is provided: read line by line, parse JSON, skip rows missing `question` or `response`, build `SFTExample(prompt_messages=[{"role":"user","content":row["question"]}], completion=row["response"], trajectory_messages=[])`
5. Merge both lists; raise `ValueError(f"Only {n} SFT examples found; need at least {min_ratings}")` if total < threshold

### 3.2 `SFTConfig` and `SFTTrainer` — `src/training/sft.py`

Added to the existing stub (which already defines `SFTExample` and `build_search_sft_example`).

**`SFTConfig`:**
```python
@dataclass(frozen=True)
class SFTConfig:
    epochs: int = 3
    lr: float = 2e-5
    batch_size: int = 4
    max_length: int = 2048
    grad_clip: float = 1.0
```

**`SFTTrainer`:**
```python
class SFTTrainer:
    def __init__(
        self,
        policy: nn.Module,
        tokenizer: Any,
        optimizer: torch.optim.Optimizer,
        config: SFTConfig | None = None,
        device: str | torch.device = "cpu",
    ) -> None: ...

    def train(self, examples: list[SFTExample]) -> list[float]:
        """Train for config.epochs, return per-step loss history."""

    def save(self, output_dir: str | Path) -> None:
        """Save policy and tokenizer in HuggingFace format."""
```

**Tokenization + masking** (inside `train`):
1. For each `SFTExample`, apply tokenizer chat template to `prompt_messages + [{"role":"assistant","content":completion}]`
2. Identify the assistant token span: `labels = input_ids.clone()`; set `labels[:prompt_len] = -100`
3. Feed `(input_ids, attention_mask, labels)` to `policy(**batch, labels=labels)`; loss is cross-entropy over assistant tokens only
4. `loss.backward()` → `clip_grad_norm_` → `optimizer.step()`

### 3.3 CLI script — `examples/run_sft_grpo.py`

**Arguments:**

```
--db_path               SQLite DB path  [default: $AGENTIC_SEARCH_WEB_DB_PATH or :memory:]
--jsonl_path            Optional JSONL file  [default: None]
--model                 HuggingFace model id or local path  [required]
--sft_epochs            Phase 1 epochs; 0 skips SFT  [default: 3]
--sft_output_dir        Intermediate SFT checkpoint  [default: data/checkpoints/sft_warmstart/]
--grpo_output_dir       Final GRPO checkpoint  [default: data/checkpoints/sft_grpo/]
--min_ratings           Abort if fewer examples  [default: 1]
--human_feedback_weight GRPO human signal weight  [default: 0.5]
--num_rollouts          G rollouts per GRPO prompt  [default: 4]
--search_url            Retrieval server URL  [default: http://localhost:8001/retrieve]
--device                cpu | mps | cuda  [default: mps]
```

**Phase logic:**
```python
if args.sft_epochs > 0:
    # Phase 1
    examples = load_sft_examples(args.db_path, args.jsonl_path, min_ratings=args.min_ratings)
    trainer = SFTTrainer(policy, tokenizer, optimizer, SFTConfig(epochs=args.sft_epochs))
    trainer.train(examples)
    trainer.save(args.sft_output_dir)
    # Reload for GRPO
    policy = AutoModelForCausalLM.from_pretrained(args.sft_output_dir).to(device)
    tokenizer = AutoTokenizer.from_pretrained(args.sft_output_dir)

# Phase 2
feedback_examples = load_feedback_examples(args.db_path, min_ratings=args.min_ratings)
grpo_trainer = SearchAgentGRPOTrainer(policy, tokenizer, ...)
await grpo_trainer.step_async(prompts, ground_truths, metadata=metadata)
grpo_trainer.save(args.grpo_output_dir)
```

**Smoke test (no GPU):**
```bash
python3 -m examples.run_sft_grpo \
  --db_path data/feedback.sqlite3 \
  --model Qwen/Qwen2.5-0.5B-Instruct \
  --sft_epochs 1 --min_ratings 1 --device cpu
```

---

## 4. File Map

| Action | File | Change |
|--------|------|--------|
| **Modify** | `src/training/data.py` | Add `load_sft_examples(db_path, jsonl_path, min_ratings)` |
| **Modify** | `src/training/sft.py` | Add `SFTConfig` dataclass + `SFTTrainer` class |
| **Create** | `examples/run_sft_grpo.py` | Two-phase CLI script |
| **Create** | `tests/unit/test_sft_examples.py` | Unit tests for `load_sft_examples` |
| **Create** | `tests/unit/test_sft_trainer.py` | Unit tests for `SFTTrainer` |

No new dataclasses beyond `SFTConfig`. No schema changes.

---

## 5. Testing

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

## 6. Error Handling

| Scenario | Behaviour |
|----------|-----------|
| No thumbs-up sessions and no JSONL path | `ValueError` before model loads |
| JSONL row missing `question` or `response` | Skip row, log warning |
| Session has no assistant turn | Skipped silently |
| `--sft_epochs 0` | Phase 1 skipped; GRPO starts from base `--model` |
| `sft_output_dir` already exists | Checkpoint overwritten (no error) |
| Phase 1 completes, Phase 2 crashes | SFT checkpoint on disk; restart Phase 2 with `--model sft_output_dir --sft_epochs 0` |
| DB path does not exist | `AgenticSearchStore` raises on open — propagates naturally |
