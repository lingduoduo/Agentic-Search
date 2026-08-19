"""Tests for SFTTrainer."""

from __future__ import annotations

from unittest.mock import MagicMock
import pytest

pytest.importorskip("torch")
import torch

from src.model.post_training.sft.trainer import SFTConfig, SFTExample, SFTTrainer


def _make_example(prompt: str = "Q?", completion: str = "A.") -> SFTExample:
    return SFTExample(
        prompt_messages=[{"role": "user", "content": prompt}],
        completion=completion,
        trajectory_messages=[{"role": "assistant", "content": completion}],
    )


def test_non_assistant_tokens_masked_to_minus_100():
    """Labels for prompt tokens must be -100; only assistant tokens get loss."""
    prompt_ids = [10, 11, 12]  # 3 prompt tokens
    full_ids = [10, 11, 12, 20, 21]  # same prompt + 2 assistant tokens

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

    tokenizer.side_effect = tok_call

    policy = _FakePolicy()
    optimizer = torch.optim.SGD(
        policy.parameters()
        if list(policy.parameters())
        else [torch.nn.Parameter(torch.zeros(1))],
        lr=1e-3,
    )
    trainer = SFTTrainer(
        policy, tokenizer, optimizer, SFTConfig(epochs=1, batch_size=1)
    )
    trainer.train([_make_example()])

    assert len(captured_labels) == 1
    labels = captured_labels[0][0]  # shape (seq_len,)
    # First 3 positions (prompt) must be -100
    assert all(labels[i].item() == -100 for i in range(3))
    # Last 2 positions (assistant) must NOT be -100
    assert all(labels[i].item() != -100 for i in range(3, 5))


def test_loss_history_length_matches_steps():
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

    tokenizer.side_effect = tok_call

    policy = _FakePolicy()
    param = torch.nn.Parameter(torch.zeros(1))
    optimizer = torch.optim.SGD([param], lr=1e-3)
    trainer = SFTTrainer(
        policy, tokenizer, optimizer, SFTConfig(epochs=2, batch_size=1)
    )
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


def test_apply_chat_template_path_masks_prompt_correctly():
    """Primary tokenizer path (apply_chat_template) must mask prompt tokens to -100."""
    prompt_ids = [10, 11, 12]  # 3 prompt tokens
    full_ids = [10, 11, 12, 20, 21]  # prompt + 2 assistant tokens

    captured_labels = []

    class _FakePolicy(torch.nn.Module):
        def forward(self, input_ids=None, attention_mask=None, labels=None, **kw):
            captured_labels.append(labels.clone())
            out = MagicMock()
            out.loss = torch.tensor(1.0, requires_grad=True)
            return out

    def mock_apply_chat_template(messages, *, tokenize=True, return_tensors=None, **kw):
        has_assistant = any(m.get("role") == "assistant" for m in messages)
        ids = full_ids if has_assistant else prompt_ids
        enc = MagicMock()
        enc.__getitem__ = lambda self, k: (
            torch.tensor([ids])
            if k == "input_ids"
            else torch.ones(1, len(ids), dtype=torch.long)
        )
        return enc

    tokenizer = MagicMock()
    tokenizer.pad_token_id = 0
    tokenizer.apply_chat_template = mock_apply_chat_template

    policy = _FakePolicy()
    param = torch.nn.Parameter(torch.zeros(1))
    optimizer = torch.optim.SGD([param], lr=1e-3)
    trainer = SFTTrainer(
        policy, tokenizer, optimizer, SFTConfig(epochs=1, batch_size=1)
    )
    trainer.train([_make_example()])

    assert len(captured_labels) == 1
    labels = captured_labels[0][0]  # shape (seq_len,)
    assert all(labels[i].item() == -100 for i in range(3)), "Prompt tokens must be -100"
    assert all(labels[i].item() != -100 for i in range(3, 5)), (
        "Assistant tokens must not be -100"
    )
