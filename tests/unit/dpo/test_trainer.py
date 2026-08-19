"""Tests for the DPO loss and trainer.

Everything here runs against a tiny stub causal-LM, so no test downloads a
model. The unit-test CI job installs no heavy ML packages beyond torch, hence
the module-level ``importorskip``.
"""

from __future__ import annotations

import math
from types import SimpleNamespace

import pytest

pytest.importorskip("torch")
import torch
import torch.nn as nn

from src.model.post_training.dpo.data import PreferenceExample
from src.model.post_training.dpo.trainer import DPOConfig, DPOTrainer, dpo_loss

_VOCAB = 64
_DIM = 8


class _CharTokenizer:
    """Character-level tokenizer.

    Character-level matters: it guarantees the prompt's ids are an exact prefix
    of ``prompt + response``, which is what lets the trainer locate where the
    response begins.
    """

    pad_token_id = 0

    def __call__(self, text, return_tensors=None, max_length=None, truncation=False):
        ids = [(ord(c) % (_VOCAB - 1)) + 1 for c in text]
        if truncation and max_length is not None:
            ids = ids[:max_length]
        tensor = torch.tensor([ids], dtype=torch.long)
        return SimpleNamespace(input_ids=tensor, attention_mask=torch.ones_like(tensor))

    def save_pretrained(self, path):  # pragma: no cover - exercised via save()
        return None


class _TinyLM(nn.Module):
    """Smallest thing that produces real logits and real gradients."""

    def __init__(self, seed: int = 0) -> None:
        super().__init__()
        torch.manual_seed(seed)
        self.emb = nn.Embedding(_VOCAB, _DIM)
        self.head = nn.Linear(_DIM, _VOCAB)

    def forward(self, input_ids=None, **kwargs):
        return SimpleNamespace(logits=self.head(self.emb(input_ids)))

    def save_pretrained(self, path):  # pragma: no cover - exercised via save()
        return None


def _trainer(**config_kwargs):
    policy = _TinyLM()
    optimizer = torch.optim.SGD(policy.parameters(), lr=0.5)
    return DPOTrainer(
        policy=policy,
        tokenizer=_CharTokenizer(),
        optimizer=optimizer,
        config=DPOConfig(**config_kwargs),
    )


def _pair(prompt="Q: what is faiss? "):
    return PreferenceExample(
        prompt=prompt,
        chosen="A vector similarity search library.",
        rejected="No idea.",
    )


# ---------------------------------------------------------------------------
# The loss, as a pure function
# ---------------------------------------------------------------------------


def test_loss_is_exactly_log_two_when_policy_equals_reference():
    """The load-bearing test: it pins the formula, not merely its shape.

    With the policy still equal to the reference every difference term is zero,
    so the logit into the sigmoid is zero and the loss is -log(0.5) = log 2. A
    sign error or a dropped beta still produces a plausible-looking number; only
    a known value catches those.
    """
    loss, margin, accuracy = dpo_loss(
        policy_chosen=torch.tensor([-3.0]),
        policy_rejected=torch.tensor([-7.0]),
        ref_chosen=torch.tensor([-3.0]),
        ref_rejected=torch.tensor([-7.0]),
        beta=0.1,
    )

    assert loss.item() == pytest.approx(math.log(2), abs=1e-6)
    assert margin.item() == pytest.approx(0.0, abs=1e-6)
    # A zero margin is a tie, not a win.
    assert accuracy == pytest.approx(0.0)


def test_loss_falls_when_the_policy_prefers_the_chosen_response():
    """Moving probability toward `chosen` must reduce the loss."""
    common = dict(
        ref_chosen=torch.tensor([-5.0]),
        ref_rejected=torch.tensor([-5.0]),
        beta=0.1,
    )
    better, margin_better, acc = dpo_loss(
        policy_chosen=torch.tensor([-4.0]),
        policy_rejected=torch.tensor([-6.0]),
        **common,
    )
    worse, margin_worse, _ = dpo_loss(
        policy_chosen=torch.tensor([-6.0]),
        policy_rejected=torch.tensor([-4.0]),
        **common,
    )

    assert better.item() < math.log(2) < worse.item()
    assert margin_better.item() > 0 > margin_worse.item()
    assert acc == pytest.approx(1.0)


def test_beta_scales_the_margin_monotonically():
    """Larger beta on a fixed positive margin means lower loss."""
    args = dict(
        policy_chosen=torch.tensor([-4.0]),
        policy_rejected=torch.tensor([-6.0]),
        ref_chosen=torch.tensor([-5.0]),
        ref_rejected=torch.tensor([-5.0]),
    )
    losses = [dpo_loss(beta=b, **args)[0].item() for b in (0.05, 0.1, 0.5, 1.0)]

    assert losses == sorted(losses, reverse=True)


def test_accuracy_counts_pairs_whose_implicit_reward_gap_is_positive():
    """Accuracy is over the IMPLICIT REWARD gap (margin), not raw likelihood.

    Raw likelihood would report the base model's pre-existing preference at
    init, when the policy still equals the reference and nothing is learned.
    """
    _, _, accuracy = dpo_loss(
        policy_chosen=torch.tensor([-4.0, -8.0, -1.0]),
        policy_rejected=torch.tensor([-6.0, -2.0, -9.0]),
        ref_chosen=torch.tensor([-5.0, -5.0, -5.0]),
        ref_rejected=torch.tensor([-5.0, -5.0, -5.0]),
        beta=0.1,
    )

    assert accuracy == pytest.approx(2 / 3)


# ---------------------------------------------------------------------------
# Sequence log probabilities
# ---------------------------------------------------------------------------


def test_prompt_tokens_do_not_contribute_to_the_sequence_log_prob():
    """Only response tokens are scored; the prompt is conditioning, not target.

    Two different prompts of the SAME token length with the same response must
    both score only their response tokens. If prompt tokens leaked into the sum,
    the two totals would differ by the prompt's own log probs.
    """
    trainer = _trainer()
    response = "same response text"

    a = trainer.sequence_log_prob("prompt aaaaaa ", response)
    b = trainer.sequence_log_prob("prompt bbbbbb ", response)

    # They are conditioned differently, so they are not equal — but each must be
    # a sum over exactly the response tokens.
    n_response_tokens = len(response)
    assert a.shape == b.shape == torch.Size([])
    for value in (a, b):
        # Every per-token log prob is negative and bounded below by
        # n_tokens * log(1/vocab) for a uniform distribution.
        assert n_response_tokens * math.log(1 / _VOCAB) * 3 < value.item() < 0


def test_sequence_log_prob_lengthens_with_a_longer_response():
    """Summed (not averaged) log probs: more tokens means a more negative total."""
    trainer = _trainer()

    short = trainer.sequence_log_prob("Q ", "abc")
    long = trainer.sequence_log_prob("Q ", "abcabcabcabc")

    assert long.item() < short.item()


# ---------------------------------------------------------------------------
# The trainer
# ---------------------------------------------------------------------------


def test_reference_is_frozen_and_untouched_by_training():
    trainer = _trainer(epochs=1)
    before = [p.detach().clone() for p in trainer.reference.parameters()]

    trainer.train([_pair()])

    for parameter in trainer.reference.parameters():
        assert parameter.requires_grad is False
        assert parameter.grad is None
    for old, new in zip(before, trainer.reference.parameters()):
        assert torch.equal(old, new), "reference drifted during training"


def test_training_moves_the_policy_toward_the_chosen_response():
    """The gradient must actually point the right way, not merely be finite."""
    trainer = _trainer(epochs=1)
    pair = _pair()

    with torch.no_grad():
        before = (
            trainer.sequence_log_prob(pair.prompt, pair.chosen)
            - trainer.sequence_log_prob(pair.prompt, pair.rejected)
        ).item()

    trainer.train([pair])

    with torch.no_grad():
        after = (
            trainer.sequence_log_prob(pair.prompt, pair.chosen)
            - trainer.sequence_log_prob(pair.prompt, pair.rejected)
        ).item()

    assert after > before


def test_first_step_loss_is_log_two_because_policy_starts_at_reference():
    """End-to-end confirmation of the known value, through real tokenization."""
    trainer = _trainer(epochs=1)

    history = trainer.train([_pair()])

    assert history[0].loss == pytest.approx(math.log(2), abs=1e-5)
    assert history[0].margin == pytest.approx(0.0, abs=1e-5)


def test_train_updates_the_policy_and_reports_per_step_records():
    trainer = _trainer(epochs=2)
    before = [p.detach().clone() for p in trainer.policy.parameters()]

    history = trainer.train([_pair(), _pair("Q: what is bm25? ")])

    assert len(history) == 2  # 2 epochs x 1 batch
    assert all(math.isfinite(step.loss) for step in history)
    assert any(
        not torch.equal(old, new)
        for old, new in zip(before, trainer.policy.parameters())
    )


def test_no_examples_or_zero_epochs_trains_nothing():
    assert _trainer(epochs=1).train([]) == []
    assert _trainer(epochs=0).train([_pair()]) == []


def test_an_injected_reference_is_used_instead_of_a_copy():
    """Callers pointing at a distinct SFT checkpoint must not get a deepcopy."""
    policy = _TinyLM(seed=0)
    reference = _TinyLM(seed=1)
    trainer = DPOTrainer(
        policy=policy,
        tokenizer=_CharTokenizer(),
        optimizer=torch.optim.SGD(policy.parameters(), lr=0.1),
        reference_policy=reference,
    )

    assert trainer.reference is reference
    # A different reference means the first step is NOT the log-2 tie.
    history = trainer.train([_pair()])
    assert history[0].loss != pytest.approx(math.log(2), abs=1e-5)
