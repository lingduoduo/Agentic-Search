"""Unit tests for the simulated-judge GRPO demo helpers.

These test only the pure helpers and must not load a model or import torch.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from examples.run_bamboogle_grpo_train import (
    cycle_prompt_batches,
    make_judge_fn,
)

REPO_ROOT = Path(__file__).resolve().parents[2]


class _StubJudge:
    """Minimal judge: score is the word count, ground truth ignored."""

    def score(self, answer: str) -> float:
        return float(len(answer.split()))


def test_make_judge_fn_ignores_ground_truth_and_returns_score():
    judge_fn = make_judge_fn(_StubJudge())
    assert judge_fn("two words", "IGNORED GOLD") == 2.0
    assert judge_fn("one two three", "") == 3.0


def test_make_judge_fn_matches_real_judge_pointwise():
    from src.training.judge import SimulatedPreferenceJudge

    judge = SimulatedPreferenceJudge()
    judge_fn = make_judge_fn(judge)
    answer = "Paris is the capital of France."
    assert judge_fn(answer, "whatever") == judge.score(answer)


def test_cycle_prompt_batches_no_wrap():
    prompts = ["a", "b", "c", "d"]
    batches = cycle_prompt_batches(prompts, steps=2, batch_size=2)
    assert batches == [["a", "b"], ["c", "d"]]


def test_cycle_prompt_batches_wraps_when_exhausted():
    prompts = ["a", "b", "c"]
    batches = cycle_prompt_batches(prompts, steps=3, batch_size=2)
    # a,b | c,a | b,c  — continuous cycle across step boundaries
    assert batches == [["a", "b"], ["c", "a"], ["b", "c"]]


def test_cycle_prompt_batches_single_prompt():
    batches = cycle_prompt_batches(["only"], steps=2, batch_size=2)
    assert batches == [["only", "only"], ["only", "only"]]


@pytest.mark.parametrize(
    "prompts,steps,batch",
    [([], 1, 1), (["a"], 0, 1), (["a"], 1, 0)],
)
def test_cycle_prompt_batches_rejects_bad_args(prompts, steps, batch):
    with pytest.raises(ValueError):
        cycle_prompt_batches(prompts, steps=steps, batch_size=batch)


def test_help_flag_prints_usage_and_exits_zero():
    """`--help` must exit cleanly and print the expected usage text."""
    result = subprocess.run(
        [sys.executable, "-m", "examples.run_bamboogle_grpo_train", "--help"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert "usage:" in result.stdout
    assert "--model" in result.stdout


def test_help_does_not_import_torch():
    """Importing the module must not pull in torch (heavy imports stay lazy in `_run`).

    Run in a fresh interpreter (not in-process) because other tests in this
    file import the real judge, which transitively imports torch and would
    make an in-process check meaningless.
    """
    probe = (
        "import sys, importlib; "
        "importlib.import_module('examples.run_bamboogle_grpo_train'); "
        "assert 'torch' not in sys.modules, "
        "sorted(m for m in sys.modules if 'torch' in m)"
    )
    result = subprocess.run(
        [sys.executable, "-c", probe],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, (
        "torch leaked into module import time (stderr below):\n" + result.stderr
    )
