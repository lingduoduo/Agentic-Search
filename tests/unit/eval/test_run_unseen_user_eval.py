"""Contracts for the unseen-user evaluation CLI."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]


def test_cli_writes_a_report_stating_it_is_simulated(tmp_path: Path):
    out = tmp_path / "report.md"

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "examples.run_unseen_user_eval",
            "--users",
            "24",
            "--sessions",
            "6",
            "--resamples",
            "100",
            "--power-replications",
            "0",
            "--output",
            str(out),
        ],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
    )

    assert result.returncode == 0, result.stderr
    text = out.read_text()
    assert "simulated" in text.lower()
    assert "held-out users" in text
    assert "Conversion alignment" in text


def test_cli_includes_power_when_replications_are_requested(tmp_path: Path):
    out = tmp_path / "report.md"

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "examples.run_unseen_user_eval",
            "--users",
            "20",
            "--sessions",
            "5",
            "--resamples",
            "50",
            "--power-replications",
            "3",
            "--output",
            str(out),
        ],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
    )

    assert result.returncode == 0, result.stderr
    assert "Achieved power" in out.read_text()


def test_cli_fails_loudly_on_an_out_of_range_holdout():
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "examples.run_unseen_user_eval",
            "--users",
            "10",
            "--sessions",
            "4",
            "--resamples",
            "20",
            "--power-replications",
            "0",
            "--holdout",
            "1.5",
        ],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
    )

    assert result.returncode != 0
    assert "holdout_fraction" in result.stderr


def test_cli_runs_without_torch():
    program = """
import sys

class _Blocker:
    def find_spec(self, name, path=None, target=None):
        if name == "torch" or name.startswith("torch."):
            raise ImportError("No module named %r (blocked)" % name)
        return None

sys.meta_path.insert(0, _Blocker())

import examples.run_unseen_user_eval  # noqa: F401
from src.model.post_training.eval import evaluate_unseen_users  # noqa: F401

assert "torch" not in sys.modules
print("eval is torch-free")
"""
    result = subprocess.run(
        [sys.executable, "-c", program],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
    )

    assert result.returncode == 0, result.stderr + result.stdout
    assert "eval is torch-free" in result.stdout
