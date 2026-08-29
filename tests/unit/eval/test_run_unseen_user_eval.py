"""Contracts for the unseen-user evaluation CLI."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from examples.run_unseen_user_eval import _build_parser
from src.model.post_training.eval.cohort import CohortConfig

REPO_ROOT = Path(__file__).resolve().parents[3]


def _run(*extra: str, out: Path) -> str:
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
            *extra,
        ],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
    )
    assert result.returncode == 0, result.stderr
    return out.read_text()


def _row(text: str, name: str) -> list[str]:
    line = next(line for line in text.splitlines() if line.startswith(f"| `{name}`"))
    return [cell.strip() for cell in line.split("|")]


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


def test_cli_defaults_match_the_cohort_defaults():
    """One constant, one value: --users 60 against CohortConfig.num_users 40
    meant the documented command and the library disagreed about the
    population the report describes."""
    defaults = _build_parser().parse_args([])

    assert defaults.users == CohortConfig.num_users
    assert defaults.sessions == CohortConfig.sessions_per_user


def test_cli_report_states_the_planted_effect_sizes(tmp_path: Path):
    text = _run(out=tmp_path / "report.md")

    assert "alignment=2" in text
    assert "behavior_shift=1.5" in text
    assert "instruction_gap=0.25" in text


def test_cli_report_carries_the_command_that_regenerates_it(tmp_path: Path):
    text = _run(out=tmp_path / "report.md")

    assert "python -m examples.run_unseen_user_eval" in text
    assert "--users 24" in text
    assert "--seed 0" in text


def test_cli_allowed_tools_flag_reaches_the_constraint(tmp_path: Path):
    """Pointed at tool names the population never emits, the constraint must
    read 0.0 -- and the flag is the only thing that can point it anywhere
    else."""
    default_text = _run(out=tmp_path / "default.md")
    retargeted = _run(
        "--allowed-tools", "wikipedia,arxiv", out=tmp_path / "retargeted.md"
    )

    assert float(_row(default_text, "tool_calls_parseable")[3]) > 0.0
    assert float(_row(retargeted, "tool_calls_parseable")[3]) == 0.0
    assert float(_row(retargeted, "tool_calls_parseable")[4]) == 0.0


def test_cli_max_search_rounds_flag_reaches_the_constraint(tmp_path: Path):
    default_text = _run(out=tmp_path / "default.md")
    # -1, not 0: a rollout that used zero rounds respects a budget of zero.
    strict = _run("--max-search-rounds", "-1", out=tmp_path / "strict.md")

    assert float(_row(default_text, "round_budget_respected")[3]) > 0.0
    assert float(_row(strict, "round_budget_respected")[3]) == 0.0


def test_cli_labels_the_baseline_as_the_caller_asked(tmp_path: Path):
    text = _run("--baseline-label", "Llama-3.1-70B", out=tmp_path / "report.md")

    assert "## Instruction following (vs Llama-3.1-70B)" in text
