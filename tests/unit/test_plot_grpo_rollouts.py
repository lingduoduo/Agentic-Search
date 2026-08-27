"""Contracts for the relocated GRPO rollout plotting CLI.

The renderer moved out of the library into ``examples`` because it is a
command-line utility, not a training concern.  These tests pin the module's new
home, its HTML output, and the CLI entrypoint.
"""

from __future__ import annotations

import json
import subprocess
import sys
from importlib.util import find_spec
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]

RECORD = {
    "reward": 0.75,
    "advantage": -0.25,
    "group_id": "g0",
    "rollout_index": 2,
    "trajectory": {
        "question": "Who wrote <Dune> & when?",
        "final_answer": "Frank Herbert, 1965",
        "steps": [
            {
                "action_type": "search",
                "action_value": "dune author",
                "observation": "Dune was written by Frank Herbert.",
            }
        ],
    },
}


def test_plotting_module_is_an_example_not_a_library_module():
    assert find_spec("src.model.post_training.grpo.plot_rollouts") is None
    assert find_spec("examples.plot_grpo_rollouts") is not None


def test_rollout_record_renders_escaped_question_answer_and_reward():
    from examples.plot_grpo_rollouts import rollout_record_to_html

    markup = rollout_record_to_html(RECORD)

    assert "&lt;Dune&gt;" in markup
    assert "<Dune>" not in markup
    assert "&amp;" in markup
    assert "Frank Herbert, 1965" in markup
    assert "0.750" in markup
    assert "-0.250" in markup
    assert "dune author" in markup


def test_save_rollout_plot_writes_a_document_for_each_record(tmp_path: Path):
    from examples.plot_grpo_rollouts import save_rollout_plot

    jsonl = tmp_path / "rollouts.jsonl"
    jsonl.write_text("\n".join(json.dumps(RECORD) for _ in range(3)) + "\n")
    out = tmp_path / "trace.html"

    count = save_rollout_plot(jsonl, out, max_records=None, title="Trace")

    assert count == 3
    markup = out.read_text()
    assert markup.count('class="record"') == 3
    assert "<title>Trace</title>" in markup


def test_cli_renders_html_from_a_jsonl_file(tmp_path: Path):
    jsonl = tmp_path / "rollouts.jsonl"
    jsonl.write_text(json.dumps(RECORD) + "\n")
    out = tmp_path / "trace.html"

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "examples.plot_grpo_rollouts",
            "--jsonl",
            str(jsonl),
            "--out",
            str(out),
        ],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
    )

    assert result.returncode == 0, result.stderr
    assert "Rendered 1 rollout record(s)" in result.stdout
    markup = out.read_text()
    assert "&lt;Dune&gt;" in markup
    assert "Frank Herbert, 1965" in markup


def test_plotting_cli_runs_without_torch():
    program = """
import sys

class _Blocker:
    def find_spec(self, name, path=None, target=None):
        if name == "torch" or name.startswith("torch."):
            raise ImportError("No module named %r (blocked)" % name)
        return None

sys.meta_path.insert(0, _Blocker())

import examples.plot_grpo_rollouts as plot

assert plot.rollout_record_to_html({"question": "q", "answer": "a"})
"""
    result = subprocess.run(
        [sys.executable, "-c", program],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
    )

    assert result.returncode == 0, result.stderr + result.stdout


@pytest.mark.parametrize("flag", ["--jsonl", "--out"])
def test_cli_requires_its_input_and_output_flags(flag: str):
    from examples.plot_grpo_rollouts import _build_parser

    with pytest.raises(SystemExit):
        _build_parser().parse_args([flag, "value"])
