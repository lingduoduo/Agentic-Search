"""Contracts for the GRPO/reward benchmark harness.

The harness is diagnostic: it records evidence for optimization decisions.
These tests pin its measurement discipline and its reproducibility metadata,
never an absolute time — a wall-clock assertion would be flaky on any shared
machine and would make the suite fail for reasons unrelated to correctness.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

pytest.importorskip("torch")

from examples.benchmark_grpo_optimization import (  # noqa: E402
    REQUIRED_CASES,
    BenchmarkResult,
    format_markdown_report,
    measure_case,
    run_smoke_benchmarks,
)


def test_measure_case_runs_warmups_outside_recorded_samples():
    calls = 0

    def operation():
        nonlocal calls
        calls += 1

    result = measure_case("operation", operation, warmup=2, iterations=5)

    # 2 warmups + 5 timed samples + 1 allocation pass.
    assert calls == 8
    assert result.name == "operation"
    assert len(result.samples_ns) == 5
    assert result.median_ns > 0
    assert result.iterations == 5
    assert result.warmup == 2


def test_measure_case_records_peak_python_allocation():
    def operation():
        return [0] * 200_000

    result = measure_case("allocating", operation, warmup=1, iterations=3)

    assert result.peak_bytes > 100_000


def test_measure_case_rejects_a_zero_iteration_run():
    with pytest.raises(ValueError, match="iterations"):
        measure_case("operation", lambda: None, warmup=0, iterations=0)


def test_benchmark_result_median_is_the_middle_sample():
    result = BenchmarkResult(
        name="fixed",
        warmup=0,
        iterations=3,
        samples_ns=(30, 10, 20),
        peak_bytes=0,
    )

    assert result.median_ns == 20


def test_smoke_benchmarks_carry_reproducibility_metadata():
    payload = run_smoke_benchmarks(warmup=1, iterations=3)

    assert payload["python_version"]
    assert payload["torch_version"]
    assert payload["iterations"] == 3
    assert payload["warmup"] == 1
    assert payload["fixtures"]
    assert {case["name"] for case in payload["cases"]} >= set(REQUIRED_CASES)
    for case in payload["cases"]:
        assert case["median_ns"] > 0
        assert case["iterations"] == 3


def test_smoke_benchmarks_cover_the_paths_this_work_optimizes():
    payload = run_smoke_benchmarks(warmup=1, iterations=2)
    names = {case["name"] for case in payload["cases"]}

    assert {
        "group_advantages",
        "group_advantages_normalized",
        "group_advantages_tensor",
        "reward_components_shaped",
        "reward_components_sparse",
        "reward_batch",
        "reward_token_advantages",
        "training_batch_assembly",
        "response_log_probs",
    } <= names


def test_smoke_benchmarks_are_json_serializable():
    payload = run_smoke_benchmarks(warmup=1, iterations=2)

    assert json.loads(json.dumps(payload))["cases"]


def test_markdown_report_renders_one_row_per_case():
    payload = run_smoke_benchmarks(warmup=1, iterations=2)

    report = format_markdown_report(payload, heading="Baseline")

    assert "## Baseline" in report
    for case in payload["cases"]:
        assert case["name"] in report
    assert "median" in report.lower()


def test_cli_appends_a_report_section_to_an_existing_file(tmp_path: Path):
    import subprocess
    import sys

    out = tmp_path / "baseline.md"
    out.write_text("# Existing\n")
    repo_root = Path(__file__).resolve().parents[2]

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "examples.benchmark_grpo_optimization",
            "--warmup",
            "1",
            "--iterations",
            "2",
            "--output",
            str(out),
            "--heading",
            "After reward optimization",
        ],
        capture_output=True,
        text=True,
        cwd=repo_root,
    )

    assert result.returncode == 0, result.stderr
    text = out.read_text()
    assert text.startswith("# Existing")
    assert "## After reward optimization" in text
    assert "group_advantages" in text
