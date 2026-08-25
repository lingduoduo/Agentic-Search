from __future__ import annotations

from importlib import import_module
from importlib.util import find_spec
from pathlib import Path

import pytest


def test_consolidated_trainer_module_owns_the_full_hierarchy():
    pytest.importorskip("torch", exc_type=ImportError)
    from src.model.post_training.grpo.trainers import (
        GRPOTrainer,
        LLMGRPOTrainer,
        SearchAgentGRPOTrainer,
    )

    assert LLMGRPOTrainer in SearchAgentGRPOTrainer.__mro__
    assert GRPOTrainer.__module__ == "src.model.post_training.grpo.trainers"
    assert LLMGRPOTrainer.__module__ == "src.model.post_training.grpo.trainers"
    assert SearchAgentGRPOTrainer.__module__ == "src.model.post_training.grpo.trainers"


@pytest.mark.parametrize(
    "module_name",
    [
        "src.model.post_training.grpo.grpo_trainer",
        "src.model.post_training.grpo.llm_grpo_trainer",
        "src.model.post_training.grpo.search_agent_grpo_trainer",
    ],
)
def test_replaced_trainer_modules_are_removed(module_name: str):
    assert find_spec(module_name) is None


def test_consolidated_training_module_owns_orchestration():
    pytest.importorskip("torch", exc_type=ImportError)
    from src.model.post_training.grpo.training import (
        LocalGRPOController,
        TrainLoopConfig,
        train_loop,
    )

    assert LocalGRPOController.__module__ == "src.model.post_training.grpo.training"
    assert TrainLoopConfig.__module__ == "src.model.post_training.grpo.training"
    assert train_loop.__module__ == "src.model.post_training.grpo.training"


@pytest.mark.parametrize(
    "module_name",
    [
        "src.model.post_training.grpo.controller",
        "src.model.post_training.grpo.train_loop",
    ],
)
def test_replaced_training_modules_are_removed(module_name: str):
    assert find_spec(module_name) is None


def test_generation_owns_its_tensor_helpers():
    pytest.importorskip("torch", exc_type=ImportError)
    from src.model.post_training.grpo.generation import TensorConfig, TensorHelper

    assert TensorConfig.__module__ == "src.model.post_training.grpo.generation"
    assert TensorHelper.__module__ == "src.model.post_training.grpo.generation"


def test_replaced_tensor_helper_module_is_removed():
    assert find_spec("src.model.post_training.grpo.tensor_helper") is None


@pytest.mark.parametrize(
    ("export_name", "module_name"),
    [
        ("LocalGRPOController", "training"),
        ("RolloutResult", "training"),
        ("GRPOTrainer", "trainers"),
        ("Policy", "trainers"),
        ("compute_group_advantages", "trainers"),
        ("grpo_clipped_policy_loss", "trainers"),
        ("make_grpo_trainer", "trainers"),
        ("reverse_kl_penalty", "trainers"),
        ("LLMGRPOConfig", "trainers"),
        ("LLMGRPOTrainer", "trainers"),
        ("LLMRolloutResult", "trainers"),
        ("get_response_log_probs", "trainers"),
        ("SearchAgentGRPOTrainer", "trainers"),
    ],
)
def test_grpo_package_lazy_exports_resolve_to_the_consolidated_modules(
    export_name: str, module_name: str
):
    pytest.importorskip("torch", exc_type=ImportError)

    grpo = import_module("src.model.post_training.grpo")
    implementation_module = import_module(f"src.model.post_training.grpo.{module_name}")

    assert export_name in grpo.__all__
    assert getattr(grpo, export_name) is getattr(implementation_module, export_name)


@pytest.mark.parametrize(
    ("export_name", "module_name"),
    [
        ("TensorConfig", "src.model.post_training.grpo.generation"),
        ("TensorHelper", "src.model.post_training.grpo.generation"),
    ],
)
def test_root_lazy_tensor_helper_exports_resolve_to_generation_by_identity(
    export_name: str, module_name: str
):
    pytest.importorskip("torch", exc_type=ImportError)

    root = import_module("src")
    implementation_module = import_module(module_name)

    assert export_name in root.__all__
    assert getattr(root, export_name) is getattr(implementation_module, export_name)


def test_grpo_package_has_only_the_approved_implementation_modules():
    package_dir = (
        Path(__file__).resolve().parents[2] / "src" / "model" / "post_training" / "grpo"
    )
    actual = {path.name for path in package_dir.glob("*.py")}
    assert actual == {
        "__init__.py",
        "core_algos.py",
        "generation.py",
        "judge.py",
        "plot_rollouts.py",
        "rollouts.py",
        "trainers.py",
        "training.py",
    }


def test_filesystem_layout_checks_run_without_torch():
    import subprocess
    import sys

    program = """
import sys

class _Blocker:
    def find_spec(self, name, path=None, target=None):
        if name == "torch" or name.startswith("torch."):
            raise ImportError("No module named %r (blocked)" % name)
        return None

sys.meta_path.insert(0, _Blocker())

import pytest

raise SystemExit(pytest.main([
    "-q",
    "tests/unit/test_grpo_module_layout.py",
    "-k",
    "replaced or approved_implementation_modules",
]))
"""
    result = subprocess.run(
        [sys.executable, "-c", program],
        capture_output=True,
        text=True,
        cwd=Path(__file__).resolve().parents[2],
    )

    assert result.returncode == 0, result.stderr + result.stdout
    assert "7 passed" in result.stdout
