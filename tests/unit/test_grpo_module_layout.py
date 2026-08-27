from __future__ import annotations

from importlib import import_module
from importlib.util import find_spec
from pathlib import Path

import pytest


def test_consolidated_trainer_module_owns_the_full_hierarchy():
    pytest.importorskip("torch", exc_type=ImportError)
    from src.model.post_training.grpo.training import (
        GRPOTrainer,
        LLMGRPOTrainer,
        SearchAgentGRPOTrainer,
    )

    assert LLMGRPOTrainer in SearchAgentGRPOTrainer.__mro__
    assert GRPOTrainer.__module__ == "src.model.post_training.grpo.training"
    assert LLMGRPOTrainer.__module__ == "src.model.post_training.grpo.training"
    assert SearchAgentGRPOTrainer.__module__ == "src.model.post_training.grpo.training"


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
        ("GRPOTrainer", "training"),
        ("Policy", "training"),
        ("compute_group_advantages", "training"),
        ("grpo_clipped_policy_loss", "training"),
        ("make_grpo_trainer", "training"),
        ("reverse_kl_penalty", "training"),
        ("LLMGRPOConfig", "training"),
        ("LLMGRPOTrainer", "training"),
        ("LLMRolloutResult", "training"),
        ("get_response_log_probs", "training"),
        ("SearchAgentGRPOTrainer", "training"),
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


def test_grpo_package_has_only_the_minimal_implementation_modules():
    package_dir = (
        Path(__file__).resolve().parents[2] / "src" / "model" / "post_training" / "grpo"
    )
    assert {path.name for path in package_dir.glob("*.py")} == {
        "__init__.py",
        "generation.py",
        "algorithms.py",
        "training.py",
    }


@pytest.mark.parametrize(
    "module_name",
    [
        "src.model.post_training.grpo.core_algos",
        "src.model.post_training.grpo.rollouts",
        "src.model.post_training.grpo.judge",
        "src.model.post_training.grpo.trainers",
        "src.model.post_training.grpo.plot_rollouts",
    ],
)
def test_second_stage_replaced_modules_are_removed(module_name: str):
    assert find_spec(module_name) is None


def test_representative_symbols_have_final_owners():
    pytest.importorskip("torch", exc_type=ImportError)
    from src.model.post_training.grpo.algorithms import (
        GRPOAdvantageConfig,
        LLMJudge,
        compute_grpo_policy_loss,
        score_prompt_group,
    )
    from src.model.post_training.grpo.training import (
        LLMGRPOTrainer,
        LocalGRPOController,
        SearchAgentGRPOTrainer,
    )

    for value in (
        GRPOAdvantageConfig,
        LLMJudge,
        compute_grpo_policy_loss,
        score_prompt_group,
    ):
        assert value.__module__ == "src.model.post_training.grpo.algorithms"
    for value in (LLMGRPOTrainer, LocalGRPOController, SearchAgentGRPOTrainer):
        assert value.__module__ == "src.model.post_training.grpo.training"


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
    "minimal_implementation_modules or second_stage_replaced_modules_are_removed",
]))
"""
    result = subprocess.run(
        [sys.executable, "-c", program],
        capture_output=True,
        text=True,
        cwd=Path(__file__).resolve().parents[2],
    )

    assert result.returncode == 0, result.stderr + result.stdout
    assert "6 passed" in result.stdout
