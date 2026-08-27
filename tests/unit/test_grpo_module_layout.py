from __future__ import annotations

from importlib import import_module
from importlib.util import find_spec
from pathlib import Path

import pytest


# Torch-free: grouped sampling, scoring and the judges.
ALGORITHM_EXPORTS = [
    "compute_grpo_outcome_advantage",
    "PromptGroupSamplingConfig",
    "GRPORolloutSample",
    "ScoredGRPORollout",
    "GRPOAdvantageConfig",
    "build_grpo_sampling_params",
    "sample_prompt_group",
    "sample_prompt_batch",
    "score_prompt_group",
    "compute_dapo_advantages",
    "score_prompt_batch",
    "OnPolicyGRPOConfig",
    "OnPolicyBatchStats",
    "filter_zero_advantage_groups",
    "assemble_on_policy_batch",
    "compute_on_policy_batch_stats",
    "SimulatedPreferenceJudge",
    "judge_gold_agreement",
    "GoldAgreementJudge",
    "JudgeParseError",
    "parse_judge_score",
    "is_degenerate_group",
    "LLMJudge",
]

# The grpo package deliberately does NOT export the scalar
# `compute_grpo_outcome_advantage`: that name resolves to the list form from
# the root and post_training only, so one name never means two functions.
# Tensor math, and therefore the package's torch boundary.
CORE_ALGO_EXPORTS = [
    "compute_grpo_token_advantages",
    "compute_reinforce_policy_loss_core",
    "compute_reinforce_policy_loss",
    "compute_grpo_policy_loss",
]

GRPO_PACKAGE_ALGORITHM_EXPORTS = [
    "GoldAgreementJudge",
    "JudgeParseError",
    "LLMJudge",
    "SimulatedPreferenceJudge",
    "is_degenerate_group",
    "judge_gold_agreement",
    "parse_judge_score",
]

ROOT_ALGORITHM_EXPORTS = [
    "GRPOAdvantageConfig",
    "GRPORolloutSample",
    "PromptGroupSamplingConfig",
    "ScoredGRPORollout",
    "build_grpo_sampling_params",
    "compute_grpo_outcome_advantage",
    "sample_prompt_group",
    "sample_prompt_batch",
    "score_prompt_group",
    "score_prompt_batch",
    "OnPolicyGRPOConfig",
    "OnPolicyBatchStats",
    "filter_zero_advantage_groups",
    "assemble_on_policy_batch",
    "compute_on_policy_batch_stats",
]

POST_TRAINING_ALGORITHM_EXPORTS = [
    "GRPOAdvantageConfig",
    "PromptGroupSamplingConfig",
    "compute_dapo_advantages",
    "compute_grpo_outcome_advantage",
    "score_prompt_group",
    "GoldAgreementJudge",
    "LLMJudge",
    "SimulatedPreferenceJudge",
    "judge_gold_agreement",
]

TRAINER_EXPORTS = [
    "Policy",
    "GRPOTrainer",
    "compute_group_advantages",
    "grpo_clipped_policy_loss",
    "make_grpo_trainer",
    "reverse_kl_penalty",
    "LLMGRPOConfig",
    "LLMRolloutResult",
    "LLMGRPOTrainer",
    "SearchAgentGRPOTrainer",
]


@pytest.mark.parametrize("export_name", TRAINER_EXPORTS)
def test_consolidated_training_module_owns_every_trainer_symbol(export_name: str):
    pytest.importorskip("torch", exc_type=ImportError)

    grpo = import_module("src.model.post_training.grpo")
    training = import_module("src.model.post_training.grpo.training")

    assert export_name in grpo.__all__
    assert getattr(grpo, export_name) is getattr(training, export_name)


def test_consolidated_training_module_preserves_trainer_hierarchy():
    pytest.importorskip("torch", exc_type=ImportError)
    from src.model.post_training.grpo.training import (
        LLMGRPOTrainer,
        SearchAgentGRPOTrainer,
    )

    assert LLMGRPOTrainer in SearchAgentGRPOTrainer.__mro__


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


@pytest.mark.parametrize("export_name", ALGORITHM_EXPORTS)
def test_algorithms_module_owns_every_moved_public_symbol(export_name: str):
    pytest.importorskip("torch", exc_type=ImportError)

    algorithms = import_module("src.model.post_training.grpo.algorithms")

    value = getattr(algorithms, export_name)
    assert value.__module__ == "src.model.post_training.grpo.algorithms"


@pytest.mark.parametrize("export_name", GRPO_PACKAGE_ALGORITHM_EXPORTS)
def test_grpo_package_algorithm_exports_resolve_by_identity(export_name: str):
    pytest.importorskip("torch", exc_type=ImportError)

    grpo = import_module("src.model.post_training.grpo")
    algorithms = import_module("src.model.post_training.grpo.algorithms")

    assert export_name in grpo.__all__
    assert getattr(grpo, export_name) is getattr(algorithms, export_name)


@pytest.mark.parametrize("export_name", CORE_ALGO_EXPORTS)
def test_core_algos_owns_every_tensor_level_loss(export_name: str):
    pytest.importorskip("torch", exc_type=ImportError)

    core_algos = import_module("src.model.post_training.grpo.core_algos")

    value = getattr(core_algos, export_name)
    assert value.__module__ == "src.model.post_training.grpo.core_algos"


@pytest.mark.parametrize("export_name", CORE_ALGO_EXPORTS)
def test_grpo_package_core_algo_exports_resolve_by_identity(export_name: str):
    pytest.importorskip("torch", exc_type=ImportError)

    grpo = import_module("src.model.post_training.grpo")
    core_algos = import_module("src.model.post_training.grpo.core_algos")

    assert export_name in grpo.__all__
    assert getattr(grpo, export_name) is getattr(core_algos, export_name)


def test_algorithms_imports_without_torch():
    """The judges and rollout scoring must stay importable with no torch.

    This is why `core_algos.py` is a separate module rather than folded into
    `algorithms.py`: the CI unit-test job installs no torch, and merging the two
    silently dropped 17 judge tests from it.
    """
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

from src.model.post_training.grpo.algorithms import (
    GoldAgreementJudge,
    LLMJudge,
    SimulatedPreferenceJudge,
    compute_grpo_outcome_advantage,
    score_prompt_group,
)

assert compute_grpo_outcome_advantage([1.0, 0.0]) == [0.5, -0.5]
assert "torch" not in sys.modules
print("algorithms is torch-free")
"""
    result = subprocess.run(
        [sys.executable, "-c", program],
        capture_output=True,
        text=True,
        cwd=Path(__file__).resolve().parents[2],
    )

    assert result.returncode == 0, result.stderr + result.stdout
    assert "algorithms is torch-free" in result.stdout


@pytest.mark.parametrize("export_name", ROOT_ALGORITHM_EXPORTS)
def test_root_algorithm_exports_resolve_by_identity(export_name: str):
    pytest.importorskip("torch", exc_type=ImportError)

    root = import_module("src")
    algorithms = import_module("src.model.post_training.grpo.algorithms")

    assert export_name in root.__all__
    assert getattr(root, export_name) is getattr(algorithms, export_name)


@pytest.mark.parametrize("export_name", POST_TRAINING_ALGORITHM_EXPORTS)
def test_post_training_algorithm_exports_resolve_by_identity(export_name: str):
    pytest.importorskip("torch", exc_type=ImportError)

    post_training = import_module("src.model.post_training")
    algorithms = import_module("src.model.post_training.grpo.algorithms")

    assert getattr(post_training, export_name) is getattr(algorithms, export_name)


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
    """Four implementation modules, and the fourth earns its place.

    The design targeted three. `core_algos.py` is split back out because it is
    the torch boundary: folding its tensor math into `algorithms.py` made the
    judges and rollout scoring unimportable without torch, which silently
    dropped 17 tests from the torch-free CI job. Module count lost to that.
    """
    package_dir = (
        Path(__file__).resolve().parents[2] / "src" / "model" / "post_training" / "grpo"
    )
    assert {path.name for path in package_dir.glob("*.py")} == {
        "__init__.py",
        "algorithms.py",
        "core_algos.py",
        "generation.py",
        "training.py",
    }


@pytest.mark.parametrize(
    "module_name",
    [
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
        score_prompt_group,
    )
    from src.model.post_training.grpo.core_algos import compute_grpo_policy_loss
    from src.model.post_training.grpo.training import (
        LLMGRPOTrainer,
        LocalGRPOController,
        SearchAgentGRPOTrainer,
    )

    for value in (GRPOAdvantageConfig, LLMJudge, score_prompt_group):
        assert value.__module__ == "src.model.post_training.grpo.algorithms"
    assert (
        compute_grpo_policy_loss.__module__ == "src.model.post_training.grpo.core_algos"
    )
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
    assert "5 passed" in result.stdout


def test_reward_imports_without_torch():
    """reward.py must never acquire a torch dependency.

    It is the torch-free half of the post-training package, and the shared
    advantage primitive lives there precisely so that stays true.

    Note what consolidation cost: ``rollouts.py`` used to be torch-free too and
    was covered by this guard. Merging it with ``core_algos.py`` — which is
    tensor math and cannot be — into ``algorithms.py`` gives that up. The
    package ``__init__`` stays lazy, so importing
    ``src.model.post_training.grpo`` still does not pull torch; only the
    submodule does.
    """
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

import src.model.post_training.reward as reward
import src.model.post_training.grpo as grpo

assert reward.group_relative_advantages([1.0, 0.0]) == [0.5, -0.5]
assert reward.grouped_relative_advantages([1.0, 0.0], ["g", "g"]) == [0.5, -0.5]
assert "torch" not in sys.modules
print("torch-free OK")
"""
    result = subprocess.run(
        [sys.executable, "-c", program],
        capture_output=True,
        text=True,
        cwd=Path(__file__).resolve().parents[2],
    )

    assert result.returncode == 0, result.stderr + result.stdout
    assert "torch-free OK" in result.stdout


def test_training_owns_the_async_step_and_generation_does_not_import_training():
    """The controller entrypoint lives with the controller.

    It was a pure delegator sitting in generation.py, which forced a
    generation <-> training cycle papered over with function-local imports.
    """
    pytest.importorskip("torch", exc_type=ImportError)

    from src.model.post_training.grpo import generation, training

    assert (
        training.async_run_grpo_training_step.__module__
        == "src.model.post_training.grpo.training"
    )
    assert not hasattr(generation, "async_run_grpo_training_step")

    import src as root

    assert root.async_run_grpo_training_step is training.async_run_grpo_training_step


def test_outcome_advantage_name_resolves_to_exactly_one_function():
    """One public name, one function, whatever the import path.

    Before this, `from ...grpo import compute_grpo_outcome_advantage` gave a
    torch tensor function while `from src import` (and `from
    src.model.post_training import`) gave a list[float] one, with
    incompatible signatures -- so a wrong import failed at call time, not
    import time. Now the torch function has its own name
    (`compute_grpo_token_advantages`), and the `grpo` package's own lazy-export
    path for the old name is gone entirely rather than repointed -- accessing
    it raises `AttributeError` immediately instead of silently resolving to
    something.

    Consolidation put both functions in one module, which is only safe because
    they no longer share a name.
    """
    pytest.importorskip("torch", exc_type=ImportError)

    import src as root
    import src.model.post_training as post_training
    from src.model.post_training.grpo import algorithms

    from src.model.post_training.grpo import core_algos

    assert hasattr(core_algos, "compute_grpo_token_advantages")
    assert not hasattr(algorithms, "compute_grpo_token_advantages")

    assert (
        root.compute_grpo_outcome_advantage is algorithms.compute_grpo_outcome_advantage
    )
    assert (
        post_training.compute_grpo_outcome_advantage
        is algorithms.compute_grpo_outcome_advantage
    )

    import src.model.post_training.grpo as grpo

    with pytest.raises(AttributeError):
        grpo.compute_grpo_outcome_advantage

    # Python's import machinery turns a module `__getattr__` AttributeError
    # into an ImportError for the `from ... import ...` form specifically.
    with pytest.raises(ImportError):
        from src.model.post_training.grpo import compute_grpo_outcome_advantage  # noqa: F401


def test_unused_batch_retriever_protocol_is_gone():
    """BatchRetriever was referenced nowhere outside its own definition --
    nothing implemented it, nothing isinstance-checked it, and no other
    docstring named it."""
    pytest.importorskip("torch", exc_type=ImportError)

    from src.model.post_training.grpo import generation

    assert not hasattr(generation, "BatchRetriever")
