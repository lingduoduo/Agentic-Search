"""`src/training/__init__` must actually export what it claims to.

That module wraps its re-exports in `try/except ImportError` so the package stays
importable in a CI job with no torch. The cost is that a *wrong* import inside
the block is indistinguishable from a *missing dependency*: it is swallowed, and
because the whole block is one `try`, every import after the failing line is
skipped too.

That is not hypothetical. When `PPORewardManager` moved from the RL package to
`ppo` (#542), the stale `from .grpo import PPORewardManager` raised, and seven
exports silently vanished from `src.training` -- `SearchRewardFunction`,
`SFTExample`, `SimulatedPreferenceJudge`, `PromptBatch`, `QLearningAgent`,
`SearchEnvironment` and the manager itself. Every test still passed, because
nothing asserted on the re-export surface.

This test reads the names out of the `try` block and checks each one resolves,
so it cannot go stale as that block changes.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

pytest.importorskip("torch")

import src.training as training

_INIT = Path(__file__).resolve().parents[2] / "src" / "training" / "__init__.py"


def _guarded_export_names() -> list[str]:
    """Every name the `try` block in src/training/__init__ imports."""
    tree = ast.parse(_INIT.read_text(encoding="utf-8"))
    names: list[str] = []
    for node in tree.body:
        if not isinstance(node, ast.Try):
            continue
        for stmt in ast.walk(node):
            if isinstance(stmt, ast.ImportFrom):
                names.extend(alias.asname or alias.name for alias in stmt.names)
    return names


def test_the_guarded_block_actually_imports_something():
    """Guard the guard: an empty list would make the test below vacuous."""
    assert len(_guarded_export_names()) > 10


@pytest.mark.parametrize("name", _guarded_export_names())
def test_guarded_export_resolves(name: str):
    """Each name survives the try/except rather than being silently swallowed."""
    assert hasattr(training, name), (
        f"src.training.{name} is missing. An ImportError inside the try/except "
        f"in src/training/__init__.py was swallowed -- and because the block is "
        f"a single `try`, every import after the failing line was skipped too. "
        f"Check that the module {name!r} is imported from still holds it."
    )
