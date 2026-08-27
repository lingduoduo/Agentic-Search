from __future__ import annotations

import ast
from pathlib import Path


PACKAGE = Path(__file__).resolve().parents[2] / "src/model/post_training/grpo"


def _relative_imports(module: str) -> set[str]:
    tree = ast.parse((PACKAGE / f"{module}.py").read_text())
    return {
        node.module or ""
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.level == 1
    }


def test_grpo_dependency_direction_is_acyclic():
    assert "generation" not in _relative_imports("algorithms")
    assert "training" not in _relative_imports("algorithms")
    assert "training" not in _relative_imports("generation")
