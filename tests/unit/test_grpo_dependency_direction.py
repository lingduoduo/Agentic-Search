from __future__ import annotations

import ast
from pathlib import Path


PACKAGE = Path(__file__).resolve().parents[2] / "src/model/post_training/grpo"
PACKAGE_NAME = "src.model.post_training.grpo"


def _internal_imports(module: str) -> set[str]:
    tree = ast.parse((PACKAGE / f"{module}.py").read_text())
    imports: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            if node.level == 1:
                if node.module:
                    imports.add(node.module.split(".")[0])
                imports.update(alias.name.split(".")[0] for alias in node.names)
            elif node.level == 0 and node.module:
                if node.module == PACKAGE_NAME:
                    imports.update(alias.name.split(".")[0] for alias in node.names)
                elif node.module.startswith(f"{PACKAGE_NAME}."):
                    imports.add(
                        node.module.removeprefix(f"{PACKAGE_NAME}.").split(".")[0]
                    )
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.startswith(f"{PACKAGE_NAME}."):
                    imports.add(
                        alias.name.removeprefix(f"{PACKAGE_NAME}.").split(".")[0]
                    )
    return imports


def test_grpo_dependency_direction_is_acyclic():
    assert "generation" not in _internal_imports("algorithms")
    assert "training" not in _internal_imports("algorithms")
    assert "training" not in _internal_imports("generation")
