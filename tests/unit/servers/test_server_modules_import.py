"""Every module under src/internal/servers/ must import.

An un-importable module can sit in the tree indefinitely: nothing imports it, so
nothing notices. That is how a 922-line eval cluster survived with a hard
dependency on `braintrust`, an SDK in neither requirements.txt nor
pyproject.toml, and a one-off script that called sys.exit(1) at import time.
"""

from __future__ import annotations

import importlib
import pathlib

import pytest

_SERVERS_ROOT = pathlib.Path(__file__).resolve().parents[3] / "src/internal/servers"


def _module_names() -> list[str]:
    names: list[str] = []
    for path in sorted(_SERVERS_ROOT.rglob("*.py")):
        if "__pycache__" in path.parts:
            continue
        relative = path.relative_to(_SERVERS_ROOT.parents[2]).with_suffix("")
        module = ".".join(relative.parts)
        names.append(module.removesuffix(".__init__"))
    return names


def test_the_server_tree_is_not_empty():
    """Guard the guard: a broken path would make every case below vacuous."""
    assert len(_module_names()) > 100


@pytest.mark.parametrize("module_name", _module_names())
def test_server_module_imports(module_name: str):
    try:
        importlib.import_module(module_name)
    except BaseException as exc:  # noqa: BLE001 - SystemExit is a real failure here
        # `except Exception` would let a module that calls sys.exit() at import
        # time pass silently, which is one of the two shapes this test exists
        # to catch.
        pytest.fail(f"{module_name} does not import: {type(exc).__name__}: {exc}")
