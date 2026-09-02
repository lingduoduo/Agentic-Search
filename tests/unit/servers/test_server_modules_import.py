"""Every module under src/internal/servers/ must import.

An un-importable module can sit in the tree indefinitely: nothing imports it, so
nothing notices. That is how a 922-line eval cluster survived with a hard
dependency on `braintrust`, an SDK in neither requirements.txt nor
pyproject.toml, and a one-off script that called sys.exit(1) at import time.

The check runs in a subprocess. Importing ~145 modules into the pytest process
costs 164 MB that is never released, and registers routes and mutates global
registries for every test that follows -- a cost this file has no business
imposing on the rest of the session.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]
_SERVERS_ROOT = _REPO_ROOT / "src/internal/servers"

# Runs in a fresh interpreter: import every module, report the ones that fail.
# BaseException, not Exception -- a module that calls sys.exit() at import time
# is one of the two shapes this guard exists to catch, and a bare
# `except Exception` would let it pass.
_PROBE = """
import importlib, pathlib, sys
root = pathlib.Path(sys.argv[1])
repo = pathlib.Path(sys.argv[2])
failures = []
count = 0
for path in sorted(root.rglob("*.py")):
    if "__pycache__" in path.parts:
        continue
    module = ".".join(path.relative_to(repo).with_suffix("").parts)
    module = module.removesuffix(".__init__")
    count += 1
    try:
        importlib.import_module(module)
    except BaseException as exc:
        failures.append(f"{module}: {type(exc).__name__}: {exc}")
print(count)
for line in failures:
    print(line)
"""


def _run_probe() -> tuple[int, list[str]]:
    result = subprocess.run(
        [sys.executable, "-c", _PROBE, str(_SERVERS_ROOT), str(_REPO_ROOT)],
        capture_output=True,
        text=True,
        cwd=_REPO_ROOT,
        timeout=300,
    )
    assert result.returncode == 0, result.stderr[-2000:]
    lines = [line for line in result.stdout.splitlines() if line.strip()]
    return int(lines[0]), lines[1:]


def test_every_server_module_imports():
    count, failures = _run_probe()

    # Guard the guard: a broken path would make the assertion below vacuous.
    assert count > 100, f"only {count} modules discovered under {_SERVERS_ROOT}"
    assert failures == [], "modules that do not import:\n  " + "\n  ".join(failures)


def test_the_guard_does_not_import_the_tree_into_this_process():
    """The check must not cost the rest of the session 164 MB and 145 imports.

    Doing it in-process inflated the pytest interpreter enough to make a
    neighbouring test -- one that spawns a 384 MB child and races a sampling
    watchdog against it -- fail intermittently on a memory-constrained machine.
    """
    before = {m for m in sys.modules if m.startswith("src.internal.servers")}

    _run_probe()

    after = {m for m in sys.modules if m.startswith("src.internal.servers")}
    assert after == before, f"guard imported into the test process: {after - before}"
