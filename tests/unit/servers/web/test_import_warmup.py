"""The first request must not pay for imports the app can make at startup."""

from __future__ import annotations

import subprocess
import sys


def test_warm_lazy_imports_loads_the_deferred_modules():
    """Both modules deferred on the request path are importable up front.

    They are imported lazily inside hot functions -- `.safety` for laziness,
    `request_capture` to break a cycle from the agent loops. Neither cycle
    exists from the web app's own startup, and leaving them cold made the
    first request of a process pay ~1.4s of import time.

    Asserted in a subprocess on purpose: `request_capture` holds a
    module-level ContextVar, so popping it from `sys.modules` in-process to
    force a cold start would mint a second ContextVar and orphan every
    reference already taken.
    """
    probe = (
        "import sys;"
        "from src.internal.servers.web.app import _warm_lazy_imports;"
        "before = ["
        "  'src.context.safety' in sys.modules,"
        "  'src.internal.servers.web.request_capture' in sys.modules,"
        "];"
        "_warm_lazy_imports();"
        "print('src.context.safety' in sys.modules,"
        "      'src.internal.servers.web.request_capture' in sys.modules)"
    )
    completed = subprocess.run(
        [sys.executable, "-c", probe],
        capture_output=True,
        text=True,
        cwd="/Users/linghuang/Git/Agentic-Search",
    )
    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.strip().endswith("True True"), completed.stdout


def test_warm_lazy_imports_never_raises(monkeypatch):
    """A failed warm-up must not take down startup."""
    import builtins

    from src.internal.servers.web.app import _warm_lazy_imports

    real_import = builtins.__import__

    def _boom(name, *args, **kwargs):
        if name == "src.context.safety":
            raise ImportError("simulated")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _boom)
    _warm_lazy_imports()  # must not raise
