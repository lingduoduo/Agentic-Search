"""Guard against blocking calls made directly inside async functions.

This bug class keeps coming back. #547 fixed answer synthesis blocking the web
backend's event loop, but the identical shape survived in the MCP chat tool and
in the license claim endpoint, because the fix was applied per-call-site by hand
and nothing checked the rest of the tree. One blocking call on the loop thread
stalls *every* concurrent request, so the failure is invisible in single-user
testing and only shows up under load.

The rule: if a coroutine needs to do blocking work, hand it to a thread with
`asyncio.to_thread`. That is what every offloaded call site here already does.
"""

from __future__ import annotations

import ast
from pathlib import Path

SOURCE_ROOT = Path(__file__).parents[2] / "src"

# Blocking HTTP verbs on the `requests` module.
BLOCKING_MODULE_CALLS = {
    ("requests", "get"),
    ("requests", "post"),
    ("requests", "put"),
    ("requests", "delete"),
    ("requests", "request"),
    ("time", "sleep"),
    ("subprocess", "run"),
    ("subprocess", "call"),
    ("subprocess", "check_output"),
}

# Synchronous functions in this repo that perform blocking LLM/network IO.
# `answer_with_retrieval` is deliberately absent: it is itself a coroutine.
BLOCKING_FUNCTIONS = {"generate_answer"}


class BlockingCallFinder(ast.NodeVisitor):
    """Collect blocking calls whose nearest enclosing function is a coroutine.

    A nested plain `def` is a boundary: its body runs wherever it is later
    called, which is frequently a worker thread, so calls inside it are not
    attributable to the enclosing coroutine.
    """

    def __init__(self) -> None:
        self.scopes: list[str | None] = []
        self.findings: list[tuple[int, str, str]] = []

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self.scopes.append(node.name)
        self.generic_visit(node)
        self.scopes.pop()

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self.scopes.append(None)
        self.generic_visit(node)
        self.scopes.pop()

    def visit_Call(self, node: ast.Call) -> None:
        enclosing = self.scopes[-1] if self.scopes else None
        if enclosing is not None:
            func = node.func
            module = getattr(getattr(func, "value", None), "id", None)
            attribute = getattr(func, "attr", None)
            name = getattr(func, "id", None)
            if (module, attribute) in BLOCKING_MODULE_CALLS:
                self.findings.append(
                    (node.lineno, enclosing, f"{module}.{attribute}()")
                )
            elif name in BLOCKING_FUNCTIONS:
                self.findings.append((node.lineno, enclosing, f"{name}()"))
        self.generic_visit(node)


def find_blocking_calls(source: str) -> list[tuple[int, str, str]]:
    finder = BlockingCallFinder()
    finder.visit(ast.parse(source))
    return finder.findings


def test_no_async_function_makes_a_blocking_call_directly() -> None:
    offenders = []
    for path in sorted(SOURCE_ROOT.rglob("*.py")):
        for lineno, function, call in find_blocking_calls(
            path.read_text(encoding="utf-8")
        ):
            relative = path.relative_to(SOURCE_ROOT.parent)
            offenders.append(f"{relative}:{lineno} async {function}() calls {call}")

    assert not offenders, (
        "blocking calls on the event loop thread — wrap them in "
        "asyncio.to_thread:\n" + "\n".join(offenders)
    )


def test_the_finder_reports_a_blocking_call_inside_a_coroutine() -> None:
    source = """
import requests
async def fetch():
    return requests.get("http://example.com")
"""
    assert find_blocking_calls(source) == [(4, "fetch", "requests.get()")]


def test_the_finder_ignores_the_same_call_in_a_plain_function() -> None:
    source = """
import requests
def fetch():
    return requests.get("http://example.com")
"""
    assert find_blocking_calls(source) == []


def test_the_finder_ignores_a_call_offloaded_to_a_thread() -> None:
    source = """
import asyncio
import requests
async def fetch():
    return await asyncio.to_thread(requests.get, "http://example.com")
"""
    assert find_blocking_calls(source) == []


def test_the_finder_ignores_a_nested_sync_function_passed_to_a_thread() -> None:
    source = """
import asyncio
import requests
async def fetch():
    def work():
        return requests.get("http://example.com")
    return await asyncio.to_thread(work)
"""
    assert find_blocking_calls(source) == []
