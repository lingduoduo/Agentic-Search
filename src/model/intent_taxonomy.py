"""The two-level intent taxonomy: three routes, each with its own modules.

Modules are route-scoped rather than orthogonal primitives. At a few hundred
canonical examples a small hierarchical taxonomy is labelable consistently and
explainable; orthogonal primitives are the right end state at several thousand.

The module names are not invented. Each is drawn from a regex cue already used
by ``src/internal/servers/web/intent_routing.py``, so the taxonomy describes
distinctions the router already makes.

This module imports nothing. Every other intent module depends on it.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

INTENT_LABELS: tuple[str, ...] = ("chat", "search", "tool")


@dataclass(frozen=True)
class ModuleSpec:
    """One module: its route, and whether it names an intent or an utterance form."""

    name: str
    route: str
    kind: str


_SPECS: tuple[ModuleSpec, ...] = (
    # search — _SEARCH_LOOKUP_RE, _CURRENCY_RE, _is_bare_lookup
    ModuleSpec("lookup_document", "search", "intent"),
    ModuleSpec("lookup_fact", "search", "intent"),
    ModuleSpec("current_info", "search", "intent"),
    # A form label, not an intent: "OpenAI" is a bare entity, "OpenAI CEO" is a
    # fact lookup. It is excluded from module macro-F1, and _is_bare_lookup
    # routes such queries at cascade step 2, before this model ever runs.
    ModuleSpec("bare_entity", "search", "form"),
    # chat — _CHAT_START_RE, _GENERATIVE_RE
    ModuleSpec("explain", "chat", "intent"),
    ModuleSpec("summarize", "chat", "intent"),
    ModuleSpec("compare", "chat", "intent"),
    ModuleSpec("generate", "chat", "intent"),
    ModuleSpec("converse", "chat", "intent"),
    # tool — _TOOL_ACTION_RE, _TOOL_OBJECT_RE
    ModuleSpec("create", "tool", "intent"),
    ModuleSpec("send", "tool", "intent"),
    ModuleSpec("schedule", "tool", "intent"),
    ModuleSpec("modify", "tool", "intent"),
    ModuleSpec("execute", "tool", "intent"),
)

MODULES: dict[str, ModuleSpec] = {spec.name: spec for spec in _SPECS}

SEMANTIC_MODULES: tuple[str, ...] = tuple(
    spec.name for spec in _SPECS if spec.kind == "intent"
)

# Composite detection keys off these: a runner-up route whose best module is an
# action is the signature of "find X and book it".
ACTION_MODULES: frozenset[str] = frozenset(
    spec.name for spec in _SPECS if spec.route == "tool"
)


def modules_for_route(route: str) -> tuple[str, ...]:
    """Every module belonging to *route*, in taxonomy order."""
    return tuple(spec.name for spec in _SPECS if spec.route == route)


def route_of_module(module: str) -> str:
    """The route *module* belongs to. Raises KeyError if unknown."""
    return MODULES[module].route


def validate_modules(route: str, modules: Sequence[str]) -> None:
    """Check a label's modules: nonempty, known, unique, and all in *route*."""
    if not modules:
        raise ValueError(f"Route {route!r} needs at least one module")
    seen: set[str] = set()
    for module in modules:
        if module in seen:
            raise ValueError(f"Found duplicate module {module!r} for route {route!r}")
        seen.add(module)
        spec = MODULES.get(module)
        if spec is None:
            raise ValueError(f"Unknown module {module!r}")
        if spec.route != route:
            raise ValueError(
                f"Module {module!r} belongs to route {spec.route!r}, not {route!r}"
            )
