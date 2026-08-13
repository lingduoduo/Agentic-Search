import pytest

from src.model.intent_taxonomy import (
    ACTION_MODULES,
    INTENT_LABELS,
    MODULES,
    SEMANTIC_MODULES,
    modules_for_route,
    route_of_module,
    validate_modules,
)


def test_routes_are_the_three_serving_labels():
    assert INTENT_LABELS == ("chat", "search", "tool")


def test_taxonomy_has_fourteen_modules_thirteen_of_them_semantic():
    assert len(MODULES) == 14
    assert len(SEMANTIC_MODULES) == 13
    assert "bare_entity" not in SEMANTIC_MODULES


def test_bare_entity_is_a_form_label_not_a_semantic_intent():
    """It describes utterance shape, so it must not enter macro-F1."""
    assert MODULES["bare_entity"].kind == "form"
    assert all(MODULES[name].kind == "intent" for name in SEMANTIC_MODULES)


def test_every_module_belongs_to_exactly_one_route():
    for name, spec in MODULES.items():
        assert spec.route in INTENT_LABELS
        assert name in modules_for_route(spec.route)
        assert route_of_module(name) == spec.route


def test_routes_partition_the_modules():
    covered = [name for route in INTENT_LABELS for name in modules_for_route(route)]
    assert sorted(covered) == sorted(MODULES)


def test_action_modules_are_exactly_the_tool_modules():
    """Composite detection keys off these; drift would silently break it."""
    assert ACTION_MODULES == frozenset(modules_for_route("tool"))


def test_validate_accepts_multiple_modules_from_the_same_route():
    validate_modules("search", ["current_info", "lookup_fact"])


def test_validate_rejects_a_module_from_another_route():
    with pytest.raises(ValueError, match="summarize"):
        validate_modules("search", ["lookup_fact", "summarize"])


def test_validate_rejects_an_unknown_module():
    with pytest.raises(ValueError, match="nonsense"):
        validate_modules("search", ["nonsense"])


def test_validate_rejects_an_empty_module_list():
    with pytest.raises(ValueError, match="at least one"):
        validate_modules("search", [])


def test_validate_rejects_duplicate_modules():
    with pytest.raises(ValueError, match="duplicate"):
        validate_modules("search", ["lookup_fact", "lookup_fact"])


def test_route_of_unknown_module_is_an_error():
    with pytest.raises(KeyError):
        route_of_module("nonsense")
