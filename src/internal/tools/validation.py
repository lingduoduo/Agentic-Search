"""Validate call arguments against a tool's JSON-schema parameters."""

from __future__ import annotations

from typing import Any


def check_json_type(value: Any, json_type: str) -> bool:
    if json_type == "string":
        return isinstance(value, str)
    if json_type == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if json_type == "number":
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if json_type == "boolean":
        return isinstance(value, bool)
    if json_type == "array":
        return isinstance(value, list)
    if json_type == "object":
        return isinstance(value, dict)
    if json_type == "null":
        return value is None
    return True  # unknown type — don't reject


def validate_arguments(
    parameters: dict[str, Any], arguments: dict[str, Any]
) -> list[str]:
    """Return a list of validation error strings (empty = valid)."""
    errors: list[str] = []
    required = parameters.get("required", [])
    props: dict[str, Any] = parameters.get("properties", {})

    for req in required:
        if req not in arguments:
            errors.append(f"Missing required argument: {req!r}")

    for key, value in arguments.items():
        if key not in props:
            continue
        expected = props[key].get("type")
        if expected and not check_json_type(value, expected):
            errors.append(
                f"Argument {key!r}: expected {expected!r}, got {type(value).__name__!r}"
            )

    return errors
