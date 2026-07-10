"""Small JSON-value semantics shared by schema and evidence comparisons."""

from __future__ import annotations

from typing import Any


def json_values_equal(left: Any, right: Any) -> bool:
    """Compare JSON-like values without Python's boolean/integer aliasing."""
    if _is_number(left) and _is_number(right):
        return left == right
    if type(left) is not type(right):
        return False
    if isinstance(left, dict):
        return left.keys() == right.keys() and all(json_values_equal(left[key], right[key]) for key in left)
    if isinstance(left, list):
        return len(left) == len(right) and all(
            json_values_equal(left_item, right_item)
            for left_item, right_item in zip(left, right, strict=True)
        )
    return left == right


def json_semantic_key(value: Any) -> tuple[Any, ...]:
    """Return a hashable key with the same equality rules as JSON values."""
    if _is_number(value):
        return ("number", value)
    if value is None:
        return ("null",)
    if isinstance(value, bool):
        return ("boolean", value)
    if isinstance(value, str):
        return ("string", value)
    if isinstance(value, list):
        return ("array", tuple(json_semantic_key(item) for item in value))
    if isinstance(value, dict):
        return (
            "object",
            tuple(
                sorted(
                    (str(key), json_semantic_key(item))
                    for key, item in value.items()
                )
            ),
        )
    return (f"python:{type(value).__module__}.{type(value).__qualname__}", repr(value))


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)
