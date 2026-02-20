"""
forge/tools/validator.py
========================
Checks whether two ML library implementations produce numerically
equivalent outputs, accounting for floating point differences.
"""

import json
import math
from typing import Any


def validate_output(output_a: str, output_b: str, tolerance: float = 1e-5) -> dict[str, Any]:
    """
    Compare two JSON-serialized outputs for numerical equivalence.

    Handles scalars, lists, and nested lists (tensors).
    """
    try:
        a = json.loads(output_a)
        b = json.loads(output_b)
    except json.JSONDecodeError as e:
        return {
            "equivalent": False,
            "error": f"Failed to parse output as JSON: {e}",
            "max_diff": None,
            "shape_match": None
        }

    return _compare(a, b, tolerance)


def _compare(a: Any, b: Any, tolerance: float) -> dict[str, Any]:
    """Recursively compare two values."""

    # Both scalars
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        diff = abs(a - b)
        return {
            "equivalent": diff <= tolerance,
            "max_diff": diff,
            "shape_match": True,
            "notes": "Scalar comparison"
        }

    # Both lists (tensors represented as nested lists)
    if isinstance(a, list) and isinstance(b, list):
        shape_a = _get_shape(a)
        shape_b = _get_shape(b)

        if shape_a != shape_b:
            return {
                "equivalent": False,
                "max_diff": None,
                "shape_match": False,
                "notes": f"Shape mismatch: {shape_a} vs {shape_b}"
            }

        flat_a = _flatten(a)
        flat_b = _flatten(b)

        if len(flat_a) == 0:
            return {"equivalent": True, "max_diff": 0.0, "shape_match": True, "notes": "Empty tensors"}

        diffs = [abs(x - y) for x, y in zip(flat_a, flat_b)
                 if not (math.isnan(x) or math.isnan(y))]
        nan_count = sum(1 for x, y in zip(flat_a, flat_b) if math.isnan(x) or math.isnan(y))

        max_diff = max(diffs) if diffs else 0.0
        mean_diff = sum(diffs) / len(diffs) if diffs else 0.0

        return {
            "equivalent": max_diff <= tolerance and nan_count == 0,
            "max_diff": round(max_diff, 10),
            "mean_diff": round(mean_diff, 10),
            "shape_match": True,
            "shape": shape_a,
            "nan_count": nan_count,
            "notes": f"Tensor comparison, {len(flat_a)} elements"
        }

    # Type mismatch
    return {
        "equivalent": False,
        "max_diff": None,
        "shape_match": False,
        "notes": f"Type mismatch: {type(a).__name__} vs {type(b).__name__}"
    }


def _get_shape(lst: list) -> list[int]:
    """Get the shape of a nested list."""
    shape = []
    current = lst
    while isinstance(current, list):
        shape.append(len(current))
        current = current[0] if current else []
    return shape


def _flatten(lst: Any) -> list[float]:
    """Flatten a nested list to a flat list of floats."""
    if isinstance(lst, (int, float)):
        return [float(lst)]
    if isinstance(lst, list):
        result = []
        for item in lst:
            result.extend(_flatten(item))
        return result
    return []
