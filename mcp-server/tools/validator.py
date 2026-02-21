"""
forge/tools/validator.py
========================
Checks whether two ML library implementations produce numerically
equivalent outputs, accounting for floating point differences using tensor signatures.
"""

import json
import math
from typing import Any

def validate_output(output_a: str, output_b: str, tolerance: float = 1e-5) -> dict[str, Any]:
    try:
        a = json.loads(output_a)
        b = json.loads(output_b)
    except json.JSONDecodeError as e:
        return {"equivalent": False, "error": f"Failed to parse JSON: {e}"}

    return _compare(a, b, tolerance)

def _compare(a: Any, b: Any, tolerance: float) -> dict[str, Any]:
    # Both are scalar numbers
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        diff = abs(a - b)
        return {"equivalent": diff <= tolerance, "max_diff": diff, "shape_match": True, "notes": "Scalar"}

    # Both are signature dictionaries (shape, mean, sum)
    if isinstance(a, dict) and isinstance(b, dict):
        if a.get("shape") != b.get("shape"):
            return {
                "equivalent": False, 
                "max_diff": None, 
                "shape_match": False, 
                "notes": f"Shape mismatch: {a.get('shape')} vs {b.get('shape')}"
            }
        
        diffs = []
        for key in ["mean", "sum", "std"]:
            if key in a and key in b:
                diffs.append(abs(a[key] - b[key]))
        
        max_diff = max(diffs) if diffs else 0.0
        return {
            "equivalent": max_diff <= tolerance,
            "max_diff": round(max_diff, 10),
            "shape_match": True,
            "shape": a.get("shape"),
            "notes": "Signature comparison (mean, sum, std)"
        }

    return {"equivalent": False, "notes": f"Type mismatch: {type(a).__name__} vs {type(b).__name__}"}