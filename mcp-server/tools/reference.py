"""
forge/tools/reference.py
========================
Loads and queries pre-loaded industry benchmark reference data.
Sources: MLPerf, DS-1000, curated research papers.
"""

import json
from pathlib import Path
from typing import Any, Optional

# Path to reference data directory (relative to mcp-server/)
REFERENCE_DIR = Path(__file__).parent.parent / "reference-data"

_cache: dict[str, Any] = {}


def _load_all() -> dict[str, Any]:
    """Load all reference data files into memory (cached)."""
    global _cache
    if _cache:
        return _cache

    data: dict[str, Any] = {}

    for json_file in REFERENCE_DIR.rglob("*.json"):
        try:
            with open(json_file) as f:
                content = json.load(f)
                source = json_file.stem
                data[source] = content
        except Exception:
            pass

    _cache = data
    return data


def get_reference_data(task_type: str, library: str) -> Optional[dict[str, Any]]:
    """
    Fetch the best matching reference benchmark for a task/library combination.

    Args:
        task_type: e.g. 'matmul', 'conv2d', 'relu'
        library: e.g. 'pytorch', 'tensorflow', 'numpy'

    Returns:
        Reference dict with keys: source, time_ms, memory_mb, hardware, date, notes
        or None if no match found
    """
    all_data = _load_all()

    # Search across all loaded datasets
    for source, dataset in all_data.items():
        if not isinstance(dataset, list):
            continue
        for entry in dataset:
            if (entry.get("task") == task_type and
                    entry.get("library", "").lower() == library.lower()):
                return {
                    "source": entry.get("source", source),
                    "task": task_type,
                    "library": library,
                    "time_ms": entry.get("time_ms"),
                    "memory_mb": entry.get("memory_mb"),
                    "hardware": entry.get("hardware", "unknown"),
                    "hardware_notes": entry.get("hardware_notes", ""),
                    "date": entry.get("date", "unknown"),
                    "notes": entry.get("notes", ""),
                    "url": entry.get("url", "")
                }

    return None


def list_reference_tasks() -> dict[str, list[str]]:
    """
    List all task/library combinations available in reference data.

    Returns:
        Dict mapping task_type -> list of libraries with coverage
    """
    all_data = _load_all()
    coverage: dict[str, list[str]] = {}

    for source, dataset in all_data.items():
        if not isinstance(dataset, list):
            continue
        for entry in dataset:
            task = entry.get("task")
            lib = entry.get("library", "").lower()
            if task and lib:
                if task not in coverage:
                    coverage[task] = []
                if lib not in coverage[task]:
                    coverage[task].append(lib)

    return coverage
