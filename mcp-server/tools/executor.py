"""
forge/tools/executor.py
=======================
Orchestrates running a benchmark across multiple libraries.
Calls the profiler per library and aggregates results.
"""

import asyncio
from typing import Any
from tools.profiler import profile_code


async def run_benchmark(
    generated_code: dict[str, str],
    libraries: list[str],
    iterations: int = 5
) -> dict[str, Any]:
    """
    Run all library implementations concurrently and return aggregated results.

    Args:
        generated_code: Dict mapping library name -> Python code string
        libraries: List of library names to benchmark
        iterations: Number of timed runs per library

    Returns:
        Dict mapping library name -> profiling result
    """
    tasks = []
    for lib in libraries:
        code = generated_code.get(lib)
        if not code:
            tasks.append(_error_result(lib, "No code provided"))
            continue
        tasks.append(profile_code(code, lib, iterations))

    results_list = await asyncio.gather(*tasks, return_exceptions=False)

    return {
        lib: result
        for lib, result in zip(libraries, results_list)
    }


async def _error_result(library: str, message: str) -> dict:
    return {
        "library": library,
        "status": "error",
        "error": message,
        "time_ms": None,
        "memory_mb": None
    }
