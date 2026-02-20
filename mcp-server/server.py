"""
Forge MCP Server
================
Put ML libraries through the fire. Find out which one is strongest.

Exposes tools to Claude Desktop for:
- Generating idiomatic ML library implementations
- Executing code safely and measuring performance
- Comparing results against industry reference data
- Launching a Textual TUI to visualize benchmark results
"""

import json
import asyncio
from mcp.server.fastmcp import FastMCP
from pydantic import BaseModel, Field, ConfigDict
from typing import Optional
from pathlib import Path

from tools.executor import run_benchmark
from tools.profiler import profile_code
from tools.validator import validate_output
from tools.reference import get_reference_data, list_reference_tasks
from tools.tui import launch_tui

# ── Server Init ───────────────────────────────────────────────────────────────

mcp = FastMCP("forge_mcp")

SUPPORTED_LIBRARIES = ["numpy", "pytorch", "tensorflow", "tinygrad", "jax"]
SUPPORTED_TASKS = ["matmul", "dot_product", "svd", "conv2d", "relu", "softmax", "norm"]

# ── Input Models ──────────────────────────────────────────────────────────────

class BenchmarkInput(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")

    task_description: str = Field(
        ...,
        description="Natural language description of the task to benchmark (e.g. 'multiply two 512x512 matrices')",
        min_length=5,
        max_length=500
    )
    libraries: list[str] = Field(
        ...,
        description=f"List of libraries to benchmark. Supported: {SUPPORTED_LIBRARIES}",
        min_length=2,
        max_length=5
    )
    generated_code: dict[str, str] = Field(
        ...,
        description="Dict mapping library name to generated implementation code"
    )
    iterations: int = Field(
        default=5,
        description="Number of benchmark iterations (median will be used)",
        ge=3,
        le=20
    )
    show_tui: bool = Field(
        default=True,
        description="Whether to launch the Textual TUI to show results visually"
    )


class ProfileInput(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")

    code: str = Field(..., description="Python code to profile", min_length=10)
    library: str = Field(..., description=f"Library being used. One of: {SUPPORTED_LIBRARIES}")
    iterations: int = Field(default=5, ge=3, le=20, description="Number of runs")


class ValidateInput(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")

    output_a: str = Field(..., description="JSON-serialized output from implementation A")
    output_b: str = Field(..., description="JSON-serialized output from implementation B")
    tolerance: float = Field(default=1e-5, description="Numerical tolerance for comparison", ge=0)


class ReferenceInput(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")

    task_type: str = Field(
        ...,
        description=f"Type of ML task. One of: {SUPPORTED_TASKS}"
    )
    library: str = Field(
        ...,
        description=f"Library name. One of: {SUPPORTED_LIBRARIES}"
    )


# ── Tools ─────────────────────────────────────────────────────────────────────

@mcp.tool(
    name="forge_run_benchmark",
    annotations={
        "title": "Run Full Benchmark",
        "readOnlyHint": False,
        "destructiveHint": False,
        "idempotentHint": False,
        "openWorldHint": False
    }
)
async def forge_run_benchmark(params: BenchmarkInput) -> str:
    """Run a full benchmark across multiple ML libraries and display results in a TUI.

    This is the primary Forge tool. Given pre-generated code implementations for
    each library, it executes them, measures performance, fetches reference data,
    and optionally launches a Textual TUI showing the comparison.

    Args:
        params (BenchmarkInput): Benchmark configuration including:
            - task_description (str): Human-readable description of the task
            - libraries (list[str]): Libraries to benchmark
            - generated_code (dict[str, str]): Code per library
            - iterations (int): Number of runs (default 5)
            - show_tui (bool): Launch TUI dashboard (default True)

    Returns:
        str: JSON with full benchmark results including timing, memory, scores,
             reference comparisons, and winner analysis
    """
    results = {}

    for library in params.libraries:
        code = params.generated_code.get(library)
        if not code:
            results[library] = {"error": f"No code provided for {library}"}
            continue

        profile = await profile_code(code, library, params.iterations)
        results[library] = profile

    # Fetch reference data for each library
    reference = {}
    task_key = _infer_task_key(params.task_description)
    for library in params.libraries:
        ref = get_reference_data(task_key, library)
        if ref:
            reference[library] = ref

    # Score and rank
    ranked = _score_and_rank(results, reference)

    final = {
        "task": params.task_description,
        "task_key": task_key,
        "libraries_tested": params.libraries,
        "results": results,
        "reference": reference,
        "ranking": ranked,
        "winner": ranked[0]["library"] if ranked else None,
        "summary": _build_summary(params.task_description, results, ranked, reference)
    }

    # Launch TUI if requested — capture result for fallback command
    if params.show_tui:
        tui_result = await launch_tui(final)
        if tui_result and tui_result != "TUI launched ✓":
            # Auto-launch failed (common from Claude Desktop background process)
            # Include the open command so Claude can show it to the user
            final["tui_command"] = tui_result
            final["tui_note"] = (
                f"TUI ready — run this in your terminal to open it: {tui_result}"
            )

    return json.dumps(final, indent=2, default=str)


@mcp.tool(
    name="forge_profile",
    annotations={
        "title": "Profile Single Implementation",
        "readOnlyHint": False,
        "destructiveHint": False,
        "idempotentHint": False,
        "openWorldHint": False
    }
)
async def forge_profile(params: ProfileInput) -> str:
    """Profile a single ML code implementation — time, memory, and correctness.

    Use this when you want to measure a specific implementation before running
    a full cross-library benchmark.

    Args:
        params (ProfileInput): Contains:
            - code (str): Python code to execute and profile
            - library (str): Library name for context
            - iterations (int): Number of timed runs

    Returns:
        str: JSON with time_ms (median), memory_mb (peak), iterations, status, output
    """
    result = await profile_code(params.code, params.library, params.iterations)
    return json.dumps(result, indent=2, default=str)


@mcp.tool(
    name="forge_validate",
    annotations={
        "title": "Validate Output Equivalence",
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": False
    }
)
async def forge_validate(params: ValidateInput) -> str:
    """Check whether two library implementations produce numerically equivalent outputs.

    Use this to verify correctness before benchmarking — ensures you're comparing
    apples to apples.

    Args:
        params (ValidateInput): Contains:
            - output_a (str): JSON-serialized output from first implementation
            - output_b (str): JSON-serialized output from second implementation
            - tolerance (float): Acceptable numerical difference (default 1e-5)

    Returns:
        str: JSON with { equivalent: bool, max_diff: float, shape_match: bool, notes: str }
    """
    result = validate_output(params.output_a, params.output_b, params.tolerance)
    return json.dumps(result, indent=2)


@mcp.tool(
    name="forge_get_reference",
    annotations={
        "title": "Get Industry Reference Benchmark",
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": False
    }
)
async def forge_get_reference(params: ReferenceInput) -> str:
    """Fetch pre-loaded industry benchmark data (MLPerf, DS-1000) for a task/library combo.

    Use this to anchor live benchmark results against known baselines, or to give
    Claude context about expected performance before generating code.

    Args:
        params (ReferenceInput): Contains:
            - task_type (str): The ML task type (e.g. 'matmul', 'conv2d')
            - library (str): Library name (e.g. 'pytorch', 'tensorflow')

    Returns:
        str: JSON with reference result including source, hardware, date, metrics
    """
    result = get_reference_data(params.task_type, params.library)
    if not result:
        return json.dumps({
            "found": False,
            "message": f"No reference data found for task='{params.task_type}' library='{params.library}'",
            "available_tasks": SUPPORTED_TASKS,
            "available_libraries": SUPPORTED_LIBRARIES
        })
    return json.dumps({"found": True, **result}, indent=2)


@mcp.tool(
    name="forge_list_tasks",
    annotations={
        "title": "List Available Reference Tasks",
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": False
    }
)
async def forge_list_tasks() -> str:
    """List all tasks and libraries that have pre-loaded reference benchmark data.

    Use this to discover what Forge can benchmark with reference comparisons.

    Returns:
        str: JSON with supported tasks, libraries, and coverage matrix
    """
    tasks = list_reference_tasks()
    return json.dumps({
        "supported_libraries": SUPPORTED_LIBRARIES,
        "supported_tasks": SUPPORTED_TASKS,
        "reference_coverage": tasks
    }, indent=2)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _infer_task_key(description: str) -> str:
    """Infer a task key from a natural language description."""
    desc = description.lower()
    if any(w in desc for w in ["matmul", "matrix mul", "matrix multiply", "mm"]):
        return "matmul"
    if any(w in desc for w in ["dot", "dot product", "inner product"]):
        return "dot_product"
    if any(w in desc for w in ["svd", "singular value"]):
        return "svd"
    if any(w in desc for w in ["conv", "convolution"]):
        return "conv2d"
    if any(w in desc for w in ["relu", "activation"]):
        return "relu"
    if any(w in desc for w in ["softmax"]):
        return "softmax"
    if any(w in desc for w in ["norm", "normalize", "normalisation"]):
        return "norm"
    return "matmul"  # default


def _score_and_rank(results: dict, reference: dict) -> list:
    """Score libraries across speed and memory, return ranked list."""
    scores = []
    valid = {lib: r for lib, r in results.items() if "error" not in r and r.get("time_ms")}

    if not valid:
        return []

    min_time = min(r["time_ms"] for r in valid.values())
    min_mem = min(r.get("memory_mb", 999) for r in valid.values()) or 1

    for lib, result in valid.items():
        time_score = (min_time / result["time_ms"]) * 60       # 60% weight
        mem_score = (min_mem / max(result.get("memory_mb", 1), 0.1)) * 20  # 20% weight

        # Reference alignment score (10%)
        ref_score = 0
        if lib in reference and "time_ms" in reference[lib]:
            ref_time = reference[lib]["time_ms"]
            ratio = min(result["time_ms"], ref_time) / max(result["time_ms"], ref_time)
            ref_score = ratio * 10

        total = time_score + mem_score + ref_score
        scores.append({
            "library": lib,
            "total_score": round(total, 1),
            "time_ms": result["time_ms"],
            "memory_mb": result.get("memory_mb"),
            "speedup_vs_slowest": None
        })

    scores.sort(key=lambda x: x["total_score"], reverse=True)

    # Add speedup vs slowest
    if scores:
        slowest_time = max(s["time_ms"] for s in scores)
        for s in scores:
            s["speedup_vs_slowest"] = round(slowest_time / s["time_ms"], 2)

    return scores


def _build_summary(task: str, results: dict, ranked: list, reference: dict) -> str:
    """Build a human-readable summary of benchmark results."""
    if not ranked:
        return "Benchmark failed — no valid results."

    winner = ranked[0]
    lines = [f"Task: {task}", f"Winner: {winner['library'].upper()} ({winner['time_ms']:.1f}ms median)"]

    if len(ranked) > 1:
        slowest = ranked[-1]
        lines.append(f"Slowest: {slowest['library'].upper()} ({slowest['time_ms']:.1f}ms) — {winner['speedup_vs_slowest']}x slower")

    if winner["library"] in reference:
        ref = reference[winner["library"]]
        lines.append(f"vs MLPerf baseline: {ref.get('time_ms', '?')}ms on {ref.get('hardware', 'unknown hardware')}")

    return " | ".join(lines)


# ── Entry Point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    mcp.run()