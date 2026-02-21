"""
Forge MCP Server
================
Put ML libraries through the fire. Find out which one is strongest.
"""

import json
import asyncio
from mcp.server.fastmcp import FastMCP
from pydantic import BaseModel, Field, ConfigDict
from typing import Optional

from tools.executor import run_benchmark
from tools.profiler import profile_code
from tools.validator import validate_output
from tools.reference import get_reference_data, list_reference_tasks
from tools.metal_profiler import profile_metal, is_metal_available

# ── Server Init ───────────────────────────────────────────────────────────────

mcp = FastMCP("forge_mcp")

SUPPORTED_LIBRARIES = ["numpy", "pytorch", "tensorflow", "tinygrad", "jax"]
SUPPORTED_TASKS = ["matmul", "dot_product", "svd", "conv2d", "relu", "softmax", "norm"]

# ── Input Models ──────────────────────────────────────────────────────────────

class CodeSnippet(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")
    setup: str = Field(..., description="Imports and data initialization. NOT timed.")
    run: str = Field(..., description="The core operation to benchmark. Timed. Must assign to 'result'.")

class BenchmarkInput(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")
    task_description: str = Field(..., min_length=5, max_length=500)
    libraries: list[str] = Field(..., min_length=2, max_length=5)
    generated_code: dict[str, CodeSnippet] = Field(...)
    iterations: int = Field(default=5, ge=3, le=20)

class MetalBenchmarkInput(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")
    task_description: str = Field(..., min_length=5, max_length=500)
    libraries: list[str] = Field(default=["pytorch"])
    generated_code: dict[str, CodeSnippet] = Field(...)
    iterations: int = Field(default=5, ge=3, le=20)

class ProfileInput(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")
    code: CodeSnippet = Field(...)
    library: str = Field(...)
    iterations: int = Field(default=5, ge=3, le=20)

class ValidateInput(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")
    output_a: str = Field(...)
    output_b: str = Field(...)
    tolerance: float = Field(default=1e-5, ge=0)

class ReferenceInput(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")
    task_type: str = Field(...)
    library: str = Field(...)

# ── Tools ─────────────────────────────────────────────────────────────────────

@mcp.tool(name="forge_run_benchmark")
async def forge_run_benchmark(params: BenchmarkInput) -> str:
    """Run a full benchmark across multiple ML libraries and return metrics."""
    results = {}

    for library in params.libraries:
        code = params.generated_code.get(library)
        if not code:
            results[library] = {"error": f"No code provided for {library}"}
            continue

        results[library] = await profile_code(code.setup, code.run, library, params.iterations)

    reference = {}
    task_key = _infer_task_key(params.task_description)
    for library in params.libraries:
        ref = get_reference_data(task_key, library)
        if ref:
            reference[library] = ref

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

    return json.dumps(final, indent=2, default=str)


@mcp.tool(name="forge_profile")
async def forge_profile(params: ProfileInput) -> str:
    """Profile a single ML code implementation — time, memory, and correctness."""
    result = await profile_code(params.code.setup, params.code.run, params.library, params.iterations)
    return json.dumps(result, indent=2, default=str)


@mcp.tool(name="forge_validate")
async def forge_validate(params: ValidateInput) -> str:
    """Check whether two library implementations produce numerically equivalent outputs."""
    result = validate_output(params.output_a, params.output_b, params.tolerance)
    return json.dumps(result, indent=2)


@mcp.tool(name="forge_get_reference")
async def forge_get_reference(params: ReferenceInput) -> str:
    """Fetch pre-loaded industry benchmark data (MLPerf, DS-1000)."""
    result = get_reference_data(params.task_type, params.library)
    if not result:
        return json.dumps({"found": False})
    return json.dumps({"found": True, **result}, indent=2)


@mcp.tool(name="forge_list_tasks")
async def forge_list_tasks() -> str:
    """List all tasks and libraries that have pre-loaded reference benchmark data."""
    tasks = list_reference_tasks()
    return json.dumps({
        "supported_libraries": SUPPORTED_LIBRARIES,
        "supported_tasks": SUPPORTED_TASKS,
        "reference_coverage": tasks
    }, indent=2)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _infer_task_key(description: str) -> str:
    desc = description.lower()
    if any(w in desc for w in ["matmul", "mm"]): return "matmul"
    if any(w in desc for w in ["dot"]): return "dot_product"
    if any(w in desc for w in ["svd"]): return "svd"
    if any(w in desc for w in ["conv"]): return "conv2d"
    if any(w in desc for w in ["relu"]): return "relu"
    if any(w in desc for w in ["softmax"]): return "softmax"
    if any(w in desc for w in ["norm"]): return "norm"
    return "matmul"

def _score_and_rank(results: dict, reference: dict) -> list:
    scores = []
    valid = {lib: r for lib, r in results.items() if "error" not in r and r.get("time_ms")}
    if not valid: return []

    min_time = min(r["time_ms"] for r in valid.values())
    min_mem = min(r.get("memory_mb", 999) for r in valid.values()) or 1

    for lib, result in valid.items():
        time_score = (min_time / result["time_ms"]) * 60
        mem_score = (min_mem / max(result.get("memory_mb", 1), 0.1)) * 20
        total = time_score + mem_score
        scores.append({
            "library": lib,
            "total_score": round(total, 1),
            "time_ms": result["time_ms"],
            "memory_mb": result.get("memory_mb")
        })

    scores.sort(key=lambda x: x["total_score"], reverse=True)
    if scores:
        slowest_time = max(s["time_ms"] for s in scores)
        for s in scores:
            s["speedup_vs_slowest"] = round(slowest_time / s["time_ms"], 2)
    return scores

def _build_summary(task: str, results: dict, ranked: list, reference: dict) -> str:
    if not ranked: return "Benchmark failed."
    winner = ranked[0]
    return f"Task: {task} | Winner: {winner['library'].upper()} ({winner['time_ms']:.1f}ms median)"


@mcp.tool(name="forge_metal_benchmark")
async def forge_metal_benchmark(params: MetalBenchmarkInput) -> str:
    """Benchmark ML operations on Apple Silicon Metal GPU vs CPU."""
    if not is_metal_available():
        return json.dumps({"error": "Metal/MPS not available."})

    cpu_results, metal_results = {}, {}

    for library in params.libraries:
        cpu_code = params.generated_code.get(library)
        metal_code = params.generated_code.get(f"{library}_metal")

        if not cpu_code:
            cpu_results[library] = {"status": "error"}
            continue

        cpu_results[library] = await profile_code(cpu_code.setup, cpu_code.run, library, params.iterations)

        if metal_code and library == "pytorch":
            metal_results[library] = await profile_metal(metal_code.setup, metal_code.run, library, params.iterations)

    speedups = {}
    for lib in params.libraries:
        cpu_t  = cpu_results.get(lib, {}).get("time_ms")
        metal_t = metal_results.get(lib, {}).get("time_ms")
        if cpu_t and metal_t and metal_t > 0:
            speedups[lib] = round(cpu_t / metal_t, 2)

    return json.dumps({
        "task": params.task_description,
        "cpu_results": cpu_results,
        "metal_results": metal_results,
        "gpu_speedups": speedups
    }, indent=2, default=str)

if __name__ == "__main__":
    mcp.run()