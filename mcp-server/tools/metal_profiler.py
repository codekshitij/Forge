"""
forge/tools/metal_profiler.py
==============================
Profiles PyTorch operations on Apple Silicon Metal GPU (MPS backend).

Cannot use Docker — Metal requires direct access to the host GPU.
Runs in an isolated subprocess using the host venv Python instead.

Architecture:
    CPU benchmarks  → Docker container (isolated)
    Metal benchmarks → Host subprocess via venv Python (GPU access)
"""

import asyncio
import json
import os
import sys
import tempfile
import textwrap
from typing import Any
from pathlib import Path

TIMEOUT_SECONDS = 120
WARMUP_RUNS = 3  # MPS needs more warmup — first run compiles Metal shaders


async def profile_metal(code: str, library: str, iterations: int = 5) -> dict[str, Any]:
    """
    Profile a Metal GPU code snippet by running it in a host subprocess.

    The code must use MPS device explicitly:
        device = torch.device("mps")
        a = torch.randn(512, 512, device=device)

    Args:
        code: Python code using MPS tensors. Must assign output to `result`.
        library: Library name (should be 'pytorch' for Metal).
        iterations: Number of timed runs. Median is reported.

    Returns:
        Dict with time_ms, status, device='metal', library.
    """
    runner_script = _build_metal_runner(code, iterations, WARMUP_RUNS)

    # Write runner to temp file
    tmp = tempfile.NamedTemporaryFile(
        mode="w", suffix=".py", delete=False, prefix="forge_metal_"
    )
    tmp.write(runner_script)
    tmp.close()

    try:
        result = await asyncio.wait_for(
            _run_subprocess(tmp.name, library),
            timeout=TIMEOUT_SECONDS
        )
        return result
    except asyncio.TimeoutError:
        return {
            "library": library,
            "device": "metal",
            "status": "timeout",
            "error": f"Metal benchmark timed out after {TIMEOUT_SECONDS}s",
            "time_ms": None,
        }
    except Exception as e:
        return {
            "library": library,
            "device": "metal",
            "status": "error",
            "error": str(e),
            "time_ms": None,
        }
    finally:
        os.unlink(tmp.name)


async def _run_subprocess(script_path: str, library: str) -> dict[str, Any]:
    """Run the Metal benchmark script as a host subprocess."""
    # Use the same Python that's running the MCP server (has torch+MPS)
    python = sys.executable

    proc = await asyncio.create_subprocess_exec(
        python, script_path,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        env={**os.environ, "PYTORCH_ENABLE_MPS_FALLBACK": "1"}
    )

    stdout, stderr = await proc.communicate()

    if proc.returncode != 0:
        err = stderr.decode().strip().splitlines()
        last_err = next((l for l in reversed(err) if l.strip()), "Unknown error")
        return {
            "library": library,
            "device": "metal",
            "status": "error",
            "error": last_err,
            "time_ms": None,
        }

    output = stdout.decode().strip()
    for line in reversed(output.splitlines()):
        line = line.strip()
        if line.startswith("{"):
            try:
                data = json.loads(line)
                data["library"] = library
                data["device"] = "metal"
                data.setdefault("status", "success")
                return data
            except json.JSONDecodeError:
                continue

    return {
        "library": library,
        "device": "metal",
        "status": "error",
        "error": f"Could not parse output: {output[:200]}",
        "time_ms": None,
    }


def _build_metal_runner(code: str, iterations: int, warmup: int) -> str:
    """Build the Metal benchmark runner script."""
    return textwrap.dedent(f'''\
import time
import json
import statistics
import sys

# Verify MPS is available before running
import torch
if not torch.backends.mps.is_available():
    print(json.dumps({{"status": "error", "error": "MPS not available on this system"}}))
    sys.exit(1)

USER_CODE = {repr(code)}

def run_once():
    ns = {{}}
    exec(compile(USER_CODE, "<forge-metal>", "exec"), ns)
    # Ensure GPU work is complete before timing ends
    if torch.backends.mps.is_available():
        torch.mps.synchronize()
    return ns.get("result", None)

# Warmup — MPS compiles Metal shaders on first run, must not be timed
for _ in range({warmup}):
    try:
        run_once()
    except Exception:
        pass

# Timed runs
times = []

for i in range({iterations}):
    torch.mps.synchronize()  # flush any pending GPU work before timing
    t0 = time.perf_counter()
    try:
        run_once()
    except Exception as e:
        print(json.dumps({{"status": "error", "error": str(e)}}))
        sys.exit(1)
    torch.mps.synchronize()  # wait for GPU to finish
    t1 = time.perf_counter()
    times.append((t1 - t0) * 1000)

print(json.dumps({{
    "time_ms": round(statistics.median(times), 3),
    "time_ms_mean": round(statistics.mean(times), 3),
    "time_ms_stdev": round(statistics.stdev(times) if len(times) > 1 else 0, 3),
    "iterations": {iterations},
    "all_times_ms": [round(t, 3) for t in times],
    "device_info": "Apple M-series GPU via MPS"
}}))
''')


def is_metal_available() -> bool:
    """Check if Metal/MPS is available on this system."""
    try:
        import torch
        return torch.backends.mps.is_available()
    except ImportError:
        return False