"""
forge/tools/profiler.py
=======================
Measures execution time (median over N runs) and peak memory usage
by running code inside an isolated Docker container per benchmark run.

Each run gets a fresh container — no state leakage between libraries.
"""

import asyncio
import json
import textwrap
import tempfile
import os
from typing import Any

import docker

DOCKER_IMAGE = "forge-sandbox:latest"
TIMEOUT_SECONDS = 90
WARMUP_RUNS = 2
MEMORY_LIMIT = "2g"     # Max RAM per container
CPU_LIMIT = 1.0         # CPU cores per container


def _get_docker_client():
    try:
        return docker.from_env()
    except Exception as e:
        raise RuntimeError(
            f"Docker is not running or not installed. "
            f"Please start Docker Desktop and try again. Error: {e}"
        )


async def profile_code(code: str, library: str, iterations: int = 5) -> dict[str, Any]:
    """
    Profile a code snippet inside an isolated Docker container.

    Spins up a fresh forge-sandbox container, runs the code with
    warmup + timed iterations, captures time and memory, then destroys
    the container.

    Args:
        code: Python code to execute. Must assign output to `result`.
        library: Library name (for labeling results).
        iterations: Number of timed runs (median will be used).

    Returns:
        Dict with time_ms, memory_mb, status, iterations, library.
    """
    runner_script = _build_runner(code, iterations, WARMUP_RUNS)

    try:
        result = await asyncio.wait_for(
            _run_in_docker(runner_script, library),
            timeout=TIMEOUT_SECONDS + 10  # extra buffer for container startup
        )
        return result
    except asyncio.TimeoutError:
        return {
            "library": library,
            "status": "timeout",
            "error": f"Container timed out after {TIMEOUT_SECONDS}s",
            "time_ms": None,
            "memory_mb": None
        }
    except Exception as e:
        return {
            "library": library,
            "status": "error",
            "error": str(e),
            "time_ms": None,
            "memory_mb": None
        }


async def _run_in_docker(runner_script: str, library: str) -> dict[str, Any]:
    """Spin up a Docker container, run the script, capture output, destroy container."""

    loop = asyncio.get_event_loop()

    def _blocking_docker_run():
        client = _get_docker_client()

        with tempfile.NamedTemporaryFile(
            mode="w",
            suffix=".py",
            delete=False,
            prefix="forge_runner_"
        ) as f:
            f.write(runner_script)
            tmp_path = f.name

        try:
            output = client.containers.run(
                image=DOCKER_IMAGE,
                command="python /forge/runner.py",
                volumes={
                    tmp_path: {"bind": "/forge/runner.py", "mode": "ro"}
                },
                mem_limit=MEMORY_LIMIT,
                nano_cpus=int(CPU_LIMIT * 1e9),
                network_disabled=True,
                remove=True,
                detach=False,
                stdout=True,
                stderr=True,
                security_opt=["no-new-privileges:true"],
                cap_drop=["ALL"],
            )

            output_str = output.decode("utf-8").strip() if isinstance(output, bytes) else str(output)
            return _parse_output(output_str, library)

        except docker.errors.ContainerError as e:
            stderr = e.stderr.decode("utf-8") if e.stderr else str(e)
            return {
                "library": library,
                "status": "error",
                "error": _clean_error(stderr),
                "time_ms": None,
                "memory_mb": None
            }
        except docker.errors.ImageNotFound:
            return {
                "library": library,
                "status": "error",
                "error": (
                    f"Docker image '{DOCKER_IMAGE}' not found. "
                    f"Run: docker build -t {DOCKER_IMAGE} ./sandbox"
                ),
                "time_ms": None,
                "memory_mb": None
            }
        finally:
            os.unlink(tmp_path)

    result = await loop.run_in_executor(None, _blocking_docker_run)
    return result


def _parse_output(output: str, library: str) -> dict[str, Any]:
    """Parse JSON output from the runner script."""
    lines = output.strip().splitlines()
    for line in reversed(lines):
        line = line.strip()
        if line.startswith("{"):
            try:
                data = json.loads(line)
                data["library"] = library
                data.setdefault("status", "success")
                return data
            except json.JSONDecodeError:
                continue

    return {
        "library": library,
        "status": "error",
        "error": f"Could not parse runner output: {output[:300]}",
        "time_ms": None,
        "memory_mb": None
    }


def _build_runner(code: str, iterations: int, warmup: int) -> str:
    """Build the runner script that executes inside the Docker container."""
    return textwrap.dedent(f'''\
import time
import json
import tracemalloc
import statistics
import sys

USER_CODE = {repr(code)}

def run_once():
    ns = {{}}
    exec(compile(USER_CODE, "<forge>", "exec"), ns)
    return ns.get("result", None)

# Warmup runs (not timed)
for _ in range({warmup}):
    try:
        run_once()
    except Exception:
        pass

# Timed runs
times = []
peak_memories = []

for i in range({iterations}):
    tracemalloc.start()
    t0 = time.perf_counter()
    try:
        run_once()
    except Exception as e:
        print(json.dumps({{"status": "error", "error": str(e)}}))
        sys.exit(1)
    t1 = time.perf_counter()
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    times.append((t1 - t0) * 1000)
    peak_memories.append(peak / 1024 / 1024)

print(json.dumps({{
    "time_ms": round(statistics.median(times), 3),
    "time_ms_mean": round(statistics.mean(times), 3),
    "time_ms_stdev": round(statistics.stdev(times) if len(times) > 1 else 0, 3),
    "memory_mb": round(statistics.median(peak_memories), 3),
    "iterations": {iterations},
    "all_times_ms": [round(t, 3) for t in times]
}}))
''')


def _clean_error(stderr: str) -> str:
    """Extract the most relevant line from a traceback."""
    lines = stderr.strip().splitlines()
    for line in reversed(lines):
        if line.strip():
            return line.strip()
    return stderr[:300]
