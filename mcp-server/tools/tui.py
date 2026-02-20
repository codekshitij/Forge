"""
forge/tools/tui.py
==================
Textual TUI for displaying benchmark results in a beautiful terminal interface.
Launched automatically after a benchmark run completes.

Key design: uses RichLog widget which accepts Rich renderables (Table, Text)
directly — avoids the MarkupError caused by passing ANSI strings to Static.
"""

import asyncio
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

from textual.app import App, ComposeResult
from textual.widgets import Header, Footer, Static, RichLog
from rich.table import Table
from rich.text import Text
from rich import box


# ── TUI App ───────────────────────────────────────────────────────────────────

class ForgeApp(App):
    """Forge benchmark results TUI."""

    CSS = """
    Screen {
        background: #0d0d14;
    }
    Header {
        background: #1a1a2e;
        color: #ff6b35;
        text-style: bold;
    }
    Footer {
        background: #1a1a2e;
        color: #666666;
    }
    #title-bar {
        height: 3;
        background: #1a1a2e;
        border-bottom: solid #ff6b35;
        padding: 0 2;
        content-align: left middle;
        color: #ff6b35;
        text-style: bold;
    }
    RichLog {
        height: auto;
        margin: 1 2;
        background: #0d0d14;
    }
    #summary-log {
        background: #1a1a2e;
        border: solid #ff6b35;
        padding: 1 2;
    }
    #reference-log {
        background: #0f1f1a;
        border: solid #00d4aa;
        padding: 1 2;
    }
    """

    BINDINGS = [("q", "quit", "Quit")]

    def __init__(self, benchmark_data: dict[str, Any]):
        super().__init__()
        self.benchmark_data = benchmark_data

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)

        task   = self.benchmark_data.get("task", "Unknown task")
        winner = self.benchmark_data.get("winner", "N/A")

        yield Static(
            f"⚒  FORGE  |  {task}  |  Winner: {winner.upper()}",
            id="title-bar"
        )
        yield RichLog(id="summary-log",  markup=True,  highlight=False, wrap=True)
        yield RichLog(id="results-log",  markup=False, highlight=False)
        yield RichLog(id="chart-log",    markup=False, highlight=False)
        yield RichLog(id="reference-log",markup=False, highlight=False)
        yield Footer()

    def on_mount(self) -> None:
        """Write Rich renderables into each log after widgets are mounted."""
        ranking   = self.benchmark_data.get("ranking", [])
        results   = self.benchmark_data.get("results", {})
        reference = self.benchmark_data.get("reference", {})
        summary   = self.benchmark_data.get("summary", "")

        # Summary
        self.query_one("#summary-log", RichLog).write(
            Text.from_markup(f"[bold #ff6b35]Summary[/]\n{summary}")
        )

        # Results table — write the Rich Table object directly (no ANSI strings)
        self.query_one("#results-log", RichLog).write(
            _build_results_table(ranking, results)
        )

        # Bar chart
        chart_log = self.query_one("#chart-log", RichLog)
        for line in _build_bar_chart_lines(ranking):
            chart_log.write(line)

        # Reference
        ref_log = self.query_one("#reference-log", RichLog)
        if reference:
            ref_log.write(Text.from_markup("[bold #00d4aa]Industry Reference Comparison[/]"))
            for line in _build_reference_lines(reference, ranking):
                ref_log.write(line)

    def action_quit(self) -> None:
        self.exit()


# ── Rich Rendering Helpers ────────────────────────────────────────────────────

def _build_results_table(ranking: list, results: dict) -> Table:
    """Return a Rich Table object — written directly into RichLog, no ANSI strings."""
    table = Table(
        title="Benchmark Results",
        box=box.SIMPLE_HEAVY,
        title_style="bold #ff6b35",
        header_style="bold #888888",
        border_style="#333333",
        show_lines=True
    )
    table.add_column("Rank",        justify="center", style="#666666", width=6)
    table.add_column("Library",     style="bold white",                width=14)
    table.add_column("Time (ms)",   justify="right",  style="#00d4ff", width=12)
    table.add_column("Memory (MB)", justify="right",  style="#a8ff78", width=14)
    table.add_column("Score",       justify="right",  style="#ffb347", width=8)
    table.add_column("Speedup",     justify="right",  style="#ff6b35", width=10)
    table.add_column("Status",                                         width=10)

    medals = ["🥇", "🥈", "🥉"]

    for i, entry in enumerate(ranking):
        lib        = entry["library"]
        result     = results.get(lib, {})
        medal      = medals[i] if i < 3 else f"#{i+1}"
        status_str = "✓ ok" if result.get("status") == "success" else f"✗ {result.get('status', '?')}"

        time_str    = f"{entry['time_ms']:.2f}"          if entry.get("time_ms")            else "—"
        mem_str     = f"{entry['memory_mb']:.1f}"        if entry.get("memory_mb")          else "—"
        score_str   = f"{entry['total_score']:.1f}"      if entry.get("total_score")        else "—"
        speedup_str = f"{entry['speedup_vs_slowest']}x"  if entry.get("speedup_vs_slowest") else "—"

        table.add_row(
            medal, lib.upper(), time_str, mem_str,
            score_str, speedup_str, status_str,
            style="bold" if i == 0 else ""
        )

    return table


def _build_bar_chart_lines(ranking: list) -> list:
    """Return a list of Rich Text objects — one per bar. No ANSI strings."""
    if not ranking:
        return []

    lines  = [Text.from_markup("[bold #ff6b35]Execution Time Comparison[/]")]
    max_t  = max((r["time_ms"] for r in ranking if r.get("time_ms")), default=1)
    colors = ["#ff6b35", "#00d4ff", "#a8ff78", "#ffb347", "#ff4d6d"]

    for i, entry in enumerate(ranking):
        if not entry.get("time_ms"):
            continue
        lib    = entry["library"].upper().ljust(12)
        t      = entry["time_ms"]
        filled = int((t / max_t) * 40)
        bar    = "█" * filled + "░" * (40 - filled)
        c      = colors[i % len(colors)]
        lines.append(Text.from_markup(
            f"[{c}]{lib}[/] [{c}]{bar}[/] [white]{t:.2f}ms[/]"
        ))

    return lines


def _build_reference_lines(reference: dict, ranking: list) -> list:
    """Return a list of Rich Text objects for reference comparison."""
    lines = []
    for lib, ref in reference.items():
        live   = next((r for r in ranking if r["library"] == lib), None)
        ref_t  = ref.get("time_ms", "?")
        live_t = live["time_ms"] if live else None

        delta_str = ""
        if live_t and isinstance(ref_t, (int, float)):
            delta     = ((live_t - ref_t) / ref_t) * 100
            sign      = "+" if delta > 0 else ""
            color     = "#ff4d6d" if delta > 20 else "#a8ff78" if delta < -5 else "#ffb347"
            delta_str = f"[{color}]({sign}{delta:.1f}% vs baseline)[/]"

        live_str = f"{live_t:.2f}ms" if live_t else "—"
        lines.append(Text.from_markup(
            f"[bold white]{lib.upper()}[/] "
            f"live=[#00d4ff]{live_str}[/] "
            f"ref=[#888888]{ref_t}ms[/] "
            f"src=[#555555]{ref.get('source', '?')}[/] "
            f"hw=[#555555]{ref.get('hardware', '?')}[/] "
            f"{delta_str}"
        ))
    return lines


# ── Launcher ──────────────────────────────────────────────────────────────────

def prepare_tui_command(benchmark_data: dict[str, Any]) -> str:
    """
    Write benchmark data + a .command file to /tmp, return its path.
    Claude Desktop MCP server cannot open GUI windows directly, so we
    prepare the file and return the open command for Claude to show the user.
    """
    server_dir  = Path(__file__).parent.parent
    python_path = sys.executable

    data_file = tempfile.NamedTemporaryFile(
        mode="w", suffix=".json", delete=False, prefix="forge_data_"
    )
    json.dump(benchmark_data, data_file)
    data_file.close()
    data_path = data_file.name

    cmd_content = (
        "#!/bin/bash\n"
        f"{python_path} -c \"\n"
        "import sys, json, os\n"
        f"sys.path.insert(0, {repr(str(server_dir))})\n"
        "from tools.tui import ForgeApp\n"
        f"with open({repr(data_path)}) as f: data = json.load(f)\n"
        "ForgeApp(data).run()\n"
        f"os.unlink({repr(data_path)})\n"
        "\"\n"
    )

    cmd_file = tempfile.NamedTemporaryFile(
        mode="w", suffix=".command", delete=False, prefix="forge_tui_"
    )
    cmd_file.write(cmd_content)
    cmd_file.close()
    os.chmod(cmd_file.name, 0o755)
    return cmd_file.name


async def launch_tui(benchmark_data: dict[str, Any]) -> str:
    """
    Prepare TUI launch, try to open automatically, return open command as fallback.
    """
    cmd_path = prepare_tui_command(benchmark_data)

    # Try direct open (works when called from terminal)
    try:
        result = subprocess.run(["open", cmd_path], capture_output=True, timeout=3)
        if result.returncode == 0:
            return "TUI launched ✓"
    except Exception:
        pass

    # Fallback — Claude will show this to the user
    return f"open {cmd_path}"


# ── Standalone test ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    demo_data = {
        "task": "Matrix multiply 512x512",
        "task_key": "matmul",
        "libraries_tested": ["numpy", "pytorch", "tensorflow"],
        "winner": "pytorch",
        "summary": "PyTorch wins | 3.2x faster than TensorFlow",
        "results": {
            "numpy":      {"status": "success", "time_ms": 8.42,  "memory_mb": 4.1},
            "pytorch":    {"status": "success", "time_ms": 2.11,  "memory_mb": 6.3},
            "tensorflow": {"status": "success", "time_ms": 6.73,  "memory_mb": 12.1},
        },
        "ranking": [
            {"library": "pytorch",    "time_ms": 2.11, "memory_mb": 6.3,
             "total_score": 91.2, "speedup_vs_slowest": 4.0},
            {"library": "numpy",      "time_ms": 8.42, "memory_mb": 4.1,
             "total_score": 62.4, "speedup_vs_slowest": 1.0},
            {"library": "tensorflow", "time_ms": 6.73, "memory_mb": 12.1,
             "total_score": 55.1, "speedup_vs_slowest": 1.25},
        ],
        "reference": {
            "pytorch": {
                "source": "MLPerf v3.1",
                "time_ms": 2.4,
                "hardware": "NVIDIA A100",
                "notes": "FP32, single GPU"
            }
        }
    }
    ForgeApp(demo_data).run()