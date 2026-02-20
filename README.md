# ⚒ Forge
> Put ML libraries through the fire. Find out which one is strongest.

Forge is an AI-powered, on-demand benchmarking tool for ML libraries. Describe a task in plain English to Claude Desktop, and Forge generates idiomatic implementations in PyTorch, TensorFlow, NumPy, JAX, and TinyGrad — then executes them, measures real performance, and compares against industry reference data (MLPerf, DS-1000).

Results appear in a beautiful Textual TUI in your terminal.

---

## Supported Libraries

| Library | Status |
|---------|--------|
| NumPy | ✅ Phase 1 |
| PyTorch | ✅ Phase 1 |
| TensorFlow | ✅ Phase 1 |
| TinyGrad | 🔜 Phase 2 |
| JAX | 🔜 Phase 2 |

## Supported Tasks

`matmul` · `dot_product` · `svd` · `conv2d` · `relu` · `softmax` · `norm`

---

## Setup

### 1. Prerequisites

- Python 3.11+
- [Docker Desktop](https://www.docker.com/products/docker-desktop/) installed and running

### 2. Build the sandbox image

This is a one-time step. The image pre-installs all ML libraries so containers start fast.

```bash
docker build -t forge-sandbox:latest ./sandbox
```

### 3. Install MCP server dependencies

```bash
cd mcp-server
pip install -r requirements.txt
```

### 4. Configure Claude Desktop

Add to your `claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "forge": {
      "command": "python",
      "args": ["/path/to/forge/mcp-server/server.py"],
      "env": {}
    }
  }
}
```

### 5. Add Skills to Claude

Copy the contents of `skills/benchmark/SKILL.md` and `skills/code-generation/SKILL.md` into your Claude project instructions.

### 6. Start benchmarking

Open Claude Desktop and say:

> "Forge: benchmark matrix multiply 512x512 in NumPy vs PyTorch vs TensorFlow"

Claude will generate implementations, spin up isolated Docker containers per library, measure performance, and open a TUI showing results.

---

## Example Prompts

```
Forge: compare relu activation across numpy and pytorch on a 1M element tensor
Forge: which is faster for SVD — numpy or pytorch?
Forge: benchmark softmax in pytorch vs tensorflow, show reference data
Forge: run the full matmul suite across all available libraries
```

---

## Project Structure

```
forge/
├── mcp-server/
│   ├── server.py          # MCP entry point (register with Claude Desktop)
│   ├── tools/
│   │   ├── profiler.py    # Time + memory measurement
│   │   ├── executor.py    # Parallel execution across libraries
│   │   ├── validator.py   # Output equivalence checking
│   │   ├── reference.py   # Industry reference data queries
│   │   └── tui.py         # Textual terminal UI
│   └── requirements.txt
│
├── skills/
│   ├── benchmark/SKILL.md       # Fair benchmarking rules
│   └── code-generation/SKILL.md # Idiomatic code per library
│
├── reference-data/
│   ├── mlperf/training.json     # MLPerf benchmarks
│   └── ds1000/tasks.json        # DS-1000 baselines
│
└── README.md
```

---

## Phases

- **Phase 1** (current) — NumPy, PyTorch, TensorFlow · Linear algebra tasks · TUI results
- **Phase 2** — TinyGrad, JAX · Neural network ops · Reference scoring
- **Phase 3** — Leaderboard · Community data · Shareable reports
- **Phase 4** — GPU profiling · Statistical confidence · Public API

---

## Contributing Reference Data

See `docs/CONTRIBUTING.md` for how to add benchmark results from papers or your own hardware.

---

*Built with Claude Desktop + MCP + Textual*
