# Forge Benchmark Skill
**When to use:** Any time you are running a benchmark comparison across ML libraries using Forge.

---

## Pre-Benchmark Checklist

Before generating any code, always:

1. **Check reference data first** — call `forge_list_tasks` to see what reference coverage exists for the requested task. Tell the user which libraries have baselines.
2. **Clarify ambiguity** — if the user says "matrix multiply", confirm the size (e.g. 512x512, 1024x1024). Size matters enormously.
3. **Pick at least 2 libraries** — a benchmark with one library is just profiling.

---

## Fairness Rules (Always Follow)

These rules ensure the benchmark is scientifically valid:

- **Same operation, same data** — every library must operate on equivalent inputs (same shape, same dtype, same values). Generate the data once and convert, don't generate independently per library.
- **Warmup is mandatory** — the profiler does 2 warmup runs automatically. Never skip this.
- **Median over mean** — always use median timing. Mean is skewed by outliers. The profiler returns both; always report median.
- **No I/O in the timed section** — file reads, print statements, and data loading must happen before the timed operation.
- **Match dtypes** — if PyTorch uses float32, NumPy must use float32 too. Don't compare float32 vs float64.
- **Assign output to `result`** — the profiler captures the variable named `result`. Always assign the output to it.

---

## Code Template Per Library

All generated code must follow this structure:

```python
# 1. Imports
import numpy as np  # or torch, tensorflow, etc.

# 2. Setup (not timed — happens in warmup too, that's fine)
import numpy as np
a = np.random.randn(512, 512).astype(np.float32)
b = np.random.randn(512, 512).astype(np.float32)

# 3. The operation — assign to `result`
result = np.matmul(a, b)
```

**Never include:**
- `time.time()` calls (the profiler handles timing)
- `print()` statements inside the benchmark
- File I/O in the timed section
- Model loading (unless benchmarking inference — then loading is in setup)

---

## Iterations Guide

| Task complexity | Recommended iterations |
|----------------|----------------------|
| Simple ops (matmul, relu) | 10 |
| Medium ops (conv2d, norm) | 7 |
| Heavy ops (SVD, training step) | 5 |
| Very heavy (full forward pass) | 3 |

---

## Reporting Results

After `forge_run_benchmark` returns, always:

1. State the **winner clearly** with its median time
2. Show the **speedup ratio** (e.g. "2.4x faster than TensorFlow")
3. If reference data exists, show the **delta vs baseline** and note hardware differences
4. Call out any **errors or timeouts** — don't silently skip failed libraries
5. Provide a **1-sentence interpretation** (e.g. "PyTorch wins here due to cuDNN's optimized GEMM kernels")
