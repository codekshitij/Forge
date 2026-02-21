# Forge Code Generation Skill
**When to use:** Any time you are generating ML library implementations for Forge benchmarks.

---

## Core Principle

Generate **idiomatic** code for each library — code that a senior engineer who knows that library well would write. Don't just port NumPy to PyTorch line-by-line. Use each library's strengths.

**CRITICAL: Two-Phase Execution**
You must always split your generated code into two distinct strings:
1. `setup`: Imports, data initialization, and model loading. This phase is **NOT timed**.
2. `run`: The core operation to benchmark. This phase is **strictly timed** and must assign its final output to a variable named `result`.

**CRITICAL: Tensor Serialization**
Never use `.numpy().tolist()` on large tensors. It will crash the server.
Instead, ALWAYS serialize the final output as a statistical signature dictionary in the `run` block.

---

## Library-Specific Patterns

### NumPy
**Setup (`setup`):**
```python
import numpy as np

# Initialize data here. Not timed.
np.random.seed(42)
a = np.random.randn(512, 512).astype(np.float32)
b = np.random.randn(512, 512).astype(np.float32)
```
**Run (`run`):**
```python
# The pure operation
res = np.matmul(a, b)

# Signature dictionary
result = {
    "shape": list(res.shape),
    "mean": float(np.mean(res)),
    "sum": float(np.sum(res)),
    "std": float(np.std(res))
}
```
**Key rules:**
- Always specify dtype explicitly (`float32` unless otherwise needed)
- Prefer `np.matmul` / `@` over `np.dot` for 2D+ arrays
- Avoid Python loops — always vectorize

---

### PyTorch
**Setup (`setup`):**
```python
import torch

torch.manual_seed(42)
a = torch.randn(512, 512, dtype=torch.float32)
b = torch.randn(512, 512, dtype=torch.float32)
```
**Run (`run`):**
```python
with torch.no_grad():
    res = torch.mm(a, b)    

res_np = res.cpu().numpy()
result = {
    "shape": list(res_np.shape),
    "mean": float(res_np.mean()),
    "sum": float(res_np.sum()),
    "std": float(res_np.std())
}
```
**Key rules:**
- Use `torch.no_grad()` for non-training operations
- Move to GPU only if benchmarking GPU performance (specify in task)
- Use `torch.linalg` for decompositions (not deprecated `torch.svd`)

---

### TensorFlow
**Setup (`setup`):**
```python
import tensorflow as tf

tf.random.set_seed(42)
a = tf.random.normal((512, 512), dtype=tf.float32)
b = tf.random.normal((512, 512), dtype=tf.float32)
```
**Run (`run`):**
```python
res = tf.linalg.matmul(a, b)

res_np = res.numpy()
result = {
    "shape": list(res_np.shape),
    "mean": float(res_np.mean()),
    "sum": float(res_np.sum()),
    "std": float(res_np.std())
}
```
**Key rules:**
- TF2 uses eager execution by default — no sessions
- Use `@tf.function` decorator only when explicitly benchmarking graph mode

---

### JAX
**Setup (`setup`):**
```python
import jax
import jax.numpy as jnp
from jax import jit

a = jnp.ones((512, 512), dtype=jnp.float32)
b = jnp.ones((512, 512), dtype=jnp.float32)

@jit
def matmul(x, y):
    return jnp.matmul(x, y)
```
**Run (`run`):**
```python
res = matmul(a, b).block_until_ready()

res_np = jax.device_get(res)
result = {
    "shape": list(res_np.shape),
    "mean": float(res_np.mean()),
    "sum": float(res_np.sum()),
    "std": float(res_np.std())
}
```
**Key rules:**
- Always use `block_until_ready()` — JAX is async by default
- Do not add manual JIT warmup calls (the profiler's untimed warmup handles this)

---

### TinyGrad
**Setup (`setup`):**
```python
from tinygrad.tensor import Tensor
import numpy as np

np.random.seed(42)
a_np = np.random.randn(512, 512).astype(np.float32)
b_np = np.random.randn(512, 512).astype(np.float32)

a = Tensor(a_np)
b = Tensor(b_np)
```
**Run (`run`):**
```python
res = a.matmul(b)
res.realize()  

res_np = res.numpy()
result = {
    "shape": list(res_np.shape),
    "mean": float(np.mean(res_np)),
    "sum": float(np.sum(res_np)),
    "std": float(np.std(res_np))
}
```
**Key rules:**
- TinyGrad uses lazy evaluation — always call `.realize()` before generating the signature dictionary.

---

## Common Mistakes to Avoid

| Mistake | Why it's wrong | Fix |
|---------|---------------|-----|
| Putting data generation in the `run` block | Ruins the benchmark timing | Move `randn()`, `ones()`, etc. to `setup` |
| Calling `.tolist()` on a tensor | Crashes the server due to JSON payload limits | Always extract a signature dictionary |
| Different random seeds per library | Results differ → validator fails | Set consistent seeds in the `setup` block |
| Float64 in NumPy vs Float32 in PyTorch | Unfair comparison | Always enforce float32/float16 parity |
| Not calling `.realize()` in TinyGrad | Time measurement invalid | Always realize before extracting values |
| Not calling `block_until_ready()` in JAX | JAX returns before compute finishes | Always block |