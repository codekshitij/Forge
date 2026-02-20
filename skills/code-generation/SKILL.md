# Forge Code Generation Skill
**When to use:** Any time you are generating ML library implementations for Forge benchmarks.

---

## Core Principle

Generate **idiomatic** code for each library — code that a senior engineer who knows that library well would write. Don't just port NumPy to PyTorch line-by-line. Use each library's strengths.

---

## Library-Specific Patterns

### NumPy
```python
import numpy as np

# ✅ Good — vectorized, idiomatic
a = np.random.randn(512, 512).astype(np.float32)
b = np.random.randn(512, 512).astype(np.float32)
result = np.matmul(a, b)      # or a @ b

# ❌ Bad — Python loops defeat the purpose
result = np.zeros((512, 512))
for i in range(512):
    for j in range(512):
        result[i, j] = np.dot(a[i], b[:, j])
```
**Key rules:**
- Always specify dtype explicitly (`float32` unless otherwise needed)
- Prefer `np.matmul` / `@` over `np.dot` for 2D+ arrays
- Use `np.linalg` for decompositions
- Avoid Python loops — always vectorize

---

### PyTorch
```python
import torch

# ✅ Good — GPU-aware, no_grad for inference
a = torch.randn(512, 512, dtype=torch.float32)
b = torch.randn(512, 512, dtype=torch.float32)

with torch.no_grad():
    result = torch.mm(a, b)    # or a @ b

result = result.numpy().tolist()  # serialize for validator
```
**Key rules:**
- Use `torch.no_grad()` for non-training operations
- Move to GPU only if benchmarking GPU performance (specify in task)
- `torch.mm` for 2D, `torch.bmm` for batched, `torch.matmul` for general
- Convert to list/numpy for `result` so validator can compare
- Use `torch.linalg` for decompositions (not deprecated `torch.svd`)

---

### TensorFlow
```python
import tensorflow as tf

# ✅ Good — eager execution, explicit dtype
a = tf.random.normal((512, 512), dtype=tf.float32)
b = tf.random.normal((512, 512), dtype=tf.float32)

result = tf.linalg.matmul(a, b).numpy().tolist()
```
**Key rules:**
- TF2 uses eager execution by default — no sessions
- Use `@tf.function` decorator only when explicitly benchmarking graph mode
- Always call `.numpy().tolist()` on result for serialization
- Use `tf.linalg` for linear algebra ops

---

### JAX
```python
import jax
import jax.numpy as jnp
from jax import jit

# ✅ Good — JIT compiled, functional style
a = jnp.ones((512, 512), dtype=jnp.float32)
b = jnp.ones((512, 512), dtype=jnp.float32)

@jit
def matmul(x, y):
    return jnp.matmul(x, y)

# Warm up JIT (first call compiles)
_ = matmul(a, b).block_until_ready()

result = matmul(a, b).block_until_ready()
result = result.tolist()
```
**Key rules:**
- Always use `block_until_ready()` — JAX is async by default
- Always include a JIT warmup call before the timed operation (profiler warmup handles this)
- Use `jax.numpy` (jnp) not numpy for operations
- JAX arrays are immutable — use functional patterns

---

### TinyGrad
```python
from tinygrad.tensor import Tensor
import numpy as np

# ✅ Good — lazy evaluation, explicit realize
a_np = np.random.randn(512, 512).astype(np.float32)
b_np = np.random.randn(512, 512).astype(np.float32)

a = Tensor(a_np)
b = Tensor(b_np)

result_tensor = a.matmul(b)
result_tensor.realize()  # trigger lazy evaluation
result = result_tensor.numpy().tolist()
```
**Key rules:**
- TinyGrad uses lazy evaluation — always call `.realize()` before timing completes
- Initialize from numpy arrays for predictable data
- Use `.numpy()` to extract results
- TinyGrad is designed for simplicity — don't use framework-specific tricks

---

## Common Mistakes to Avoid

| Mistake | Why it's wrong | Fix |
|---------|---------------|-----|
| Different random seeds per library | Results differ → validator fails | Use `np.random.seed(42)` and convert |
| Float64 in NumPy vs Float32 in PyTorch | Unfair comparison | Always `astype(np.float32)` |
| Not calling `.realize()` in TinyGrad | Time measurement invalid | Always realize before `result =` |
| Not calling `block_until_ready()` in JAX | JAX returns before compute finishes | Always block |
| Python loops anywhere | Defeats vectorization | Use library ops |
| Not assigning to `result` | Profiler can't capture output | Always `result = ...` |
