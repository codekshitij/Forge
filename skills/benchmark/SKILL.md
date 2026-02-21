## Tool Selection

**forge_run_benchmark** — use for CPU-only comparisons across multiple libraries

**forge_metal_benchmark** — use when:
- User mentions "Metal", "MPS", "GPU", "Apple Silicon", or "M1/M2/M3/M4"
- User asks to compare CPU vs GPU
- User wants to see GPU speedup

When using forge_metal_benchmark, generated_code must include BOTH:
- `"pytorch"` key → standard CPU code (runs in Docker)
- `"pytorch_metal"` key → MPS code with `device = torch.device("mps")`

## Metal Code Template

\```python
import torch

device = torch.device("mps")
a = torch.randn(1024, 1024, dtype=torch.float32, device=device)
b = torch.randn(1024, 1024, dtype=torch.float32, device=device)

result = torch.mm(a, b)
torch.mps.synchronize()  # REQUIRED — waits for GPU to finish
result = result.cpu().tolist()
\```
```

But there's a deeper issue — Claude Desktop doesn't automatically read your skill files. You need to paste the skill content into your **Project Instructions** in Claude Desktop so Claude sees it on every message.

Go to your Forge project in Claude Desktop → Project Instructions → paste the contents of both `skills/benchmark/SKILL.md` and `skills/code-generation/SKILL.md` there.

Once that's done, try:
```
Forge: use forge_metal_benchmark to compare PyTorch CPU vs Metal GPU on 1024x1024 matmul