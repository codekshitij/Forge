import sys
import os
import asyncio

# 1. Fix the path so it finds the mcp-server folder from ~/Forge
sys.path.append(os.path.abspath("./mcp-server"))

# 2. Now import the profile_code function
from tools.profiler import profile_code

async def main():
    print("🚀 Starting standalone profiler test...")
    
    setup_code = """
import numpy as np
print("Initializing 4096x4096 matrices (this takes a moment...)")
a = np.random.randn(4096, 4096).astype(np.float32)
b = np.random.randn(4096, 4096).astype(np.float32)
"""

    run_code = """
result = np.matmul(a, b)
"""

    print("Running Docker container...")
    
    result = await profile_code(
        setup_code=setup_code,
        run_code=run_code,
        library="numpy",
        iterations=5
    )
    
    print("\n📊 Benchmark Results:")
    print(f"Library:    {result.get('library')}")
    print(f"Status:     {result.get('status')}")
    print(f"Time (ms):  {result.get('time_ms')} ms")
    print(f"Memory:     {result.get('memory_mb')} MB")
    
    if result.get("error"):
        print(f"Error:      {result.get('error')}")

if __name__ == "__main__":
    asyncio.run(main())