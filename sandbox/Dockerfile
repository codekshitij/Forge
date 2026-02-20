# Forge Sandbox — isolated execution environment
# Build: docker build -t forge-sandbox:latest ./sandbox
# Fixed for Apple Silicon (aarch64/M1/M2/M3)

FROM python:3.11-slim

# Security: run as non-root user
RUN useradd -m -u 1000 forge
WORKDIR /forge

# Install base scientific packages
RUN pip install --no-cache-dir numpy==1.26.4 scipy==1.13.0

# PyTorch — standard PyPI index works for ARM (no --index-url needed)
RUN pip install --no-cache-dir torch==2.2.2

# TensorFlow — try in order: generic → macos → latest
# tensorflow 2.16+ supports ARM natively via the main package
RUN pip install --no-cache-dir "tensorflow==2.16.1" || \
    pip install --no-cache-dir "tensorflow-macos" || \
    pip install --no-cache-dir "tensorflow"

# Phase 2 libraries (uncomment when ready)
# RUN pip install --no-cache-dir tinygrad
# RUN pip install --no-cache-dir "jax[cpu]"

# Ensure forge user owns the workdir
RUN chown forge:forge /forge

USER forge

CMD ["python"]