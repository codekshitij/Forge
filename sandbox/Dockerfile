# Forge Sandbox — isolated execution environment
# Build: docker build -t forge-sandbox:latest ./sandbox
#
# Key fix: installs OpenBLAS + libgomp so PyTorch uses optimized BLAS
# instead of falling back to a naive single-threaded C++ loop.
# Without this, PyTorch matmul on ARM can be 100x slower than expected.

FROM python:3.11-slim

# Install system-level BLAS/LAPACK libraries BEFORE pip installing torch
# - libopenblas-dev: OpenBLAS (fast matrix ops for PyTorch)
# - libgomp1: OpenMP (multi-threaded parallelism)
# - gfortran: required by some LAPACK builds
RUN apt-get update && apt-get install -y --no-install-recommends \
    libopenblas-dev \
    libgomp1 \
    gfortran \
    && rm -rf /var/lib/apt/lists/*

# Security: run as non-root user
RUN useradd -m -u 1000 forge
WORKDIR /forge

# Install psutil — used for accurate RSS memory measurement
RUN pip install --no-cache-dir psutil==5.9.8

# Base scientific packages
RUN pip install --no-cache-dir numpy==1.26.4 scipy==1.13.0

# PyTorch — with OpenBLAS installed above, torch will use it automatically
RUN pip install --no-cache-dir torch==2.2.2

# TensorFlow
RUN pip install --no-cache-dir "tensorflow==2.16.1" || \
    pip install --no-cache-dir "tensorflow-macos" || \
    pip install --no-cache-dir "tensorflow"

# Cap thread counts to match the CPU limit we give each container (2 cores)
# Prevents over-subscription when running multiple containers in parallel
ENV OPENBLAS_NUM_THREADS=2
ENV OMP_NUM_THREADS=2
ENV MKL_NUM_THREADS=2

# Phase 2 (uncomment when ready)
# RUN pip install --no-cache-dir tinygrad
# RUN pip install --no-cache-dir "jax[cpu]"

RUN chown forge:forge /forge
USER forge

CMD ["python"]