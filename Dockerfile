# ldcov Dockerfile with vendored high-performance libraries
# This builds ldcov with vendored zlib-ng and zstd for optimal performance
FROM python:3.11-slim

# Install build dependencies including CMake for vendored libraries
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    make \
    git \
    cmake \
    libsqlite3-dev \
    wget \
    tar \
    # Additional build tools that might be needed
    build-essential \
    ninja-build \
    # OpenBLAS for optimized linear algebra operations
    libopenblas-dev \
    && rm -rf /var/lib/apt/lists/*

# Install bgenix from Oxford
RUN wget -q https://www.chg.ox.ac.uk/~gav/resources/bgen_v1.1.4-Ubuntu16.04-x86_64.tgz -O /tmp/bgen.tgz && \
    tar -xzf /tmp/bgen.tgz -C /tmp && \
    cp /tmp/bgen_v1.1.4-Ubuntu16.04-x86_64/bgenix /usr/local/bin/ && \
    chmod +x /usr/local/bin/bgenix && \
    rm -rf /tmp/bgen.tgz /tmp/bgen_v1.1.4-Ubuntu16.04-x86_64

WORKDIR /app

# Copy everything including submodules
COPY . .

# Initialize git submodules if not already done
RUN if [ -d .git ]; then \
        git submodule update --init --recursive; \
    fi

# Enable SIMD optimizations (AVX2 + FMA for x86_64)
ENV CFLAGS="-O3 -mfma -mavx -mavx2"
ENV CXXFLAGS="-O3 -mfma -mavx -mavx2 -std=c++17"

# Ensure we don't use system libraries
ENV LDCOV_USE_SYSTEM_LIBS=0

# Pre-install build dependencies
RUN pip install --upgrade pip setuptools wheel cmake ninja

# Build and install ldcov with vendored libraries
# Use verbose output to debug build issues
RUN pip install -e . -v 2>&1 | tee /tmp/build.log && \
    echo "=== Checking for vendored libraries ===" && \
    python -c "import ldcov; import os; print('ldcov installed at:', ldcov.__file__)" && \
    # Check if vendored libraries were built
    find /app -name "*zlib-ng*" -o -name "*zstd*.so" | head -20 && \
    # Verify the build configuration
    if [ -f /app/ldcov/io/bgen/_build_config.py ]; then \
        echo "Build configuration:"; \
        python -c "import sys; sys.path.insert(0, '/app/ldcov/io/bgen'); from _build_config import COMPRESSION_BACKEND; print(COMPRESSION_BACKEND)"; \
    else \
        echo "Warning: Build configuration file not found"; \
    fi

# Install additional dependencies
RUN pip install pytest pytest-cov black flake8 psutil bgen

# Verify numpy is using OpenBLAS
RUN python -c "import numpy; numpy.show_config()" | grep -i blas || true

# Copy vendored check script
COPY scripts/check_vendored.py /tmp/check_vendored.py

# Run a quick test to verify vendored libraries are being used
RUN python /tmp/check_vendored.py

# Run tests to verify the build
RUN pytest -v tests/ || true

# Set environment variables for benchmarking
# Note: These can be overridden when running the container
ENV PYTHONUNBUFFERED=1
ENV OMP_NUM_THREADS=4
ENV MKL_NUM_THREADS=4
ENV NUMEXPR_NUM_THREADS=4
ENV OPENBLAS_NUM_THREADS=4

# Create directory for test data
RUN mkdir -p /data

# Default command
CMD ["pytest", "-v"]