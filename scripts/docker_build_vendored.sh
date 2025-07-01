#!/bin/bash
# Script to build Docker image with vendored libraries properly included

set -e

echo "Building ldcov Docker image with vendored libraries..."

# Ensure we're in the project root
cd "$(dirname "$0")/.."

# Check if submodules are initialized
if [ ! -f "ldcov/io/bgen/zlib-ng/CMakeLists.txt" ] || [ ! -f "ldcov/io/bgen/zstd/CMakeLists.txt" ]; then
    echo "Initializing git submodules..."
    git submodule update --init --recursive
fi

# Build the Docker image
echo "Building Docker image..."
docker build -f Dockerfile -t ldcov:vendored . --no-cache

echo "Build complete! Testing vendored library configuration..."

# Test that vendored libraries are being used
docker run --rm ldcov:vendored python -c "
import os
import sys
try:
    from ldcov.io.bgen import BgenReader
    sys.path.insert(0, '/app/ldcov/io/bgen')
    try:
        from _build_config import COMPRESSION_BACKEND, get_build_info
        info = get_build_info()
        print(f'Build configuration: {COMPRESSION_BACKEND}')
        print(f'Build info: {info}')
        if COMPRESSION_BACKEND == 'vendored':
            print('✅ SUCCESS: Using vendored libraries!')
            sys.exit(0)
        else:
            print('❌ ERROR: Using system libraries instead of vendored')
            sys.exit(1)
    except ImportError as e:
        print(f'❌ ERROR: Build configuration module not found: {e}')
        sys.exit(1)
except Exception as e:
    print(f'❌ ERROR: {e}')
    sys.exit(1)
"

if [ $? -eq 0 ]; then
    echo "✅ Docker image successfully built with vendored libraries!"
    echo "Run benchmarks with: docker run --rm -v \$(pwd):/data -w /data ldcov:vendored python benchmarks/quick_comparison.py"
else
    echo "❌ Failed to build with vendored libraries"
    exit 1
fi