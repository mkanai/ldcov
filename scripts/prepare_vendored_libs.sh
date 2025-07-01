#!/bin/bash
# Script to prepare vendored libraries for Docker builds
# This ensures git submodules are fully materialized in the build context

set -e

echo "Preparing vendored libraries for Docker build..."

# Get the script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_ROOT"

# Check if submodules are initialized
if [ ! -f "ldcov/io/bgen/zlib-ng/CMakeLists.txt" ] || [ ! -f "ldcov/io/bgen/zstd/lib/zstd.h" ]; then
    echo "Initializing git submodules..."
    git submodule update --init --recursive
fi

# Create a temporary directory for the Docker build context
BUILD_CONTEXT_DIR="${PROJECT_ROOT}/.docker-build-context"
# Ensure we're in the project root
cd "$PROJECT_ROOT"
rm -rf "$BUILD_CONTEXT_DIR"
mkdir -p "$BUILD_CONTEXT_DIR"

echo "Copying project files to build context..."
# Use tar to preserve permissions and handle special files
tar --exclude='.git' \
    --exclude='.docker-build-context' \
    --exclude='*.pyc' \
    --exclude='__pycache__' \
    --exclude='*.so' \
    --exclude='build' \
    --exclude='dist' \
    --exclude='*.egg-info' \
    --exclude='.pytest_cache' \
    --exclude='.coverage' \
    -cf - . | (cd "$BUILD_CONTEXT_DIR" && tar -xf -)

# Now we need to fully materialize the submodules (remove .git files and copy actual content)
echo "Materializing vendored libraries..."

# zlib-ng
if [ -d "ldcov/io/bgen/zlib-ng/.git" ]; then
    echo "  - Materializing zlib-ng..."
    rm -rf "$BUILD_CONTEXT_DIR/ldcov/io/bgen/zlib-ng"
    mkdir -p "$BUILD_CONTEXT_DIR/ldcov/io/bgen/zlib-ng"
    (cd "ldcov/io/bgen/zlib-ng" && tar --exclude='.git' -cf - .) | \
        (cd "$BUILD_CONTEXT_DIR/ldcov/io/bgen/zlib-ng" && tar -xf -)
fi

# zstd
if [ -d "ldcov/io/bgen/zstd/.git" ]; then
    echo "  - Materializing zstd..."
    rm -rf "$BUILD_CONTEXT_DIR/ldcov/io/bgen/zstd"
    mkdir -p "$BUILD_CONTEXT_DIR/ldcov/io/bgen/zstd"
    (cd "ldcov/io/bgen/zstd" && tar --exclude='.git' -cf - .) | \
        (cd "$BUILD_CONTEXT_DIR/ldcov/io/bgen/zstd" && tar -xf -)
fi

echo "Build context prepared at: $BUILD_CONTEXT_DIR"
echo ""
echo "To build the Docker image with vendored libraries:"
echo "  docker build -f Dockerfile -t ldcov $BUILD_CONTEXT_DIR"
echo ""
echo "Or use the build script:"
echo "  ./scripts/docker_build_with_vendored.sh"