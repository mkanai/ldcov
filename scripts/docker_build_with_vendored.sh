#!/bin/bash
# Docker build script that ensures vendored libraries are included

set -e

# Get the script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_ROOT"

# Prepare vendored libraries
echo "Preparing vendored libraries..."
./scripts/prepare_vendored_libs.sh

# Build Docker image
echo ""
echo "Building Docker image with vendored libraries..."
docker build -f Dockerfile -t ldcov:latest .docker-build-context/

# Cleanup
read -p "Remove temporary build context? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    rm -rf .docker-build-context/
    echo "Build context cleaned up."
fi

echo ""
echo "Docker image built successfully: ldcov:latest"
echo "Run tests with: docker run ldcov:latest pytest -v"