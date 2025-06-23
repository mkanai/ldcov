.PHONY: clean lint test coverage docs dist install

# Default target
all: clean lint test coverage

# Clean build files
clean:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info
	find . -name __pycache__ -exec rm -rf {} +
	find . -name "*.pyc" -delete
	find . -name "*.pyo" -delete
	find . -name "*.pyd" -delete
	find . -name "*.so" -delete
	# IMPORTANT: Only remove Cython-generated files, NOT hand-written C++ files
	# Cython-generated files (safe to remove):
	rm -f ldcov/io/bgen/_bgen.cpp
	rm -f ldcov/io/bgen/_decompressor.cpp
	rm -f ldcov/io/bgen/header.cpp
	rm -f ldcov/io/bgen/reader.cpp
	rm -f ldcov/io/bgen/reader_v1.cpp
	rm -f ldcov/io/bgen/variant.cpp
	# Remove only Cython-generated .c files (NOT vendored library files)
	# Only remove .c files in the main ldcov directory, not in vendor subdirectories
	find ldcov -name "*.c" -not -path "*/zlib-ng/*" -not -path "*/zstd/*" -not -path "reference_only/*" -delete
	# Remove Cython HTML output (but NOT from vendored libraries)
	find ldcov/io/bgen -name "*.html" -not -path "*/zlib-ng/*" -not -path "*/zstd/*" -delete
	# Remove coverage files
	find . -name ".coverage" -delete
	find . -name "coverage.xml" -delete
	find . -name ".pytest_cache" -exec rm -rf {} +
	rm -rf ldcov.egg-info/

# Lint code
lint:
	flake8 ldcov tests
	black --check ldcov tests

# Format code
format:
	black ldcov tests

# Run tests
test:
	pytest tests/

# Run tests with coverage
coverage:
	pytest --cov=ldcov tests/ --cov-report=xml

# Build distribution packages
dist: clean
	python -m build

# Install for development
install:
	pip install -e .[dev]

# Docker build with vendored libraries
docker-build:
	./scripts/docker_build_with_vendored.sh

# Prepare vendored libraries for Docker build (without building)
docker-prepare:
	./scripts/prepare_vendored_libs.sh
