.PHONY: clean lint format test coverage dist install

# Default target
all: clean lint test coverage

# Clean build files
clean:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info
	rm -rf ldcov.egg-info/
	find . -name __pycache__ -exec rm -rf {} +
	find . -name "*.pyc" -delete
	find . -name "*.pyo" -delete
	find . -name ".coverage" -delete
	find . -name "coverage.xml" -delete
	find . -name ".pytest_cache" -exec rm -rf {} +

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
