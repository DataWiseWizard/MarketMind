.PHONY: check lint test format clean

# Run both linting and tests
check: lint test

# Run ruff linter
lint:
	ruff check src/ tests/

# Run pytest unit and integration test suite
test:
	pytest tests/ -v

# Auto-fix linting issues
format:
	ruff check --fix src/ tests/

# Clean temporary cache files
clean:
	rm -rf .pytest_cache .ruff_cache __pycache__ */__pycache__ */*/__pycache__
