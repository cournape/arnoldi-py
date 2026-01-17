.PHONY: lint tests

lint:
	ruff check src tests

tests:
	uv run --no-dev --group test pytest tests -s

tests-all: lint tests
