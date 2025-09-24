.PHONY: lint tests

lint:
	ruff check src tests

tests:
	uv run pytest tests -s

tests-all: lint tests
