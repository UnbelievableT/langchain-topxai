.PHONY: test tests integration_tests lint format type build

test tests:
	uv run --locked --group test pytest --disable-socket --allow-unix-socket tests/unit_tests/

integration_tests:
	uv run --group test --group test_integration pytest -v --tb=short --timeout=180 tests/integration_tests/

lint:
	uv run --group lint ruff check .
	uv run --group lint ruff format --check .

format:
	uv run --group lint ruff format .
	uv run --group lint ruff check --fix .

type:
	uv run --all-groups mypy langchain_topxai tests

build:
	uv build
