.PHONY: install test lint typecheck fmt demo bench check clean

PYTHON ?= python3
VENV   ?= .venv
BIN    := $(VENV)/bin

install: $(BIN)/evalkit

$(BIN)/evalkit: pyproject.toml
	$(PYTHON) -m venv $(VENV)
	$(BIN)/pip install -q --upgrade pip
	$(BIN)/pip install -q -e '.[dev]'
	@touch $@

test: install
	$(BIN)/pytest -q

lint: install
	$(BIN)/ruff check .
	$(BIN)/ruff format --check .
	$(BIN)/mypy

typecheck: install
	$(BIN)/mypy

fmt: install
	$(BIN)/ruff format .
	$(BIN)/ruff check --fix .

## Offline: baseline vs candidate on the scripted `fake` provider. No API key.
demo: install
	EVALKIT=$(BIN)/evalkit bash scripts/demo.sh

## Harness overhead and statistics timings on the fake provider.
bench: install
	$(BIN)/python -m bench.bench_harness

check: lint test

clean:
	rm -rf runs .evalkit-cache .pytest_cache .mypy_cache .ruff_cache .hypothesis build dist
