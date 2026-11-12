# EE6405 Table 8 -- common tasks.  Run `make help` for the list.
SHELL := /bin/bash
PY ?= python
CONDA_ENV ?= ee6405

.DEFAULT_GOAL := help

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | awk 'BEGIN{FS=":.*?## "}{printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}'

setup: ## Install core + dev deps into the active environment
	$(PY) -m pip install --upgrade pip
	$(PY) -m pip install -r requirements.txt -r requirements-dev.txt
	$(PY) -m pip install -e .

setup-ml: ## Install the optional heavy ML deps (local embeddings / local LLM baselines)
	$(PY) -m pip install -r requirements-ml.txt

ingest: ## Build the RAG index from data/raw -> data/processed/index.sqlite
	$(PY) -m ee6405.ingest.pipeline --source data/raw --rebuild

app: ## Launch the Streamlit front-end
	$(PY) -m streamlit run src/ee6405/app/main.py

eval: ## Run the baseline / ablation evaluation harness
	$(PY) -m ee6405.eval.run_eval --config configs/eval.yaml

test: ## Run unit tests
	$(PY) -m pytest

lint: ## Lint and format-check
	$(PY) -m ruff check src tests
	$(PY) -m ruff format --check src tests

fmt: ## Auto-format
	$(PY) -m ruff format src tests
	$(PY) -m ruff check --fix src tests

clean: ## Remove caches and the built index
	rm -rf .pytest_cache .ruff_cache **/__pycache__ src/**/__pycache__
	rm -f data/processed/index.sqlite data/processed/memory.sqlite

.PHONY: help setup setup-ml ingest app eval test lint fmt clean
