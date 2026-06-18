# Try bash first, fall back to zsh (for macOS)
ifeq ($(shell which bash 2>/dev/null),)
SHELL := /bin/zsh
else
SHELL := /bin/bash
endif

# Project variables
PACKAGE      := sdasim
DIST_NAME    := sdasim
VERSION      := $(shell sed -n 's/^version = "\(.*\)"/\1/p' pyproject.toml)
# Non-conflicting extras (compat/satsim conflicts with catalogs/fits, so it is
# installed on demand, never as part of the standard dev environment).
EXTRAS       := --extra dev --extra catalogs --extra fits --extra calibrate
# Export variables
EXPORT_BASE  := dist/export
EXPORT_NAME  := $(PACKAGE)-$(VERSION)
EXPORT_DIR   := $(EXPORT_BASE)/$(EXPORT_NAME)

.DEFAULT_GOAL := help

###################
# Setup           #
###################

.PHONY: check-uv sync

check-uv:
	@which uv > /dev/null 2>&1 || ( \
		echo "Error: uv is not installed. Please install it first:"; \
		echo "curl -LsSf https://astral.sh/uv/install.sh | sh"; \
		exit 1 \
	)

sync: check-uv  ## Sync dependencies (dev + catalogs + fits + calibrate)
	uv sync $(EXTRAS)

###################
# Testing         #
###################

.PHONY: test coverage

test: ## Run the full test suite
	uv run $(EXTRAS) pytest -v

coverage: ## Run tests with coverage report (htmlcov/index.html)
	uv run $(EXTRAS) --with pytest-cov pytest \
		--cov=$(PACKAGE) \
		--cov-report=term-missing \
		--cov-report=html
	@echo "Coverage report: htmlcov/index.html"

###################
# Code Quality    #
###################

.PHONY: lint format

lint: ## Run linter (ruff)
	uv run ruff check src/ tests/

format: ## Format code and fix lint issues (ruff)
	uv run ruff format src/ tests/
	uv run ruff check --fix src/ tests/

###################
# Release         #
###################

.PHONY: version build check-clean-tree check-version-unpublished check-tag-free tag publish-test publish

version: ## Print the version from pyproject.toml
	@echo $(VERSION)

build: ## Build wheel and sdist
	rm -rf dist/
	uv build

check-clean-tree:
	@test -z "$$(git status --porcelain)" || { echo "ERROR: git tree is dirty; commit or stash first"; git status --short; exit 1; }

check-version-unpublished:
	@if curl -sf https://pypi.org/pypi/$(DIST_NAME)/$(VERSION)/json > /dev/null; then \
		echo "ERROR: $(DIST_NAME) $(VERSION) is already on PyPI; bump version in pyproject.toml first"; exit 1; \
	fi
	@echo "PyPI check OK: $(DIST_NAME) $(VERSION) not yet published"

check-tag-free:
	@if git rev-parse -q --verify "refs/tags/v$(VERSION)" > /dev/null; then \
		echo "ERROR: local tag v$(VERSION) already exists"; exit 1; \
	fi
	@if git ls-remote --exit-code --tags origin "v$(VERSION)" > /dev/null 2>&1; then \
		echo "ERROR: tag v$(VERSION) already exists on origin"; exit 1; \
	fi
	@echo "Tag check OK: v$(VERSION) is free"

tag: check-clean-tree check-tag-free ## git tag v<version> from pyproject.toml and push it
	git tag v$(VERSION)
	git push origin v$(VERSION)

publish-test: check-clean-tree test build ## Test + build + upload to TestPyPI (token from ~/.pypirc)
	uvx twine upload --repository testpypi dist/*

# Publish to PyPI. Guarded: refuses on a dirty tree or an already-published
# version, always rebuilds from scratch, and runs the test suite first.
# PyPI uploads are irreversible per version - bump pyproject.toml first.
publish: check-clean-tree check-version-unpublished test build ## Guarded upload to PyPI (token from ~/.pypirc)
	uvx twine upload dist/*

###################
# Export          #
###################

.PHONY: export export-zip

export: ## Export clean source snapshot (no git history)
	@echo "Exporting clean source snapshot..."
	rm -rf $(EXPORT_DIR)
	mkdir -p $(EXPORT_DIR)
	git archive --format=tar HEAD | tar -x -C $(EXPORT_DIR)
	@echo "Exported to $(EXPORT_DIR)"

export-zip: ## Export clean source snapshot as zip
	@echo "Exporting clean source snapshot (zip)..."
	mkdir -p $(EXPORT_BASE)
	git archive --format=zip -o $(EXPORT_BASE)/$(EXPORT_NAME).zip HEAD
	@echo "Exported to $(EXPORT_BASE)/$(EXPORT_NAME).zip"

###################
# Cleanup         #
###################

.PHONY: clean clean-all

clean: ## Remove build artifacts, caches, and reports
	rm -rf build/ dist/ .eggs/ .pytest_cache/ .ruff_cache/ .coverage .coverage.* htmlcov/
	find . -type d -name '*.egg-info' -exec rm -rf {} +
	find . -type d -name '__pycache__' -exec rm -rf {} +
	find . -type f -name '*.py[cod]' -delete
	rm -f coverage.xml

clean-all: clean ## clean + remove the virtual environment and lock file
	rm -rf .venv uv.lock

###################
# Help            #
###################

.PHONY: help

help: ## Show this help message
	@echo 'Usage: make [target]'
	@echo ''
	@echo 'Targets:'
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## / {printf "  %-24s %s\n", $$1, $$2}' $(MAKEFILE_LIST)
