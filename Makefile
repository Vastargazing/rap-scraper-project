# Rap Analyzer - Production Poetry Makefile
# For ML Platform Engineer best practices

.PHONY: help install install-dev install-full install-prod clean test lint format check build release ci-lint ci-test ci-build ci-all pre-commit test-wheel deps-update deps-outdated deps-export docker-build-prod docker-build-dev docker-run-prod docker-run-dev docker-test docker-clean

# Default target
help:  ## Show this help message
	@echo "Rap Analyzer - Poetry Commands"
	@echo "=============================="
	@echo "Installation:"
	@echo "  install           - Install production dependencies only"
	@echo "  install-dev       - Install main + development dependencies"
	@echo "  install-analysis  - Install main + analysis (pandas, jupyter)"
	@echo "  install-ml        - Install main + heavy ML dependencies"
	@echo "  install-full      - Install ALL dependencies"
	@echo "  install-prod      - Production deployment (minimal)"
	@echo ""
	@echo "Development:"
	@echo "  dev               - Setup development environment"
	@echo "  shell             - Activate Poetry shell"
	@echo "  format            - Format code with black"
	@echo "  lint              - Run linting (flake8 + mypy)"
	@echo "  check             - Run all quality checks"
	@echo ""
	@echo "Testing:"
	@echo "  test              - Run all tests"
	@echo "  test-unit         - Run unit tests only"
	@echo "  test-integration  - Run integration tests only"
	@echo "  test-coverage     - Run tests with coverage"
	@echo ""
	@echo "Application:"
	@echo "  run-scraper       - Run main scraper"
	@echo "  run-api           - Start ML API service"
	@echo "  run-fastapi       - Start FastAPI web service"
	@echo "  run-ml-training   - Run QWEN ML training"
	@echo ""
	@echo "Docker Compose:"
	@echo "  docker-up         - Start production stack"
	@echo "  docker-dev        - Start development stack"
	@echo "  docker-db         - Start only database (for local dev)"
	@echo "  docker-down       - Stop all services"
	@echo "  docker-logs       - Show API logs"
	@echo "  docker-ps         - Show running containers"
	@echo ""
	@echo "Database:"
	@echo "  db-up             - Start PostgreSQL + Redis (alias for docker-db)"
	@echo "  db-down           - Stop database containers (alias for docker-down)"
	@echo "  db-status         - Check database status"
	@echo ""
	@echo "Build & Release:"
	@echo "  build             - Build wheel package"
	@echo "  semantic-release  - Automated versioning"
	@echo ""
	@echo "Docker:"
	@echo "  docker-build-prod - Build production Docker image"
	@echo "  docker-build-dev  - Build development Docker image"
	@echo "  docker-run-prod   - Run production container"
	@echo "  docker-run-dev    - Run development container"
	@echo "  docker-test       - Test Docker image"
	@echo "  docker-clean      - Clean Docker artifacts"
	@echo ""
	@echo "CI/CD Simulation:"
	@echo "  ci-lint           - CI: Linting (black, flake8, mypy)"
	@echo "  ci-test           - CI: Tests with coverage"
	@echo "  ci-build          - CI: Build production wheel"
	@echo "  ci-all            - CI: Full pipeline locally"
	@echo "  pre-commit        - Fast checks before git commit"
	@echo "  test-wheel        - Test wheel installation"
	@echo ""
	@echo "Dependencies:"
	@echo "  deps-update       - Update all dependencies"
	@echo "  deps-outdated     - Show outdated packages"
	@echo "  deps-export       - Export to requirements.txt"
	@echo ""
	@echo "Quick Workflows:"
	@echo "  quick-start       - Quick start for new developers"
	@echo "  prod-deploy       - Production deployment setup"
	@echo "  ml-experiment     - Setup for ML experiments"

# Installation targets
install:  ## Install production dependencies only
	poetry install --only main

install-dev:  ## Install main + development dependencies
	poetry install --only main --with dev

install-analysis:  ## Install main + analysis (pandas, jupyter, visualization)
	poetry install --only main --with analysis

install-ml:  ## Install main + heavy ML dependencies (torch, transformers)
	poetry install --only main --with ml-heavy

install-full:  ## Install ALL dependencies (main + dev + analysis + ml-heavy + release)
	poetry install --with dev,analysis,ml-heavy,release

install-prod:  ## Production deployment - minimal dependencies
	poetry install --only main --no-dev

# Development workflow
dev:  ## Start development environment (install dev deps + activate shell)
	poetry install --with dev,analysis
	@echo "Run 'poetry shell' to activate environment"

shell:  ## Activate Poetry shell
	poetry shell

clean:  ## Clean virtual environment and caches
	poetry env remove --all
	@find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	@find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	@find . -type f -name "*.pyc" -delete 2>/dev/null || true
	@rm -rf .coverage .mypy_cache htmlcov dist build 2>/dev/null || true

# Code quality
format:  ## Format code with black
	poetry run black src/ tests/ --line-length 88

lint:  ## Run linting (flake8 + mypy)
	poetry run flake8 src/ tests/
	poetry run mypy src/

check:  ## Run all quality checks (format + lint)
	poetry run black src/ tests/ --line-length 88
	poetry run flake8 src/ tests/
	poetry run mypy src/

# Testing
test:  ## Run all tests
	poetry run pytest tests/ -v

test-unit:  ## Run unit tests only
	poetry run pytest tests/ -v -m "unit"

test-integration:  ## Run integration tests only
	poetry run pytest tests/ -v -m "integration"

test-benchmark:  ## Run performance benchmarks
	poetry run pytest tests/ -v -m "benchmark" --benchmark-only

test-coverage:  ## Run tests with coverage
	poetry run pytest tests/ --cov=src --cov-report=html --cov-report=term

# Application commands
run-scraper:  ## Run main scraper
	poetry run python main.py

run-api:  ## Start ML API service
	poetry run python src/models/ml_api_service.py

run-fastapi:  ## Start FastAPI web service
	poetry run uvicorn api:app --reload --host 127.0.0.1 --port 8000

run-ml-training:  ## Run QWEN ML training
	poetry run python models/test_qwen.py --all

# Docker Compose commands
docker-up:  ## Start production stack
	docker-compose up -d

docker-dev:  ## Start development stack
	docker-compose -f docker-compose.yml -f docker-compose.dev.yml up

docker-db:  ## Start only database (for local dev)
	docker-compose -f docker-compose.pgvector.yml up -d

docker-down:  ## Stop all services
	docker-compose down
	docker-compose -f docker-compose.dev.yml down 2>/dev/null || true
	docker-compose -f docker-compose.pgvector.yml down 2>/dev/null || true

docker-logs:  ## Show logs
	docker-compose logs -f rap-analyzer-api

docker-ps:  ## Show running containers
	docker-compose ps

# Database and infrastructure (legacy aliases)
db-up: docker-db  ## Start PostgreSQL + Redis with Docker (alias)

db-down: docker-down  ## Stop database containers (alias)

db-status:  ## Check database status
	poetry run python scripts/tools/database_diagnostics.py --quick

# Build and release
build:  ## Build wheel package
	poetry build

semantic-release:  ## Run semantic-release (automated versioning)
	poetry run semantic-release version
	poetry run semantic-release publish

# Docker operations
docker-build-prod:  ## Build production Docker image
	@echo "Building production Docker image with BuildKit..."
ifeq ($(OS),Windows_NT)
	cmd /c "set DOCKER_BUILDKIT=1&& docker build -f Dockerfile.prod -t rap-analyzer:latest ."
else
	DOCKER_BUILDKIT=1 docker build -f Dockerfile.prod -t rap-analyzer:latest .
endif

docker-build-dev:  ## Build development Docker image
	@echo "Building development Docker image with BuildKit..."
ifeq ($(OS),Windows_NT)
	cmd /c "set DOCKER_BUILDKIT=1&& docker build -f Dockerfile.dev -t rap-analyzer:dev ."
else
	DOCKER_BUILDKIT=1 docker build -f Dockerfile.dev -t rap-analyzer:dev .
endif

docker-run-prod:  ## Run production Docker container
	docker run -p 8000:8000 --env-file .env rap-analyzer:latest

docker-run-dev:  ## Run development Docker container with hot reload
	docker run -p 8000:8000 -v $(PWD):/app rap-analyzer:dev

docker-test:  ## Test Docker image works
	docker run --rm rap-analyzer:latest python -c "import src; print('✅ Docker production image OK')"

docker-clean:  ## Clean Docker images and containers
	docker system prune -f
	docker rmi rap-analyzer:latest rap-analyzer:dev 2>/dev/null || true

# Quick shortcuts for common workflows
quick-start:  ## Quick start for new developers
	poetry install --with dev,analysis
	docker-compose -f docker-compose.pgvector.yml up -d
	@echo "✅ Development environment ready!"
	@echo "Run 'poetry shell' to activate Poetry environment"
	@echo "Run 'make docker-dev' for full development stack"
	@echo "Run 'make run-fastapi' to start FastAPI locally"

prod-deploy:  ## Production deployment setup
	poetry install --only main --no-dev
	poetry build
	@echo "✅ Production build ready!"

ml-experiment:  ## Setup for ML experiments
	poetry install --with analysis,ml-heavy
	@echo "✅ ML experiment environment ready!"
	@echo "All analysis and ML libraries available"

# CI/CD simulation (exactly as in GitLab CI)
ci-lint:  ## CI: Linting (exactly as CI/CD pipeline)
	poetry run black --check src/ tests/
	poetry run flake8 src/ tests/
	poetry run mypy src/

ci-test:  ## CI: Tests with coverage
	poetry run pytest tests/ -v --cov=src --cov-report=xml --cov-report=term

ci-build:  ## CI: Build production wheel
	poetry build
	@ls -lh dist/ 2>/dev/null || dir dist\

ci-all: ci-lint ci-test ci-build  ## CI: Full pipeline locally
	@echo "✅ All CI checks passed!"

pre-commit:  ## Fast checks before git commit
	@echo "🔍 Running pre-commit checks..."
	@poetry run black src/ tests/ --check --quiet || (echo "❌ Run 'make format'" && exit 1)
	@poetry run flake8 src/ tests/ --count
	@echo "✅ Pre-commit passed!"

test-wheel:  ## Test wheel installation (production simulation)
	poetry build
	@echo "Installing wheel package..."
	@pip install --force-reinstall dist/rap_analyzer-0.0.0-py3-none-any.whl || pip install --force-reinstall dist/*.whl
	python -c "import src; print('✅ Wheel installation OK')"

# Dependency management
deps-update:  ## Update all dependencies
	poetry update

deps-outdated:  ## Show outdated packages
	poetry show --outdated

deps-export:  ## Export to requirements.txt (legacy compatibility)
	poetry export -f requirements.txt --output requirements.txt --without-hashes
	poetry export -f requirements.txt --with dev --output requirements-dev.txt --without-hashes

# Project information
info:  ## Show project information
	@echo "Project: rap-analyzer"
	@poetry version
	@echo "Virtual Env: " && poetry env info --path
	@echo ""
	@echo "Main Dependencies:"
	@poetry show --only main

deps:  ## Show dependency tree
	poetry show --tree
