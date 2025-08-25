# Makefile for Test-Driven Development Workflow
# Windows PowerShell compatible commands

.PHONY: test lint type-check format install clean dev-setup commit-check

# 🧪 Testing
test:
	python -m unittest discover tests -v

test-coverage:
	python -m coverage run -m unittest discover tests
	python -m coverage report -m
	python -m coverage html

test-spotify:
	python -m unittest tests.test_spotify_enhancer -v

test-models:
	python -m unittest tests.test_models -v

# 📝 Code Quality
lint:
	python -m flake8 *.py --max-line-length=120 --exclude=__pycache__,tests/conftest.py

type-check:
	python -m mypy spotify_enhancer.py models.py --ignore-missing-imports

format:
	python -m black *.py tests/*.py --line-length=120

format-check:
	python -m black *.py tests/*.py --check --line-length=120

# 🛠️ Development Setup
install:
	pip install -r requirements.txt

dev-setup: install
	@echo "🚀 Setting up development environment..."
	python -c "import sqlite3; print('✅ SQLite available')"
	python -c "import pydantic; print('✅ Pydantic available')" 
	python -c "import pytest; print('✅ Pytest available')"
	@echo "🧪 Running initial tests..."
	$(MAKE) test

# 💾 Git Workflow  
commit-check: format lint type-check test
	@echo "✅ All checks passed! Ready to commit."

quick-check: test-models test-spotify
	@echo "⚡ Quick checks completed!"

# 🧹 Cleanup
clean:
	Remove-Item -Recurse -Force __pycache__ -ErrorAction SilentlyContinue
	Remove-Item -Recurse -Force .mypy_cache -ErrorAction SilentlyContinue
	Remove-Item -Recurse -Force htmlcov -ErrorAction SilentlyContinue
	Remove-Item -Force .coverage -ErrorAction SilentlyContinue

# 📊 Project Stats
stats:
	@echo "📊 Project Statistics:"
	@echo "Python files: $$(Get-ChildItem -Filter '*.py' | Measure-Object).Count"
	@echo "Test files: $$(Get-ChildItem tests -Filter '*.py' | Measure-Object).Count"
	@echo "Lines of code:"
	Get-ChildItem -Filter '*.py' -Exclude setup.py | ForEach-Object { Get-Content $_.FullName | Measure-Object -Line } | ForEach-Object { $_.Lines } | Measure-Object -Sum

# 🎯 TDD Workflow Commands
tdd-cycle: test format lint type-check
	@echo "🔄 TDD Cycle completed - ready for next iteration!"

tdd-commit: commit-check
	@echo "💾 Code quality verified - ready to commit!"
	@echo "📝 Use: git add . && git commit -m 'feat: description'"

# Help
help:
	@echo "🧪 Test-Driven Development Commands:"
	@echo ""
	@echo "Testing:"
	@echo "  make test           - Run all tests"
	@echo "  make test-coverage  - Run tests with coverage report"
	@echo "  make test-spotify   - Test Spotify integration only"
	@echo "  make test-models    - Test Pydantic models only"
	@echo ""
	@echo "Code Quality:"
	@echo "  make lint          - Check code style (flake8)"
	@echo "  make type-check    - Check types (mypy)"
	@echo "  make format        - Format code (black)"
	@echo "  make format-check  - Check formatting without changes"
	@echo ""
	@echo "Development:"
	@echo "  make dev-setup     - Setup development environment"
	@echo "  make commit-check  - Full check before commit"
	@echo "  make tdd-cycle     - Complete TDD iteration"
	@echo "  make quick-check   - Fast essential checks"
	@echo ""
	@echo "Utilities:"
	@echo "  make clean         - Remove cache files"
	@echo "  make stats         - Show project statistics"
