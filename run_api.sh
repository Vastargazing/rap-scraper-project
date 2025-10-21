#!/bin/bash
# 🚀 API Launcher Script
# Запускает Unified FastAPI Application с правильными путями

# Set working directory to project root
cd "$(dirname "$0")" || exit 1

# Set PYTHONPATH
export PYTHONPATH="$(pwd)"

# Activate virtual environment and run
echo "🚀 Starting Rap Analyzer API..."
echo "📁 Working directory: $(pwd)"
echo "🐍 Python: .venv/bin/python"
echo "⚙️  Config: config.yaml"
echo "🔑 ENV: .env"
echo ""

.venv/bin/python -m src.api.main
