# 🐳 Production-ready Docker image for Rap Lyrics Analyzer API
FROM python:3.11-slim

# Metadata
LABEL maintainer="Rap Lyrics Analyzer Team"
LABEL description="Enterprise microservices API for rap lyrics analysis"
LABEL version="1.0.0"

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Security: Create non-root user
RUN useradd -m -u 1000 rapuser

# Set working directory
WORKDIR /app

# Install Poetry
RUN pip install --no-cache-dir poetry==1.7.1

# Configure Poetry - no virtual env needed in Docker
ENV POETRY_VIRTUALENVS_CREATE=false \
    POETRY_NO_INTERACTION=1

# Copy dependency files first for better caching
COPY pyproject.toml poetry.lock ./

# Install Python dependencies (production only)
RUN poetry install --only main --no-root --no-cache

# Copy application code
COPY . .

# Install the project itself (scripts and package)
RUN poetry install --only-root

# Create necessary directories
RUN mkdir -p logs data results temp && \
    chown -R rapuser:rapuser /app

# Security: Switch to non-root user
USER rapuser

# Environment variables
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1
ENV RAP_ANALYZER_ENV=production

# Expose ports
EXPOSE 8000

# Health check for API
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Default command - FastAPI server
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4"]

# Alternative commands:
# CLI: docker run rap-analyzer python main.py
# Single analysis: docker run rap-analyzer python main.py --analyze "text"
# Batch: docker run rap-analyzer python main.py --batch input.json
