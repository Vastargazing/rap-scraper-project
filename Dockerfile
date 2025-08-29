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
    sqlite3 \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Security: Create non-root user
RUN useradd -m -u 1000 rapuser

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

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
