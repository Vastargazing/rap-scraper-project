# Configuration Module - Complete Guide

**Type-safe, production-ready configuration with Pydantic validation**

---

## Table of Contents

- [Quick Start (2 minutes)](#quick-start-2-minutes)
- [Testing Configuration](#testing-configuration)
- [Core Concepts](#core-concepts)
- [Configuration Files](#configuration-files)
- [Usage Examples](#usage-examples)
- [Environment Variables](#environment-variables)
- [Multi-Environment Support](#multi-environment-support)
- [Available Config Sections](#available-config-sections)
- [Best Practices](#best-practices)
- [Troubleshooting](#troubleshooting)
- [Testing](#testing)
- [Related Documentation](#related-documentation)

---

## Quick Start (2 minutes)

### 1. Setup Files
```bash
cp .env.example .env          # Add your secrets here
cp config.example.yaml config.yaml
```

### 2. Validate Config
```bash
# Quick check (needs real ENV)
python src/config/config_loader.py

# Full test (works without ENV)
python src/config/test_loader.py
```

### 3. Use in Code
```python
from src.config import get_config

config = get_config()  # Cached singleton
db_url = config.database.connection_string
api_key = config.analyzers.get_qwen().api_key  # From ENV
```

---

## Testing Configuration

### Two Testing Modes

| Script | Purpose | Output | ENV Required | When to Use |
|--------|---------|--------|--------------|-------------|
| **config_loader.py** | Production validation | 10 lines summary | Yes | Before deployment, CI/CD, quick check |
| **test_loader.py** | Development testing | 200+ lines detailed | No (has defaults) | Development, debugging, learning |

### Quick Check (Production)
```bash
python src/config/config_loader.py

# Output:
# ⚡ Quick Config Check
# ✅ Config Valid!
#    • App: Rap Scraper v2.0.0
#    • DB: postgresql://localhost:5432/rap_lyrics
#    • Redis: localhost:6379
```

**Use when:** Before deployment, quick sanity check, CI/CD  
**Requires:** Real ENV variables (DB_PASSWORD, API keys)

### Full Test (Development)
```bash
python src/config/test_loader.py

# Output: Detailed report of ALL sections:
# Application, Database, Vector Search, API, Redis, Analyzers, Monitoring
```

**Use when:** Development setup, debugging, learning config  
**Requires:** Nothing (has fallback defaults)

### Common Error

**"Environment variable DB_PASSWORD not set!"**
```bash
# For config_loader.py → Set real ENV
export DB_PASSWORD="your_password"

# For test_loader.py → Already works (has defaults)
python src/config/test_loader.py
```

---

## Core Concepts

### Why This Config System?

**Before (Manual ENV reading):**
```python
# ❌ No validation, can crash, might be None
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = int(os.getenv("DB_PORT", "5432"))  # Can crash!
API_KEY = os.getenv("API_KEY")  # Might be None!
```

**After (Type-safe config):**
```python
# ✅ Type-safe, validated, single source of truth
from src.config import get_config

config = get_config()
db_host = config.database.host  # str, guaranteed
api_key = config.analyzers.get_qwen().api_key  # Validated
```

### Architecture

```
config.yaml  ──┐
.env file    ──┼──> Pydantic Validation ──> Config Object ──> Your App
ENV vars     ──┘
```

**Key Features:**
- Type-safe - IDE autocomplete, no typos
- Validated - Fails fast on startup if invalid
- Secure - Secrets from ENV, never hardcoded
- Multi-environment - dev/staging/prod overrides
- Singleton - One config instance, cached

---

## Configuration Files

### config.yaml (Main Settings)

**Location:** Project root  
**Purpose:** Main configuration with all settings  
**Security:** Gitignore if contains secrets

```yaml
application:
  name: "Rap Scraper Project"
  version: "2.0.0"
  environment: "production"  # Or: development, staging

database:
  host_env: "DB_HOST"          # Reads from ENV
  password_env: "DB_PASSWORD"  # Reads from ENV
  pool_size: 20                # Direct value

analyzers:
  qwen:
    config:
      api_key_env: "NOVITA_API_KEY"  # From ENV
      temperature: 0.1                # Direct value
```

### .env (Secrets)

**Location:** Project root  
**Purpose:** Environment variables (secrets, credentials)  
**Security:** MUST be gitignored!

```bash
# Database
DB_HOST=localhost
DB_PASSWORD=your_secure_password

# AI APIs
NOVITA_API_KEY=your_api_key
OLLAMA_BASE_URL=http://localhost:11434

# Redis (optional)
REDIS_PASSWORD=optional_password
```

### Templates (Safe to commit)

- `config.example.yaml` - Config template
- `.env.example` - ENV variables template

---

## Usage Examples

### Database Connection
```python
from sqlalchemy import create_engine
from src.config import get_config

config = get_config()
engine = create_engine(
    config.database.connection_string,
    pool_size=config.database.pool_size,
    max_overflow=config.database.max_overflow
)
```

### Redis Cache
```python
import redis
from src.config import get_config

config = get_config()
client = redis.Redis(
    host=config.redis.host,
    port=config.redis.port,
    password=config.redis.password  # None if not set
)

# Use cache TTL from config
ttl = config.redis.cache.artist_ttl  # 3600 seconds
client.setex("key", ttl, "value")
```

### FastAPI Application
```python
from fastapi import FastAPI
from src.config import get_config

config = get_config()
app = FastAPI(
    title=config.api.docs.title,
    version=config.api.docs.version
)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app,
        host=config.api.host,
        port=config.api.port,
        workers=config.api.workers
    )
```

### AI Analyzers
```python
from openai import OpenAI
from src.config import get_config

config = get_config()

# QWEN Analyzer
qwen_config = config.analyzers.get_qwen()
qwen_client = OpenAI(
    base_url=qwen_config.base_url,
    api_key=qwen_config.api_key  # From NOVITA_API_KEY ENV
)

# Ollama Analyzer
ollama_config = config.analyzers.get_ollama()
ollama_url = ollama_config.base_url  # From OLLAMA_BASE_URL ENV
```

### Logging Setup
```python
import logging
from logging.handlers import RotatingFileHandler
from src.config import get_config

def setup_logger(name: str):
    config = get_config()
    logger = logging.getLogger(name)
    logger.setLevel(config.logging.level)
    
    handler = RotatingFileHandler(
        config.logging.file_path,
        maxBytes=config.logging.max_file_size,
        backupCount=config.logging.backup_count
    )
    logger.addHandler(handler)
    return logger
```

### Health Check Endpoint
```python
from fastapi import FastAPI
from src.config import get_config

app = FastAPI()
config = get_config()

@app.get(config.monitoring.health.endpoint)
async def health_check():
    return {
        "status": "healthy",
        "app": config.application.name,
        "version": config.application.version,
        "components": {}  # Add component checks here
    }
```

**For more detailed examples, see [EXAMPLES.md](EXAMPLES.md)**

---

## Environment Variables

### Required
```bash
DB_PASSWORD=...      # Database password
NOVITA_API_KEY=...   # QWEN API key (if using QWEN)
```

### Optional (have defaults)
```bash
DB_HOST=localhost
DB_NAME=rap_lyrics
OLLAMA_BASE_URL=http://localhost:11434
REDIS_HOST=localhost
REDIS_PASSWORD=      # Empty = no password
```

### Loading Priority

1. Environment variables (highest priority)
2. `.env` file (loaded by python-dotenv)
3. `config.yaml` (structure and defaults)
4. Pydantic defaults (in model definitions)

---

## Multi-Environment Support

### Configuration Structure

```yaml
# Base config (all environments)
database:
  pool_size: 20

# Environment overrides
environments:
  development:
    database:
      pool_size: 5      # Smaller in dev
      echo: true        # SQL logging
    api:
      reload: true      # Hot reload
  
  production:
    database:
      pool_size: 20     # Larger in prod
      echo: false
    monitoring:
      prometheus:
        enabled: true
```

### Usage

```bash
# Development
export APP_ENVIRONMENT=development
python main.py
# Uses: pool_size=5, echo=true, reload=true

# Production (default)
export APP_ENVIRONMENT=production
python main.py
# Uses: pool_size=20, echo=false, monitoring=true
```

---

## Available Config Sections

| Section | Access | Example |
|---------|--------|---------|
| **Application** | `config.application` | `.name`, `.version`, `.environment` |
| **Database** | `config.database` | `.connection_string`, `.pool_size` |
| **Vector Search** | `config.vector_search` | `.dimension`, `.embedding_model` |
| **API** | `config.api` | `.host`, `.port`, `.workers` |
| **Redis** | `config.redis` | `.host`, `.port`, `.cache.artist_ttl` |
| **Analyzers** | `config.analyzers` | `.get_qwen()`, `.get_ollama()` |
| **Logging** | `config.logging` | `.level`, `.file_path` |
| **Monitoring** | `config.monitoring` | `.prometheus`, `.grafana` |
| **CI/CD** | `config.ci_cd` | `.testing.required_coverage` |

---

## Best Practices

### 1. Never Hardcode Secrets
```python
# ❌ BAD
API_KEY = "sk-1234567890"

# ✅ GOOD
config = get_config()
api_key = config.analyzers.get_qwen().api_key  # From ENV
```

### 2. Validate Early
```python
# ✅ App startup
from src.config import get_config

try:
    config = get_config()
    print("✅ Config valid")
except ValueError as e:
    print(f"❌ Config error: {e}")
    exit(1)
```

### 3. Use Type Hints
```python
# ✅ IDE autocomplete works
from src.config import Config

def setup_db(config: Config):
    return create_engine(config.database.connection_string)
```

### 4. Environment-Specific Settings
```yaml
# ✅ Different settings per environment
environments:
  development:
    logging:
      level: "DEBUG"
  production:
    logging:
      level: "WARNING"
```

### 5. Document Configuration Changes
```yaml
# ✅ Comments in config.yaml
database:
  pool_size: 20  # Increased from 10 on 2025-10-01 for production load
```

---

## Troubleshooting

### "Environment variable DB_PASSWORD not set"

```bash
# Check .env exists
ls -la .env

# Check contains variable
cat .env | grep DB_PASSWORD

# Load manually
export DB_PASSWORD="your_password"

# Or use test_loader.py (has defaults)
python src/config/test_loader.py
```

### "FileNotFoundError: config.yaml"

```bash
# Must run from project root
cd /path/to/rap-scraper-project

# Verify file exists
ls -la config.yaml

# Copy from example
cp config.example.yaml config.yaml
```

### "Invalid log level"

```yaml
# ❌ Invalid
logging:
  level: "INVALID"

# ✅ Valid (must be one of these)
logging:
  level: "INFO"  # DEBUG, INFO, WARNING, ERROR, CRITICAL
```

### "Redis connection refused"

```bash
# Check Redis running
docker ps | grep redis

# Start Redis
docker run -d -p 6379:6379 redis:7-alpine

# Test connection
redis-cli ping  # Should return "PONG"
```

### "Cannot import get_config"

```bash
# Option 1: Run from project root
cd /path/to/rap-scraper-project
python -c "from src.config import get_config"

# Option 2: Add to PYTHONPATH
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
```

---

## Testing

### Quick Validation
```bash
# Production check (needs real ENV)
python src/config/config_loader.py

# Development test (works without ENV)
python src/config/test_loader.py
```

### Unit Tests
```bash
# Run all config tests
pytest tests/test_config/ -v

# With coverage
pytest tests/test_config/ --cov=src.config
```

### CI/CD Integration
```yaml
# .github/workflows/test.yml
steps:
  - name: Validate Config
    run: python src/config/config_loader.py
  
  - name: Full Config Test
    run: python src/config/test_loader.py
```

---

## Related Documentation

### Project Docs
- [Main README](../../README.md) - Project overview
- [AI Onboarding](../../AI_ONBOARDING_CHECKLIST.md) - Quick start
- [PostgreSQL Setup](../../docs/postgresql_setup.md) - Database
- [Redis Architecture](../../docs/redis_architecture.md) - Caching
- [Monitoring Guide](../../docs/monitoring_guide.md) - Metrics

### External Resources
- [Pydantic Documentation](https://docs.pydantic.dev/)
- [Pydantic Settings](https://docs.pydantic.dev/latest/concepts/pydantic_settings/)
- [Python dotenv](https://pypi.org/project/python-dotenv/)

---

## Contributing

When adding new configuration:

1. Update Pydantic models in `config_loader.py`
2. Add to `config.yaml` with comments
3. Update `config.example.yaml` with example values
4. Add to `.env.example` if ENV variable
5. Write tests in `tests/test_config/`
6. Update this README with usage examples

---

## Pydantic Models Reference

### Available Models

- **Config** - Root config object
- **ApplicationConfig** - App metadata
- **DatabaseConfig** - PostgreSQL settings
- **VectorSearchConfig** - pgvector settings
- **APIConfig** - FastAPI settings
- **RedisConfig** - Redis cache settings
- **AnalyzersConfig** - All analyzers
- **MonitoringConfig** - Prometheus/Grafana
- **LoggingConfig** - Logging settings
- **CICDConfig** - CI/CD settings

### Type-Safe Properties

```python
config = get_config()

# Properties (read from ENV automatically)
db_host: str = config.database.host              # From DB_HOST
db_password: str = config.database.password      # From DB_PASSWORD
redis_host: str = config.redis.host              # From REDIS_HOST
qwen_key: str = config.analyzers.get_qwen().api_key  # From NOVITA_API_KEY

# Direct values (from YAML)
pool_size: int = config.database.pool_size       # 20
api_port: int = config.api.port                  # 8000
cache_ttl: int = config.redis.cache.artist_ttl   # 3600
```

### Built-in Validators

```python
# Log level validation
config.logging.level  # Must be: DEBUG, INFO, WARNING, ERROR, CRITICAL

# Vector dimension validation
config.vector_search.dimension  # Must be: 384, 512, 768, 1536, 3072

# Distance metric validation
config.vector_search.distance_metric  # Must be: cosine, l2, inner_product
```

---

## Architecture

### Module Structure

```
src/config/
├── __init__.py          # Exports: get_config, load_config, Config
├── config_loader.py     # Main config loader + Quick Check CLI
├── test_loader.py       # Full test suite
└── README.md           # This file
```

### Configuration Flow

```
config.yaml  ──┐
.env file    ──┼──> Pydantic Validation ──> Config Object ──> Your Application
ENV vars     ──┘                                     │
                                                     ├──> Database
                                                     ├──> Redis
                                                     ├──> API
                                                     └──> Analyzers
```

---

## Version History

- **v2.0.0** (2025-10-01) - Full Pydantic rewrite with type safety
- **v1.0.0** - Initial config system

---

**Questions? Check the troubleshooting section or contact the team!**