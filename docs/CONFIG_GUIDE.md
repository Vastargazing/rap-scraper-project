# 🔧 Configuration Guide

## 📋 Overview

This project uses a **two-tier configuration system** for maximum security and flexibility:

1. **`config.yaml`** - Main configuration with structure (uses ENV variables for secrets)
2. **`.env`** - Environment variables with actual credentials (NEVER committed to git)

## 🚀 Quick Setup

### 1. Create Configuration Files

```bash
# Copy example files
cp config.example.yaml config.yaml
cp .env.example .env
```

### 2. Fill in Your Credentials

Edit `.env` with your actual API keys and passwords:

```bash
# Required for QWEN ML model
NOVITA_API_KEY=your_actual_novita_key_here

# Required for PostgreSQL
DB_PASSWORD=your_secure_database_password

# Optional services
REDIS_PASSWORD=your_redis_password_if_needed
GRAFANA_ADMIN_PASSWORD=your_grafana_password
```

### 3. Verify Configuration

```bash
# Test configuration loading
python -c "import yaml; print(yaml.safe_load(open('config.yaml')))"

# Check environment variables
python -c "from dotenv import load_dotenv; load_dotenv(); import os; print('✅ DB_HOST:', os.getenv('DB_HOST'))"
```

## 🔐 Security Best Practices

### ⚠️ CRITICAL: Never Commit Secrets!

```bash
# These files are in .gitignore (NEVER commit them):
.env
.env.local
.env.*.local
config.yaml  # Contains sensitive data via ENV variables

# These files ARE safe to commit:
config.example.yaml  # Template without secrets
.env.example         # Template without actual keys
```

### 🛡️ Production Security Checklist

- [ ] **Strong Passwords**: Minimum 16 characters, mixed case, numbers, symbols
- [ ] **API Key Rotation**: Rotate keys every 90 days
- [ ] **Environment Isolation**: Different credentials for dev/staging/prod
- [ ] **2FA Enabled**: On all external services (GitHub, cloud providers)
- [ ] **Secrets Manager**: Use cloud secrets (AWS/Azure/GCP) in production
- [ ] **Read-Only Access**: Use minimum required permissions
- [ ] **Monitoring**: Set up alerts for unusual API usage

## 📚 Configuration Structure

### Application Settings

```yaml
application:
  name: "Rap Scraper Project"
  version: "2.0.0"
  environment: "production"  # dev, staging, production
```

**Environment-specific behavior:**
- `development`: Debug mode, hot reload, verbose logging
- `staging`: Production-like with test data
- `production`: Optimized, secure, monitored

### Database Configuration

```yaml
database:
  type: "postgresql"
  host_env: "DB_HOST"        # Reads from .env
  password_env: "DB_PASSWORD" # ⚠️ SECURE: Never hardcoded!
  pool_size: 20              # Connection pool size
```

**Required ENV variables:**
- `DB_HOST` - Database host (localhost or container name)
- `DB_NAME` - Database name
- `DB_USERNAME` - Database user
- `DB_PASSWORD` - Database password

### PgVector (Vector Search)

```yaml
vector_search:
  enabled: true
  embedding_model: "text-embedding-3-small"
  dimension: 1536
  distance_metric: "cosine"
```

### AI/ML Models

```yaml
analyzers:
  qwen:  # Primary ML model
    config:
      api_key_env: "NOVITA_API_KEY"  # ⚠️ From .env
      validate_api_key: true
      
  ollama:  # Local alternative
    config:
      base_url_env: "OLLAMA_BASE_URL"
```

**Required ENV variables:**
- `NOVITA_API_KEY` - For QWEN model (primary)
- `OLLAMA_BASE_URL` - For local Ollama (optional)
- `GOOGLE_API_KEY` - For Google AI (optional)

### FastAPI Settings

```yaml
api:
  host: "0.0.0.0"
  port: 8000
  workers: 4
  cors:
    origins:
      - "http://localhost:3000"
```

### Redis Cache

```yaml
redis:
  enabled: true
  host_env: "REDIS_HOST"
  password_env: "REDIS_PASSWORD"  # Optional
```

### Monitoring (Prometheus + Grafana)

```yaml
monitoring:
  prometheus:
    enabled: true
    port: 9090
  grafana:
    enabled: true
    admin_password_env: "GRAFANA_ADMIN_PASSWORD"  # ⚠️ From .env
```

## 🌍 Environment-Specific Configs

### Development

```yaml
environments:
  development:
    database:
      pool_size: 5      # Smaller pool
      echo: true        # SQL logging
    api:
      reload: true      # Hot reload
      workers: 1        # Single worker
    development:
      debug_mode: true
```

**Activate:**
```bash
export ENVIRONMENT=development
# or in .env:
ENVIRONMENT=development
```

### Staging

```yaml
environments:
  staging:
    database:
      pool_size: 10
    api:
      workers: 2
    ci_cd:
      github_actions:
        auto_deploy: true  # Auto-deploy to staging
```

### Production

```yaml
environments:
  production:
    database:
      pool_size: 20
      backup_enabled: true
    api:
      workers: 4
      reload: false
    production:
      auto_scaling: true
      security:
        enable_https_only: true
```

## 🔑 Required API Keys

### Essential (Required)

| Service | Key Name | Purpose | Get It From |
|---------|----------|---------|-------------|
| **Novita AI** | `NOVITA_API_KEY` | QWEN ML model (primary) | https://novita.ai |
| **PostgreSQL** | `DB_PASSWORD` | Database access | Your DB setup |

### Optional (Recommended)

| Service | Key Name | Purpose | Get It From |
|---------|----------|---------|-------------|
| **Genius** | `GENIUS_TOKEN` | Lyrics scraping | https://genius.com/api-clients |
| **Spotify** | `SPOTIFY_CLIENT_ID`, `SPOTIFY_CLIENT_SECRET` | Metadata enrichment | https://developer.spotify.com |
| **Redis** | `REDIS_PASSWORD` | Cache security | Your Redis setup |
| **Grafana** | `GRAFANA_ADMIN_PASSWORD` | Monitoring dashboard | Your Grafana setup |

### Optional (Alternative AI)

| Service | Key Name | Purpose | Cost |
|---------|----------|---------|------|
| **Ollama** | `OLLAMA_BASE_URL` | Free local AI | Free (local) |
| **DeepSeek** | `DEEPSEEK_API_KEY` | Cheap API fallback | $0.003/song |
| **Google AI** | `GOOGLE_API_KEY` | Gemini/Gemma models | Free tier available |

## 📝 Configuration Loading

### Order of Precedence

1. **Environment variables** (highest priority)
2. **`.env` file** (loaded by python-dotenv)
3. **`config.yaml`** (structure and defaults)
4. **`config.example.yaml`** (template only, not loaded)

### Example: Database Password

```python
# config.yaml specifies:
database:
  password_env: "DB_PASSWORD"

# Code reads from environment:
import os
db_password = os.getenv("DB_PASSWORD")  # From .env file

# ✅ SECURE: Password never in config.yaml
# ✅ SAFE: .env is in .gitignore
```

## 🚨 Troubleshooting

### "API key not found" Error

```bash
# Check if .env file exists
ls -la .env

# Verify environment variables are loaded
python -c "from dotenv import load_dotenv; load_dotenv(); import os; print(os.getenv('NOVITA_API_KEY'))"

# Make sure key name matches exactly (case-sensitive!)
```

### "Database connection failed"

```bash
# Check PostgreSQL is running
docker ps | grep postgres

# Verify DB credentials in .env
cat .env | grep DB_

# Test connection manually
psql -h localhost -U rap_user -d rap_lyrics
```

### "Config validation failed"

```bash
# Validate YAML syntax
python -c "import yaml; yaml.safe_load(open('config.yaml'))"

# Check for missing required fields
python scripts/validate_config.py  # If available
```

## 📦 Docker Setup

When using Docker Compose, environment variables are passed automatically:

```yaml
# docker-compose.yml
services:
  api:
    env_file:
      - .env  # Automatically loads all ENV variables
```

**No changes needed** - just create your `.env` file!

## 🔄 Updating Configuration

### Adding New API Key

1. Add to `.env.example` with placeholder:
   ```bash
   NEW_SERVICE_API_KEY=your_new_service_key_here
   ```

2. Add to `config.yaml`:
   ```yaml
   new_service:
     api_key_env: "NEW_SERVICE_API_KEY"
   ```

3. Update your actual `.env`:
   ```bash
   NEW_SERVICE_API_KEY=actual_key_here
   ```

4. Document in this README!

### Rotating API Keys

```bash
# 1. Generate new key from service provider
# 2. Update .env file
# 3. Restart application
# 4. Verify new key works
# 5. Revoke old key from provider
```

## 📖 Related Documentation

- **[DOCKER_SETUP.md](DOCKER_SETUP.md)** - Docker configuration details
- **[API.md](API.md)** - API endpoints and authentication
- **[SECURITY.md](../SECURITY.md)** - Security policies and guidelines
- **[POETRY_GUIDE.md](POETRY_GUIDE.md)** - Dependency management

## ⚡ Quick Commands

```bash
# Validate configuration
make validate-config  # If available

# Test API connection
python models/test_qwen.py --test-api

# Check database connection
make test-db  # If available

# View current environment
python -c "import os; print('Environment:', os.getenv('ENVIRONMENT', 'not set'))"
```

## 🆘 Need Help?

- **Configuration errors**: Check YAML syntax and ENV variable names
- **API key issues**: Verify keys are valid and not expired
- **Database problems**: Ensure PostgreSQL is running and credentials are correct
- **Security questions**: Review SECURITY.md and follow best practices

---

**Last Updated:** 2025-10-01  
**Version:** 2.0.0
