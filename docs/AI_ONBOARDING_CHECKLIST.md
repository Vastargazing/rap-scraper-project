# Kubernetes-Native Rap Analyzer - AI Agent Guide

> **Enterprise-grade Kubernetes lyrics analysis platform** with PostgreSQL + pgvector backend, container orchestration, monitoring stack, and production-ready infrastructure

## 🎯 QUICK START FOR AI AGENTS

### Critical Commands (Run These First)
```bash
# PRIMARY DIAGNOSTIC (always run first)
python scripts/tools/database_diagnostics.py --quick

# AI ANALYSIS TEST
python scripts/mass_qwen_analysis.py --test

# INTERACTIVE DATABASE BROWSER
python scripts/db_browser.py

# CONCURRENT ACCESS TEST
python scripts/tools/database_diagnostics.py --connections
```

---

## 📊 DATABASE SCHEMA & CURRENT STATUS

### Production Metrics (Updated 2025-01-19)
- **Tracks**: 57,718 total (100% with lyrics)
- **Artists**: 345+ scraped from Genius.com
- **Database**: PostgreSQL 15 with pgvector extension
- **Analyses**: 269,646 total across multiple analyzers ✅ **COMPLETE**
- **AI Coverage**: 100% Qwen ✅, 100% Algorithmic ✅, 59.4% Gemma
- **Infrastructure**: Kubernetes-native with Helm chart deployment
- **Monitoring**: Prometheus + Grafana dashboards
- **Architecture**: Production-ready container orchestration

### Database Tables

#### `tracks` Table (57,718 records)
```sql
CREATE TABLE tracks (
    id                SERIAL PRIMARY KEY,
    title             VARCHAR(500),
    artist            VARCHAR(200),
    lyrics            TEXT,                    -- Primary analysis field
    url               VARCHAR(500),
    created_at        TIMESTAMP DEFAULT NOW(),
    spotify_data      JSONB,                  -- Spotify metadata
    audio_features    JSONB                   -- Audio characteristics
);
```

#### `analysis_results` Table (256,021 analyses)
```sql
CREATE TABLE analysis_results (
    id                   SERIAL PRIMARY KEY,
    track_id             INTEGER REFERENCES tracks(id),
    analyzer_type        VARCHAR(50),
    sentiment            VARCHAR,
    confidence           NUMERIC(5,4),        -- 0.0-1.0 confidence
    themes               TEXT,                -- JSON array of themes
    analysis_data        JSONB,               -- Full analysis results
    complexity_score     NUMERIC(5,4),        -- 0.0-1.0 complexity
    processing_time_ms   INTEGER,
    model_version        VARCHAR,
    created_at           TIMESTAMP DEFAULT NOW()
);
```

### Complete Analyzer Coverage ✅
```
┌─────────────────────────┬───────────┬───────────┬─────────┐
│ Analyzer Type           │ Analyses  │ Tracks    │ Share   │
├─────────────────────────┼───────────┼───────────┼─────────┤
│ simplified_features     │ 115,434   │ 57,717    │ 42.8%   │
│ qwen-3-4b-fp8          │ 61,933    │ 57,716    │ 23.0%   │✅
│ simplified_features_v2  │ 57,717    │ 57,717    │ 21.4%   │
│ gemma-3-27b-it         │ 34,320    │ 34,320    │ 12.7%   │
│ emotion_analyzer_v2     │ 242       │ 242       │ 0.1%    │
└─────────────────────────┼───────────┼───────────┼─────────┤
│ TOTAL                   │ 269,646   │ 57,716    │ 100%    │✅
└─────────────────────────┴───────────┴───────────┴─────────┘

Analysis Status: COMPLETE ✅ (100% track coverage achieved)
Performance: 8ms average processing time, 76.3% confidence
```

---

## 🏗️ SYSTEM ARCHITECTURE

### Core Components

| Component | File | Purpose | Status |
|-----------|------|---------|--------|
| **Database Layer** | `src/database/postgres_adapter.py` | Connection pooling & management | ✅ Production |
| **Main Analysis** | `scripts/mass_qwen_analysis.py` | AI-powered lyric analysis | ✅ Production |
| **Diagnostics** | `scripts/tools/database_diagnostics.py` | Health monitoring | ✅ Production |
| **Interactive Browser** | `scripts/db_browser.py` | Database exploration | ✅ Production |
| **Scraping Engine** | `main.py` | Genius.com data collection | ✅ Production |

### PostgreSQL Infrastructure
- **Connection Pool**: 20 max concurrent connections
- **Drivers**: `asyncpg` (async) + `psycopg2` (sync)
- **Migration**: Complete SQLite → PostgreSQL (100% data integrity)
- **Concurrent Safe**: Multiple scripts can run simultaneously
- **ACID Compliant**: Full transaction isolation

### AI Analysis Pipeline
- **Qwen AI**: Cloud-based advanced analysis (Novita API)
- **Algorithmic**: Fast baseline analysis (pure Python)
- **Emotion AI**: 6-emotion detection (Hugging Face)
- **Hybrid**: Multi-model approach combining all analyzers
- **Ollama**: Local LLM integration

---

## 🤖 AI AGENT INVESTIGATION PROTOCOL

### Step 1: Database Health Check
```python
def investigate_issue(problem_description):
    # ALWAYS START HERE
    run_command("python scripts/tools/database_diagnostics.py --quick")
    
    # Expected output:
    # ✅ Подключение к PostgreSQL успешно!
    # 🎵 Треков: 57,718 (с текстами: 57,718)
    # 🤖 Анализ: 57,718/57,718 (100.0%)
    # 💾 Размер БД: 392 MB
```

### Step 2: Specific Diagnostics
```python
if "analysis" in problem_description.lower():
    run_command("python scripts/mass_qwen_analysis.py --test")
    run_command("python scripts/tools/database_diagnostics.py --analysis")
    
elif "connection" in problem_description.lower():
    run_command("python scripts/tools/database_diagnostics.py --connections")
    run_command("python scripts/db_browser.py")
    
elif "concurrent" in problem_description.lower():
    # Test multiple script execution
    run_command("python scripts/mass_qwen_analysis.py --batch 10 &")
    run_command("python scripts/db_browser.py")
```

### Step 3: Configuration Validation
```python
def check_configuration():
    # PostgreSQL credentials
    check_file(".env")  # Database connection params
    check_file("config.yaml")  # System configuration
    
    # API keys validation
    check_env_var("NOVITA_API_KEY")    # Qwen AI analysis
    check_env_var("GENIUS_ACCESS_TOKEN")  # Lyrics scraping
    check_env_var("SPOTIFY_CLIENT_ID")    # Metadata enhancement
```

### Step 4: Code Compatibility Check
```python
def verify_postgresql_compatibility(script_path):
    indicators = {
        "good": [
            "from src.database.postgres_adapter import PostgreSQLManager",
            "import asyncpg",
            "import psycopg2",
            "async with db_manager.get_connection()"
        ],
        "bad": [
            "import sqlite3",
            "sqlite3.connect",
            "cursor.execute"
        ]
    }
    return scan_file_for_patterns(script_path, indicators)
```

---

## 🔧 ESSENTIAL COMMANDS FOR AI AGENTS

### Level 1: Health & Diagnostics
```bash
# MAIN diagnostic tool (PostgreSQL)
python scripts/tools/database_diagnostics.py --quick
python scripts/tools/database_diagnostics.py         # Full stats
python scripts/tools/database_diagnostics.py --analysis  # AI analysis only
python scripts/tools/database_diagnostics.py --unanalyzed  # Find remaining

# Connection testing
python scripts/db_browser.py                    # Interactive browser
psql -h localhost -U rap_user -d rap_lyrics -p 5433  # Direct connection
```

### Level 2: Kubernetes Operations
```bash
# Deploy complete stack
helm install rap-analyzer ./helm/rap-analyzer --create-namespace --namespace rap-analyzer

# Check deployment status
kubectl get pods -n rap-analyzer
kubectl get svc -n rap-analyzer
kubectl logs deployment/rap-analyzer -f -n rap-analyzer

# Access applications
kubectl port-forward svc/rap-analyzer-service 8000:8000 -n rap-analyzer
kubectl port-forward svc/grafana-service 3000:3000 -n rap-analyzer

# Monitoring URLs
# API: http://localhost:8000/docs
# Grafana: http://localhost:3000 (admin/admin123)
```

### Level 3: Legacy Analysis & Testing (Development Mode)
```bash
# AI analysis pipeline (for local development)
python scripts/mass_qwen_analysis.py --test     # Test mode (10 tracks)
python scripts/mass_qwen_analysis.py            # Full analysis
python scripts/mass_qwen_analysis.py --batch 25 --max 100  # Custom params

# Concurrent processing test
python scripts/mass_qwen_analysis.py --batch 25 &    # Background
python scripts/db_browser.py                         # Foreground
```

### Level 3: Data Operations
```bash
# Scraping new data
python main.py                                  # Main scraper interface
python scripts/rap_scraper_cli.py scraping --debug  # Debug scraping

# Database operations
python scripts/migrate_to_postgresql.py        # Migration (if needed)
python scripts/check_overlap.py               # Analysis coverage check
```

### Level 4: Configuration & Validation
```bash
# Environment check
cat .env | grep POSTGRES                       # Database credentials  
cat .env | grep -E "(NOVITA|GENIUS|SPOTIFY)"   # API keys
python -c "from src.utils.config import get_db_config; print(get_db_config())"

# Performance validation
python -c "
from src.database.postgres_adapter import PostgreSQLManager
import asyncio
async def test():
    db = PostgreSQLManager()
    await db.initialize()
    print('✅ PostgreSQL connection OK')
    await db.close()
asyncio.run(test())
"
```

---

## 📊 KEY SQL QUERIES FOR AI AGENTS

### Find Unanalyzed Tracks
```sql
-- Tracks without Qwen analysis
SELECT COUNT(*) FROM tracks t
LEFT JOIN analysis_results ar ON t.id = ar.track_id 
  AND ar.analyzer_type = 'qwen-3-4b-fp8'
WHERE ar.id IS NULL AND t.lyrics IS NOT NULL;

-- Get specific unanalyzed tracks
SELECT t.id, t.artist, t.title 
FROM tracks t 
LEFT JOIN analysis_results ar ON t.id = ar.track_id 
  AND ar.analyzer_type = 'qwen-3-4b-fp8'
WHERE ar.id IS NULL AND t.lyrics IS NOT NULL
ORDER BY t.id LIMIT 100;
```

### Analysis Statistics
```sql
-- Analyzer performance metrics
SELECT 
    analyzer_type,
    COUNT(*) as total_analyses,
    COUNT(DISTINCT track_id) as unique_tracks,
    AVG(confidence) as avg_confidence,
    AVG(processing_time_ms) as avg_time_ms,
    ROUND(100.0 * COUNT(DISTINCT track_id) / 57718.0, 2) as coverage_percent
FROM analysis_results 
GROUP BY analyzer_type 
ORDER BY total_analyses DESC;

-- Recent analysis activity
SELECT t.artist, t.title, ar.analyzer_type, ar.created_at, ar.confidence
FROM analysis_results ar
JOIN tracks t ON ar.track_id = t.id
ORDER BY ar.created_at DESC LIMIT 20;
```

### Artist Statistics
```sql
-- Top artists by track count
SELECT artist, COUNT(*) as tracks_count,
       COUNT(CASE WHEN ar.analyzer_type = 'qwen-3-4b-fp8' THEN 1 END) as qwen_analyzed
FROM tracks t
LEFT JOIN analysis_results ar ON t.id = ar.track_id
WHERE t.lyrics IS NOT NULL
GROUP BY artist 
ORDER BY tracks_count DESC LIMIT 20;
```

---

## 🚨 TROUBLESHOOTING GUIDE FOR AI AGENTS

### Problem: Database Connection Issues
```bash
# Diagnosis steps
cat .env | grep POSTGRES                        # Check credentials
docker ps | grep postgres                      # Check if container running
python scripts/tools/database_diagnostics.py --quick  # Test connection

# Common solutions
docker-compose -f docker-compose.pgvector.yml up -d  # Start PostgreSQL
# Check port conflicts (default: 5433)
# Verify firewall settings
```

### Problem: AI Analysis Failures
```bash
# Diagnosis
python scripts/mass_qwen_analysis.py --test --batch 1  # Single track test
cat .env | grep NOVITA_API_KEY                  # Check API key
curl -s "https://api.novita.ai/v3/health"       # API status

# Common solutions
# 1. Verify API key validity
# 2. Check rate limits (Qwen: ~2-5 tracks/min)
# 3. Network connectivity issues
# 4. API service outages
```

### Problem: Concurrent Access Issues
```bash
# Diagnosis
python scripts/tools/database_diagnostics.py --connections  # Pool status
python -c "
from src.database.postgres_adapter import PostgreSQLManager
import asyncio
async def test_pool():
    db = PostgreSQLManager()
    await db.initialize()
    print(f'Pool: {db.pool.get_size()} connections')
    await db.close()
asyncio.run(test_pool())
"

# Solutions
# 1. Increase pool_size in config
# 2. Close unused connections
# 3. Check for connection leaks
# 4. Review async/await patterns
```

### Problem: Performance Issues
```bash
# Diagnosis
python scripts/tools/database_diagnostics.py   # Full statistics
htop                                           # System resources
docker stats rap-analyzer-postgres-vector      # Container resources

# Optimization
# 1. Adjust batch_size in analysis scripts
# 2. Monitor connection pool usage
# 3. Index optimization for frequent queries
# 4. Memory allocation for large datasets
```

---

## 🚀 KUBERNETES DEPLOYMENT CHECKLIST

### Phase 1: Infrastructure Complete ✅
- [x] **PostgreSQL StatefulSet** - pgvector enabled, persistent storage
- [x] **FastAPI Deployment** - Auto-scaling (3-10 replicas), health probes
- [x] **Monitoring Stack** - Prometheus metrics + Grafana dashboards
- [x] **Helm Chart** - Complete package with 80+ configuration parameters
- [x] **Documentation** - Installation guide and operational procedures

### Phase 2: Advanced Features (In Progress)
- [ ] **GitOps Integration** - ArgoCD for automated deployments
- [ ] **Multi-region Setup** - Data replication and geographical distribution
- [ ] **Advanced Monitoring** - Jaeger distributed tracing
- [ ] **Security Hardening** - Pod Security Standards, RBAC refinement
- [ ] **Backup Automation** - Scheduled database snapshots

### Deployment Commands
```bash
# Quick deployment
helm install rap-analyzer ./helm/rap-analyzer --create-namespace --namespace rap-analyzer

# Verify deployment
kubectl get all -n rap-analyzer

# Access services
kubectl port-forward svc/rap-analyzer-service 8000:8000 -n rap-analyzer
kubectl port-forward svc/grafana-service 3000:3000 -n rap-analyzer
```

## 🎯 AI AGENT RESPONSE TEMPLATE

```markdown
## 🔍 INVESTIGATION SUMMARY
**Kubernetes Status**: [Pod health, service availability, resource usage]
**Database Status**: [PostgreSQL connection, pool health, query performance]
**Analysis Pipeline**: [API status, processing rate, error rate]
**Monitoring**: [Metrics collection, alert status, dashboard availability]

## 📋 FINDINGS
**Root Cause**: [Specific issue with code references]
**Impact Assessment**: [Affected components, data integrity, performance]
**Configuration Issues**: [Helm values, secrets, ingress settings]

## 🚀 SOLUTION PLAN
1. **Immediate Actions**: [Commands to run right now]
   ```bash
   kubectl get pods -n rap-analyzer
   helm status rap-analyzer -n rap-analyzer
   ```

2. **Code/Configuration Changes**: [Specific file modifications]
3. **Testing Strategy**: [Validation commands and expected results]
4. **Monitoring**: [How to track the solution effectiveness]

## ✅ VALIDATION COMMANDS
```bash
# Verify solution
python scripts/tools/database_diagnostics.py --quick
python scripts/mass_qwen_analysis.py --test
python scripts/db_browser.py  # Interactive validation
```

## 📊 SUCCESS METRICS
- PostgreSQL connection: < 100ms
- Query response: < 500ms  
- Analysis success rate: > 90%
- Concurrent scripts: No database locks
```

---

## 💡 OPTIMIZATION NOTES FOR AI AGENTS

### Don't Request These (Already Documented):
- Database schema (provided above)
- Table statistics (current metrics included)
- Analyzer types (coverage table provided)
- SQL examples (comprehensive set included)

### Always Use These First:
- `helm status rap-analyzer -n rap-analyzer` - Kubernetes deployment status
- `kubectl get pods -n rap-analyzer` - Pod health check
- `python scripts/tools/database_diagnostics.py --quick` - Database diagnostic
- Ready SQL queries from this guide

### Key Files Priority:
1. `helm/rap-analyzer/values.yaml` - Kubernetes configuration (~400 lines)
2. `k8s/api/fastapi-deployment.yaml` - API service deployment (~150 lines)
3. `k8s/postgres/postgresql-deployment.yaml` - Database deployment (~200 lines)
4. `INSTALLATION.md` - Deployment guide
5. `src/database/postgres_adapter.py` - Database layer (~200 lines)

### Production Success Indicators:
- ✅ All pods Running status in `kubectl get pods -n rap-analyzer`
- ✅ API accessible via port-forward on :8000
- ✅ Grafana dashboards showing metrics on :3000
- ✅ Database connections < 100ms latency
- ✅ Auto-scaling working (3-10 replicas based on load)
- ✅ No CrashLoopBackOff or ImagePullBackOff errors

---

## 📦 PROJECT STRUCTURE REFERENCE

```
rap-scraper-project/
├── helm/rap-analyzer/                  # 📦 Helm chart for complete deployment
│   ├── Chart.yaml                     # Chart metadata and dependencies
│   ├── values.yaml                    # Configuration parameters (80+)
│   └── templates/                     # Kubernetes manifest templates
├── k8s/                               # 🚀 Kubernetes manifests
│   ├── postgres/postgresql-deployment.yaml  # Database with pgvector
│   ├── api/fastapi-deployment.yaml    # API service with auto-scaling
│   ├── monitoring/                    # Prometheus + Grafana stack
│   └── ingress.yaml                   # Load balancing & external access
├── INSTALLATION.md                    # 📋 Complete deployment guide
├── src/database/postgres_adapter.py   # 🔧 PostgreSQL connection management
├── scripts/mass_qwen_analysis.py      # 🤖 Main AI analysis script (legacy)
├── scripts/tools/database_diagnostics.py # 📊 Primary diagnostic tool
├── main.py                            # 🕷️ Genius.com scraper entry point
├── config.yaml                        # ⚙️ System configuration
├── .env                              # 🔐 PostgreSQL + API credentials
├── docker-compose.pgvector.yml       # 🐳 PostgreSQL with pgvector
└── scripts/archive/                   # 📦 SQLite legacy scripts
```

---

**REMEMBER**: This system uses PostgreSQL with connection pooling for enterprise-grade concurrent processing. Always start with `database_diagnostics.py --quick` and use the provided SQL queries instead of ad-hoc database requests.