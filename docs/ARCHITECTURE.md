# 🏗️ Rap Scraper Project - Architecture Documentation

**Version:** 2.0.0
**Last Updated:** October 1, 2025
**Status:** Production-Ready ML Platform

---

## 📑 Table of Contents

- [Overview](#overview)
- [High-Level Architecture](#high-level-architecture)
- [Core Components](#core-components)
- [Data Flow](#data-flow)
- [Technology Stack](#technology-stack)
- [Directory Structure](#directory-structure)
- [Module Documentation](#module-documentation)
- [Infrastructure](#infrastructure)
- [Deployment](#deployment)
- [Development Workflow](#development-workflow)

---

## 🎯 Overview

**Rap Scraper Project** is a production-ready ML platform for comprehensive rap lyrics analysis. The system processes 57,718+ tracks using 5 AI analyzers, stores data in PostgreSQL with pgvector for semantic search, and provides a FastAPI service for real-time analysis.

### Key Statistics

- **Tracks:** 57,718 with complete lyrics
- **AI Analyses:** 269,646 completed analyses
- **Coverage:** 100% analysis coverage
- **AI Models:** 5 analyzers (QWEN, Gemma, Emotion, Algorithmic, Ollama)
- **Database:** PostgreSQL 15 + pgvector for vector similarity
- **API:** FastAPI with 20-connection pool
- **Caching:** Redis with 85% hit rate
- **Monitoring:** Prometheus + Grafana

### Architecture Principles

1. **Type-Safe Configuration** - Pydantic-based config management
2. **Async-First** - Asynchronous operations for scalability
3. **Concurrent Processing** - Multiple scripts run simultaneously
4. **Production-Ready** - Docker, Kubernetes, monitoring included
5. **ML-Focused** - QWEN as primary model for training
6. **Semantic Search** - pgvector for similarity queries

---

## 🏛️ High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    PRESENTATION LAYER                           │
├─────────────────────────────────────────────────────────────────┤
│  • FastAPI REST API (port 8000)                                 │
│  • Web Interface                                                │
│  • Prometheus Metrics Endpoint (/metrics)                       │
│  • Health Check Endpoint (/health)                              │
└─────────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│                    APPLICATION LAYER                            │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌───────────────────┐  ┌───────────────────┐                  │
│  │  Configuration    │  │   API Service     │                  │
│  │  (Pydantic)       │  │   (FastAPI)       │                  │
│  └───────────────────┘  └───────────────────┘                  │
│                                                                 │
│  ┌───────────────────────────────────────────────────────┐     │
│  │              AI Analysis Engine                       │     │
│  ├───────────────────────────────────────────────────────┤     │
│  │  • QWEN Analyzer (Primary - qwen3-4b-fp8)           │     │
│  │  • Gemma Analyzer (gemma-3-27b-it)                  │     │
│  │  • Emotion Analyzer (Hugging Face)                  │     │
│  │  • Algorithmic Analyzer (Rule-based)                │     │
│  │  • Ollama Analyzer (Local LLM)                      │     │
│  └───────────────────────────────────────────────────────┘     │
│                                                                 │
│  ┌───────────────────────────────────────────────────────┐     │
│  │              Data Processing Pipeline                 │     │
│  ├───────────────────────────────────────────────────────┤     │
│  │  • Scrapers (Genius.com)                            │     │
│  │  • Enhancers (Spotify API)                          │     │
│  │  • Batch Processing                                 │     │
│  └───────────────────────────────────────────────────────┘     │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│                       DATA LAYER                                │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌──────────────────┐  ┌──────────────────┐  ┌──────────────┐  │
│  │   PostgreSQL     │  │   Redis Cache    │  │  pgvector    │  │
│  │   (Primary DB)   │  │   (85% hit rate) │  │  (Semantic)  │  │
│  ├──────────────────┤  ├──────────────────┤  ├──────────────┤  │
│  │ • 57,718 tracks  │  │ • Artist cache   │  │ • Embeddings │  │
│  │ • 269,646 anal.  │  │ • Rate limiting  │  │ • Similarity │  │
│  │ • 20 conn pool   │  │ • Circuit break  │  │ • Search     │  │
│  └──────────────────┘  └──────────────────┘  └──────────────┘  │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│                    ML MODELS LAYER                              │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌───────────────────┐  ┌───────────────────┐                  │
│  │  QWEN Training    │  │  Quality Predict  │                  │
│  │  (Primary Model)  │  │  (RandomForest)   │                  │
│  └───────────────────┘  └───────────────────┘                  │
│                                                                 │
│  ┌───────────────────┐  ┌───────────────────┐                  │
│  │  Style Transfer   │  │  Trend Analysis   │                  │
│  │  (T5-based)       │  │  (KMeans+PCA)     │                  │
│  └───────────────────┘  └───────────────────┘                  │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│                  INFRASTRUCTURE LAYER                           │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐          │
│  │   Docker     │  │  Kubernetes  │  │  Monitoring  │          │
│  │  Containers  │  │  Orchestrat  │  │  (Prom+Graf) │          │
│  └──────────────┘  └──────────────┘  └──────────────┘          │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## 🧩 Core Components

### 1. Configuration System (`src/config/`)

**Type-safe Pydantic-based configuration management**

- **Purpose:** Centralized, validated configuration
- **Technology:** Pydantic + YAML + python-dotenv
- **Key Features:**
  - Type-safe configuration with IDE autocomplete
  - Automatic ENV variable substitution
  - Multi-environment support (dev/staging/prod)
  - Validation on startup (fail-fast)

**Files:**
```
src/config/
├── __init__.py           # Exports: get_config(), Config
├── config_loader.py      # Main Pydantic models + CLI
├── test_loader.py        # Full test suite
└── README.md            # Complete documentation
```

**Usage:**
```python
from src.config import get_config

config = get_config()  # Cached singleton
db_url = config.database.connection_string
api_port = config.api.port
qwen_api_key = config.analyzers.get_qwen().api_key
```

### 2. Database Layer (`src/database/`)

**PostgreSQL + pgvector for data storage and semantic search**

- **Purpose:** Primary data storage with vector similarity
- **Technology:** PostgreSQL 15, pgvector, asyncpg
- **Key Features:**
  - 20-connection pool for concurrent access
  - Vector embeddings for semantic search
  - ACID transactions for data integrity
  - Async operations for scalability

**Files:**
```
src/database/
├── __init__.py
├── connection.py         # Connection pooling
└── postgres_adapter.py   # Main database interface
```

**Schema:**
```sql
-- Main tracks table
CREATE TABLE tracks (
    id                SERIAL PRIMARY KEY,
    title             VARCHAR(500),
    artist            VARCHAR(200),
    lyrics            TEXT,
    spotify_data      JSONB,
    lyrics_embedding  vector(384),
    audio_embedding   vector(128)
);

-- AI analysis results
CREATE TABLE analysis_results (
    id                   SERIAL PRIMARY KEY,
    track_id             INTEGER REFERENCES tracks(id),
    analyzer_type        VARCHAR(50),
    sentiment            VARCHAR,
    confidence           NUMERIC,
    analysis_data        JSONB,
    analysis_embedding   vector(256)
);
```

### 3. AI Analyzers (`src/analyzers/`)

**5 AI models for comprehensive rap analysis**

#### 3.1 QWEN Analyzer (Primary)
- **Model:** qwen/qwen3-4b-fp8 via Novita AI
- **Purpose:** Primary AI analysis with highest quality
- **Coverage:** 57,716 tracks (100%)
- **Status:** ✅ WORKING (100% success rate)
- **File:** `src/analyzers/qwen_analyzer.py`

#### 3.2 Gemma Analyzer
- **Model:** gemma-3-27b-it
- **Purpose:** Alternative AI analysis
- **Coverage:** 34,320 tracks (59.4%)
- **File:** `src/analyzers/` (legacy location)

#### 3.3 Emotion Analyzer
- **Model:** j-hartmann/emotion-english-distilroberta-base (Hugging Face)
- **Purpose:** 6-emotion detection (joy, sadness, anger, fear, surprise, love)
- **Technology:** Transformers, PyTorch
- **File:** `src/analyzers/emotion_analyzer.py`

#### 3.4 Algorithmic Analyzer
- **Purpose:** Rule-based baseline analysis
- **Features:** Sentiment scoring, complexity metrics, theme detection
- **Status:** Production-ready, 100% coverage
- **File:** `src/analyzers/algorithmic_analyzer.py`

#### 3.5 Ollama Analyzer
- **Model:** llama3.1:8b (local)
- **Purpose:** Local LLM analysis (no API costs)
- **Status:** Development/testing
- **File:** `src/analyzers/ollama_analyzer.py`

### 4. ML Models (`models/`)

**Research & Training components**

#### 4.1 QWEN Training System (`models/test_qwen.py`)
- **Purpose:** PRIMARY ML MODEL for training
- **Capabilities:**
  - API testing and validation
  - Dataset preparation from PostgreSQL
  - Training simulation (prompt engineering)
  - Model evaluation (MAE: 0.450)
- **Commands:**
  ```bash
  python models/test_qwen.py --test-api
  python models/test_qwen.py --prepare-dataset
  python models/test_qwen.py --train
  python models/test_qwen.py --evaluate
  python models/test_qwen.py --all
  ```

#### 4.2 Quality Predictor (`models/quality_prediction.py`)
- **Purpose:** Commercial potential prediction
- **Technology:** RandomForest, GradientBoosting
- **Targets:** quality_score, commercial_potential, viral_potential, longevity
- **Output:** `quality_predictor.pkl`

#### 4.3 Style Transfer (`models/style_transfer.py`)
- **Purpose:** Transfer rap style between artists
- **Technology:** T5-small (Transformers)
- **Task:** "transfer rap style: source: [lyrics] target_style: [artist]"

#### 4.4 Trend Analysis (`models/trend_analysis.py`)
- **Purpose:** Trend analysis and prediction
- **Technology:** KMeans, PCA, TF-IDF
- **Outputs:** `trend_analysis_report.json`, `trend_dashboard.html`

### 5. Data Collection (`src/scrapers/`)

**Web scraping and data enrichment**

- **Genius.com Scraper:** 345+ artists, 57,717 tracks
- **Spotify Enricher:** Metadata, audio features, popularity
- **Smart Resume:** Checkpoint-based scraping
- **Data Validation:** Duplicate detection, quality control

### 6. API Service (`src/api/`)

**FastAPI REST API for production deployment**

- **Technology:** FastAPI + Uvicorn
- **Port:** 8000 (configurable)
- **Features:**
  - RESTful endpoints for analysis
  - Web interface
  - Prometheus metrics
  - Health checks
  - CORS support
  - Rate limiting

**Key Endpoints:**
```
GET  /                  # Web interface
GET  /health            # Health check
GET  /metrics           # Prometheus metrics
POST /analyze           # Analyze lyrics
POST /search            # Semantic search
GET  /track/{id}        # Get track details
```

### 7. Caching Layer (Redis)

**Intelligent caching strategy**

- **Technology:** Redis 7 Alpine
- **Hit Rate:** 85%+
- **Features:**
  - Artist cache (TTL: 1 hour)
  - Song hashes (infinite TTL for deduplication)
  - Rate limit state (60s TTL)
  - Analysis results (24h TTL)

### 8. Monitoring (`monitoring/`)

**Production observability**

- **Prometheus:** Metrics collection (25+ custom metrics)
- **Grafana:** Dashboards and visualization
- **Metrics:**
  - Response times, throughput
  - Error rates, success rates
  - Cache hit ratios
  - Database pool status
  - Memory/CPU usage

---

## 🔄 Data Flow

### 1. Data Ingestion Flow

```
Genius.com
    │
    ├─> Scraper (src/scrapers/)
    │
    ├─> Lyrics Extraction
    │
    ├─> PostgreSQL (tracks table)
    │
    └─> Spotify API
        │
        └─> Metadata Enrichment
            │
            └─> PostgreSQL (updated tracks)
```

### 2. Analysis Flow

```
Track (from DB)
    │
    ├─> AI Analyzer Selection
    │   ├─> QWEN (Primary)
    │   ├─> Gemma
    │   ├─> Emotion
    │   ├─> Algorithmic
    │   └─> Ollama
    │
    ├─> Analysis Processing
    │   ├─> API Call / Local Inference
    │   ├─> Result Validation
    │   └─> Confidence Scoring
    │
    ├─> PostgreSQL (analysis_results table)
    │
    └─> Cache Update (Redis)
```

### 3. Query Flow

```
API Request
    │
    ├─> Cache Check (Redis)
    │   ├─> HIT: Return cached result
    │   └─> MISS: Continue
    │
    ├─> PostgreSQL Query
    │   ├─> Standard Query
    │   └─> Vector Search (pgvector)
    │
    ├─> Result Processing
    │
    ├─> Cache Update (Redis)
    │
    └─> API Response
```

### 4. ML Training Flow

```
PostgreSQL (57,718 tracks + 269,646 analyses)
    │
    ├─> Dataset Preparation (models/test_qwen.py)
    │   ├─> Filter: confidence > 0.5, length > 100
    │   ├─> Sample: 1000 tracks
    │   └─> Split: 800 train / 200 eval
    │
    ├─> Training Pipeline
    │   ├─> Prompt Engineering
    │   ├─> API Calls (QWEN)
    │   └─> Result Collection
    │
    ├─> Evaluation
    │   ├─> MAE Calculation: 0.450
    │   ├─> Baseline Comparison: 62% better
    │   └─> Metrics Logging
    │
    └─> Model Artifacts
        └─> results/qwen_training/
```

---

## 🛠️ Technology Stack

### Backend

| Component | Technology | Version | Purpose |
|-----------|-----------|---------|---------|
| **Language** | Python | 3.10+ | Primary language |
| **Web Framework** | FastAPI | 0.104+ | REST API |
| **Web Server** | Uvicorn | 0.24+ | ASGI server |
| **Config** | Pydantic | 2.0+ | Type-safe config |

### Database & Storage

| Component | Technology | Version | Purpose |
|-----------|-----------|---------|---------|
| **Primary DB** | PostgreSQL | 15 | Main data storage |
| **Vector Search** | pgvector | Latest | Semantic similarity |
| **Cache** | Redis | 7 | Intelligent caching |
| **Connection Pool** | psycopg2 | 2.9+ | DB connections |

### AI/ML

| Component | Technology | Version | Purpose |
|-----------|-----------|---------|---------|
| **Primary Model** | QWEN | qwen3-4b-fp8 | Main AI analyzer |
| **API** | OpenAI SDK | 1.0+ | LLM API calls |
| **Emotion AI** | Transformers | Latest | Emotion detection |
| **ML Models** | scikit-learn | Latest | Quality prediction |
| **Style Transfer** | T5 | t5-small | Style adaptation |

### Infrastructure

| Component | Technology | Version | Purpose |
|-----------|-----------|---------|---------|
| **Containerization** | Docker | 24+ | Application packaging |
| **Orchestration** | Kubernetes | 1.28+ | Container management |
| **Package Manager** | Helm | Latest | K8s deployment |
| **GitOps** | ArgoCD | Latest | Continuous deployment |

### Monitoring

| Component | Technology | Version | Purpose |
|-----------|-----------|---------|---------|
| **Metrics** | Prometheus | Latest | Metrics collection |
| **Visualization** | Grafana | Latest | Dashboards |
| **Client** | prometheus-client | 0.19+ | Python metrics |

### Development Tools

| Component | Technology | Purpose |
|-----------|-----------|---------|
| **Dependency Mgmt** | Poetry | Package management |
| **Testing** | pytest | Unit/integration tests |
| **Linting** | black, flake8 | Code quality |
| **Type Checking** | mypy | Static type checking |

---

## 📁 Directory Structure

```
rap-scraper-project/
│
├── 📄 Configuration Files
│   ├── config.yaml              # Main config (Pydantic-based)
│   ├── config.example.yaml      # Config template
│   ├── .env                     # Environment variables (secrets)
│   ├── .env.example             # ENV template
│   ├── pyproject.toml           # Poetry dependencies
│   └── docker-compose.yml       # Docker orchestration
│
├── 📚 Documentation
│   ├── README.md                # Main project documentation
│   ├── ARCHITECTURE.md          # This file
│   ├── SECURITY.md              # Security policies
│   └── docs/
│       ├── claude.md            # AI assistant context
│       ├── PROGRESS.md          # Development log
│       ├── TO_DO.md             # Future tasks
│       └── specs/               # Technical specifications
│
├── 🐍 Source Code (src/)
│   ├── config/                  # ⚙️ Type-safe configuration
│   │   ├── config_loader.py    # Pydantic models + CLI
│   │   ├── test_loader.py      # Test suite
│   │   └── README.md           # Config documentation
│   │
│   ├── database/                # 🐘 PostgreSQL integration
│   │   ├── connection.py       # Connection pooling
│   │   └── postgres_adapter.py # Main DB interface
│   │
│   ├── analyzers/               # 🤖 AI Analysis engines
│   │   ├── qwen_analyzer.py    # Primary QWEN analyzer
│   │   ├── emotion_analyzer.py # Emotion detection
│   │   ├── algorithmic_analyzer.py # Rule-based analysis
│   │   └── ollama_analyzer.py  # Local LLM
│   │
│   ├── api/                     # 🌐 FastAPI service
│   │   └── ml_api_service_v2.py
│   │
│   ├── scrapers/                # 🕷️ Data collection
│   ├── enhancers/               # 🎵 Spotify enrichment
│   ├── models/                  # 📊 Data models
│   ├── utils/                   # 🔧 Utilities
│   └── cli/                     # 💻 CLI interfaces
│
├── 🤖 ML Models (models/)
│   ├── test_qwen.py             # PRIMARY ML MODEL
│   ├── quality_prediction.py   # Quality predictor
│   ├── style_transfer.py       # Style transfer
│   ├── trend_analysis.py       # Trend analysis
│   ├── quality_predictor.pkl   # Trained model
│   └── backups/                # Model backups
│
├── 📜 Scripts (scripts/)
│   ├── tools/
│   │   └── database_diagnostics.py  # Main diagnostic tool
│   ├── postgres/               # PostgreSQL utilities
│   ├── ml/                     # ML scripts
│   └── utils/                  # Helper scripts
│
├── 🐳 Infrastructure
│   ├── docker-compose.yml      # Production stack
│   ├── docker-compose.dev.yml  # Development stack
│   ├── Dockerfile              # Main image
│   ├── Dockerfile.prod         # Production image
│   │
│   ├── k8s/                    # Kubernetes manifests
│   │   ├── namespace-and-config.yaml
│   │   ├── api/                # API deployment
│   │   ├── postgres/           # Database
│   │   └── monitoring/         # Prometheus/Grafana
│   │
│   ├── helm/                   # Helm charts
│   │   └── rap-analyzer/       # Main chart
│   │
│   ├── gitops/                 # ArgoCD configuration
│   │   ├── install-argocd.ps1
│   │   └── applications/
│   │
│   └── multi-region/           # Multi-region deployment
│       └── deploy-multi-region.ps1
│
├── 📊 Monitoring
│   ├── monitoring/
│   │   ├── prometheus.yml      # Prometheus config
│   │   ├── grafana/            # Dashboards
│   │   └── scripts/            # Monitoring scripts
│   │
│   └── logs/                   # Application logs
│
├── 💾 Data Directories
│   ├── data/                   # Raw data
│   │   └── ml/                 # ML datasets
│   │
│   ├── results/                # ML results
│   │   └── qwen_training/      # QWEN training outputs
│   │
│   ├── cache/                  # Cache files
│   └── enhanced_data/          # Processed data
│
└── 🧪 Testing
    ├── tests/                  # Test suite
    │   ├── test_config/        # Config tests
    │   └── test_ml/            # ML tests
    │
    └── .benchmarks/            # Performance benchmarks
```

---

## 📖 Module Documentation

### Core Modules

#### `src.config`
Type-safe configuration management using Pydantic.

**Key Classes:**
- `Config` - Root configuration object
- `DatabaseConfig` - PostgreSQL settings
- `AnalyzersConfig` - AI analyzer settings
- `APIConfig` - FastAPI settings

**Usage:**
```python
from src.config import get_config
config = get_config()
```

**Documentation:** [src/config/README.md](src/config/README.md)

#### `src.database`
PostgreSQL integration with connection pooling.

**Key Classes:**
- `PostgreSQLManager` - Main database interface
- Connection pool management (20 max connections)

**Usage:**
```python
from src.database.postgres_adapter import PostgreSQLManager
db = PostgreSQLManager()
await db.initialize()
```

#### `src.analyzers`
AI analysis engines for rap lyrics.

**Available Analyzers:**
- `QwenAnalyzer` - Primary QWEN-based analyzer
- `EmotionAnalyzer` - Emotion detection (6 emotions)
- `AlgorithmicAnalyzer` - Rule-based analysis
- `OllamaAnalyzer` - Local LLM analysis

**Usage:**
```python
from src.analyzers.qwen_analyzer import QwenAnalyzer
analyzer = QwenAnalyzer()
result = await analyzer.analyze(lyrics)
```

### ML Models

#### `models.test_qwen`
PRIMARY ML MODEL for training and evaluation.

**Commands:**
```bash
python models/test_qwen.py --test-api          # Test API
python models/test_qwen.py --prepare-dataset   # Prepare data
python models/test_qwen.py --train             # Train model
python models/test_qwen.py --evaluate          # Evaluate
python models/test_qwen.py --all               # Full pipeline
```

**Key Classes:**
- `QwenTrainingSystem` - Main training system
- `QwenConfig` - Model configuration

#### `models.quality_prediction`
ML model for predicting track quality and commercial potential.

**Features:**
- Multi-target regression
- SHAP interpretability
- Cross-validation

**Usage:**
```python
from models.quality_prediction import RapQualityPredictor
predictor = RapQualityPredictor()
quality_score = predictor.predict(track_features)
```

### Scripts

#### `scripts/tools/database_diagnostics.py`
Main diagnostic tool for database health checks.

**Usage:**
```bash
python scripts/tools/database_diagnostics.py --quick      # Quick check
python scripts/tools/database_diagnostics.py --analysis   # Analysis status
python scripts/tools/database_diagnostics.py --connections # Pool status
```

---

## 🏗️ Infrastructure

### Docker Setup

**Production Stack:**
```bash
docker-compose -f docker-compose.full.yml up -d
```

**Components:**
- `rap-analyzer-api` - FastAPI service
- `rap-analyzer-postgres-vector` - PostgreSQL + pgvector
- `rap-analyzer-redis` - Redis cache
- `rap-analyzer-prometheus` - Metrics
- `rap-analyzer-grafana` - Dashboards

### Kubernetes Deployment

**Helm Installation:**
```bash
helm install rap-analyzer ./helm/rap-analyzer \
  --create-namespace \
  --namespace rap-analyzer
```

**Key Resources:**
- Deployment: API with HPA (3-10 replicas)
- StatefulSet: PostgreSQL with persistence
- Service: LoadBalancer for external access
- ConfigMap: Configuration management
- Secret: Sensitive credentials

### GitOps with ArgoCD

**Installation:**
```bash
./gitops/install-argocd.ps1
```

**Features:**
- Automated deployments
- Self-healing applications
- Multi-cluster support
- Git-based source of truth

---

## 🚀 Deployment

### Local Development

```bash
# 1. Setup configuration
cp .env.example .env
cp config.example.yaml config.yaml

# 2. Validate config
python src/config/test_loader.py

# 3. Start infrastructure
docker-compose -f docker-compose.pgvector.yml up -d

# 4. Run application
poetry install
poetry run python main.py
```

### Production Deployment

```bash
# 1. Build Docker image
docker build -t rap-analyzer:latest -f Dockerfile.prod .

# 2. Deploy to Kubernetes
helm install rap-analyzer ./helm/rap-analyzer \
  --namespace rap-analyzer \
  --create-namespace

# 3. Verify deployment
kubectl get pods -n rap-analyzer
kubectl get svc -n rap-analyzer

# 4. Access application
kubectl port-forward svc/rap-analyzer-service 8000:8000 -n rap-analyzer
```

### Multi-Region Deployment

```bash
# Deploy to multiple regions
.\multi-region\deploy-multi-region.ps1 -Action deploy

# Check status
.\multi-region\deploy-multi-region.ps1 -Action status

# Test deployment
python multi-region/test-multi-region.py
```

---

## 🔧 Development Workflow

### 1. Setup Development Environment

```bash
# Install Poetry
curl -sSL https://install.python-poetry.org | python3 -

# Install dependencies
poetry install --with dev

# Activate virtual environment
poetry shell

# Verify setup
python src/config/test_loader.py
python scripts/tools/database_diagnostics.py --quick
```

### 2. Configuration Management

```bash
# Create/update configuration
cp config.example.yaml config.yaml
# Edit config.yaml with your settings

# Create/update environment variables
cp .env.example .env
# Edit .env with your secrets

# Validate configuration
python src/config/config_loader.py
```

### 3. Database Setup

```bash
# Start PostgreSQL + pgvector
docker-compose -f docker-compose.pgvector.yml up -d

# Verify connection
python -c "
from src.database.postgres_adapter import PostgreSQLManager
import asyncio
async def test():
    db = PostgreSQLManager()
    await db.initialize()
    print('✅ PostgreSQL OK')
    await db.close()
asyncio.run(test())
"

# Run diagnostics
python scripts/tools/database_diagnostics.py --quick
```

### 4. Running Analysis

```bash
# Test QWEN API
python models/test_qwen.py --test-api

# Run full ML pipeline
python models/test_qwen.py --all

# Start ML API service
python src/models/ml_api_service.py --host 127.0.0.1 --port 8001

# Test ML API
python test_ml_api.py
```

### 5. Testing

```bash
# Run all tests
poetry run pytest tests/ -v

# Run with coverage
poetry run pytest tests/ --cov=src --cov-report=html

# Run specific test
poetry run pytest tests/test_config/test_loader.py -v
```

### 6. Code Quality

```bash
# Format code
poetry run black src/

# Lint code
poetry run flake8 src/

# Type checking
poetry run mypy src/
```

---

## 📊 Performance Metrics

### Current System Performance

| Metric | Value | Target |
|--------|-------|--------|
| **Tracks Stored** | 57,718 | - |
| **Analyses Completed** | 269,646 | - |
| **Analysis Coverage** | 100% | 100% |
| **PostgreSQL Query Time** | <500ms | <1000ms |
| **Redis Hit Rate** | 85%+ | >80% |
| **API Response Time (p95)** | <1s | <2s |
| **Connection Pool Utilization** | <75% | <80% |
| **QWEN API Success Rate** | 100% | >90% |
| **ML Model MAE** | 0.450 | <0.5 |

### Scalability Limits

| Component | Current | Max Capacity | Scale Strategy |
|-----------|---------|--------------|----------------|
| **PostgreSQL** | 57K tracks | 500K tracks | Read replicas + PgBouncer |
| **Redis** | 512MB | 1M+ entries | Cluster mode (sharding) |
| **API** | 1K req/min | 10K req/min | Horizontal scaling |
| **Connection Pool** | 20 conn | 100 conn | Pool size increase |

---

## 🔐 Security Considerations

### Configuration Security

1. **Environment Variables**
   - All secrets in `.env` (gitignored)
   - No hardcoded credentials
   - Pydantic validation on startup

2. **API Security**
   - Rate limiting (100 req/min)
   - CORS configuration
   - Health check endpoints

3. **Database Security**
   - SSL/TLS connections
   - Connection pooling limits
   - Non-root Docker containers

### Best Practices

- ✅ Type-safe configuration (Pydantic)
- ✅ Environment-specific settings
- ✅ Secrets management (ENV variables)
- ✅ Input validation (Pydantic models)
- ✅ Error handling and logging
- ✅ Monitoring and alerting

---

## 📈 Future Roadmap

### Phase 5: Advanced AI Integration (Current)

- [ ] Fine-tuning QWEN model (when API supports)
- [ ] Real-time ML inference (WebSocket streaming)
- [ ] Advanced embeddings (musical features + lyrics)
- [ ] Cross-modal analysis (lyrics + audio)

### Phase 6: Enterprise Features

- [ ] Security enhancement (Redis AUTH, SSL/TLS, RBAC)
- [ ] Advanced analytics (ML insights dashboard)
- [ ] API rate limiting (Redis-backed throttling)
- [ ] Backup automation (Redis + PostgreSQL)

### Phase 7: ML Platform

- [ ] Feature store implementation
- [ ] Model versioning and registry
- [ ] A/B testing framework
- [ ] Automated retraining pipeline

---

## 📚 Additional Resources

### Documentation

- **Main README:** [README.md](README.md)
- **Configuration Guide:** [src/config/README.md](src/config/README.md)
- **AI Context:** [docs/claude.md](docs/claude.md)
- **Progress Log:** [docs/PROGRESS.md](docs/PROGRESS.md)
- **Security Policy:** [SECURITY.md](SECURITY.md)

### External Resources

- [Pydantic Documentation](https://docs.pydantic.dev/)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [PostgreSQL Documentation](https://www.postgresql.org/docs/)
- [pgvector Documentation](https://github.com/pgvector/pgvector)
- [Kubernetes Documentation](https://kubernetes.io/docs/)

---

## 🤝 Contributing

When contributing to this project:

1. **Update Configuration**: Add new settings to `config.yaml` and Pydantic models
2. **Update Documentation**: Keep this ARCHITECTURE.md in sync with changes
3. **Add Tests**: Write tests for new functionality
4. **Follow Conventions**: Use existing patterns and code style
5. **Update Logs**: Add entries to `docs/PROGRESS.md` (new entries at top)

---

## 📝 Version History

- **2.1.0** (2025-10-27) - **Legacy Cleanup**: Removed all SQLite code (3,805 lines), full PostgreSQL migration
- **2.0.0** (2025-10-01) - Pydantic Config System, QWEN Primary Model, 100% analysis
- **1.5.0** (2025-09-28) - ML Models system, Quality Predictor, Style Transfer
- **1.0.0** (2025-01-19) - Kubernetes deployment, Multi-region, GitOps
- **0.5.0** - PostgreSQL migration, pgvector integration
- **0.1.0** - Initial release (SQLite-based, deprecated)

---

**Last Updated:** October 27, 2025
**Maintained by:** Vastargazing
**Status:** Production-Ready ✅
