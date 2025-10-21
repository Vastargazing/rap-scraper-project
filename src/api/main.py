"""
🚀 Unified FastAPI Application (v3.0.0)

Consolidated main application combining:
- Config-based API (from ml_api_service_v2.py)
- ML models logic (from src/models/ml_api_service.py)
- Web interface + Batch processing (from api.py)

Architecture:
- Type-safe Pydantic configuration
- Modular route structure
- Redis caching throughout
- Unified health checks
- Single entry point for production

Author: ML Platform Team
Date: October 2025
"""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# Configure logging FIRST
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Graceful imports with fallbacks
try:
    from src.config import get_config
    config = get_config()
    CONFIG_AVAILABLE = True
except Exception as e:
    logger.warning(f"⚠️ Config not available: {e}")
    # Fallback config
    class FallbackConfig:
        class application:
            environment = "development"
            name = "Rap ML API"
            version = "3.0.0"
        class api:
            host = "127.0.0.1"
            port = 8000
            workers = 1
            reload = True
            log_level = "info"
            class cors:
                enabled = True
                origins = ["*"]
                allow_credentials = True
                allow_methods = ["*"]
                allow_headers = ["*"]
            class docs:
                enabled = True
                title = "Rap ML API"
                version = "3.0.0"
                swagger_url = "/docs"
                redoc_url = "/redoc"
        class database:
            type = "postgresql"
            pool_size = 20
        class redis:
            enabled = False
    config = FallbackConfig()
    CONFIG_AVAILABLE = False

# Import routes with graceful fallback
ROUTES_AVAILABLE = {}
try:
    from .routes import health
    ROUTES_AVAILABLE['health'] = health
except Exception as e:
    logger.warning(f"⚠️ Health routes not available: {e}")
    ROUTES_AVAILABLE['health'] = None

try:
    from .routes import analyze
    ROUTES_AVAILABLE['analyze'] = analyze
except Exception as e:
    logger.warning(f"⚠️ Analyze routes not available: {e}")
    ROUTES_AVAILABLE['analyze'] = None

try:
    from .routes import ml_models
    ROUTES_AVAILABLE['ml_models'] = ml_models
except Exception as e:
    logger.warning(f"⚠️ ML models routes not available: {e}")
    ROUTES_AVAILABLE['ml_models'] = None

try:
    from .routes import batch
    ROUTES_AVAILABLE['batch'] = batch
except Exception as e:
    logger.warning(f"⚠️ Batch routes not available: {e}")
    ROUTES_AVAILABLE['batch'] = None

try:
    from .routes import web
    ROUTES_AVAILABLE['web'] = web
except Exception as e:
    logger.warning(f"⚠️ Web routes not available: {e}")
    ROUTES_AVAILABLE['web'] = None

try:
    from .routes import models_info
    ROUTES_AVAILABLE['models_info'] = models_info
except Exception as e:
    logger.warning(f"⚠️ Models info routes not available: {e}")
    ROUTES_AVAILABLE['models_info'] = None

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Load configuration
config = get_config()


# ============================================================================
# STARTUP / SHUTDOWN EVENTS
# ============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    # STARTUP
    logger.info("=" * 80)
    logger.info("🚀 STARTING UNIFIED RAP ML API (v3.0.0)")
    logger.info("=" * 80)
    logger.info(f"   Environment: {config.application.environment}")
    logger.info(f"   API Host: {config.api.host}:{config.api.port}")
    logger.info(f"   Workers: {config.api.workers}")
    logger.info(f"   Database: {config.database.type} (pool: {config.database.pool_size})")
    logger.info(f"   Redis: {'enabled' if config.redis.enabled else 'disabled'}")
    logger.info("=" * 80)
    
    yield
    
    # SHUTDOWN
    logger.info("=" * 80)
    logger.info("🛑 SHUTTING DOWN RAP ML API")
    logger.info("=" * 80)


# ============================================================================
# CREATE FASTAPI APP
# ============================================================================

app = FastAPI(
    title=config.api.docs.title,
    description="Unified ML API for rap lyrics analysis with RAG systems",
    version=config.api.docs.version,
    docs_url=config.api.docs.swagger_url if config.api.docs.enabled else None,
    redoc_url=config.api.docs.redoc_url if config.api.docs.enabled else None,
    lifespan=lifespan,
)

# ============================================================================
# MIDDLEWARE SETUP
# ============================================================================

# CORS middleware
if config.api.cors.enabled:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=config.api.cors.origins,
        allow_credentials=config.api.cors.allow_credentials,
        allow_methods=config.api.cors.allow_methods,
        allow_headers=config.api.cors.allow_headers,
    )
    logger.debug(f"✅ CORS enabled for origins: {config.api.cors.origins}")

# ============================================================================
# INCLUDE ROUTES (All route modules)
# ============================================================================

# Register all available route modules
routes_registered = []

if ROUTES_AVAILABLE.get('health'):
    app.include_router(ROUTES_AVAILABLE['health'].router, tags=["health"])
    routes_registered.append("health")
if ROUTES_AVAILABLE.get('analyze'):
    app.include_router(ROUTES_AVAILABLE['analyze'].router, tags=["analysis"])
    routes_registered.append("analyze")
if ROUTES_AVAILABLE.get('ml_models'):
    app.include_router(ROUTES_AVAILABLE['ml_models'].router, tags=["ml-models"])
    routes_registered.append("ml_models")
if ROUTES_AVAILABLE.get('batch'):
    app.include_router(ROUTES_AVAILABLE['batch'].router, tags=["batch"])
    routes_registered.append("batch")
if ROUTES_AVAILABLE.get('web'):
    app.include_router(ROUTES_AVAILABLE['web'].router, tags=["web"])
    routes_registered.append("web")
if ROUTES_AVAILABLE.get('models_info'):
    app.include_router(ROUTES_AVAILABLE['models_info'].router, tags=["models"])
    routes_registered.append("models_info")

logger.info(f"✅ Unified FastAPI application initialized (v3.0.0)")
logger.info(f"   Routes registered: {', '.join(routes_registered) if routes_registered else 'NONE'}")

# ============================================================================
# RUN SERVER
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    
    logger.info("Starting uvicorn server...")
    uvicorn.run(
        "src.api.main:app",
        host=config.api.host,
        port=config.api.port,
        workers=config.api.workers if not config.api.reload else 1,
        reload=config.api.reload,
        log_level=config.api.log_level,
    )
