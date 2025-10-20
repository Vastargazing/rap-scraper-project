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

from src.config import get_config
from .routes import (
    analyze,
    batch,
    health,
    ml_models,
    models_info,
    web,
)

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

# Register all route modules
app.include_router(health.router, tags=["health"])
app.include_router(analyze.router, tags=["analysis"])
app.include_router(ml_models.router, tags=["ml-models"])
app.include_router(batch.router, tags=["batch"])
app.include_router(web.router, tags=["web"])
app.include_router(models_info.router, tags=["models"])

logger.info("✅ Unified FastAPI application initialized (v3.0.0)")
logger.info("   Routes registered:")
logger.info("   - GET  /health          - Health check")
logger.info("   - GET  /config/info     - Configuration info")
logger.info("   - POST /analyze         - QWEN analysis")
logger.info("   - GET  /cache/stats     - Cache statistics")
logger.info("   - POST /generate        - Lyrics generation")
logger.info("   - POST /style-transfer  - Style transfer")
logger.info("   - POST /predict-quality - Quality prediction")
logger.info("   - POST /analyze-trends  - Trend analysis")
logger.info("   - POST /batch           - Batch processing")
logger.info("   - GET  /batch/{id}/status - Batch status")
logger.info("   - GET  /                - Web interface")
logger.info("   - GET  /models/info     - Models info")
logger.info("   - GET  /models/status   - Models status")

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
