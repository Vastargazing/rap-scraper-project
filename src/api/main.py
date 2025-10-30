"""Main FastAPI application for Rap ML API (v3.0.0).

This is the entry point and core orchestrator of the Rap ML API, responsible for:
- Application initialization and configuration
- ML model loading and lifecycle management
- Route registration and middleware setup
- CORS configuration and security
- Graceful error handling and fallbacks

Architecture:
    ✅ Config-based initialization with Pydantic validation
    ✅ Graceful import fallbacks for resilience
    ✅ Modular route system (6 route modules)
    ✅ Lifespan context manager for startup/shutdown
    ✅ Comprehensive error handling and logging

Components:
    1. Configuration Management - Load settings from .env via Pydantic
    2. Route Imports - Dynamic import with graceful fallbacks
    3. Lifespan Manager - Initialize resources on startup, cleanup on shutdown
    4. ML Model Initialization - Load QWEN analyzer during startup (3-5s)
    5. CORS Middleware - Enable cross-origin requests
    6. FastAPI Application - Create and configure main app instance

Execution Flow:
    1️⃣ Python loads main.py and defines all components
    2️⃣ Configuration and routes imported with fallback handling
    3️⃣ Uvicorn starts FastAPI application
    4️⃣ lifespan.__aenter__() executes (startup phase):
       - Log configuration details
       - Initialize QWEN analyzer (takes 3-5 seconds)
       - Handle initialization errors gracefully
    5️⃣ API ready to serve requests
    6️⃣ On Ctrl+C: lifespan.__aexit__() executes (shutdown phase):
       - Log shutdown message
       - Cleanup resources (if needed)

Key Technologies:
    📦 FastAPI - Modern async web framework
    🐍 asynccontextmanager - Resource lifecycle management
    ⚡  async/await - Asynchronous request handling
    🤖 QWEN Analyzer - AI model for lyrics analysis
    🔐 CORS - Cross-origin resource sharing
    📝 Pydantic - Configuration validation

Usage:
    Development (with auto-reload):
        python -m uvicorn src.api.main:app --reload

    Production (multi-worker):
        uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --workers 4

Endpoints:
    Documentation:
        - http://localhost:8000/docs - Swagger UI
        - http://localhost:8000/redoc - ReDoc
    Health:
        - http://localhost:8000/health - System health check
    API:
        - http://localhost:8000/analyze - Lyrics analysis
        - http://localhost:8000/batch - Batch processing
        - http://localhost:8000/generate - Lyrics generation
        - http://localhost:8000/models/info - Model information

Important Notes:
    - QWEN analyzer initializes on EVERY startup (3-5 seconds delay)
    - If initialization fails, API starts anyway but analysis endpoints return 503
    - Graceful imports prevent total failure if a route module is broken
    - With --workers, each process initializes QWEN independently
    - Configuration is loaded from .env file via Pydantic validation

Author: ML Platform Team
Date: October 2025
Version: 3.0.0
"""

import logging
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# ============================================================================
# LOGGING CONFIGURATION
# ============================================================================

# Configure logging FIRST (before any other imports)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


# ============================================================================
# CONFIGURATION LOADING
# ============================================================================

# Load configuration with graceful fallback
try:
    from src.config import get_config

    config = get_config()
    CONFIG_AVAILABLE = True
    logger.debug("✅ Configuration loaded successfully")
except Exception as e:
    logger.warning(f"⚠️ Config loading failed: {e}")

    # Fallback configuration for resilience
    class FallbackConfig:
        """Fallback configuration when primary config fails to load.

        Provides sensible defaults to allow API to start even if
        configuration loading fails. Used for development and debugging.
        """

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
    logger.warning("⚠️ Using fallback configuration")


# ============================================================================
# ROUTE IMPORTS WITH GRACEFUL FALLBACKS
# ============================================================================

# Import route modules with error handling to prevent total failure
ROUTES_AVAILABLE: dict[str, Any] = {}

try:
    from .routes import health

    ROUTES_AVAILABLE["health"] = health
    logger.debug("✅ Health routes imported")
except Exception as e:
    logger.warning(f"⚠️ Health routes not available: {e}")
    ROUTES_AVAILABLE["health"] = None

try:
    from .routes import analyze

    ROUTES_AVAILABLE["analyze"] = analyze
    logger.debug("✅ Analyze routes imported")
except Exception as e:
    logger.warning(f"⚠️ Analyze routes not available: {e}")
    ROUTES_AVAILABLE["analyze"] = None

try:
    from .routes import ml_models

    ROUTES_AVAILABLE["ml_models"] = ml_models
    logger.debug("✅ ML models routes imported")
except Exception as e:
    logger.warning(f"⚠️ ML models routes not available: {e}")
    ROUTES_AVAILABLE["ml_models"] = None

try:
    from .routes import batch

    ROUTES_AVAILABLE["batch"] = batch
    logger.debug("✅ Batch routes imported")
except Exception as e:
    logger.warning(f"⚠️ Batch routes not available: {e}")
    ROUTES_AVAILABLE["batch"] = None

try:
    from .routes import web

    ROUTES_AVAILABLE["web"] = web
    logger.debug("✅ Web routes imported")
except Exception as e:
    logger.warning(f"⚠️ Web routes not available: {e}")
    ROUTES_AVAILABLE["web"] = None

try:
    from .routes import models_info

    ROUTES_AVAILABLE["models_info"] = models_info
    logger.debug("✅ Models info routes imported")
except Exception as e:
    logger.warning(f"⚠️ Models info routes not available: {e}")
    ROUTES_AVAILABLE["models_info"] = None


# ============================================================================
# INITIALIZATION FUNCTIONS
# ============================================================================

# Import ML model initialization functions
INIT_FUNCTIONS: dict[str, Any] = {}

try:
    from .routes.analyze import initialize_analyzer

    INIT_FUNCTIONS["initialize_analyzer"] = initialize_analyzer
    logger.debug("✅ initialize_analyzer imported successfully")
except Exception as e:
    logger.warning(f"⚠️ initialize_analyzer not available: {e}")
    INIT_FUNCTIONS["initialize_analyzer"] = None


# ============================================================================
# APPLICATION LIFESPAN MANAGEMENT
# ============================================================================


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifespan with startup and shutdown events.

    This context manager handles:
    - Startup: Initialize ML models, log configuration, setup resources
    - Shutdown: Cleanup resources, log shutdown message

    The lifespan pattern ensures proper resource management and graceful
    startup/shutdown behavior for production deployments.

    Args:
        app: FastAPI application instance

    Yields:
        None: Control is yielded to the application during normal operation

    Note:
        - QWEN analyzer initialization takes 3-5 seconds
        - Initialization errors are logged but don't prevent API startup
        - Multiple workers each run their own initialization
    """
    # ========================================================================
    # STARTUP PHASE
    # ========================================================================
    logger.info("=" * 80)
    logger.info("🚀 STARTING RAP ML API (v3.0.0)")
    logger.info("=" * 80)
    logger.info(f"   Environment: {config.application.environment}")
    logger.info(f"   API Host: {config.api.host}:{config.api.port}")
    logger.info(f"   Workers: {config.api.workers}")
    logger.info(
        f"   Database: {config.database.type} (pool: {config.database.pool_size})"
    )
    logger.info(f"   Redis: {'enabled' if config.redis.enabled else 'disabled'}")
    logger.info("=" * 80)

    # Initialize ML models
    logger.info("\n📊 INITIALIZING ML MODELS...")
    if INIT_FUNCTIONS.get("initialize_analyzer"):
        try:
            logger.info("   🤖 Initializing QWEN analyzer...")
            success = await INIT_FUNCTIONS["initialize_analyzer"]()
            if success:
                logger.info("   ✅ QWEN analyzer initialized successfully!")
            else:
                logger.warning(
                    "   ⚠️ Failed to initialize QWEN analyzer (check logs above)"
                )
        except Exception as e:
            logger.error(f"   ❌ Error initializing QWEN analyzer: {e}", exc_info=True)
    else:
        logger.warning("   ⚠️ Initialize function not available (check route imports)")

    logger.info("=" * 80)
    logger.info("✅ API READY TO SERVE REQUESTS")
    logger.info("=" * 80 + "\n")

    yield

    # ========================================================================
    # SHUTDOWN PHASE
    # ========================================================================
    logger.info("=" * 80)
    logger.info("🛑 SHUTTING DOWN RAP ML API")
    logger.info("=" * 80)
    # Add cleanup logic here if needed (e.g., close database connections)


# ============================================================================
# FASTAPI APPLICATION CREATION
# ============================================================================

app = FastAPI(
    title=config.api.docs.title,
    description="""
    🎤 **Rap ML API** - Unified ML platform for rap lyrics analysis

    ## Features
    - 🤖 AI-powered lyrics analysis using QWEN model
    - 🎨 Artist style transfer and generation
    - 📊 Quality prediction and trend analysis
    - ⚡ Redis caching for performance optimization
    - 🔄 Batch processing for large-scale operations

    ## Quick Start
    1. Check system health: `GET /health`
    2. Analyze lyrics: `POST /analyze`
    3. View models info: `GET /models/info`

    ## Documentation
    - **Swagger UI**: Interactive API testing
    - **ReDoc**: Detailed API reference
    - **Web Interface**: Browser-based testing at `/`
    """,
    version=config.api.docs.version,
    docs_url=config.api.docs.swagger_url if config.api.docs.enabled else None,
    redoc_url=config.api.docs.redoc_url if config.api.docs.enabled else None,
    lifespan=lifespan,
    openapi_tags=[
        {
            "name": "Health",
            "description": "System health checks and configuration",
        },
        {
            "name": "Analysis",
            "description": "Lyrics analysis and sentiment detection",
        },
        {
            "name": "ML Models",
            "description": "Model management and generation endpoints",
        },
        {
            "name": "Batch Processing",
            "description": "Asynchronous batch operations",
        },
        {
            "name": "Web Interface",
            "description": "Interactive web interface and model information",
        },
        {
            "name": "Models",
            "description": "Model status and monitoring",
        },
    ],
)


# ============================================================================
# MIDDLEWARE CONFIGURATION
# ============================================================================

# Configure CORS middleware for cross-origin requests
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
# ROUTE REGISTRATION
# ============================================================================

# Register all available route modules
routes_registered: list[str] = []

if ROUTES_AVAILABLE.get("health"):
    app.include_router(ROUTES_AVAILABLE["health"].router)
    routes_registered.append("health")
    logger.debug("✅ Health routes registered")

if ROUTES_AVAILABLE.get("analyze"):
    app.include_router(ROUTES_AVAILABLE["analyze"].router)
    routes_registered.append("analyze")
    logger.debug("✅ Analyze routes registered")

if ROUTES_AVAILABLE.get("ml_models"):
    app.include_router(ROUTES_AVAILABLE["ml_models"].router)
    routes_registered.append("ml_models")
    logger.debug("✅ ML models routes registered")

if ROUTES_AVAILABLE.get("batch"):
    app.include_router(ROUTES_AVAILABLE["batch"].router)
    routes_registered.append("batch")
    logger.debug("✅ Batch routes registered")

if ROUTES_AVAILABLE.get("web"):
    app.include_router(ROUTES_AVAILABLE["web"].router)
    routes_registered.append("web")
    logger.debug("✅ Web routes registered")

if ROUTES_AVAILABLE.get("models_info"):
    app.include_router(ROUTES_AVAILABLE["models_info"].router)
    routes_registered.append("models_info")
    logger.debug("✅ Models info routes registered")

logger.info("✅ FastAPI application initialized (v3.0.0)")
logger.info(
    f"   Routes registered: {', '.join(routes_registered) if routes_registered else 'NONE'}"
)


# ============================================================================
# MAIN ENTRY POINT
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
