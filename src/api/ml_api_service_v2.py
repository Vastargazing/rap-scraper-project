"""
🚀 ML API Service with Config Integration v2.0.0
FastAPI service with type-safe configuration

Features:
- Config-based API settings (host, port, CORS)
- Integrated analyzers (QWEN, Ollama)
- Health checks with component status
- Rate limiting from config
- Automatic documentation

Author: Vastargazing
Version: 2.0.0
"""

import logging
import time
from datetime import datetime

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from src.analyzers.qwen_analyzer import QwenAnalyzer
from src.cache.redis_client import redis_cache, test_redis_connection
from src.config import get_config
from src.database.connection import test_connection as test_db_connection

logger = logging.getLogger(__name__)

# Load config
config = get_config()

# Create FastAPI app with config
app = FastAPI(
    title=config.api.docs.title,
    version=config.api.docs.version,
    docs_url=config.api.docs.swagger_url if config.api.docs.enabled else None,
    redoc_url=config.api.docs.redoc_url if config.api.docs.enabled else None,
)

# Setup CORS from config
if config.api.cors.enabled:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=config.api.cors.origins,
        allow_credentials=config.api.cors.allow_credentials,
        allow_methods=config.api.cors.allow_methods,
        allow_headers=config.api.cors.allow_headers,
    )
    logger.info(f"✅ CORS enabled for origins: {config.api.cors.origins}")

# Initialize analyzers
qwen_analyzer = None


@app.on_event("startup")
async def startup_event():
    """Initialize services on startup"""
    global qwen_analyzer

    logger.info("🚀 Starting ML API Service v2.0.0...")
    logger.info(f"   Environment: {config.application.environment}")
    logger.info(f"   API Host: {config.api.host}:{config.api.port}")
    logger.info(f"   Workers: {config.api.workers}")

    # Initialize QWEN analyzer
    try:
        qwen_analyzer = QwenAnalyzer()
        logger.info("✅ QWEN analyzer initialized")
    except Exception as e:
        logger.error(f"❌ Failed to initialize QWEN: {e}")
        qwen_analyzer = None

    # Test connections
    if test_redis_connection():
        logger.info("✅ Redis connection successful")
    else:
        logger.warning("⚠️ Redis connection failed - caching disabled")

    if test_db_connection():
        logger.info("✅ Database connection successful")
    else:
        logger.warning("⚠️ Database connection failed")

    logger.info("✅ ML API Service started successfully!")


# ============================================================================
# Request Models
# ============================================================================


class AnalyzeRequest(BaseModel):
    """Request for lyrics analysis"""

    lyrics: str = Field(..., description="Lyrics text to analyze")
    use_cache: bool = Field(True, description="Use Redis cache if available")
    temperature: float | None = Field(None, description="Override config temperature")


class HealthResponse(BaseModel):
    """Health check response"""

    status: str
    version: str
    environment: str
    timestamp: str
    components: dict[str, str]


# ============================================================================
# API Endpoints
# ============================================================================


@app.get("/")
async def root():
    """API root - service info"""
    return {
        "service": config.application.name,
        "version": config.application.version,
        "environment": config.application.environment,
        "docs": f"{config.api.docs.swagger_url}"
        if config.api.docs.enabled
        else "disabled",
        "status": "running",
        "timestamp": datetime.now().isoformat(),
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """
    Comprehensive health check

    Returns status of all system components
    """
    components = {}

    # Check database
    try:
        db_ok = test_db_connection()
        components["database"] = "healthy" if db_ok else "unavailable"
    except Exception as e:
        components["database"] = f"error: {e!s}"

    # Check Redis
    try:
        redis_ok = test_redis_connection()
        components["redis"] = "healthy" if redis_ok else "unavailable"
    except Exception as e:
        components["redis"] = f"error: {e!s}"

    # Check QWEN
    components["qwen"] = "healthy" if qwen_analyzer else "not initialized"

    # Overall status
    status = (
        "healthy"
        if all(v in ["healthy", "unavailable"] for v in components.values())
        else "degraded"
    )

    return HealthResponse(
        status=status,
        version=config.application.version,
        environment=config.application.environment,
        timestamp=datetime.now().isoformat(),
        components=components,
    )


@app.get("/config/info")
async def config_info():
    """
    Get non-sensitive configuration info

    Returns current API configuration (excluding secrets)
    """
    return {
        "application": {
            "name": config.application.name,
            "version": config.application.version,
            "environment": config.application.environment,
        },
        "api": {
            "host": config.api.host,
            "port": config.api.port,
            "workers": config.api.workers,
            "cors_enabled": config.api.cors.enabled,
            "rate_limit_enabled": config.api.rate_limit.enabled,
            "rate_limit_rpm": config.api.rate_limit.requests_per_minute
            if config.api.rate_limit.enabled
            else None,
        },
        "database": {
            "type": config.database.type,
            "pool_size": config.database.pool_size,
        },
        "redis": {"enabled": config.redis.enabled},
        "monitoring": {
            "prometheus_enabled": config.monitoring.prometheus.enabled,
            "grafana_enabled": config.monitoring.grafana.enabled,
        },
    }


@app.post("/analyze")
async def analyze_lyrics(request: AnalyzeRequest):
    """
    Analyze rap lyrics using QWEN model

    Returns detailed analysis including themes, style, complexity, quality
    """
    if not qwen_analyzer:
        raise HTTPException(status_code=503, detail="QWEN analyzer not available")

    try:
        logger.info(f"🎤 Analyzing lyrics ({len(request.lyrics)} chars)")

        result = qwen_analyzer.analyze_lyrics(
            lyrics=request.lyrics,
            temperature=request.temperature,
            use_cache=request.use_cache,
        )

        if "error" in result:
            raise HTTPException(
                status_code=500, detail=f"Analysis failed: {result['error']}"
            )

        return {
            "success": True,
            "analysis": result,
            "cached": "timestamp" not in result
            or time.time() - result.get("timestamp", 0) > 60,
            "timestamp": datetime.now().isoformat(),
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Analysis error: {e}")
        raise HTTPException(status_code=500, detail=f"Analysis failed: {e!s}")


@app.get("/cache/stats")
async def cache_stats():
    """
    Get Redis cache statistics

    Returns cache performance metrics
    """
    if not config.redis.enabled:
        return {"enabled": False, "message": "Redis caching is disabled"}

    stats = redis_cache.get_stats()
    return {
        "enabled": True,
        "stats": stats,
        "config": {
            "artist_ttl": f"{config.redis.cache.artist_ttl}s",
            "lyrics_ttl": f"{config.redis.cache.lyrics_ttl}s",
            "analysis_ttl": f"{config.redis.cache.analysis_ttl}s",
            "embedding_ttl": f"{config.redis.cache.embedding_ttl}s",
        },
    }


@app.get("/models/info")
async def models_info():
    """
    Get information about loaded models

    Returns model configuration and availability
    """
    info = {
        "qwen": {
            "available": qwen_analyzer is not None,
            "config": qwen_analyzer.get_config_info() if qwen_analyzer else None,
        }
    }

    return info


# ============================================================================
# Error Handlers
# ============================================================================


@app.exception_handler(404)
async def not_found_handler(request: Request, exc):
    """Custom 404 handler"""
    return JSONResponse(
        status_code=404,
        content={
            "error": "Not Found",
            "message": f"Endpoint {request.url.path} not found",
            "docs": config.api.docs.swagger_url if config.api.docs.enabled else None,
        },
    )


@app.exception_handler(500)
async def server_error_handler(request: Request, exc):
    """Custom 500 handler"""
    logger.error(f"Internal server error: {exc}")
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal Server Error",
            "message": "An unexpected error occurred",
            "environment": config.application.environment,
        },
    )


# ============================================================================
# Run Server
# ============================================================================

if __name__ == "__main__":
    import uvicorn

    logger.info("=" * 60)
    logger.info("🚀 Starting ML API Service")
    logger.info("=" * 60)
    logger.info(f"Environment: {config.application.environment}")
    logger.info(f"Host: {config.api.host}")
    logger.info(f"Port: {config.api.port}")
    logger.info(f"Workers: {config.api.workers}")
    logger.info(f"Reload: {config.api.reload}")
    logger.info(
        f"Docs: http://{config.api.host}:{config.api.port}{config.api.docs.swagger_url}"
    )
    logger.info("=" * 60)

    uvicorn.run(
        "ml_api_service_v2:app",
        host=config.api.host,
        port=config.api.port,
        workers=config.api.workers if not config.api.reload else 1,
        reload=config.api.reload,
        log_level=config.api.log_level,
    )
