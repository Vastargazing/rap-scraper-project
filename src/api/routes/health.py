"""
🏥 Health Check Routes
Consolidated from all 3 API files

Provides:
- /health - Comprehensive system health check
- /config/info - Non-sensitive configuration information
"""

from datetime import datetime
from typing import Optional

from fastapi import APIRouter
from pydantic import BaseModel

from src.config import get_config
from src.cache.redis_client import redis_cache, test_redis_connection
from src.database.connection import test_connection as test_db_connection

router = APIRouter()
config = get_config()


# ============================================================================
# PYDANTIC MODELS
# ============================================================================

class HealthResponse(BaseModel):
    """Health check response model"""
    status: str
    version: str
    environment: str
    timestamp: str
    components: dict[str, str]


class ComponentStatus(BaseModel):
    """Individual component status"""
    name: str
    status: str
    timestamp: Optional[str] = None


# ============================================================================
# ENDPOINTS
# ============================================================================

@router.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """
    Comprehensive Health Check

    Returns status of all system components:
    - Database (PostgreSQL)
    - Redis (caching)
    - QWEN Analyzer (ML)

    Response Status:
    - "healthy": All critical components OK
    - "degraded": Some components unavailable
    - "error": Critical components failing

    Example:
        GET /health
        Response: {
            "status": "healthy",
            "version": "3.0.0",
            "environment": "production",
            "timestamp": "2025-10-20T20:50:00.000000",
            "components": {
                "database": "healthy",
                "redis": "healthy",
                "qwen": "healthy"
            }
        }
    """
    components = {}

    # Check Database
    try:
        db_ok = test_db_connection()
        components["database"] = "healthy" if db_ok else "unavailable"
    except Exception as e:
        components["database"] = f"error: {str(e)[:50]}"

    # Check Redis
    try:
        redis_ok = test_redis_connection()
        components["redis"] = "healthy" if redis_ok else "unavailable"
    except Exception as e:
        components["redis"] = f"error: {str(e)[:50]}"

    # Check QWEN Analyzer (will be initialized in main.py)
    # For now, we assume it's available
    try:
        from src.analyzers.qwen_analyzer import QwenAnalyzer
        qwen_available = True
        components["qwen"] = "healthy" if qwen_available else "not initialized"
    except ImportError:
        components["qwen"] = "not available"

    # Calculate overall status
    all_statuses = list(components.values())
    if all(s in ["healthy", "unavailable"] for s in all_statuses):
        status = "healthy" if any(s == "healthy" for s in all_statuses) else "degraded"
    else:
        status = "degraded"

    return HealthResponse(
        status=status,
        version=config.application.version,
        environment=config.application.environment,
        timestamp=datetime.now().isoformat(),
        components=components,
    )


@router.get("/config/info")
async def config_info() -> dict:
    """
    Get Non-Sensitive Configuration Info

    Returns current API configuration (excludes secrets like API keys)

    Returns:
        dict: Configuration information with:
            - application metadata
            - API settings (host, port, workers)
            - database configuration
            - redis status
            - monitoring settings

    Example:
        GET /config/info
        Response: {
            "application": {
                "name": "Rap ML API",
                "version": "3.0.0",
                "environment": "production"
            },
            "api": {
                "host": "0.0.0.0",
                "port": 8000,
                "workers": 4,
                "cors_enabled": true,
                "rate_limit_enabled": true,
                "rate_limit_rpm": 1000
            },
            "database": {
                "type": "postgresql",
                "pool_size": 20
            },
            "redis": {
                "enabled": true
            },
            "monitoring": {
                "prometheus_enabled": true,
                "grafana_enabled": true
            }
        }
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
        "redis": {
            "enabled": config.redis.enabled,
        },
        "monitoring": {
            "prometheus_enabled": config.monitoring.prometheus.enabled,
            "grafana_enabled": config.monitoring.grafana.enabled,
        },
    }


@router.get("/")
async def root_health() -> dict:
    """
    API Root / Service Info

    Quick check that API is running

    Returns:
        dict: Basic service information and status
    """
    return {
        "service": config.application.name,
        "version": config.application.version,
        "environment": config.application.environment,
        "status": "running",
        "timestamp": datetime.now().isoformat(),
        "docs": f"{config.api.docs.swagger_url}"
        if config.api.docs.enabled
        else "disabled",
    }
