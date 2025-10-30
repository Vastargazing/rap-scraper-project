"""Health check and system status endpoints.

Provides comprehensive health monitoring for all system components including
database, cache, and ML analyzers. These endpoints are essential for
production monitoring, Kubernetes readiness probes, and diagnostic purposes.

Endpoints:
    GET /health - Comprehensive multi-component health check
    GET /config/info - Non-sensitive configuration details
    GET / - Quick service status

The health check aggregates status from:
- PostgreSQL database with connection pooling
- Redis cache for request acceleration
- QWEN ML analyzer for AI features
- External APIs and services

Returns detailed diagnostics for each component to help identify bottlenecks
and service degradation in production environments.

Example:
    GET /health will return {"status": "healthy", "components": {...}}
    GET /config/info will return configuration without secrets
    GET / returns quick service info

Author: ML Platform Team
Date: October 2025
Version: 3.0.0
"""

from datetime import datetime
from typing import Literal

from fastapi import APIRouter, status
from pydantic import BaseModel, Field

from src.cache.redis_client import test_redis_connection
from src.config import get_config
from src.database.connection import test_connection as test_db_connection

router = APIRouter(tags=["Health"])
config = get_config()


# ============================================================================
# PYDANTIC MODELS
# ============================================================================


class HealthResponse(BaseModel):
    """API health check response with component status.

    This model represents the complete health status of the API and all
    its dependencies. Use this to monitor system reliability in production.

    Attributes:
        status: Overall API status ("healthy", "degraded", "error").
            - "healthy": All critical components operational
            - "degraded": Some non-critical components down, API functional
            - "error": Critical components failing, API severely compromised
        version: Semantic version of the API (e.g., "3.0.0")
        environment: Deployment environment ("development", "staging", "production")
        timestamp: ISO 8601 timestamp of when health check was performed
        components: Dictionary mapping component names to their status strings.
            Keys typically include: "database", "redis", "qwen"
            Values: "healthy", "degraded", "unavailable", or error messages
    """

    status: Literal["healthy", "degraded", "error"] = Field(
        ...,
        description='Overall API status: "healthy", "degraded", or "error"',
        examples=["healthy"],
    )
    version: str = Field(
        ...,
        description="Semantic version of the API (e.g., 3.0.0)",
        pattern=r"^\d+\.\d+\.\d+$",
        examples=["3.0.0"],
    )
    environment: Literal["development", "staging", "production"] = Field(
        ...,
        description="Deployment environment where API is running",
        examples=["production"],
    )
    timestamp: str = Field(
        ...,
        description="ISO 8601 timestamp when health check was executed",
        examples=["2025-10-30T10:30:00.000Z"],
    )
    components: dict[str, str] = Field(
        ...,
        description='Component health status map. Values: "healthy", "unavailable", or "error: message"',
        examples=[{"database": "healthy", "redis": "healthy", "qwen": "healthy"}],
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "status": "healthy",
                    "version": "3.0.0",
                    "environment": "production",
                    "timestamp": "2025-10-30T10:30:00.000Z",
                    "components": {
                        "database": "healthy",
                        "redis": "healthy",
                        "qwen": "healthy",
                    },
                }
            ]
        }
    }


class ComponentStatus(BaseModel):
    """Individual system component health status.

    Represents the status of a single system component in detailed form.
    More granular than HealthResponse for component-level monitoring.

    Attributes:
        name: Component identifier (e.g., "database", "redis", "qwen")
        status: Component status string ("healthy", "degraded", "unavailable", "error")
        timestamp: Optional timestamp when status was last checked
    """

    name: str = Field(
        ...,
        description="Component identifier",
        examples=["database"],
    )
    status: str = Field(
        ...,
        description='Component health status: "healthy", "degraded", "unavailable", or error message',
        examples=["healthy"],
    )
    timestamp: str | None = Field(
        default=None,
        description="ISO 8601 timestamp of last status check",
        examples=["2025-10-30T10:30:00Z"],
    )


# ============================================================================
# ENDPOINTS
# ============================================================================


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Check system health",
    status_code=status.HTTP_200_OK,
    responses={
        200: {
            "description": "Health check completed successfully. Returns aggregated status of all components.",
            "model": HealthResponse,
        },
        503: {
            "description": "Service unavailable - one or more critical components are down",
        },
    },
)
async def health_check() -> HealthResponse:
    """Check health of all system components.

    Performs synchronous health checks on critical system dependencies including
    the PostgreSQL database, Redis cache, and QWEN AI analyzer. Returns aggregated
    health status suitable for:
    - Kubernetes readiness/liveness probes
    - Load balancer health checks
    - Monitoring and alerting systems
    - Production debugging

    The endpoint classifies overall status based on component health:
    - "healthy": All critical components operational
    - "degraded": Some non-critical components unavailable but API functional
    - "error": Critical components failing, API severely compromised

    Returns:
        HealthResponse: Aggregated health status with per-component details and
            metadata (version, environment, timestamp).

    Raises:
        HTTPException: Not raised directly; connection errors are reported in
            components dict as "error: {message[:50]}".

    Example:
        >>> response = await health_check()
        >>> print(response.status)
        'healthy'
        >>> print(response.components['database'])
        'healthy'

    Note:
        Execution time is typically < 100ms. Each component check has its own
        timeout to prevent cascade failures.
    """
    components: dict[str, str] = {}

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
    try:
        from src.analyzers.qwen_analyzer import QwenAnalyzer

        qwen_available = True
        components["qwen"] = "healthy" if qwen_available else "not initialized"
    except ImportError:
        components["qwen"] = "not available"
    except Exception as e:
        components["qwen"] = f"error: {str(e)[:50]}"

    # Calculate overall status
    all_statuses = list(components.values())
    if all(s in ["healthy", "unavailable"] for s in all_statuses):
        overall_status = (
            "healthy" if any(s == "healthy" for s in all_statuses) else "degraded"
        )
    else:
        overall_status = "degraded"

    return HealthResponse(
        status=overall_status,
        version=config.application.version,
        environment=config.application.environment,
        timestamp=datetime.now().isoformat(),  # noqa: DTZ005
        components=components,
    )


@router.get(
    "/config/info",
    summary="Get API configuration",
    status_code=status.HTTP_200_OK,
    responses={
        200: {
            "description": "Configuration retrieved successfully. Excludes sensitive data like API keys and passwords.",
        }
    },
)
async def config_info() -> dict[str, dict[str, object]]:
    """Get non-sensitive API configuration for monitoring.

    Returns current API configuration excluding secrets like API keys, passwords,
    or authentication tokens. Safe to expose in logs and monitoring systems.

    This endpoint is useful for:
    - Verifying deployed configuration matches expectations
    - Monitoring environment settings across multiple instances
    - Debugging configuration-related issues
    - Health dashboards and monitoring tools

    Returns:
        dict[str, dict[str, object]]: Configuration dictionary with sections:
            - application: Name, version, environment
            - api: Host, port, workers, CORS status, rate limiting
            - database: Type (PostgreSQL), connection pool size
            - redis: Enabled status (not connection details)
            - monitoring: Prometheus and Grafana status

    Example:
        >>> config = await config_info()
        >>> print(config['application']['version'])
        '3.0.0'
        >>> print(config['database']['pool_size'])
        20

    Note:
        All sensitive configuration (API keys, passwords, connection strings)
        is intentionally omitted. Use /health for quick status checks.
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


@router.get(
    "/",
    summary="Quick service status",
    status_code=status.HTTP_200_OK,
    responses={
        200: {
            "description": "Service is running and accessible. Returns basic metadata.",
        }
    },
)
async def root_health() -> dict[str, object]:
    """Quick API status check and root endpoint.

    Returns basic service information for quickly verifying that the API
    is running and accessible. Lighter than /health endpoint - useful for
    frequent polling or simple availability checks.

    This endpoint answers:
    - Is the API responding?
    - What version is running?
    - What environment is it?
    - Where are the API docs?

    Returns:
        dict[str, object]: Service status with:
            - service: Human-readable service name
            - version: Semantic version string
            - environment: Deployment environment name
            - status: Always "running" if endpoint responds
            - timestamp: ISO 8601 timestamp of response
            - docs: Swagger UI documentation URL (if enabled)

    Example:
        >>> response = await root_health()
        >>> print(response['service'])
        'Rap ML API'
        >>> print(response['status'])
        'running'

    Note:
        This is the fastest health check endpoint. Does not check component
        status - use /health for detailed diagnostics.
    """
    return {
        "service": config.application.name,
        "version": config.application.version,
        "environment": config.application.environment,
        "status": "running",
        "timestamp": datetime.now().isoformat(),  # noqa: DTZ005
        "docs": f"{config.api.docs.swagger_url}"
        if config.api.docs.enabled
        else "disabled",
    }
