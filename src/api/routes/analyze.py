"""Lyrics analysis endpoints with QWEN ML model integration.

Provides comprehensive rap lyrics analysis using state-of-the-art QWEN AI model
with Redis caching for optimal performance. These endpoints power the core
analysis features of the Content Intelligence Platform.

Endpoints:
    POST /analyze - Analyze individual lyrics with QWEN model
    GET /cache/stats - Redis cache performance metrics

The analysis pipeline includes:
- Sentiment analysis and emotional detection
- Theme extraction and content categorization
- Complexity scoring and readability metrics
- Quality prediction using ML models
- Trend analysis and temporal patterns

Features:
- Redis caching for repeated analysis requests
- Configurable model temperature for creativity control
- Comprehensive error handling and graceful degradation
- Type-safe Pydantic models with validation
- Performance monitoring and timing metrics

Example:
    POST /analyze analyzes lyrics and returns detailed analysis
    GET /cache/stats shows cache hit rates and performance

Author: ML Platform Team
Date: October 2025
Version: 3.0.0
"""

from datetime import datetime, timezone
from typing import Any

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field

# Graceful imports with fallbacks
try:
    from src.analyzers.qwen_analyzer import QwenAnalyzer

    qwen_available = True
except ImportError:
    qwen_available = False

try:
    from src.cache.redis_client import test_redis_connection

    redis_available = True
except ImportError:
    redis_available = False

try:
    from src.config import get_config

    config = get_config()
except Exception:
    config = None

router = APIRouter(tags=["Analysis"])
if config is None:
    # Fallback if config loading failed
    from src.config import get_config

    config = get_config()

# Global QWEN analyzer instance (initialized in main.py startup)
qwen_analyzer: "QwenAnalyzer | None" = None


# ============================================================================
# PYDANTIC MODELS
# ============================================================================


class AnalyzeRequest(BaseModel):
    """Request model for lyrics analysis with validation.

    This model validates and structures the input for lyrics analysis requests.
    Includes configurable parameters for cache usage and model temperature.

    Attributes:
        lyrics: The rap lyrics text to analyze (10-5000 characters).
            Should be clean text without excessive formatting.
        use_cache: Whether to use Redis cache for performance optimization.
            Significantly speeds up repeated analysis of same lyrics.
        temperature: Optional model temperature override (0.1-1.0).
            Lower values (0.1-0.3) = more conservative/focused analysis.
            Higher values (0.7-1.0) = more creative/diverse analysis.
    """

    lyrics: str = Field(
        ...,
        description="Rap lyrics text to analyze using QWEN model",
        min_length=10,
        max_length=5000,
        examples=["Yeah, I'm rising to the top, no time for games..."],
    )
    use_cache: bool = Field(
        default=True,
        description="Use Redis cache to speed up repeated analysis requests",
        examples=[True],
    )
    temperature: float | None = Field(
        default=None,
        description="Override default model temperature for analysis creativity (0.1=conservative, 1.0=creative)",
        ge=0.1,
        le=1.0,
        examples=[0.7],
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "lyrics": "Yeah, I'm rising to the top, no time for games, stacking paper while they play...",
                    "use_cache": True,
                    "temperature": 0.7,
                }
            ]
        }
    }


class AnalysisResult(BaseModel):
    """Response model for comprehensive analysis results.

    Contains the complete analysis output from QWEN model with metadata
    about caching, timing, and processing status.

    Attributes:
        success: Whether the analysis completed successfully without errors.
        analysis: Dictionary containing detailed analysis results from QWEN including:
            - sentiment: Overall emotional tone ("positive", "negative", "neutral")
            - themes: List of detected themes/topics in lyrics
            - complexity_score: Lyrical complexity rating (0-10)
            - quality_score: Overall quality prediction (0-1)
            - emotional_profile: Detailed emotional breakdown with confidence scores
        cached: Whether the result came from Redis cache (True) or fresh analysis (False).
        timestamp: ISO 8601 timestamp of when analysis was performed.
        processing_time_ms: Time taken for analysis in milliseconds (None if cached).
    """

    success: bool = Field(
        ...,
        description="Whether the lyrics analysis completed successfully",
        examples=[True],
    )
    analysis: dict[str, Any] = Field(
        ...,
        description="Detailed analysis results from QWEN model including sentiment, themes, complexity, and quality scores",
        examples=[
            {
                "sentiment": "positive",
                "themes": ["success", "ambition", "wealth"],
                "complexity_score": 8.5,
                "quality_score": 0.85,
                "emotional_profile": {"confidence": 0.9, "energy": 0.8},
            }
        ],
    )
    cached: bool = Field(
        ...,
        description="Whether this result was retrieved from Redis cache",
        examples=[False],
    )
    timestamp: str = Field(
        ...,
        description="ISO 8601 timestamp when analysis was completed",
        examples=["2025-10-30T10:30:00.000Z"],
    )
    processing_time_ms: int | None = Field(
        default=None,
        description="Time taken for analysis in milliseconds (None if cached)",
        examples=[245],
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "success": True,
                    "analysis": {
                        "sentiment": "positive",
                        "themes": ["success", "ambition", "wealth"],
                        "complexity_score": 8.5,
                        "quality_score": 0.85,
                        "emotional_profile": {"confidence": 0.9, "energy": 0.8},
                    },
                    "cached": False,
                    "timestamp": "2025-10-30T10:30:00.000Z",
                    "processing_time_ms": 245,
                }
            ]
        }
    }


class CacheStatsResponse(BaseModel):
    """Response model for cache statistics endpoint.

    Provides detailed metrics about Redis cache performance and configuration.

    Attributes:
        enabled: Whether Redis caching is currently enabled.
        stats: Optional dictionary with cache performance metrics (hits, misses, hit_rate).
        config: Optional dictionary with cache TTL configuration for different data types.
        message: Optional message when cache is disabled.
        error: Optional error message if stats retrieval failed.
    """

    enabled: bool = Field(
        ...,
        description="Whether Redis caching is enabled in configuration",
        examples=[True],
    )
    stats: dict[str, Any] | None = Field(
        default=None,
        description="Cache performance metrics including hits, misses, and hit rate",
        examples=[{"hits": 42, "misses": 8, "hit_rate": 0.84}],
    )
    config: dict[str, str] | None = Field(
        default=None,
        description="Cache TTL configuration for different data types in seconds",
        examples=[
            {
                "artist_ttl": "3600s",
                "lyrics_ttl": "7200s",
                "analysis_ttl": "86400s",
            }
        ],
    )
    message: str | None = Field(
        default=None,
        description="Message when cache is disabled or unavailable",
        examples=["Redis caching is disabled in configuration"],
    )
    error: str | None = Field(
        default=None,
        description="Error message if stats retrieval failed",
        examples=["Could not retrieve cache stats: Connection timeout"],
    )


async def initialize_analyzer() -> bool:
    """Initialize QWEN analyzer at application startup.

    Loads and configures the QWEN model for lyrics analysis. This function
    is called from main.py during the application lifespan startup phase.

    Returns:
        bool: True if initialization successful, False otherwise.

    Note:
        Initialization typically takes 3-5 seconds as the model loads into memory.
        Failure to initialize will result in 503 errors on /analyze endpoint.
    """
    global qwen_analyzer
    try:
        if qwen_available:
            from src.analyzers.qwen_analyzer import QwenAnalyzer

            qwen_analyzer = QwenAnalyzer()
            print("✅ QWEN analyzer initialized")
            return True
        print("❌ QWEN analyzer not available")
        return False
    except Exception as e:
        print(f"❌ Failed to initialize QWEN: {e}")
        return False


# ============================================================================
# ENDPOINTS
# ============================================================================


@router.post(
    "/analyze",
    response_model=AnalysisResult,
    summary="Analyze rap lyrics using QWEN AI model",
    status_code=status.HTTP_200_OK,
    responses={
        200: {
            "description": "Lyrics analysis completed successfully. Returns detailed analysis with sentiment, themes, and quality metrics.",
            "model": AnalysisResult,
        },
        422: {
            "description": "Validation error - lyrics too short/long or invalid temperature parameter",
        },
        500: {
            "description": "Analysis failed due to internal error or QWEN model processing issue",
        },
        503: {
            "description": "Service unavailable - QWEN analyzer not initialized or temporarily offline",
        },
    },
)
async def analyze_lyrics(request: AnalyzeRequest) -> AnalysisResult:
    """Analyze rap lyrics using QWEN AI model with comprehensive analysis.

    Performs detailed analysis of rap lyrics using the QWEN AI model, providing
    sentiment analysis, theme detection, complexity scoring, quality prediction,
    and emotional profiling. Supports Redis caching for performance optimization
    and configurable model temperature for analysis creativity control.

    This endpoint powers the core analysis features of the Content Intelligence
    Platform, enabling deep understanding of lyrical content for research,
    content moderation, and creative analysis applications.

    Args:
        request: AnalyzeRequest containing:
            - lyrics: Rap lyrics text (10-5000 characters)
            - use_cache: Whether to use Redis cache (default: True)
            - temperature: Optional model temperature override (0.1-1.0)

    Returns:
        AnalysisResult: Comprehensive analysis results with:
            - success: Boolean indicating successful analysis
            - analysis: Dictionary with detailed QWEN analysis results
            - cached: Whether result came from Redis cache
            - timestamp: ISO 8601 timestamp of analysis completion
            - processing_time_ms: Analysis duration in milliseconds

    Raises:
        HTTPException 422: Invalid input parameters (lyrics length, temperature range)
        HTTPException 500: Analysis failed due to model error or processing issue
        HTTPException 503: QWEN analyzer not available (service initializing)

    Example:
        >>> request = AnalyzeRequest(
        ...     lyrics="Yeah, I'm rising to the top, no time for games...",
        ...     use_cache=True,
        ...     temperature=0.7
        ... )
        >>> result = await analyze_lyrics(request)
        >>> print(result.success)
        True
        >>> print(result.analysis['sentiment'])
        'positive'
        >>> print(result.processing_time_ms)
        245

    Note:
        - Analysis typically takes 200-500ms for fresh requests
        - Cached results return in <10ms
        - Model temperature affects analysis creativity (lower = more conservative)
        - Redis caching significantly improves performance for repeated lyrics
        - First request after server restart may take longer (model loading)
    """
    if not qwen_analyzer:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="QWEN analyzer not available - service initializing",
        )

    try:
        start_time = datetime.now(timezone.utc)

        # Call QWEN analyzer
        result = qwen_analyzer.analyze_lyrics(
            lyrics=request.lyrics,
            temperature=request.temperature,
            use_cache=request.use_cache,
        )

        if "error" in result:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Analysis failed: {result['error']}",
            )

        processing_time = int(
            (datetime.now(timezone.utc) - start_time).total_seconds() * 1000
        )

        return AnalysisResult(
            success=True,
            analysis=result,
            cached=request.use_cache and "timestamp" in result,
            timestamp=datetime.now(timezone.utc).isoformat(),
            processing_time_ms=processing_time,
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Analysis failed: {e!r}",
        ) from e


@router.get(
    "/cache/stats",
    response_model=CacheStatsResponse,
    summary="Get Redis cache performance statistics",
    status_code=status.HTTP_200_OK,
    responses={
        200: {
            "description": "Cache statistics retrieved successfully. Returns performance metrics and configuration.",
            "model": CacheStatsResponse,
        },
    },
)
async def cache_stats() -> CacheStatsResponse:
    """Get Redis cache performance statistics and configuration.

    Returns detailed metrics about Redis cache performance including hit/miss rates,
    cache configuration (TTL settings), and availability status. Useful for:
    - Monitoring cache effectiveness
    - Debugging performance issues
    - Capacity planning and optimization
    - Understanding cache behavior in production

    Returns:
        CacheStatsResponse: Cache statistics with:
            - enabled: Whether Redis caching is enabled
            - stats: Performance metrics (hits, misses, hit_rate)
            - config: TTL configuration for different cache types
            - message: Status message if cache disabled
            - error: Error message if stats retrieval failed

    Example:
        >>> stats = await cache_stats()
        >>> print(stats.enabled)
        True
        >>> print(stats.stats['hit_rate'])
        0.84
        >>> print(stats.config['analysis_ttl'])
        '86400s'

    Note:
        - High hit rate (>70%) indicates effective caching
        - Low hit rate may indicate cache TTL too short or diverse queries
        - Stats are cumulative since last Redis restart
    """
    if not config or not config.redis.enabled:
        return CacheStatsResponse(
            enabled=False,
            message="Redis caching is disabled in configuration",
        )

    try:
        from src.cache.redis_client import redis_cache

        stats = redis_cache.get_stats()
        return CacheStatsResponse(
            enabled=True,
            stats=stats,
            config={
                "artist_ttl": f"{config.redis.cache.artist_ttl}s",
                "lyrics_ttl": f"{config.redis.cache.lyrics_ttl}s",
                "analysis_ttl": f"{config.redis.cache.analysis_ttl}s",
                "embedding_ttl": f"{config.redis.cache.embedding_ttl}s",
            },
        )
    except Exception as e:
        return CacheStatsResponse(
            enabled=True,
            error=f"Could not retrieve cache stats: {e!r}",
        )
