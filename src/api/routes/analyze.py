"""
🎤 Lyrics Analysis Routes
QWEN ML-powered analysis with Redis caching

Provides:
- /analyze - Analyze rap lyrics using QWEN model
- Integrated Redis caching for performance
- Type-safe Pydantic models
"""

from datetime import datetime
from typing import Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

# Graceful imports with fallbacks
try:
    from src.analyzers.qwen_analyzer import QwenAnalyzer
    QWEN_AVAILABLE = True
except ImportError:
    QwenAnalyzer = None
    QWEN_AVAILABLE = False

try:
    from src.cache.redis_client import test_redis_connection
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False

try:
    from src.config import get_config
    config = get_config()
except Exception:
    config = None

router = APIRouter()
config = get_config()

# Global QWEN analyzer instance (initialized in main.py startup)
qwen_analyzer: Optional[QwenAnalyzer] = None


# ============================================================================
# PYDANTIC MODELS
# ============================================================================

class AnalyzeRequest(BaseModel):
    """Request model for lyrics analysis"""
    lyrics: str = Field(
        ...,
        description="Rap lyrics text to analyze",
        min_length=10,
        max_length=5000,
    )
    use_cache: bool = Field(
        True,
        description="Use Redis cache if available",
    )
    temperature: float | None = Field(
        None,
        description="Override model temperature (0.1-1.0)",
        ge=0.1,
        le=1.0,
    )


class AnalysisResult(BaseModel):
    """Response model for analysis results"""
    success: bool
    analysis: dict
    cached: bool
    timestamp: str
    processing_time_ms: int | None = None


# ============================================================================
# INITIALIZATION (Called from main.py)
# ============================================================================

async def initialize_analyzer():
    """Initialize QWEN analyzer (called at app startup)"""
    global qwen_analyzer
    try:
        qwen_analyzer = QwenAnalyzer()
        print("✅ QWEN analyzer initialized")
        return True
    except Exception as e:
        print(f"❌ Failed to initialize QWEN: {e}")
        return False


# ============================================================================
# ENDPOINTS
# ============================================================================

@router.post("/analyze", response_model=AnalysisResult)
async def analyze_lyrics(request: AnalyzeRequest) -> AnalysisResult:
    """
    Analyze Rap Lyrics Using QWEN Model

    Provides detailed analysis including:
    - Sentiment analysis
    - Theme detection
    - Complexity scoring
    - Quality prediction
    - Trend analysis

    Features:
    - Redis caching for performance
    - Type-safe configuration
    - Comprehensive error handling

    Args:
        request: AnalyzeRequest with lyrics and optional parameters

    Returns:
        AnalysisResult with analysis data and metadata

    Raises:
        HTTPException 503: QWEN analyzer not available
        HTTPException 422: Invalid input (too short/long)
        HTTPException 500: Analysis failed

    Example:
        POST /analyze
        {
            "lyrics": "Yeah, I'm rising to the top...",
            "use_cache": true,
            "temperature": 0.7
        }

        Response: {
            "success": true,
            "analysis": {
                "sentiment": "positive",
                "themes": ["success", "ambition"],
                "complexity_score": 8.5,
                "quality_score": 0.85
            },
            "cached": false,
            "timestamp": "2025-10-20T20:50:00.000000",
            "processing_time_ms": 245
        }
    """
    if not qwen_analyzer:
        raise HTTPException(
            status_code=503,
            detail="QWEN analyzer not available - service initializing",
        )

    try:
        start_time = datetime.now()

        # Call QWEN analyzer
        result = qwen_analyzer.analyze_lyrics(
            lyrics=request.lyrics,
            temperature=request.temperature,
            use_cache=request.use_cache,
        )

        if "error" in result:
            raise HTTPException(
                status_code=500,
                detail=f"Analysis failed: {result['error']}",
            )

        processing_time = int((datetime.now() - start_time).total_seconds() * 1000)

        return AnalysisResult(
            success=True,
            analysis=result,
            cached=request.use_cache and "timestamp" in result,
            timestamp=datetime.now().isoformat(),
            processing_time_ms=processing_time,
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Analysis failed: {str(e)}",
        )


@router.get("/cache/stats")
async def cache_stats() -> dict:
    """
    Get Redis Cache Statistics

    Returns cache performance metrics and configuration

    Returns:
        dict: Cache statistics including:
            - enabled: whether caching is enabled
            - stats: cache hit/miss metrics
            - config: TTL settings

    Example:
        GET /cache/stats
        Response: {
            "enabled": true,
            "stats": {
                "hits": 42,
                "misses": 8,
                "hit_rate": 0.84
            },
            "config": {
                "artist_ttl": "3600s",
                "lyrics_ttl": "7200s",
                "analysis_ttl": "86400s"
            }
        }
    """
    if not config.redis.enabled:
        return {
            "enabled": False,
            "message": "Redis caching is disabled in configuration",
        }

    try:
        from src.cache.redis_client import redis_cache

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
    except Exception as e:
        return {
            "enabled": True,
            "error": f"Could not retrieve cache stats: {str(e)}",
        }
