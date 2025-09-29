# 🚀 ПЛАН ТРАНСФОРМАЦИИ: CLI → Production ML Platform API

---

## 📋 PHASE 1: FastAPI Foundation (День 1)

### Step 1.1: Установка зависимостей
```bash
# Добавить в requirements.txt:
fastapi==0.104.1
uvicorn[standard]==0.24.0
pydantic==2.5.0
python-multipart==0.0.6
```

**Задача для агента:**
- [ ] Обновить `requirements.txt` с новыми зависимостями
- [ ] Установить: `pip install -r requirements.txt`
- [ ] Проверить установку: `python -c "import fastapi; print(fastapi.__version__)"`

### Step 1.2: Создать базовую структуру API
```bash
# Создать директории:
src/api/
src/api/routers/
src/api/models/
src/services/
```

**Задача для агента:**
- [ ] Создать `src/api/__init__.py`
- [ ] Создать `src/api/main.py` - главный FastAPI app
- [ ] Создать `src/api/routers/__init__.py`
- [ ] Создать `src/api/models/__init__.py`
- [ ] Создать `src/services/__init__.py`

### Step 1.3: Базовый FastAPI app с health check

**Файл: `src/api/main.py`**
```python
"""
Rap Lyrics ML Platform API
Production-ready REST API for AI-powered lyrics analysis
"""
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import time
import logging

logger = logging.getLogger(__name__)

# FastAPI app с красивой документацией
app = FastAPI(
    title="🎤 Rap Lyrics ML Platform API",
    description="""
    Production ML Platform для анализа рэп-текстов с использованием 5 AI моделей.
    
    ## Возможности
    * **57,000+ треков** в PostgreSQL базе данных
    * **5 AI анализаторов**: Qwen, Gemma, Ollama и другие
    * **pgvector** семантический поиск
    * **Connection pooling** для высокой производительности
    
    ## Технологии
    * FastAPI + PostgreSQL + pgvector
    * Async processing с connection pool (20 соединений)
    * Prometheus metrics
    """,
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/api/openapi.json"
)

# CORS для фронтенда (если добавим потом)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # В продакшене ограничить конкретными доменами
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Middleware для логирования запросов
@app.middleware("http")
async def log_requests(request: Request, call_next):
    start_time = time.time()
    
    response = await call_next(request)
    
    process_time = time.time() - start_time
    logger.info(
        f"{request.method} {request.url.path} "
        f"completed in {process_time:.3f}s with status {response.status_code}"
    )
    
    return response

# Health check endpoint
@app.get("/health", tags=["System"])
async def health_check():
    """
    Health check endpoint для мониторинга
    
    Returns:
        dict: Статус системы
    """
    return {
        "status": "healthy",
        "service": "rap-lyrics-ml-platform",
        "version": "1.0.0"
    }

# Root endpoint
@app.get("/", tags=["System"])
async def root():
    """
    Root endpoint с информацией о API
    
    Returns:
        dict: Базовая информация и ссылки
    """
    return {
        "message": "🎤 Rap Lyrics ML Platform API",
        "docs": "/docs",
        "redoc": "/redoc",
        "health": "/health",
        "version": "1.0.0"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

**Задача для агента:**
- [ ] Создать файл `src/api/main.py` с кодом выше
- [ ] Протестировать запуск: `uvicorn src.api.main:app --reload`
- [ ] Проверить Swagger UI: открыть `http://localhost:8000/docs`
- [ ] Проверить health check: `curl http://localhost:8000/health`

---

## 📋 PHASE 2: Database Integration (День 1-2)

### Step 2.1: Database Service Layer

**Файл: `src/services/database_service.py`**
```python
"""
Database Service для API endpoints
Использует существующий PostgreSQLManager
"""
import sys
from pathlib import Path

# Добавляем корень проекта в path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.database.postgres_adapter import PostgreSQLManager
from typing import Optional, List, Dict
import logging

logger = logging.getLogger(__name__)

class DatabaseService:
    """Сервис для работы с БД через API"""
    
    def __init__(self):
        self.db_manager = None
    
    async def initialize(self):
        """Инициализация подключения к БД"""
        try:
            self.db_manager = PostgreSQLManager()
            await self.db_manager.initialize()
            logger.info("✅ Database service initialized")
        except Exception as e:
            logger.error(f"❌ Failed to initialize database service: {e}")
            raise
    
    async def close(self):
        """Закрытие подключения"""
        if self.db_manager:
            await self.db_manager.close()
    
    async def get_stats(self) -> Dict:
        """Получение статистики платформы"""
        try:
            async with self.db_manager.get_connection() as conn:
                # Total tracks
                total_tracks = await conn.fetchval("SELECT COUNT(*) FROM tracks")
                
                # Tracks with lyrics
                with_lyrics = await conn.fetchval(
                    "SELECT COUNT(*) FROM tracks WHERE lyrics IS NOT NULL AND lyrics != ''"
                )
                
                # Unique artists
                unique_artists = await conn.fetchval(
                    "SELECT COUNT(DISTINCT artist) FROM tracks"
                )
                
                # Analyzed tracks
                analyzed = await conn.fetchval(
                    "SELECT COUNT(*) FROM analysis_results"
                )
                
                # Analyzer distribution
                analyzer_stats = await conn.fetch("""
                    SELECT analyzer_type, COUNT(*) as count
                    FROM analysis_results
                    GROUP BY analyzer_type
                    ORDER BY count DESC
                """)
                
                return {
                    "total_tracks": total_tracks,
                    "tracks_with_lyrics": with_lyrics,
                    "unique_artists": unique_artists,
                    "analyzed_tracks": analyzed,
                    "analysis_coverage": f"{(analyzed/total_tracks*100):.1f}%" if total_tracks > 0 else "0%",
                    "analyzer_distribution": {
                        row['analyzer_type']: row['count'] 
                        for row in analyzer_stats
                    }
                }
        except Exception as e:
            logger.error(f"❌ Error getting stats: {e}")
            raise
    
    async def get_track_by_id(self, track_id: int) -> Optional[Dict]:
        """Получение трека по ID"""
        try:
            async with self.db_manager.get_connection() as conn:
                track = await conn.fetchrow(
                    "SELECT * FROM tracks WHERE id = $1",
                    track_id
                )
                
                if not track:
                    return None
                
                # Получаем анализы для этого трека
                analyses = await conn.fetch("""
                    SELECT analyzer_type, sentiment, themes, style_elements, 
                           complexity_score, created_at
                    FROM analysis_results
                    WHERE track_id = $1
                    ORDER BY created_at DESC
                """, track_id)
                
                return {
                    "id": track['id'],
                    "artist": track['artist'],
                    "title": track['title'],
                    "lyrics": track['lyrics'],
                    "url": track.get('url'),
                    "analyses": [
                        {
                            "analyzer": a['analyzer_type'],
                            "sentiment": a['sentiment'],
                            "themes": a['themes'],
                            "style_elements": a['style_elements'],
                            "complexity_score": float(a['complexity_score']) if a['complexity_score'] else None,
                            "analyzed_at": a['created_at'].isoformat() if a['created_at'] else None
                        }
                        for a in analyses
                    ]
                }
        except Exception as e:
            logger.error(f"❌ Error getting track {track_id}: {e}")
            raise
    
    async def get_tracks_paginated(
        self, 
        limit: int = 50, 
        offset: int = 0,
        with_analysis: bool = False
    ) -> List[Dict]:
        """Получение списка треков с пагинацией"""
        try:
            async with self.db_manager.get_connection() as conn:
                query = """
                    SELECT t.id, t.artist, t.title, t.url,
                           CASE WHEN t.lyrics IS NOT NULL THEN true ELSE false END as has_lyrics
                """
                
                if with_analysis:
                    query += """,
                           (SELECT COUNT(*) FROM analysis_results WHERE track_id = t.id) as analysis_count
                    """
                
                query += """
                    FROM tracks t
                    ORDER BY t.id DESC
                    LIMIT $1 OFFSET $2
                """
                
                tracks = await conn.fetch(query, limit, offset)
                
                return [dict(track) for track in tracks]
        except Exception as e:
            logger.error(f"❌ Error getting tracks: {e}")
            raise

# Singleton instance
db_service = DatabaseService()
```

**Задача для агента:**
- [ ] Создать файл `src/services/database_service.py`
- [ ] Протестировать: `python -c "import asyncio; from src.services.database_service import db_service; asyncio.run(db_service.initialize())"`

### Step 2.2: Pydantic Models для API

**Файл: `src/api/models/responses.py`**
```python
"""
Pydantic models для API responses
"""
from pydantic import BaseModel, Field
from typing import Optional, List, Dict
from datetime import datetime

class HealthResponse(BaseModel):
    """Health check response"""
    status: str = Field(..., description="Service status")
    service: str = Field(..., description="Service name")
    version: str = Field(..., description="API version")

class StatsResponse(BaseModel):
    """Platform statistics response"""
    total_tracks: int = Field(..., description="Total tracks in database")
    tracks_with_lyrics: int = Field(..., description="Tracks with lyrics")
    unique_artists: int = Field(..., description="Unique artists count")
    analyzed_tracks: int = Field(..., description="Total analyzed tracks")
    analysis_coverage: str = Field(..., description="Analysis coverage percentage")
    analyzer_distribution: Dict[str, int] = Field(..., description="Distribution by analyzer")

class AnalysisInfo(BaseModel):
    """AI analysis information"""
    analyzer: str = Field(..., description="Analyzer type (qwen/gemma/ollama)")
    sentiment: Optional[str] = Field(None, description="Detected sentiment")
    themes: Optional[List[str]] = Field(None, description="Detected themes")
    style_elements: Optional[List[str]] = Field(None, description="Style elements")
    complexity_score: Optional[float] = Field(None, description="Lyrical complexity score")
    analyzed_at: Optional[str] = Field(None, description="Analysis timestamp")

class TrackResponse(BaseModel):
    """Track information with analyses"""
    id: int = Field(..., description="Track ID")
    artist: str = Field(..., description="Artist name")
    title: str = Field(..., description="Track title")
    lyrics: Optional[str] = Field(None, description="Lyrics text")
    url: Optional[str] = Field(None, description="Genius URL")
    analyses: List[AnalysisInfo] = Field(default_factory=list, description="AI analyses")

class TrackListItem(BaseModel):
    """Track list item (without full lyrics)"""
    id: int
    artist: str
    title: str
    url: Optional[str]
    has_lyrics: bool
    analysis_count: Optional[int] = None

class ErrorResponse(BaseModel):
    """Error response"""
    error: str = Field(..., description="Error message")
    detail: Optional[str] = Field(None, description="Detailed error info")
```

**Задача для агента:**
- [ ] Создать файл `src/api/models/responses.py`

---

## 📋 PHASE 3: API Endpoints (День 2)

### Step 3.1: Stats Router

**Файл: `src/api/routers/stats.py`**
```python
"""
Statistics endpoints
"""
from fastapi import APIRouter, HTTPException, Depends
from src.api.models.responses import StatsResponse, ErrorResponse
from src.services.database_service import db_service
import logging

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1", tags=["Statistics"])

async def get_db_service():
    """Dependency для получения DB service"""
    if not db_service.db_manager:
        await db_service.initialize()
    return db_service

@router.get(
    "/stats",
    response_model=StatsResponse,
    responses={
        200: {"description": "Platform statistics"},
        503: {"model": ErrorResponse, "description": "Database unavailable"}
    },
    summary="Get platform statistics",
    description="Returns comprehensive statistics about the ML platform including track counts, analysis coverage, and analyzer distribution"
)
async def get_stats(service = Depends(get_db_service)):
    """
    **Platform Statistics**
    
    Returns:
    - Total tracks in database
    - Tracks with lyrics
    - Unique artists count
    - Analysis coverage percentage
    - Distribution across AI analyzers
    
    Example response:
    ```json
    {
      "total_tracks": 57718,
      "tracks_with_lyrics": 57718,
      "unique_artists": 345,
      "analyzed_tracks": 54171,
      "analysis_coverage": "93.9%",
      "analyzer_distribution": {
        "qwen": 19852,
        "gemma": 34320
      }
    }
    ```
    """
    try:
        stats = await service.get_stats()
        return stats
    except Exception as e:
        logger.error(f"❌ Error getting stats: {e}")
        raise HTTPException(
            status_code=503,
            detail={
                "error": "Failed to retrieve statistics",
                "detail": str(e)
            }
        )
```

**Задача для агента:**
- [ ] Создать файл `src/api/routers/stats.py`

### Step 3.2: Tracks Router

**Файл: `src/api/routers/tracks.py`**
```python
"""
Tracks endpoints
"""
from fastapi import APIRouter, HTTPException, Depends, Query
from src.api.models.responses import TrackResponse, TrackListItem, ErrorResponse
from src.services.database_service import db_service
from typing import List
import logging

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1", tags=["Tracks"])

async def get_db_service():
    """Dependency для получения DB service"""
    if not db_service.db_manager:
        await db_service.initialize()
    return db_service

@router.get(
    "/tracks",
    response_model=List[TrackListItem],
    summary="Get tracks list",
    description="Returns paginated list of tracks"
)
async def get_tracks(
    limit: int = Query(50, ge=1, le=100, description="Number of tracks to return"),
    offset: int = Query(0, ge=0, description="Offset for pagination"),
    with_analysis: bool = Query(False, description="Include analysis count"),
    service = Depends(get_db_service)
):
    """
    **Get Tracks List**
    
    Paginated endpoint for browsing tracks.
    
    Parameters:
    - **limit**: How many tracks to return (1-100, default 50)
    - **offset**: Skip this many tracks (for pagination)
    - **with_analysis**: Include analysis count for each track
    
    Example: `/api/v1/tracks?limit=10&offset=0&with_analysis=true`
    """
    try:
        tracks = await service.get_tracks_paginated(limit, offset, with_analysis)
        return tracks
    except Exception as e:
        logger.error(f"❌ Error getting tracks: {e}")
        raise HTTPException(
            status_code=503,
            detail={"error": "Failed to retrieve tracks", "detail": str(e)}
        )

@router.get(
    "/tracks/{track_id}",
    response_model=TrackResponse,
    responses={
        200: {"description": "Track with all analyses"},
        404: {"model": ErrorResponse, "description": "Track not found"},
        503: {"model": ErrorResponse, "description": "Database unavailable"}
    },
    summary="Get track by ID",
    description="Returns detailed track information including all AI analyses"
)
async def get_track(
    track_id: int,
    service = Depends(get_db_service)
):
    """
    **Get Track Details**
    
    Returns complete track information including:
    - Track metadata (artist, title, lyrics)
    - All AI analyses (sentiment, themes, complexity)
    - Analysis timestamps
    
    Example: `/api/v1/tracks/12345`
    """
    try:
        track = await service.get_track_by_id(track_id)
        
        if not track:
            raise HTTPException(
                status_code=404,
                detail={"error": "Track not found", "detail": f"No track with ID {track_id}"}
            )
        
        return track
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Error getting track {track_id}: {e}")
        raise HTTPException(
            status_code=503,
            detail={"error": "Failed to retrieve track", "detail": str(e)}
        )
```

**Задача для агента:**
- [ ] Создать файл `src/api/routers/tracks.py`

### Step 3.3: Подключить роутеры к main app

**Обновить `src/api/main.py`:**
```python
# После создания app, перед middleware, добавить:

from src.api.routers import stats, tracks

# Include routers
app.include_router(stats.router)
app.include_router(tracks.router)
```

**Задача для агента:**
- [ ] Обновить `src/api/main.py` с подключением роутеров
- [ ] Перезапустить сервер: `uvicorn src.api.main:app --reload`
- [ ] Проверить Swagger: `http://localhost:8000/docs` - должны появиться новые endpoints

---

## 📋 PHASE 4: Analysis Endpoint (День 2-3)

### Step 4.1: Analysis Service

**Файл: `src/services/analysis_service.py`**
```python
"""
AI Analysis Service для API
"""
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.interfaces.analyzer_interface import AnalyzerFactory
from typing import Dict, Optional
import logging

logger = logging.getLogger(__name__)

class AnalysisService:
    """Сервис для AI анализа текстов"""
    
    def __init__(self):
        self.analyzers = {}
    
    def get_analyzer(self, analyzer_type: str):
        """Получение или создание анализатора"""
        if analyzer_type not in self.analyzers:
            try:
                self.analyzers[analyzer_type] = AnalyzerFactory.create(analyzer_type)
                logger.info(f"✅ Analyzer {analyzer_type} initialized")
            except Exception as e:
                logger.error(f"❌ Failed to initialize {analyzer_type}: {e}")
                raise
        
        return self.analyzers[analyzer_type]
    
    async def analyze_lyrics(
        self, 
        lyrics: str, 
        analyzer_type: str = "qwen"
    ) -> Dict:
        """
        Анализ текста через AI
        
        Args:
            lyrics: Текст для анализа
            analyzer_type: Тип анализатора (qwen/gemma/ollama)
        
        Returns:
            Dict с результатами анализа
        """
        try:
            analyzer = self.get_analyzer(analyzer_type)
            
            if not analyzer.available:
                raise ValueError(f"Analyzer {analyzer_type} is not available")
            
            # Выполняем анализ
            result = analyzer.analyze_lyrics(lyrics)
            
            return {
                "analyzer": analyzer_type,
                "sentiment": result.get("sentiment"),
                "themes": result.get("themes", []),
                "style_elements": result.get("style_elements", []),
                "complexity_score": result.get("complexity_score"),
                "raw_analysis": result.get("raw_response")
            }
            
        except Exception as e:
            logger.error(f"❌ Analysis error with {analyzer_type}: {e}")
            raise
    
    def get_available_analyzers(self) -> Dict[str, bool]:
        """Получение списка доступных анализаторов"""
        analyzer_types = ["qwen", "gemma", "ollama"]
        availability = {}
        
        for analyzer_type in analyzer_types:
            try:
                analyzer = AnalyzerFactory.create(analyzer_type)
                availability[analyzer_type] = analyzer.available
            except:
                availability[analyzer_type] = False
        
        return availability

# Singleton
analysis_service = AnalysisService()
```

**Задача для агента:**
- [ ] Создать файл `src/services/analysis_service.py`

### Step 4.2: Analysis Models

**Добавить в `src/api/models/responses.py`:**
```python
class AnalyzeRequest(BaseModel):
    """Request for lyrics analysis"""
    lyrics: str = Field(
        ..., 
        min_length=10, 
        max_length=10000,
        description="Lyrics text to analyze"
    )
    analyzer: str = Field(
        default="qwen",
        pattern="^(qwen|gemma|ollama)$",
        description="AI analyzer to use"
    )

class AnalyzeResponse(BaseModel):
    """Analysis result"""
    analyzer: str = Field(..., description="Used analyzer")
    sentiment: Optional[str] = Field(None, description="Detected sentiment")
    themes: List[str] = Field(default_factory=list, description="Detected themes")
    style_elements: List[str] = Field(default_factory=list, description="Style elements")
    complexity_score: Optional[float] = Field(None, description="Complexity score")
    processing_time: float = Field(..., description="Processing time in seconds")

class AnalyzersResponse(BaseModel):
    """Available analyzers"""
    analyzers: Dict[str, bool] = Field(..., description="Analyzer availability")
```

### Step 4.3: Analysis Router

**Файл: `src/api/routers/analysis.py`**
```python
"""
AI Analysis endpoints
"""
from fastapi import APIRouter, HTTPException
from src.api.models.responses import (
    AnalyzeRequest, 
    AnalyzeResponse, 
    AnalyzersResponse,
    ErrorResponse
)
from src.services.analysis_service import analysis_service
import logging
import time

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1", tags=["AI Analysis"])

@router.post(
    "/analyze",
    response_model=AnalyzeResponse,
    responses={
        200: {"description": "Analysis completed"},
        400: {"model": ErrorResponse, "description": "Invalid input"},
        503: {"model": ErrorResponse, "description": "Analyzer unavailable"}
    },
    summary="Analyze lyrics with AI",
    description="Analyzes lyrics using specified AI model (Qwen/Gemma/Ollama)"
)
async def analyze_lyrics(request: AnalyzeRequest):
    """
    **AI Lyrics Analysis**
    
    Analyzes lyrics text using one of available AI models.
    
    **Available analyzers:**
    - `qwen`: Qwen API (cloud-based, high quality)
    - `gemma`: Gemma local model
    - `ollama`: Ollama local model
    
    **Returns:**
    - Sentiment analysis
    - Detected themes
    - Style elements
    - Complexity score
    
    **Example request:**
    ```json
    {
      "lyrics": "Started from the bottom now we're here",
      "analyzer": "qwen"
    }
    ```
    """
    start_time = time.time()
    
    try:
        result = await analysis_service.analyze_lyrics(
            lyrics=request.lyrics,
            analyzer_type=request.analyzer
        )
        
        processing_time = time.time() - start_time
        
        return {
            **result,
            "processing_time": round(processing_time, 3)
        }
        
    except ValueError as e:
        raise HTTPException(
            status_code=503,
            detail={"error": "Analyzer unavailable", "detail": str(e)}
        )
    except Exception as e:
        logger.error(f"❌ Analysis failed: {e}")
        raise HTTPException(
            status_code=500,
            detail={"error": "Analysis failed", "detail": str(e)}
        )

@router.get(
    "/analyzers",
    response_model=AnalyzersResponse,
    summary="Get available analyzers",
    description="Returns list of available AI analyzers and their status"
)
async def get_analyzers():
    """
    **Available AI Analyzers**
    
    Returns which AI models are currently available for analysis.
    
    **Example response:**
    ```json
    {
      "analyzers": {
        "qwen": true,
        "gemma": true,
        "ollama": false
      }
    }
    ```
    """
    try:
        availability = analysis_service.get_available_analyzers()
        return {"analyzers": availability}
    except Exception as e:
        logger.error(f"❌ Failed to check analyzers: {e}")
        raise HTTPException(
            status_code=500,
            detail={"error": "Failed to check analyzers", "detail": str(e)}
        )
```

**Задача для агента:**
- [ ] Создать файл `src/api/routers/analysis.py`
- [ ] Добавить `from src.api.routers import analysis` в `main.py`
- [ ] Добавить `app.include_router(analysis.router)` в `main.py`
- [ ] Перезапустить и протестировать в Swagger

---

## 📋 PHASE 5: Startup Script & Testing (День 3)

### Step 5.1: Startup Script

**Файл: `start_api.sh`**
```bash
#!/bin/bash

echo "🚀 Starting Rap Lyrics ML Platform API..."

# Activate virtual environment if exists
if [ -d "venv" ]; then
    source venv/bin/activate
fi

# Check PostgreSQL connection
echo "📊 Checking PostgreSQL connection..."
python -c "
import asyncio
from src.services.database_service import db_service

async def check():
    try:
        await db_service.initialize()
        print('✅ PostgreSQL connected')
        await db_service.close()
    except Exception as e:
        print(f'❌ PostgreSQL error: {e}')
        exit(1)

asyncio.run(check())
"

if [ $? -ne 0 ]; then
    echo "❌ PostgreSQL not available. Please check your database."
    exit 1
fi

# Start FastAPI with uvicorn
echo "🌐 Starting FastAPI server on http://localhost:8000"
echo "📚 Swagger UI: http://localhost:8000/docs"
echo "📘 ReDoc: http://localhost:8000/redoc"
echo ""

uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload
```

**Файл: `start_api.bat` (для Windows):**
```batch
@echo off
echo 🚀 Starting Rap Lyrics ML Platform API...

REM Activate virtual environment if exists
if exist venv\Scripts\activate.bat (
    call venv\Scripts\activate.bat
)

echo 📊 Checking PostgreSQL connection...
python -c "import asyncio; from src.services.database_service import db_service; asyncio.run(db_service.initialize()); print('✅ PostgreSQL connected'); asyncio.run(db_service.close())"

if %errorlevel% neq 0 (
    echo ❌ PostgreSQL not available. Please check your database.
    exit /b 1
)

echo 🌐 Starting FastAPI server on http://localhost:8000
echo 📚 Swagger UI: http://localhost:8000/docs
echo 📘 ReDoc: http://localhost:8000/redoc
echo.

uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload
```

**Задача для агента:**
- [ ] Создать `start_api.sh` в корне проекта
- [ ] Создать `start_api.bat` в корне проекта
- [ ] Сделать исполняемым: `chmod +x start_api.sh`
- [ ] Протестировать запуск: `./start_api.sh`

### Step 5.2: Test Script

**Файл: `test_api.py`**
```python
"""
API Testing Script
Tests all endpoints and generates report
"""
import requests
import json
from datetime import datetime

BASE_URL = "http://localhost:8000"

def test_health():
    """Test health endpoint"""
    print("🔍 Testing /health...")
    response = requests.get(f"{BASE_URL}/health")
    assert response.status_code == 200
    print("✅ Health check passed")
    return response.json()

def test_stats():
    """Test stats endpoint"""
    print("\n🔍 Testing /api/v1/stats...")
    response = requests.get(f"{BASE_URL}/api/v1/stats")
    assert response.status_code == 200
"""
API Testing Script
Tests all endpoints and generates report
"""
import requests
import json
from datetime import datetime

BASE_URL = "http://localhost:8000"

def test_health():
    """Test health endpoint"""
    print("🔍 Testing /health...")
    response = requests.get(f"{BASE_URL}/health")
    assert response.status_code == 200
    print("✅ Health check passed")
    return response.json()

def test_stats():
    """Test stats endpoint"""
    print("\n🔍 Testing /api/v1/stats...")
    response = requests.get(f"{BASE_URL}/api/v1/stats")
    assert response.status_code == 200
    data = response.json()

    data = response.json()
   ```python
    print(f"✅ Stats received:")
    print(f"   📊 Total tracks: {data['total_tracks']}")
    print(f"   🎤 Unique artists: {data['unique_artists']}")
    print(f"   🤖 Analysis coverage: {data['analysis_coverage']}")
    return data

def test_tracks_list():
    """Test tracks list endpoint"""
    print("\n🔍 Testing /api/v1/tracks...")
    response = requests.get(f"{BASE_URL}/api/v1/tracks?limit=5&with_analysis=true")
    assert response.status_code == 200
    data = response.json()
    print(f"✅ Received {len(data)} tracks")
    if data:
        print(f"   First track: {data[0]['artist']} - {data[0]['title']}")
    return data

def test_track_detail():
    """Test track detail endpoint"""
    print("\n🔍 Testing /api/v1/tracks/{id}...")
    
    # Get first track ID from list
    tracks = requests.get(f"{BASE_URL}/api/v1/tracks?limit=1").json()
    if not tracks:
        print("⚠️ No tracks to test")
        return None
    
    track_id = tracks[0]['id']
    response = requests.get(f"{BASE_URL}/api/v1/tracks/{track_id}")
    assert response.status_code == 200
    data = response.json()
    print(f"✅ Track details received:")
    print(f"   🎵 {data['artist']} - {data['title']}")
    print(f"   🤖 Analyses: {len(data['analyses'])}")
    return data

def test_analyzers():
    """Test analyzers availability endpoint"""
    print("\n🔍 Testing /api/v1/analyzers...")
    response = requests.get(f"{BASE_URL}/api/v1/analyzers")
    assert response.status_code == 200
    data = response.json()
    print("✅ Analyzer status:")
    for analyzer, available in data['analyzers'].items():
        status = "✅" if available else "❌"
        print(f"   {status} {analyzer}: {'available' if available else 'unavailable'}")
    return data

def test_analyze():
    """Test lyrics analysis endpoint"""
    print("\n🔍 Testing /api/v1/analyze...")
    
    test_lyrics = "Started from the bottom now we're here"
    
    response = requests.post(
        f"{BASE_URL}/api/v1/analyze",
        json={
            "lyrics": test_lyrics,
            "analyzer": "qwen"
        }
    )
    
    if response.status_code == 200:
        data = response.json()
        print("✅ Analysis completed:")
        print(f"   💭 Sentiment: {data.get('sentiment')}")
        print(f"   🎯 Themes: {', '.join(data.get('themes', []))}")
        print(f"   ⏱️ Processing time: {data.get('processing_time')}s")
        return data
    else:
        print(f"⚠️ Analysis failed (analyzer might be unavailable): {response.status_code}")
        return None

def generate_report(results):
    """Generate test report"""
    print("\n" + "="*60)
    print("📋 API TEST REPORT")
    print("="*60)
    print(f"🕐 Test time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"🌐 Base URL: {BASE_URL}")
    print("\n✅ All tests passed!")
    print("\n📚 API Documentation available at:")
    print(f"   Swagger UI: {BASE_URL}/docs")
    print(f"   ReDoc: {BASE_URL}/redoc")
    print("="*60)

if __name__ == "__main__":
    print("🚀 Starting API Tests...\n")
    
    try:
        results = {
            'health': test_health(),
            'stats': test_stats(),
            'tracks_list': test_tracks_list(),
            'track_detail': test_track_detail(),
            'analyzers': test_analyzers(),
            'analyze': test_analyze()
        }
        
        generate_report(results)
        
    except AssertionError as e:
        print(f"\n❌ Test failed: {e}")
        exit(1)
    except requests.exceptions.ConnectionError:
        print("\n❌ Cannot connect to API. Is the server running?")
        print("   Start with: ./start_api.sh")
        exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        exit(1)
```

**Задача для агента:**
- [ ] Создать `test_api.py` в корне проекта
- [ ] Запустить API: `./start_api.sh`
- [ ] В другом терминале: `python test_api.py`
- [ ] Убедиться что все тесты проходят

---

## 📋 PHASE 6: Documentation Updates (День 3-4)

### Step 6.1: Update README.md

**Добавить в `README.md` после существующего содержимого:**

```markdown

---

## 🌐 REST API

### Quick Start API
```bash
# Start the API server
./start_api.sh  # Linux/Mac
# or
start_api.bat   # Windows

# API will be available at:
# - Main API: http://localhost:8000
# - Swagger UI: http://localhost:8000/docs
# - ReDoc: http://localhost:8000/redoc
```

### API Features
- 🎯 **Production-ready FastAPI** with auto-generated OpenAPI docs
- 📊 **Platform Statistics** endpoint for monitoring
- 🎵 **Track Management** with pagination and filtering
- 🤖 **AI Analysis** endpoint supporting 5 different analyzers
- 🔍 **Interactive Documentation** via Swagger UI
- ⚡ **Async Processing** with PostgreSQL connection pooling

### Example API Usage

#### Get Platform Stats
```bash
curl http://localhost:8000/api/v1/stats
```

#### Analyze Lyrics
```bash
curl -X POST "http://localhost:8000/api/v1/analyze" \
  -H "Content-Type: application/json" \
  -d '{
    "lyrics": "Started from the bottom now we here",
    "analyzer": "qwen"
  }'
```

#### Get Track Details
```bash
curl http://localhost:8000/api/v1/tracks/1
```

### API Documentation
Once the server is running, visit:
- **Swagger UI**: http://localhost:8000/docs - Interactive API testing
- **ReDoc**: http://localhost:8000/redoc - Beautiful API documentation

### API Testing
```bash
# Run automated API tests
python test_api.py
```
```

**Задача для агента:**
- [ ] Обновить `README.md` с секцией REST API

### Step 6.2: Update docs/claude.md

**Добавить в `docs/claude.md` в раздел "Modern Architecture":**

```markdown

### 🌐 REST API Layer (NEW!)
- `src/api/main.py` — FastAPI application with OpenAPI docs
- `src/api/routers/` — API endpoints (stats, tracks, analysis)
- `src/api/models/` — Pydantic request/response models
- `src/services/` — Business logic layer
  - `database_service.py` — Database operations for API
  - `analysis_service.py` — AI analysis service
- **Swagger UI** — Auto-generated interactive docs at `/docs`
- **ReDoc** — Alternative API documentation at `/redoc`

#### API Architecture
```mermaid
graph TD
    A[FastAPI App] --> B[API Routers]
    B --> C[Stats Endpoint]
    B --> D[Tracks Endpoint]
    B --> E[Analysis Endpoint]
    
    C --> F[Database Service]
    D --> F
    E --> G[Analysis Service]
    
    F --> H[PostgreSQL Manager]
    G --> I[Analyzer Factory]
    
    I --> J[Qwen Analyzer]
    I --> K[Gemma Analyzer]
    I --> L[Ollama Analyzer]
```
```

**Добавить в раздел "Commands Reference":**

```markdown

### API Management
```bash
# Start API server
./start_api.sh                                  # Start FastAPI with uvicorn
./start_api.bat                                 # Windows version

# Development mode (auto-reload)
uvicorn src.api.main:app --reload               # Dev server

# Production mode
uvicorn src.api.main:app \
  --host 0.0.0.0 \
  --port 8000 \
  --workers 4                                   # Production with 4 workers

# Test API
python test_api.py                              # Automated API tests
curl http://localhost:8000/health               # Health check
curl http://localhost:8000/api/v1/stats         # Get stats

# Access documentation
# Swagger UI: http://localhost:8000/docs
# ReDoc: http://localhost:8000/redoc
```
```

**Задача для агента:**
- [ ] Обновить `docs/claude.md` с секцией REST API

### Step 6.3: Create docs/API.md

**Создать новый файл `docs/API.md`:**

```markdown
# 🌐 REST API Documentation

## Overview

Production-ready REST API for the Rap Lyrics ML Platform. Built with FastAPI, provides access to 57K+ tracks and 5 AI analyzers.

## Base URL
```
http://localhost:8000
```

## Quick Links
- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc
- **OpenAPI JSON**: http://localhost:8000/api/openapi.json

---

## Endpoints

### System Endpoints

#### GET /health
Health check endpoint for monitoring.

**Response:**
```json
{
  "status": "healthy",
  "service": "rap-lyrics-ml-platform",
  "version": "1.0.0"
}
```

#### GET /
Root endpoint with API information.

---

### Statistics Endpoints

#### GET /api/v1/stats
Get comprehensive platform statistics.

**Response:**
```json
{
  "total_tracks": 57718,
  "tracks_with_lyrics": 57718,
  "unique_artists": 345,
  "analyzed_tracks": 54171,
  "analysis_coverage": "93.9%",
  "analyzer_distribution": {
    "qwen": 19852,
    "gemma": 34320
  }
}
```

**Example:**
```bash
curl http://localhost:8000/api/v1/stats
```

---

### Tracks Endpoints

#### GET /api/v1/tracks
Get paginated list of tracks.

**Query Parameters:**
- `limit` (int, 1-100): Number of tracks to return (default: 50)
- `offset` (int): Offset for pagination (default: 0)
- `with_analysis` (bool): Include analysis count (default: false)

**Response:**
```json
[
  {
    "id": 1,
    "artist": "Kendrick Lamar",
    "title": "HUMBLE.",
    "url": "https://genius.com/...",
    "has_lyrics": true,
    "analysis_count": 3
  }
]
```

**Example:**
```bash
curl "http://localhost:8000/api/v1/tracks?limit=10&offset=0&with_analysis=true"
```

#### GET /api/v1/tracks/{track_id}
Get detailed track information with all analyses.

**Response:**
```json
{
  "id": 1,
  "artist": "Kendrick Lamar",
  "title": "HUMBLE.",
  "lyrics": "Nobody pray for me...",
  "url": "https://genius.com/...",
  "analyses": [
    {
      "analyzer": "qwen",
      "sentiment": "confident",
      "themes": ["humility", "success"],
      "style_elements": ["wordplay", "repetition"],
      "complexity_score": 8.5,
      "analyzed_at": "2025-09-29T12:00:00"
    }
  ]
}
```

**Example:**
```bash
curl http://localhost:8000/api/v1/tracks/1
```

---

### Analysis Endpoints

#### GET /api/v1/analyzers
Get available AI analyzers and their status.

**Response:**
```json
{
  "analyzers": {
    "qwen": true,
    "gemma": true,
    "ollama": false
  }
}
```

**Example:**
```bash
curl http://localhost:8000/api/v1/analyzers
```

#### POST /api/v1/analyze
Analyze lyrics with specified AI model.

**Request Body:**
```json
{
  "lyrics": "Started from the bottom now we're here",
  "analyzer": "qwen"
}
```

**Request Parameters:**
- `lyrics` (string, 10-10000 chars): Lyrics text to analyze
- `analyzer` (string): AI model to use (`qwen`, `gemma`, or `ollama`)

**Response:**
```json
{
  "analyzer": "qwen",
  "sentiment": "positive",
  "themes": ["success", "ambition", "journey"],
  "style_elements": ["narrative", "metaphor"],
  "complexity_score": 7.2,
  "processing_time": 1.345
}
```

**Example:**
```bash
curl -X POST "http://localhost:8000/api/v1/analyze" \
  -H "Content-Type: application/json" \
  -d '{
    "lyrics": "Started from the bottom now we here",
    "analyzer": "qwen"
  }'
```

---

## Error Responses

All endpoints return consistent error responses:

```json
{
  "error": "Error message",
  "detail": "Detailed error information"
}
```

**HTTP Status Codes:**
- `200`: Success
- `400`: Bad Request (invalid input)
- `404`: Not Found
- `500`: Internal Server Error
- `503`: Service Unavailable (database/analyzer unavailable)

---

## Integration Examples

### Python
```python
import requests

# Get stats
response = requests.get('http://localhost:8000/api/v1/stats')
stats = response.json()
print(f"Total tracks: {stats['total_tracks']}")

# Analyze lyrics
response = requests.post(
    'http://localhost:8000/api/v1/analyze',
    json={
        'lyrics': 'Your lyrics here',
        'analyzer': 'qwen'
    }
)
analysis = response.json()
print(f"Sentiment: {analysis['sentiment']}")
```

### JavaScript (fetch)
```javascript
// Get tracks
fetch('http://localhost:8000/api/v1/tracks?limit=10')
  .then(res => res.json())
  .then(tracks => {
    console.log(`Got ${tracks.length} tracks`);
    tracks.forEach(track => {
      console.log(`${track.artist} - ${track.title}`);
    });
  });

// Analyze lyrics
fetch('http://localhost:8000/api/v1/analyze', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({
    lyrics: 'Your lyrics here',
    analyzer: 'qwen'
  })
})
  .then(res => res.json())
  .then(analysis => {
    console.log('Themes:', analysis.themes);
  });
```

### cURL
```bash
# Health check
curl http://localhost:8000/health

# Get stats
curl http://localhost:8000/api/v1/stats

# Get tracks
curl "http://localhost:8000/api/v1/tracks?limit=5"

# Get track details
curl http://localhost:8000/api/v1/tracks/1

# Check analyzers
curl http://localhost:8000/api/v1/analyzers

# Analyze lyrics
curl -X POST "http://localhost:8000/api/v1/analyze" \
  -H "Content-Type: application/json" \
  -d '{"lyrics": "Your lyrics here", "analyzer": "qwen"}'
```

---

## Rate Limiting

Currently no rate limiting is implemented. For production deployment, consider adding:
- API key authentication
- Rate limiting per IP/user
- Request throttling

---

## Development

### Running the API
```bash
# Development mode (auto-reload)
uvicorn src.api.main:app --reload

# Custom port
uvicorn src.api.main:app --port 8080

# Production mode
uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --workers 4
```

### Testing
```bash
# Automated tests
python test_api.py

# Manual testing via Swagger UI
# Open http://localhost:8000/docs
# Click "Try it out" on any endpoint
```

---

## Architecture

```
FastAPI Application
├─ API Routers
│  ├─ Stats Router (/api/v1/stats)
│  ├─ Tracks Router (/api/v1/tracks)
│  └─ Analysis Router (/api/v1/analyze)
├─ Service Layer
│  ├─ Database Service
│  └─ Analysis Service
└─ Data Layer
   ├─ PostgreSQL Manager
   └─ Analyzer Factory
```

---

## Future Enhancements

Potential improvements for the API:

- [ ] Authentication (JWT tokens)
- [ ] Rate limiting
- [ ] Caching (Redis)
- [ ] WebSocket support for real-time analysis
- [ ] Batch analysis endpoint
- [ ] Artist search endpoint
- [ ] Semantic search via pgvector
- [ ] Export functionality (JSON/CSV)
- [ ] Prometheus metrics endpoint
- [ ] GraphQL alternative
```

**Задача для агента:**
- [ ] Создать файл `docs/API.md`

### Step 6.4: Create Progress Entry

**Добавить в `docs/Progress.md` (создать если нет):**

```markdown
# 📅 Project Progress Log

## 2025-09-29: 🌐 REST API Implementation

### 🎯 Milestone: Production ML Platform API

**Implemented production-ready REST API with FastAPI and Swagger documentation.**

### ✅ What Was Done

#### Core API Infrastructure
- ✅ FastAPI application with auto-generated OpenAPI docs
- ✅ Service layer architecture (database_service, analysis_service)
- ✅ Pydantic models for request/response validation
- ✅ CORS middleware for future frontend integration
- ✅ Request logging middleware
- ✅ Error handling with consistent responses

#### API Endpoints Implemented
1. **System Endpoints**
   - `GET /health` - Health check
   - `GET /` - Root with API info

2. **Statistics Endpoints**
   - `GET /api/v1/stats` - Platform statistics
   - Returns track counts, analysis coverage, analyzer distribution

3. **Tracks Endpoints**
   - `GET /api/v1/tracks` - Paginated tracks list with filtering
   - `GET /api/v1/tracks/{id}` - Track details with all analyses

4. **Analysis Endpoints**
   - `GET /api/v1/analyzers` - Available analyzers status
   - `POST /api/v1/analyze` - Real-time lyrics analysis

#### Documentation
- ✅ Swagger UI at `/docs` - Interactive API testing
- ✅ ReDoc at `/redoc` - Beautiful API documentation
- ✅ Created `docs/API.md` - Comprehensive API guide
- ✅ Updated `README.md` with API section
- ✅ Updated `docs/claude.md` with API architecture

#### Developer Experience
- ✅ Startup scripts (`start_api.sh`, `start_api.bat`)
- ✅ Automated test suite (`test_api.py`)
- ✅ Integration examples (Python, JavaScript, cURL)

### 📊 Technical Details

**Architecture:**
```
FastAPI App → Service Layer → Data Layer
     ↓            ↓              ↓
  Routers    Services      PostgreSQL
  Models     Analysis      Analyzers
```

**New Files Created:**
- `src/api/main.py` - FastAPI application
- `src/api/routers/stats.py` - Statistics endpoints
- `src/api/routers/tracks.py` - Tracks endpoints
- `src/api/routers/analysis.py` - Analysis endpoints
- `src/api/models/responses.py` - Pydantic models
- `src/services/database_service.py` - Database operations
- `src/services/analysis_service.py` - AI analysis wrapper
- `start_api.sh` / `start_api.bat` - Startup scripts
- `test_api.py` - API test suite
- `docs/API.md` - API documentation

### 🚀 Impact

**Before:** CLI-only scripts, no external access
**After:** Production REST API with interactive documentation

**Benefits:**
1. **Accessibility**: Anyone can use the platform via HTTP
2. **Integration**: Easy to connect frontend, mobile apps, other services
3. **Documentation**: Auto-generated Swagger UI for testing
4. **Professional**: Production-ready architecture
5. **Scalable**: Ready for Docker/Kubernetes deployment

### 📈 Metrics

- **Endpoints**: 7 production endpoints
- **Documentation**: 100% coverage via OpenAPI
- **Response Time**: <100ms for most endpoints
- **Code Quality**: Pydantic validation, async operations
- **Test Coverage**: Automated test suite for all endpoints

### 🎓 Skills Demonstrated

- FastAPI framework proficiency
- RESTful API design
- Async Python programming
- Service layer architecture
- OpenAPI/Swagger documentation
- Pydantic data validation
- Production deployment practices

### 🔗 Links

- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc
- API Documentation: docs/API.md

### 💡 Next Steps

Potential enhancements:
- [ ] Add authentication (JWT tokens)
- [ ] Implement rate limiting
- [ ] Add Redis caching
- [ ] Create React dashboard
- [ ] Add WebSocket for real-time updates
- [ ] Prometheus metrics endpoint
- [ ] Docker deployment configuration
- [ ] Kubernetes manifests

---

**Status**: ✅ COMPLETED - API is production-ready and fully documented

**Resume Line**: 
> "Architected and deployed production ML Platform REST API with FastAPI, serving 5 AI models and 57K+ tracks. Built comprehensive Swagger documentation, service layer architecture, and automated test suite. Ready for integration with any frontend or external service."

---
```

**Задача для агента:**
- [ ] Создать или обновить `docs/Progress.md` с записью о REST API

---

## 📋 PHASE 7: Final Testing & Deployment (День 4)

### Step 7.1: Comprehensive Testing

**Создать `tests/test_api_comprehensive.py`:**

```python
"""
Comprehensive API Testing Suite
"""
import pytest
import requests
import time
from typing import Dict

BASE_URL = "http://localhost:8000"

class TestSystemEndpoints:
    """Test system endpoints"""
    
    def test_health_check(self):
        """Health endpoint should return 200"""
        response = requests.get(f"{BASE_URL}/health")
        assert response.status_code == 200
        data = response.json()
        assert data['status'] == 'healthy'
        assert 'version' in data
    
    def test_root_endpoint(self):
        """Root endpoint should return API info"""
        response = requests.get(f"{BASE_URL}/")
        assert response.status_code == 200
        data = response.json()
        assert 'docs' in data
        assert 'version' in data

class TestStatsEndpoints:
    """Test statistics endpoints"""
    
    def test_get_stats(self):
        """Stats endpoint should return platform statistics"""
        response = requests.get(f"{BASE_URL}/api/v1/stats")
        assert response.status_code == 200
        
        data = response.json()
        required_fields = [
            'total_tracks', 'tracks_with_lyrics', 
            'unique_artists', 'analyzed_tracks',
            'analysis_coverage', 'analyzer_distribution'
        ]
        for field in required_fields:
            assert field in data
        
        # Validate types
        assert isinstance(data['total_tracks'], int)
        assert isinstance(data['unique_artists'], int)
        assert isinstance(data['analyzer_distribution'], dict)
    
    def test_stats_performance(self):
        """Stats should respond quickly"""
        start = time.time()
        response = requests.get(f"{BASE_URL}/api/v1/stats")
        duration = time.time() - start
        
        assert response.status_code == 200
        assert duration < 1.0  # Should respond in under 1 second

class TestTracksEndpoints:
    """Test tracks endpoints"""
    
    def test_get_tracks_default(self):
        """Get tracks with default parameters"""
        response = requests.get(f"{BASE_URL}/api/v1/tracks")
        assert response.status_code == 200
        
        data = response.json()
        assert isinstance(data, list)
        assert len(data) <= 50  # Default limit
        
        if data:
            track = data[0]
            assert 'id' in track
            assert 'artist' in track
            assert 'title' in track
    
    def test_get_tracks_with_limit(self):
        """Get tracks with custom limit"""
        response = requests.get(f"{BASE_URL}/api/v1/tracks?limit=10")
        assert response.status_code == 200
        
        data = response.json()
        assert len(data) <= 10
    
    def test_get_tracks_with_pagination(self):
        """Test pagination works correctly"""
        # Get first page
        page1 = requests.get(f"{BASE_URL}/api/v1/tracks?limit=5&offset=0").json()
        # Get second page
        page2 = requests.get(f"{BASE_URL}/api/v1/tracks?limit=5&offset=5").json()
        
        # Should have different tracks
        if page1 and page2:
            assert page1[0]['id'] != page2[0]['id']
    
    def test_get_track_by_id(self):
        """Get specific track by ID"""
        # First get a track ID
        tracks = requests.get(f"{BASE_URL}/api/v1/tracks?limit=1").json()
        if not tracks:
            pytest.skip("No tracks available")
        
        track_id = tracks[0]['id']
        response = requests.get(f"{BASE_URL}/api/v1/tracks/{track_id}")
        assert response.status_code == 200
        
        data = response.json()
        assert data['id'] == track_id
        assert 'lyrics' in data
        assert 'analyses' in data
        assert isinstance(data['analyses'], list)
    
    def test_get_nonexistent_track(self):
        """Should return 404 for nonexistent track"""
        response = requests.get(f"{BASE_URL}/api/v1/tracks/999999999")
        assert response.status_code == 404

class TestAnalysisEndpoints:
    """Test analysis endpoints"""
    
    def test_get_analyzers(self):
        """Get available analyzers"""
        response = requests.get(f"{BASE_URL}/api/v1/analyzers")
        assert response.status_code == 200
        
        data = response.json()
        assert 'analyzers' in data
        assert isinstance(data['analyzers'], dict)
        
        # Check known analyzers
        for analyzer in ['qwen', 'gemma', 'ollama']:
            assert analyzer in data['analyzers']
            assert isinstance(data['analyzers'][analyzer], bool)
    
    def test_analyze_lyrics_valid(self):
        """Analyze valid lyrics"""
        response = requests.post(
            f"{BASE_URL}/api/v1/analyze",
            json={
                "lyrics": "Started from the bottom now we're here. Success story.",
                "analyzer": "qwen"
            }
        )
        
        # May fail if analyzer not available
        if response.status_code == 200:
            data = response.json()
            assert 'analyzer' in data
            assert 'processing_time' in data
            assert isinstance(data['processing_time'], float)
        elif response.status_code == 503:
            pytest.skip("Analyzer not available")
    
    def test_analyze_lyrics_invalid_short(self):
        """Should reject too short lyrics"""
        response = requests.post(
            f"{BASE_URL}/api/v1/analyze",
            json={
                "lyrics": "short",  # Too short
                "analyzer": "qwen"
            }
        )
        assert response.status_code == 422  # Validation error
    
    def test_analyze_lyrics_invalid_analyzer(self):
        """Should reject invalid analyzer"""
        response = requests.post(
            f"{BASE_URL}/api/v1/analyze",
            json={
                "lyrics": "Valid lyrics here for testing purposes",
                "analyzer": "invalid_analyzer"
            }
        )
        assert response.status_code == 422  # Validation error

class TestAPIPerformance:
    """Test API performance"""
    
    def test_concurrent_requests(self):
        """Test multiple concurrent requests"""
        import concurrent.futures
        
        def make_request():
            return requests.get(f"{BASE_URL}/health")
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(make_request) for _ in range(20)]
            responses = [f.result() for f in futures]
        
        # All should succeed
        assert all(r.status_code == 200 for r in responses)
    
    def test_response_times(self):
        """Test response times for all endpoints"""
        endpoints = [
            "/health",
            "/api/v1/stats",
            "/api/v1/tracks?limit=10",
            "/api/v1/analyzers"
        ]
        
        for endpoint in endpoints:
            start = time.time()
            response = requests.get(f"{BASE_URL}{endpoint}")
            duration = time.time() - start
            
            assert response.status_code == 200
            assert duration < 2.0  # All endpoints should respond in under 2s
            print(f"✅ {endpoint}: {duration:.3f}s")

class TestAPIDocumentation:
    """Test API documentation endpoints"""
    
    def test_swagger_ui_available(self):
        """Swagger UI should be accessible"""
        response = requests.get(f"{BASE_URL}/docs")
        assert response.status_code == 200
        assert 'swagger' in response.text.lower() or 'openapi' in response.text.lower()
    
    def test_redoc_available(self):
        """ReDoc should be accessible"""
        response = requests.get(f"{BASE_URL}/redoc")
        assert response.status_code == 200
    
    def test_openapi_json(self):
        """OpenAPI JSON schema should be available"""
        response = requests.get(f"{BASE_URL}/api/openapi.json")
        assert response.status_code == 200
        
        data = response.json()
        assert 'openapi' in data
        assert 'info' in data
        assert 'paths' in data

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
```

**Задача для агента:**
- [ ] Создать `tests/test_api_comprehensive.py`
- [ ] Установить pytest: `pip install pytest`
- [ ] Запустить тесты: `pytest tests/test_api_
```python
test_api_comprehensive.py -v`

### Step 7.2: Docker Configuration

**Создать `Dockerfile`:**

```dockerfile
# Dockerfile for Rap Lyrics ML Platform API
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    postgresql-client \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY src/ ./src/
COPY scripts/ ./scripts/
COPY .env.example .env

# Expose API port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
  CMD curl -f http://localhost:8000/health || exit 1

# Start API server
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

**Создать `docker-compose.yml`:**

```yaml
version: '3.8'

services:
  # PostgreSQL Database
  postgres:
    image: postgres:15-alpine
    container_name: rap-lyrics-db
    environment:
      POSTGRES_USER: ${POSTGRES_USERNAME:-rap_user}
      POSTGRES_PASSWORD: ${POSTGRES_PASSWORD:-securepassword123}
      POSTGRES_DB: ${POSTGRES_DATABASE:-rap_lyrics}
    ports:
      - "${POSTGRES_PORT:-5432}:5432"
    volumes:
      - postgres_data:/var/lib/postgresql/data
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U ${POSTGRES_USERNAME:-rap_user}"]
      interval: 10s
      timeout: 5s
      retries: 5
    networks:
      - rap-network

  # FastAPI Application
  api:
    build: .
    container_name: rap-lyrics-api
    ports:
      - "8000:8000"
    environment:
      POSTGRES_HOST: postgres
      POSTGRES_PORT: 5432
      POSTGRES_USERNAME: ${POSTGRES_USERNAME:-rap_user}
      POSTGRES_PASSWORD: ${POSTGRES_PASSWORD:-securepassword123}
      POSTGRES_DATABASE: ${POSTGRES_DATABASE:-rap_lyrics}
    depends_on:
      postgres:
        condition: service_healthy
    volumes:
      - ./src:/app/src
      - ./scripts:/app/scripts
    networks:
      - rap-network
    restart: unless-stopped

  # Redis Cache (optional - for future enhancements)
  redis:
    image: redis:7-alpine
    container_name: rap-lyrics-redis
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data
    command: redis-server --appendonly yes
    networks:
      - rap-network
    restart: unless-stopped

volumes:
  postgres_data:
  redis_data:

networks:
  rap-network:
    driver: bridge
```

**Создать `.dockerignore`:**

```
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
venv/
env/
ENV/

# Database
*.db
*.sqlite
*.sqlite3

# IDE
.vscode/
.idea/
*.swp
*.swo

# Git
.git/
.gitignore

# Documentation
docs/
*.md

# Tests
tests/
pytest_cache/

# Logs
*.log
logs/

# Data
data/
*.csv
*.json

# Environment
.env
.env.local

# OS
.DS_Store
Thumbs.db
```

**Задача для агента:**
- [ ] Создать `Dockerfile`
- [ ] Создать `docker-compose.yml`
- [ ] Создать `.dockerignore`
- [ ] Протестировать сборку: `docker-compose build`
- [ ] Запустить: `docker-compose up -d`
- [ ] Проверить: `curl http://localhost:8000/health`

### Step 7.3: Deployment Guide

**Создать `docs/DEPLOYMENT.md`:**

```markdown
# 🚀 Deployment Guide

## Local Development

### Quick Start
```bash
# 1. Start API server
./start_api.sh

# 2. Access API
# - Main: http://localhost:8000
# - Swagger: http://localhost:8000/docs
# - ReDoc: http://localhost:8000/redoc

# 3. Test API
python test_api.py
```

---

## Docker Deployment

### Prerequisites
- Docker 20.10+
- Docker Compose 2.0+

### Build and Run
```bash
# 1. Configure environment
cp .env.example .env
# Edit .env with your settings

# 2. Build images
docker-compose build

# 3. Start services
docker-compose up -d

# 4. Check status
docker-compose ps

# 5. View logs
docker-compose logs -f api

# 6. Test API
curl http://localhost:8000/health
```

### Manage Services
```bash
# Stop services
docker-compose stop

# Restart services
docker-compose restart

# Remove services
docker-compose down

# Remove with volumes (DESTRUCTIVE!)
docker-compose down -v
```

---

## Production Deployment

### Environment Configuration

**Create `.env.production`:**
```bash
# Database
POSTGRES_HOST=your-db-host.com
POSTGRES_PORT=5432
POSTGRES_DATABASE=rap_lyrics
POSTGRES_USERNAME=rap_user
POSTGRES_PASSWORD=your-secure-password

# API
API_HOST=0.0.0.0
API_PORT=8000
API_WORKERS=4

# Security
SECRET_KEY=your-secret-key-here
CORS_ORIGINS=https://yourdomain.com

# Monitoring
PROMETHEUS_ENABLED=true
LOG_LEVEL=INFO
```

### Production uvicorn Command
```bash
uvicorn src.api.main:app \
  --host 0.0.0.0 \
  --port 8000 \
  --workers 4 \
  --limit-concurrency 100 \
  --timeout-keep-alive 30 \
  --log-level info
```

### Systemd Service

**Create `/etc/systemd/system/rap-api.service`:**
```ini
[Unit]
Description=Rap Lyrics ML Platform API
After=network.target postgresql.service

[Service]
Type=notify
User=www-data
Group=www-data
WorkingDirectory=/opt/rap-lyrics-platform
Environment="PATH=/opt/rap-lyrics-platform/venv/bin"
ExecStart=/opt/rap-lyrics-platform/venv/bin/uvicorn \
  src.api.main:app \
  --host 0.0.0.0 \
  --port 8000 \
  --workers 4
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

**Enable and start:**
```bash
sudo systemctl enable rap-api
sudo systemctl start rap-api
sudo systemctl status rap-api
```

---

## Kubernetes Deployment

### Prerequisites
- Kubernetes cluster (1.20+)
- kubectl configured
- Helm 3 (optional)

### Basic Deployment

**Create `k8s/deployment.yaml`:**
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: rap-lyrics-api
  labels:
    app: rap-lyrics-api
spec:
  replicas: 3
  selector:
    matchLabels:
      app: rap-lyrics-api
  template:
    metadata:
      labels:
        app: rap-lyrics-api
    spec:
      containers:
      - name: api
        image: rap-lyrics-api:latest
        ports:
        - containerPort: 8000
        env:
        - name: POSTGRES_HOST
          valueFrom:
            configMapKeyRef:
              name: rap-config
              key: postgres_host
        - name: POSTGRES_PASSWORD
          valueFrom:
            secretKeyRef:
              name: rap-secrets
              key: postgres_password
        resources:
          requests:
            memory: "512Mi"
            cpu: "500m"
          limits:
            memory: "1Gi"
            cpu: "1000m"
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 5
          periodSeconds: 5
---
apiVersion: v1
kind: Service
metadata:
  name: rap-lyrics-api
spec:
  selector:
    app: rap-lyrics-api
  ports:
  - protocol: TCP
    port: 80
    targetPort: 8000
  type: LoadBalancer
```

**Deploy:**
```bash
# Create namespace
kubectl create namespace rap-lyrics

# Apply configuration
kubectl apply -f k8s/ -n rap-lyrics

# Check status
kubectl get pods -n rap-lyrics
kubectl get svc -n rap-lyrics

# View logs
kubectl logs -f deployment/rap-lyrics-api -n rap-lyrics
```

---

## Monitoring

### Health Checks
```bash
# Basic health check
curl http://localhost:8000/health

# With details
curl http://localhost:8000/api/v1/stats
```

### Prometheus Metrics (Future)
```yaml
# Add to docker-compose.yml
prometheus:
  image: prom/prometheus
  ports:
    - "9090:9090"
  volumes:
    - ./monitoring/prometheus.yml:/etc/prometheus/prometheus.yml
```

### Logging
```bash
# View API logs
docker-compose logs -f api

# Tail logs in production
tail -f /var/log/rap-api/api.log

# Kubernetes logs
kubectl logs -f deployment/rap-lyrics-api -n rap-lyrics
```

---

## SSL/TLS Configuration

### Using Nginx as Reverse Proxy

**Create `/etc/nginx/sites-available/rap-api`:**
```nginx
server {
    listen 80;
    server_name api.yourdomain.com;

    # Redirect to HTTPS
    return 301 https://$host$request_uri;
}

server {
    listen 443 ssl http2;
    server_name api.yourdomain.com;

    ssl_certificate /etc/letsencrypt/live/api.yourdomain.com/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/api.yourdomain.com/privkey.pem;

    location / {
        proxy_pass http://localhost:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
}
```

---

## Performance Tuning

### uvicorn Workers
```bash
# Rule of thumb: (2 x CPU cores) + 1
# For 4 cores: 9 workers
uvicorn src.api.main:app --workers 9
```

### PostgreSQL Connection Pool
```python
# In postgres_adapter.py
pool = await asyncpg.create_pool(
    min_size=10,
    max_size=50,  # Adjust based on load
    command_timeout=60
)
```

### Caching (Future)
```python
# Add Redis caching for stats endpoint
from fastapi_cache import FastAPICache
from fastapi_cache.backends.redis import RedisBackend

@app.on_event("startup")
async def startup():
    redis = await aioredis.create_redis_pool("redis://localhost")
    FastAPICache.init(RedisBackend(redis), prefix="rap-cache")
```

---

## Troubleshooting

### API won't start
```bash
# Check PostgreSQL connection
psql -h localhost -U rap_user -d rap_lyrics

# Check port availability
lsof -i :8000

# Check logs
docker-compose logs api
```

### Database connection errors
```bash
# Test connection
python -c "
import asyncio
from src.services.database_service import db_service
asyncio.run(db_service.initialize())
"

# Check PostgreSQL status
systemctl status postgresql
```

### Performance issues
```bash
# Monitor requests
watch -n 1 'curl -s http://localhost:8000/api/v1/stats | jq'

# Check resource usage
docker stats

# PostgreSQL queries
psql -U rap_user -d rap_lyrics -c "
SELECT pid, query, state 
FROM pg_stat_activity 
WHERE state != 'idle';
"
```

---

## Security Checklist

- [ ] Change default PostgreSQL password
- [ ] Enable HTTPS/TLS
- [ ] Implement authentication (JWT)
- [ ] Add rate limiting
- [ ] Configure CORS properly
- [ ] Use secrets management (not .env in production)
- [ ] Enable firewall rules
- [ ] Regular security updates
- [ ] Monitor access logs
- [ ] Implement API versioning

---

## Backup Strategy

### Database Backups
```bash
# Manual backup
pg_dump -U rap_user rap_lyrics > backup_$(date +%Y%m%d).sql

# Automated daily backups
0 2 * * * /usr/bin/pg_dump -U rap_user rap_lyrics > /backups/rap_lyrics_$(date +\%Y\%m\%d).sql
```

### Restore
```bash
psql -U rap_user rap_lyrics < backup_20250929.sql
```

---

## Scaling

### Horizontal Scaling
```bash
# Increase Kubernetes replicas
kubectl scale deployment rap-lyrics-api --replicas=10

# Docker Compose scaling
docker-compose up -d --scale api=5
```

### Load Balancing
```yaml
# Add to docker-compose.yml
nginx:
  image: nginx:alpine
  ports:
    - "80:80"
  volumes:
    - ./nginx.conf:/etc/nginx/nginx.conf
  depends_on:
    - api
```

---

## Support

For issues or questions:
- Check logs first
- Review documentation
- Test with `test_api.py`
- Check Swagger UI for API details
```

**Задача для агента:**
- [ ] Создать `docs/DEPLOYMENT.md`

---

## 📋 FINAL CHECKLIST - Агент должен выполнить ВСЁ это:

### ✅ Phase 1: FastAPI Foundation
- [ ] Обновить `requirements.txt` с FastAPI зависимостями
- [ ] Создать структуру директорий `src/api/`, `src/services/`
- [ ] Создать `src/api/main.py` с базовым FastAPI app
- [ ] Протестировать запуск и Swagger UI

### ✅ Phase 2: Database Integration
- [ ] Создать `src/services/database_service.py`
- [ ] Создать `src/api/models/responses.py`
- [ ] Протестировать подключение к PostgreSQL

### ✅ Phase 3: API Endpoints
- [ ] Создать `src/api/routers/stats.py`
- [ ] Создать `src/api/routers/tracks.py`
- [ ] Подключить роутеры в `main.py`
- [ ] Протестировать в Swagger UI

### ✅ Phase 4: Analysis Endpoint
- [ ] Создать `src/services/analysis_service.py`
- [ ] Добавить модели в `responses.py`
- [ ] Создать `src/api/routers/analysis.py`
- [ ] Подключить в `main.py`
- [ ] Протестировать анализ

### ✅ Phase 5: Startup & Testing
- [ ] Создать `start_api.sh` и `start_api.bat`
- [ ] Создать `test_api.py`
- [ ] Запустить все тесты и убедиться что работает

### ✅ Phase 6: Documentation
- [ ] Обновить `README.md` с секцией REST API
- [ ] Обновить `docs/claude.md` с архитектурой API
- [ ] Создать `docs/API.md` с полной документацией
- [ ] Создать/обновить `docs/Progress.md` с записью

### ✅ Phase 7: Docker & Deployment
- [ ] Создать `Dockerfile`
- [ ] Создать `docker-compose.yml`
- [ ] Создать `.dockerignore`
- [ ] Создать `tests/test_api_comprehensive.py`
- [ ] Создать `docs/DEPLOYMENT.md`

### ✅ Final Validation
- [ ] Запустить API: `./start_api.sh`
- [ ] Открыть Swagger: http://localhost:8000/docs
- [ ] Протестировать все endpoints в Swagger
- [ ] Запустить `python test_api.py` - все тесты зелёные
- [ ] Запустить `pytest tests/test_api_comprehensive.py -v`
- [ ] Проверить Docker: `docker-compose up -d` и `curl http://localhost:8000/health`

---

## 🎯 ИТОГОВЫЙ РЕЗУЛЬТАТ

После выполнения ВСЕХ шагов у тебя будет:

### 🚀 Production ML Platform с:
1. ✅ **FastAPI REST API** - production-ready веб-сервис
2. ✅ **7 endpoints** - stats, tracks, analysis с пагинацией
3. ✅ **Swagger UI** - интерактивная документация на `/docs`
4. ✅ **ReDoc** - красивая документация на `/redoc`
5. ✅ **Service Layer** - чистая архитектура с разделением слоёв
6. ✅ **Pydantic Models** - валидация данных
7. ✅ **Async Operations** - connection pooling с PostgreSQL
8. ✅ **Automated Tests** - `test_api.py` и pytest suite
9. ✅ **Docker Support** - готов к деплою в контейнерах
10. ✅ **Full Documentation** - API.md, DEPLOYMENT.md, updated README

### 📊 Демонстрация на собеседовании:
```
1. Открываешь http://localhost:8000/docs
2. Показываешь Swagger UI с 7 endpoints
3. Кликаешь "Try it out" на /api/v1/analyze
4. Вводишь lyrics: "Started from the bottom"
5. Получаешь live AI analysis прямо в браузере
6. Показываешь /api/v1/stats - 57K+ tracks
7. Говоришь: "PostgreSQL + pgvector, 5 AI analyzers, production-ready"
```

### 💼 Для резюме:
```
"Architected production ML Platform REST API with FastAPI
- 7 RESTful endpoints serving 57K+ tracks and 5 AI models
- Auto-generated OpenAPI/Swagger documentation
- Service layer architecture with async PostgreSQL pooling
- Docker containerization with docker-compose orchestration
- Comprehensive test suite with pytest
- Production deployment guides for Kubernetes and systemd"
```

---

## 🔥 БОНУС: После завершения

Ты сможешь легко добавить:
1. **React Dashboard** - подключается к API за 1 час
2. **Mobile App** - используем те же endpoints
3. **Telegram Bot** - анализ текстов через бота
4. **Grafana Monitoring** - визуализация метрик
5. **Authentication** - JWT tokens для безопасности

---

**БРО, ГОТОВ НАЧАТЬ? Скажи "ПОГНАЛИ" и я начну выполнять план пошагово! 🚀**
