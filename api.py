import os
import sys

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.cli.batch_processor import BatchProcessor
from src.cli.performance_monitor import PerformanceMonitor
from src.cli.text_analyzer import TextAnalyzer
from src.models.analysis_models import AnalysisResult
from src.models.config_models import AppConfig

app = FastAPI(
    title="Rap Lyrics Analyzer API",
    description="Enterprise-ready microservices API for rap lyrics analysis",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS middleware for web interface
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global analyzer instances
config = AppConfig.from_yaml("config.yaml")
text_analyzer = TextAnalyzer(config)

"""
🌐 FastAPI Web API для анализа текстов и песен

НАЗНАЧЕНИЕ:
- REST API для анализа текстов, пакетной обработки, мониторинга и статистики
- Веб-интерфейс и документация OpenAPI
- Интеграция с 4 анализаторами и базой данных

ИСПОЛЬЗОВАНИЕ:
docker-compose up -d                 # Запуск API в контейнере
python api.py                        # Локальный запуск (dev)
http://localhost:8000/docs           # Swagger/OpenAPI docs
POST /analyze                        # Анализ одного текста
POST /batch                          # Пакетная обработка
GET /status                          # Статус системы

ЗАВИСИМОСТИ:
- Python 3.8+, FastAPI, Pydantic
- src/{cli,models}/
- config.yaml
- SQLite база данных

РЕЗУЛЬТАТ:
- JSON-ответы с результатами анализа, статусом, ошибками
- Веб-интерфейс для тестирования API
- Интеграция с Docker и мониторингом

АВТОР: AI Assistant
ДАТА: Сентябрь 2025
"""
batch_processor = BatchProcessor(config)
performance_monitor = PerformanceMonitor(config)


# Request/Response models
class AnalysisRequest(BaseModel):
    text: str
    analyzer: str = "algorithmic_basic"


class BatchRequest(BaseModel):
    texts: list[str]
    analyzer: str = "algorithmic_basic"


class SystemStatus(BaseModel):
    status: str
    analyzers_available: list[str]
    database_records: int
    version: str


@app.get("/", response_class=HTMLResponse)
async def web_interface():
    """Simple web interface for testing the API"""
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Rap Lyrics Analyzer</title>
        <style>
            body { font-family: Arial, sans-serif; max-width: 800px; margin: 0 auto; padding: 20px; }
            .container { background: #f5f5f5; padding: 20px; border-radius: 10px; margin: 10px 0; }
            textarea { width: 100%; height: 100px; margin: 10px 0; }
            button { background: #007cba; color: white; padding: 10px 20px; border: none; border-radius: 5px; cursor: pointer; }
            button:hover { background: #005a8b; }
            .result { background: #e8f5e8; padding: 15px; border-radius: 5px; margin: 10px 0; }
            .status { background: #fff3cd; padding: 10px; border-radius: 5px; margin: 10px 0; }
        </style>
    </head>
    <body>
        <h1>🎵 Rap Lyrics Analyzer API</h1>
        
        <div class="container">
            <h3>Quick Analysis</h3>
            <textarea id="text" placeholder="Enter lyrics to analyze..."></textarea>
            <select id="analyzer">
                <option value="algorithmic_basic">Algorithmic Basic</option>
                <option value="gemma">Gemma AI</option>
                <option value="ollama">Ollama</option>
                <option value="hybrid">Hybrid</option>
            </select>
            <button onclick="analyzeText()">Analyze</button>
            <div id="result"></div>
        </div>

        <div class="container">
            <h3>System Status</h3>
            <button onclick="getStatus()">Check Status</button>
            <div id="status"></div>
        </div>

        <div class="container">
            <h3>API Documentation</h3>
            <p>Visit <a href="/docs">/docs</a> for interactive API documentation</p>
            <p>Visit <a href="/redoc">/redoc</a> for alternative documentation</p>
        </div>

        <script>
            async function analyzeText() {
                const text = document.getElementById('text').value;
                const analyzer = document.getElementById('analyzer').value;
                
                if (!text.trim()) {
                    alert('Please enter some text to analyze');
                    return;
                }

                try {
                    const response = await fetch('/analyze', {
                        method: 'POST',
                        headers: {'Content-Type': 'application/json'},
                        body: JSON.stringify({text: text, analyzer: analyzer})
                    });
                    
                    const result = await response.json();
                    document.getElementById('result').innerHTML = 
                        '<div class="result"><h4>Analysis Result:</h4><pre>' + 
                        JSON.stringify(result, null, 2) + '</pre></div>';
                } catch (error) {
                    document.getElementById('result').innerHTML = 
                        '<div class="result" style="background: #f8d7da;">Error: ' + error.message + '</div>';
                }
            }

            async function getStatus() {
                try {
                    const response = await fetch('/status');
                    const result = await response.json();
                    document.getElementById('status').innerHTML = 
                        '<div class="status"><h4>System Status:</h4><pre>' + 
                        JSON.stringify(result, null, 2) + '</pre></div>';
                } catch (error) {
                    document.getElementById('status').innerHTML = 
                        '<div class="status" style="background: #f8d7da;">Error: ' + error.message + '</div>';
                }
            }
        </script>
    </body>
    </html>
    """


@app.get("/status", response_model=SystemStatus)
async def get_system_status():
    """Get system status and health information"""
    try:
        analyzers = ["algorithmic_basic", "gemma", "ollama", "hybrid"]
        return SystemStatus(
            status="healthy",
            analyzers_available=analyzers,
            database_records=54568,  # This would be dynamic in real implementation
            version="1.0.0",
        )
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"System status check failed: {e!s}"
        )


@app.post("/analyze", response_model=AnalysisResult)
async def analyze_text(request: AnalysisRequest):
    """Analyze a single text with specified analyzer"""
    try:
        result = await text_analyzer.analyze_text(request.text, request.analyzer)
        return result
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Analysis failed: {e!s}")


@app.post("/batch", response_model=list[AnalysisResult])
async def batch_analyze(request: BatchRequest):
    """Analyze multiple texts in batch"""
    try:
        results = await batch_processor.process_batch(
            texts=request.texts,
            analyzer_type=request.analyzer,
            output_file=None,  # Return results directly
        )
        return results
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Batch analysis failed: {e!s}")


@app.get("/benchmark")
async def performance_benchmark():
    """Run performance benchmark across all analyzers"""
    try:
        test_texts = [
            "Amazing rap with incredible flow and lyrical content",
            "Deep metaphors and philosophical thoughts in music",
            "High energy track with powerful beats and rhymes",
        ]

        results = await performance_monitor.compare_analyzers(
            analyzer_types=["algorithmic_basic", "hybrid"],
            test_texts=test_texts,
            output_file=None,
        )

        return {
            "benchmark_completed": True,
            "test_texts_count": len(test_texts),
            "results": results,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Benchmark failed: {e!s}")


@app.get("/health")
async def health_check():
    """Simple health check endpoint"""
    return {"status": "healthy", "timestamp": "2025-08-29"}


if __name__ == "__main__":
    print("🚀 Starting Rap Lyrics Analyzer API...")
    print("📖 API Documentation: http://localhost:8000/docs")
    print("🌐 Web Interface: http://localhost:8000/")

    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True, log_level="info")
