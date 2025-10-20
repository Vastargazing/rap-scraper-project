"""
🌐 Web Interface Routes
HTML web interface and models information

Provides:
- / - Main web interface
- /models/info - Available models information
"""

from datetime import datetime

from fastapi import APIRouter
from fastapi.responses import HTMLResponse

from src.config import get_config

router = APIRouter()
config = get_config()


# ============================================================================
# ENDPOINTS
# ============================================================================

@router.get("/", response_class=HTMLResponse)
async def web_interface() -> str:
    """
    Main Web Interface

    Returns interactive HTML interface for testing the API

    Returns:
        str: HTML content for web interface
    """
    return """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>🎤 Rap ML API v3.0.0</title>
        <style>
            * {
                margin: 0;
                padding: 0;
                box-sizing: border-box;
            }
            
            body {
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                min-height: 100vh;
                display: flex;
                align-items: center;
                justify-content: center;
                padding: 20px;
            }
            
            .container {
                background: white;
                border-radius: 10px;
                box-shadow: 0 10px 40px rgba(0,0,0,0.2);
                max-width: 800px;
                width: 100%;
                padding: 40px;
            }
            
            h1 {
                color: #333;
                margin-bottom: 10px;
                display: flex;
                align-items: center;
                gap: 10px;
            }
            
            .status {
                color: #666;
                font-size: 14px;
                margin-bottom: 30px;
            }
            
            .section {
                margin: 30px 0;
                padding: 20px;
                background: #f5f5f5;
                border-radius: 5px;
            }
            
            h2 {
                color: #667eea;
                font-size: 18px;
                margin-bottom: 15px;
            }
            
            textarea {
                width: 100%;
                height: 120px;
                padding: 10px;
                border: 1px solid #ddd;
                border-radius: 5px;
                font-family: monospace;
                font-size: 14px;
                margin-bottom: 10px;
            }
            
            select, input {
                width: 100%;
                padding: 10px;
                border: 1px solid #ddd;
                border-radius: 5px;
                margin-bottom: 10px;
                font-size: 14px;
            }
            
            button {
                background: #667eea;
                color: white;
                padding: 12px 24px;
                border: none;
                border-radius: 5px;
                cursor: pointer;
                font-size: 16px;
                font-weight: bold;
                transition: background 0.3s;
                width: 100%;
            }
            
            button:hover {
                background: #764ba2;
            }
            
            .result {
                background: white;
                padding: 15px;
                border-radius: 5px;
                margin-top: 15px;
                border-left: 4px solid #667eea;
                display: none;
            }
            
            .result.show {
                display: block;
            }
            
            .result pre {
                overflow-x: auto;
                font-size: 12px;
            }
            
            .links {
                display: flex;
                gap: 10px;
                margin-top: 20px;
                justify-content: center;
            }
            
            .links a {
                color: #667eea;
                text-decoration: none;
                padding: 10px 20px;
                border: 1px solid #667eea;
                border-radius: 5px;
                transition: all 0.3s;
            }
            
            .links a:hover {
                background: #667eea;
                color: white;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🎤 Rap ML API v3.0.0</h1>
            <div class="status">Unified API for rap analysis | Status: Running ✅</div>
            
            <div class="section">
                <h2>Analyze Lyrics</h2>
                <textarea id="lyrics" placeholder="Enter rap lyrics here..."></textarea>
                <button onclick="analyzeLyrics()">🔍 Analyze</button>
                <div id="result" class="result">
                    <pre id="resultText"></pre>
                </div>
            </div>
            
            <div class="section">
                <h2>Quick Actions</h2>
                <button onclick="getHealth()" style="margin-bottom: 10px;">💚 Health Check</button>
                <button onclick="getConfig()">⚙️ Config Info</button>
            </div>
            
            <div class="links">
                <a href="/docs">📚 API Docs</a>
                <a href="/redoc">📖 ReDoc</a>
                <a href="/cache/stats">📊 Cache Stats</a>
            </div>
        </div>
        
        <script>
            async function analyzeLyrics() {
                const lyrics = document.getElementById('lyrics').value;
                if (!lyrics.trim()) {
                    alert('Please enter lyrics to analyze');
                    return;
                }
                
                try {
                    const response = await fetch('/analyze', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({ lyrics, use_cache: true })
                    });
                    const data = await response.json();
                    showResult(data);
                } catch (e) {
                    showResult({ error: e.message });
                }
            }
            
            async function getHealth() {
                try {
                    const response = await fetch('/health');
                    const data = await response.json();
                    showResult(data);
                } catch (e) {
                    showResult({ error: e.message });
                }
            }
            
            async function getConfig() {
                try {
                    const response = await fetch('/config/info');
                    const data = await response.json();
                    showResult(data);
                } catch (e) {
                    showResult({ error: e.message });
                }
            }
            
            function showResult(data) {
                const result = document.getElementById('result');
                const resultText = document.getElementById('resultText');
                resultText.textContent = JSON.stringify(data, null, 2);
                result.classList.add('show');
            }
        </script>
    </body>
    </html>
    """


@router.get("/models/info")
async def models_info() -> dict:
    """
    Available Models Information

    Returns information about available ML models

    Returns:
        dict: Model names, versions, and capabilities
    """
    return {
        "models": {
            "qwen": {
                "name": "QWEN 3 4B FP8",
                "provider": "Novita AI",
                "capabilities": ["lyrics_analysis", "sentiment", "theme_detection"],
                "status": "active",
            },
            "style_transfer": {
                "name": "T5 Style Transfer",
                "provider": "HuggingFace",
                "capabilities": ["style_transfer_between_artists"],
                "status": "active",
            },
            "quality_prediction": {
                "name": "Ensemble Quality Predictor",
                "provider": "Internal",
                "capabilities": ["quality_score", "commercial_potential"],
                "status": "active",
            },
            "trend_analysis": {
                "name": "RapTrendAnalyzer",
                "provider": "Internal",
                "capabilities": ["trend_analysis", "forecasting"],
                "status": "active",
            },
        },
        "version": "3.0.0",
        "timestamp": datetime.now().isoformat(),
    }
