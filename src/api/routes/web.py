"""Web interface and model information endpoints for the Rap ML API.

Provides interactive web interface for API testing and exploration, along with
comprehensive information about available ML models and their capabilities.
These endpoints enable easy testing, debugging, and understanding of the API
without requiring external tools or complex setup.

Endpoints:
    GET / - Interactive HTML web interface for API testing and exploration
    GET /models/info - Detailed information about available ML models and versions

The web interface includes:
- Interactive forms for testing all major API endpoints
- Real-time API response display with syntax highlighting
- Quick action buttons for health checks and configuration
- Links to API documentation (Swagger/ReDoc) and cache statistics
- Modern, responsive design with gradient backgrounds and animations

Features:
- No external dependencies - pure HTML/CSS/JavaScript
- Client-side API testing without CORS issues
- Real-time JSON response formatting and display
- Mobile-responsive design for testing on any device
- Direct links to all API documentation formats

Example:
    Visit GET / in browser for interactive API testing interface
    GET /models/info returns structured model capability information

Author: ML Platform Team
Date: October 2025
Version: 3.0.0
"""

from datetime import datetime, timezone
from typing import Literal

from fastapi import APIRouter, status
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from src.config import get_config

router = APIRouter(tags=["Web Interface"])
config = get_config()


# ============================================================================
# PYDANTIC MODELS
# ============================================================================


class ModelCapabilities(BaseModel):
    """Information about a single ML model's capabilities.

    Attributes:
        name: Human-readable model name and version
        provider: Model provider (Novita AI, HuggingFace, Internal)
        capabilities: List of operations this model supports
        status: Current operational status ("active", "maintenance", "deprecated")
    """

    name: str = Field(
        ...,
        description="Human-readable model name and version",
        examples=["QWEN 3 4B FP8"],
    )
    provider: str = Field(
        ...,
        description="Model provider or source",
        examples=["Novita AI"],
    )
    capabilities: list[str] = Field(
        ...,
        description="List of operations this model can perform",
        examples=[["lyrics_analysis", "sentiment", "theme_detection"]],
    )
    status: Literal["active", "maintenance", "deprecated"] = Field(
        ...,
        description="Current operational status of the model",
        examples=["active"],
    )


class ModelsInfoResponse(BaseModel):
    """Response model for available models information.

    Attributes:
        models: Dictionary mapping model IDs to their capabilities
        version: API version string
        timestamp: ISO 8601 timestamp of when info was retrieved
    """

    models: dict[str, ModelCapabilities] = Field(
        ...,
        description="Dictionary of available models and their capabilities",
    )
    version: str = Field(
        ...,
        description="Current API version",
        examples=["3.0.0"],
    )
    timestamp: str = Field(
        ...,
        description="ISO 8601 timestamp when information was retrieved",
        examples=["2025-10-30T10:30:00.000Z"],
    )


# ============================================================================
# ENDPOINTS
# ============================================================================


# TODO(FAANG): Security - XSS and injection vulnerabilities
#   - Move HTML to separate template file (Jinja2)
#   - Add Content-Security-Policy headers
#   - Sanitize all user inputs in JavaScript
#   - Add CSRF protection for forms
#   - Use nonce for inline scripts
@router.get(
    "/",
    response_class=HTMLResponse,
    summary="Interactive web interface for API testing",
    status_code=status.HTTP_200_OK,
    responses={
        200: {
            "description": "Interactive HTML interface for testing API endpoints",
            "content": {"text/html": {}},
        }
    },
)
async def web_interface() -> str:
    """Interactive HTML web interface for API testing and exploration.

    Provides a modern, user-friendly web interface for testing all major API
    endpoints without requiring external tools like Postman or curl. Features
    real-time response display, syntax highlighting, and quick access to all
    API documentation.

    The interface includes:
    - Lyrics analysis form with textarea input
    - Quick action buttons for health check and configuration
    - Real-time JSON response display with formatting
    - Direct links to Swagger docs, ReDoc, and cache stats
    - Responsive design that works on mobile and desktop

    Returns:
        str: Complete HTML page with embedded CSS and JavaScript.
            No external dependencies required - works offline.

    Example:
        Open http://localhost:8000/ in browser to access interface.
        Enter lyrics in textarea and click "Analyze" to test the API.

    Note:
        - Interface uses fetch API for requests (modern browsers only)
        - All requests are made client-side (no server-side rendering)
        - CORS must be enabled for API calls to work
        - Mobile-responsive with gradient background design
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

            @media (max-width: 600px) {
                .container {
                    padding: 20px;
                }
                .links {
                    flex-direction: column;
                }
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


@router.get(
    "/models/info",
    response_model=ModelsInfoResponse,
    summary="Get information about available ML models",
    status_code=status.HTTP_200_OK,
    responses={
        200: {
            "description": "Model information retrieved successfully. Returns details about all available models.",
            "model": ModelsInfoResponse,
        }
    },
)
async def models_info() -> ModelsInfoResponse:
    """Get comprehensive information about available ML models and their capabilities.

    Returns detailed metadata about all ML models currently available in the API,
    including their names, providers, supported operations, and operational status.
    Useful for understanding which models to use for specific tasks.

    # TODO(FAANG): Replace hardcoded model data with dynamic discovery
    #   - Query actual model registry/database for real-time status
    #   - Add model version tracking and deprecation warnings
    #   - Include model performance metrics (latency, accuracy)
    #   - Add pagination for large model lists
    #   - Cache response with short TTL (5-10 min)

    This endpoint supports:
    - Model discovery and capability exploration
    - API integration planning and decision-making
    - Monitoring model availability and status
    - Documentation and API reference generation

    Returns:
        ModelsInfoResponse: Complete model information with:
            - models: Dictionary of model_id -> ModelCapabilities
            - version: Current API version string
            - timestamp: ISO 8601 timestamp of info retrieval

    Example:
        >>> info = await models_info()
        >>> print(info.models['qwen'].name)
        'QWEN 3 4B FP8'
        >>> print(info.models['qwen'].capabilities)
        ['lyrics_analysis', 'sentiment', 'theme_detection']
        >>> print(info.version)
        '3.0.0'

    Note:
        - Model status "active" means ready for production use
        - Model status "maintenance" means temporarily unavailable
        - Model status "deprecated" means will be removed in future version
        - Capabilities list shows all operations the model supports
    """
    # TODO(FAANG): Move static model data to configuration or database
    #   - Create ModelRegistry service for centralized model management
    #   - Add health checks to verify model availability
    #   - Include circuit breaker pattern for failing models
    return ModelsInfoResponse(
        models={
            "qwen": ModelCapabilities(
                name="QWEN 3 4B FP8",
                provider="Novita AI",
                capabilities=["lyrics_analysis", "sentiment", "theme_detection"],
                status="active",
            ),
            "style_transfer": ModelCapabilities(
                name="T5 Style Transfer",
                provider="HuggingFace",
                capabilities=["style_transfer_between_artists"],
                status="active",
            ),
            "quality_prediction": ModelCapabilities(
                name="Ensemble Quality Predictor",
                provider="Internal",
                capabilities=["quality_score", "commercial_potential"],
                status="active",
            ),
            "trend_analysis": ModelCapabilities(
                name="RapTrendAnalyzer",
                provider="Internal",
                capabilities=["trend_analysis", "forecasting"],
                status="active",
            ),
        },
        version="3.0.0",
        timestamp=datetime.now(timezone.utc).isoformat(),
    )
