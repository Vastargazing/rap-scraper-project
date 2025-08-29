# 📚 API Documentation

## Overview
FastAPI-based REST API for rap lyrics analysis with 4 AI analyzers and web interface.

**Base URL**: `http://localhost:8000`  
**Interactive Docs**: `/docs`  
**Alternative Docs**: `/redoc`

## Authentication
Currently no authentication required for local deployment.

## Endpoints

### 🏠 Web Interface
```http
GET /
```
**Description**: HTML web interface for testing the API  
**Response**: HTML page with interactive forms

### 📊 System Status
```http
GET /status
```
**Description**: Get system health and configuration  
**Response Model**: `SystemStatus`

**Example Response**:
```json
{
  "status": "healthy",
  "analyzers_available": [
    "algorithmic_basic",
    "gemma", 
    "ollama",
    "hybrid"
  ],
  "database_records": 54568,
  "version": "1.0.0"
}
```

### 🧠 Single Text Analysis
```http
POST /analyze
```
**Description**: Analyze a single text with specified analyzer  
**Request Model**: `AnalysisRequest`  
**Response Model**: `AnalysisResult`

**Request Body**:
```json
{
  "text": "Amazing rap with incredible flow and lyrical content",
  "analyzer": "hybrid"
}
```

**Response**:
```json
{
  "text": "Amazing rap with incredible flow and lyrical content",
  "analyzer_type": "hybrid",
  "complexity_score": 8.5,
  "mood_category": "positive",
  "quality_rating": 9.2,
  "technical_metrics": {
    "word_count": 8,
    "unique_words": 8,
    "syllable_density": 2.1,
    "rhyme_scheme": "AABB"
  },
  "analysis_timestamp": "2025-08-29T10:30:00Z",
  "processing_time_ms": 245
}
```

### 📦 Batch Processing
```http
POST /batch
```
**Description**: Analyze multiple texts simultaneously  
**Request Model**: `BatchRequest`  
**Response Model**: `List[AnalysisResult]`

**Request Body**:
```json
{
  "texts": [
    "First rap lyrics to analyze",
    "Second text for analysis", 
    "Third sample text"
  ],
  "analyzer": "gemma"
}
```

**Response**: Array of `AnalysisResult` objects

### ⚡ Performance Benchmark
```http
GET /benchmark
```
**Description**: Run performance comparison across analyzers  
**Parameters**: None  
**Response**: Benchmark results with timing and accuracy metrics

**Example Response**:
```json
{
  "benchmark_completed": true,
  "test_texts_count": 3,
  "results": {
    "algorithmic_basic": {
      "avg_processing_time_ms": 45,
      "accuracy_score": 0.85,
      "throughput_per_second": 22
    },
    "hybrid": {
      "avg_processing_time_ms": 300,
      "accuracy_score": 0.94,
      "throughput_per_second": 3.3
    }
  }
}
```

### 🏥 Health Check
```http
GET /health
```
**Description**: Simple health check for monitoring  
**Response**:
```json
{
  "status": "healthy",
  "timestamp": "2025-08-29T10:30:00Z"
}
```

## Data Models

### AnalysisRequest
```python
class AnalysisRequest(BaseModel):
    text: str                    # Text to analyze (required)
    analyzer: str = "algorithmic_basic"  # Analyzer type (optional)
```

### BatchRequest  
```python
class BatchRequest(BaseModel):
    texts: List[str]             # List of texts (required)
    analyzer: str = "algorithmic_basic"  # Analyzer type (optional)
```

### AnalysisResult
```python
class AnalysisResult(BaseModel):
    text: str                    # Original text
    analyzer_type: str           # Used analyzer
    complexity_score: float      # 0-10 complexity rating
    mood_category: str           # positive/negative/neutral
    quality_rating: float        # 0-10 quality score
    technical_metrics: dict      # Detailed metrics
    analysis_timestamp: str      # ISO timestamp
    processing_time_ms: int      # Processing duration
```

### SystemStatus
```python
class SystemStatus(BaseModel):
    status: str                  # System health status
    analyzers_available: List[str]  # Available analyzers
    database_records: int        # Total records in DB
    version: str                 # API version
```

## Error Handling

### HTTP Status Codes
- `200 OK`: Successful request
- `400 Bad Request`: Invalid input data
- `422 Unprocessable Entity`: Validation errors
- `500 Internal Server Error`: System errors

### Error Response Format
```json
{
  "detail": "Error description message"
}
```

## Analyzer Types

| Analyzer | Description | Speed | Accuracy | Use Case |
|----------|-------------|-------|----------|----------|
| `algorithmic_basic` | Fast rule-based analysis | Very Fast | Good | Production speed |
| `gemma` | Gemma AI model analysis | Medium | Very High | Quality analysis |
| `ollama` | Local Ollama model | Fast | High | Privacy/offline |
| `hybrid` | Multi-model combination | Medium | Highest | Best results |

## Rate Limiting
- Currently no rate limiting implemented
- For production: Implement per-IP rate limiting
- Recommended: 100 requests/minute per client

## CORS Configuration
- Currently allows all origins (`*`) 
- For production: Restrict to specific domains
- Credentials and headers allowed

## Example Integration

### Python Client
```python
import requests

class RapAnalyzerClient:
    def __init__(self, base_url="http://localhost:8000"):
        self.base_url = base_url
    
    def analyze(self, text, analyzer="hybrid"):
        response = requests.post(
            f"{self.base_url}/analyze",
            json={"text": text, "analyzer": analyzer}
        )
        return response.json()
    
    def batch_analyze(self, texts, analyzer="hybrid"):
        response = requests.post(
            f"{self.base_url}/batch", 
            json={"texts": texts, "analyzer": analyzer}
        )
        return response.json()
    
    def get_status(self):
        response = requests.get(f"{self.base_url}/status")
        return response.json()

# Usage
client = RapAnalyzerClient()
result = client.analyze("Amazing rap lyrics here")
print(f"Quality: {result['quality_rating']}/10")
```

### JavaScript/Browser
```javascript
class RapAnalyzerAPI {
    constructor(baseUrl = 'http://localhost:8000') {
        this.baseUrl = baseUrl;
    }
    
    async analyze(text, analyzer = 'hybrid') {
        const response = await fetch(`${this.baseUrl}/analyze`, {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({text, analyzer})
        });
        return await response.json();
    }
    
    async getStatus() {
        const response = await fetch(`${this.baseUrl}/status`);
        return await response.json();
    }
}

// Usage
const api = new RapAnalyzerAPI();
const result = await api.analyze('Your lyrics here');
console.log(`Complexity: ${result.complexity_score}/10`);
```

## Deployment Notes

### Docker
```bash
# Build and run
docker build -t rap-analyzer-api .
docker run -p 8000:8000 rap-analyzer-api

# With docker-compose
docker-compose up -d
```

### Production Considerations
- Add authentication middleware
- Implement rate limiting  
- Configure CORS properly
- Add request/response logging
- Set up health monitoring
- Use HTTPS in production

### Environment Variables
```bash
# Optional configuration
export RAP_ANALYZER_PORT=8000
export RAP_ANALYZER_HOST=0.0.0.0
export RAP_ANALYZER_LOG_LEVEL=info
export RAP_ANALYZER_ENV=production
```
