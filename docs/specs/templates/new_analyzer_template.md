# 🧩 New Analyzer Component Template

## Specification Template

```yaml
# specs/analyzers/{analyzer_name}.spec.yaml
component_type: "analyzer"
name: "{analyzer_name}"
version: "1.0.0"

purpose: |
  Brief description of what this analyzer does
  Example: "Sentiment analysis using transformer models"

interface:
  base_class: "BaseAnalyzer"
  required_methods:
    - analyze(text: str) -> AnalysisResult
    - batch_analyze(texts: List[str]) -> List[AnalysisResult]
    - get_config() -> Dict[str, Any]

inputs:
  text:
    type: "string"
    max_length: 10000
    encoding: "utf-8"
    required: true

outputs:
  sentiment:
    type: "float"
    range: [0.0, 1.0]
    description: "Sentiment score"
  confidence:
    type: "float"
    range: [0.0, 1.0]
    description: "Confidence in analysis"
  genre:
    type: "string"
    enum: ["hip-hop", "rap", "r&b", "other"]
    description: "Detected genre"

configuration:
  api_key:
    type: "string"
    source: "environment"
    required: true
  temperature:
    type: "float"
    default: 0.1
    range: [0.0, 2.0]
  timeout:
    type: "integer"
    default: 30
    unit: "seconds"

performance:
  target_latency: "< 500ms"
  batch_size: 100
  rate_limit: "100/minute"

dependencies:
  - "requests >= 2.28.0"
  - "pydantic >= 1.10.0"

testing:
  unit_tests: true
  integration_tests: true
  performance_tests: true
  
integration:
  config_path: "config.yaml"
  registration: "src/analyzers/__init__.py"
  cli_support: true
```

## Implementation Template

```python
#!/usr/bin/env python3
"""
🎯 {Analyzer Name}
{Brief description of analyzer purpose}

НАЗНАЧЕНИЕ:
- {Main functionality 1}
- {Main functionality 2}
- {Main functionality 3}

ИСПОЛЬЗОВАНИЕ:
python src/analyzers/{analyzer_name}.py --text "sample text"
# Or via main.py interface

ЗАВИСИМОСТИ:
- Python 3.8+
- src/interfaces/analyzer_interface.py
- {Specific dependencies}

РЕЗУЛЬТАТ:
- AnalysisResult with sentiment, confidence, genre
- Integration with main pipeline

АВТОР: {Your Name} | ДАТА: {Current Date}
"""

import logging
from typing import Dict, Any, Optional
from datetime import datetime

from interfaces.analyzer_interface import BaseAnalyzer, AnalysisResult, register_analyzer

logger = logging.getLogger(__name__)

@register_analyzer("{analyzer_name}")
class {AnalyzerClass}(BaseAnalyzer):
    """
    {Detailed analyzer description}
    
    Features:
    - {Feature 1}
    - {Feature 2}
    - {Feature 3}
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize {analyzer_name} analyzer"""
        super().__init__(config)
        
        # Configuration
        self.api_key = self.config.get('api_key')
        self.temperature = self.config.get('temperature', 0.1)
        self.timeout = self.config.get('timeout', 30)
        
        # Validation
        if not self.api_key:
            raise ValueError("API key required for {analyzer_name}")
    
    async def analyze(self, text: str, **kwargs) -> AnalysisResult:
        """
        Analyze text using {analyzer_name}
        
        Args:
            text: Input text to analyze
            **kwargs: Additional parameters
            
        Returns:
            AnalysisResult with sentiment, confidence, genre
        """
        start_time = datetime.now()
        
        try:
            # TODO: Implement analysis logic
            result = await self._perform_analysis(text)
            
            analysis_time = (datetime.now() - start_time).total_seconds()
            
            return AnalysisResult(
                analyzer_name=self.name,
                sentiment=result.get('sentiment', 0.5),
                confidence=result.get('confidence', 0.0),
                genre=result.get('genre', 'unknown'),
                analysis_time=analysis_time,
                metadata={
                    'model_version': self.config.get('model_version', '1.0'),
                    'temperature': self.temperature,
                    'text_length': len(text)
                }
            )
            
        except Exception as e:
            logger.error(f"Analysis failed: {e}")
            return AnalysisResult(
                analyzer_name=self.name,
                sentiment=0.5,
                confidence=0.0,
                genre='error',
                analysis_time=(datetime.now() - start_time).total_seconds(),
                error=str(e)
            )
    
    async def _perform_analysis(self, text: str) -> Dict[str, Any]:
        """
        Internal analysis implementation
        
        TODO: Replace with actual analysis logic
        """
        # Placeholder implementation
        return {
            'sentiment': 0.5,
            'confidence': 0.8,
            'genre': 'hip-hop'
        }
    
    def is_available(self) -> bool:
        """Check if analyzer is ready to use"""
        return self.api_key is not None
    
    def get_info(self) -> Dict[str, Any]:
        """Get analyzer information"""
        return {
            'name': self.name,
            'version': '1.0.0',
            'description': '{Analyzer description}',
            'available': self.is_available(),
            'config': {
                'temperature': self.temperature,
                'timeout': self.timeout
            }
        }

# Example usage
if __name__ == "__main__":
    analyzer = {AnalyzerClass}({
        'api_key': 'your-api-key',
        'temperature': 0.1
    })
    
    # Test analysis
    import asyncio
    result = asyncio.run(analyzer.analyze("Sample rap lyrics"))
    print(f"Result: {result}")
```

## Testing Template

```python
#!/usr/bin/env python3
"""Tests for {analyzer_name}"""

import pytest
import asyncio
from unittest.mock import Mock, patch

from src.analyzers.{analyzer_name} import {AnalyzerClass}

class Test{AnalyzerClass}:
    """Test suite for {analyzer_name}"""
    
    @pytest.fixture
    def analyzer(self):
        """Create analyzer instance for testing"""
        return {AnalyzerClass}({
            'api_key': 'test-key',
            'temperature': 0.1,
            'timeout': 10
        })
    
    @pytest.mark.asyncio
    async def test_analyze_success(self, analyzer):
        """Test successful analysis"""
        text = "Test rap lyrics with positive vibes"
        result = await analyzer.analyze(text)
        
        assert result.analyzer_name == "{analyzer_name}"
        assert 0.0 <= result.sentiment <= 1.0
        assert 0.0 <= result.confidence <= 1.0
        assert result.genre in ["hip-hop", "rap", "r&b", "other"]
    
    @pytest.mark.asyncio
    async def test_analyze_empty_text(self, analyzer):
        """Test analysis with empty text"""
        result = await analyzer.analyze("")
        assert result.confidence == 0.0
    
    def test_is_available(self, analyzer):
        """Test availability check"""
        assert analyzer.is_available() == True
        
        # Test without API key
        analyzer_no_key = {AnalyzerClass}({})
        assert analyzer_no_key.is_available() == False
    
    def test_get_info(self, analyzer):
        """Test info retrieval"""
        info = analyzer.get_info()
        assert info['name'] == "{analyzer_name}"
        assert 'version' in info
        assert 'available' in info
```

## Integration Checklist

- [ ] Create analyzer file: `src/analyzers/{analyzer_name}.py`
- [ ] Add configuration to `config.yaml`
- [ ] Register in `src/analyzers/__init__.py`
- [ ] Create tests: `tests/test_{analyzer_name}.py`
- [ ] Update documentation
- [ ] Add to main.py analyzer selection
- [ ] Test via CLI: `python main.py --analyze "test" --analyzer {analyzer_name}`

## Configuration Addition

```yaml
# Add to config.yaml
analyzers:
  {analyzer_name}:
    enabled: true
    config:
      api_key_env: "{API_KEY_ENV_VAR}"
      temperature: 0.1
      timeout: 30
      model_version: "1.0"
```
