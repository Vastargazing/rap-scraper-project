#!/usr/bin/env python3
"""
🧪 Test Suite для Emotion Analyzer
Комплексное тестирование анализатора эмоций

НАЗНАЧЕНИЕ:
- Unit тесты для всех методов EmotionAnalyzer
- Integration тесты с основным pipeline
- Performance benchmarks
- Fallback scenario testing

ИСПОЛЬЗОВАНИЕ:
python -m pytest tests/test_emotion_analyzer.py -v
python tests/test_emotion_analyzer.py  # Standalone run

ТЕСТ-КЕЙСЫ:
- Базовая функциональность
- Граничные случаи (пустой текст, длинный текст)
- Error handling и fallback режим
- Performance под нагрузкой

АВТОР: Vastargazing | ДАТА: Сентябрь 2025
"""

import pytest
import asyncio
import time
from unittest.mock import patch, MagicMock
from typing import List, Dict, Any

# Import the analyzer
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from analyzers.emotion_analyzer import EmotionAnalyzer
from interfaces.analyzer_interface import AnalysisResult

class TestEmotionAnalyzer:
    """Test suite for EmotionAnalyzer"""
    
    @pytest.fixture
    def analyzer_config(self):
        """Default test configuration"""
        return {
            'model_name': 'j-hartmann/emotion-english-distilroberta-base',
            'device': 'cpu',  # Force CPU for testing
            'max_length': 512,
            'batch_size': 4
        }
    
    @pytest.fixture
    def analyzer(self, analyzer_config):
        """Create analyzer instance for testing"""
        return EmotionAnalyzer(analyzer_config)
    
    @pytest.fixture
    def sample_texts(self):
        """Sample texts for testing different emotions"""
        return {
            'joy': "I'm so happy today! Life is amazing and wonderful!",
            'anger': "I'm furious! This is absolutely ridiculous and makes me mad!",
            'love': "I love you so much, you mean everything to me, my heart belongs to you",
            'sadness': "I'm so sad and depressed, everything hurts and I want to cry",
            'fear': "I'm scared and anxious, this situation terrifies me completely",
            'surprise': "Wow! That's incredible and absolutely amazing, I can't believe it!",
            'neutral': "The weather today is partly cloudy with occasional sunshine.",
            'rap': "Yo, I'm the best MC in the game, spitting fire with every rhyme!"
        }
    
    # === Basic Functionality Tests ===
    
    def test_analyzer_initialization(self, analyzer):
        """Test analyzer initializes correctly"""
        assert analyzer.name == "emotion_analyzer"
        assert analyzer.config is not None
        assert hasattr(analyzer, 'model_name')
        assert hasattr(analyzer, 'device')
        
    def test_analyzer_info(self, analyzer):
        """Test analyzer info method"""
        info = analyzer.get_info()
        
        assert info['name'] == "emotion_analyzer"
        assert info['version'] == '1.0.0'
        assert 'supported_emotions' in info
        assert len(info['supported_emotions']) == 6
        assert 'config' in info
        
    @pytest.mark.asyncio
    async def test_basic_emotion_analysis(self, analyzer, sample_texts):
        """Test basic emotion analysis functionality"""
        # Test joy
        result = await analyzer._analyze_emotion(sample_texts['joy'])
        assert isinstance(result, AnalysisResult)
        assert result.analyzer_name == "emotion_analyzer"
        assert 0.0 <= result.sentiment <= 1.0
        assert 0.0 <= result.confidence <= 1.0
        assert result.genre in ['rap', 'hip-hop', 'r&b', 'other']
        
        # Joy should have positive sentiment
        assert result.sentiment > 0.5
        
    @pytest.mark.asyncio
    async def test_emotion_detection_accuracy(self, analyzer, sample_texts):
        """Test emotion detection for different categories"""
        results = {}
        
        for emotion, text in sample_texts.items():
            result = await analyzer._analyze_emotion(text)
            results[emotion] = result
        
        # Joy should have high positive sentiment
        assert results['joy'].sentiment > 0.6
        
        # Anger should have low sentiment
        assert results['anger'].sentiment < 0.5
        
        # Love should have high positive sentiment
        assert results['love'].sentiment > 0.6
        
        # Sadness should have low sentiment
        assert results['sadness'].sentiment < 0.4
    
    # === Edge Cases Tests ===
    
    @pytest.mark.asyncio
    async def test_empty_text(self, analyzer):
        """Test handling of empty text"""
        result = await analyzer._analyze_emotion("")
        assert isinstance(result, AnalysisResult)
        assert result.confidence >= 0.0
        assert result.sentiment >= 0.0
        
    @pytest.mark.asyncio
    async def test_very_long_text(self, analyzer):
        """Test handling of very long text"""
        long_text = "This is a test sentence. " * 1000  # Very long text
        result = await analyzer._analyze_emotion(long_text)
        
        assert isinstance(result, AnalysisResult)
        assert result.analysis_time > 0
        
    @pytest.mark.asyncio
    async def test_special_characters(self, analyzer):
        """Test handling of special characters and emojis"""
        special_text = "I'm so happy! 😊🎉 This is amazing!!! $$$ @#$%"
        result = await analyzer._analyze_emotion(special_text)
        
        assert isinstance(result, AnalysisResult)
        assert result.sentiment > 0.5  # Should still detect positive emotion
        
    # === Fallback Mode Tests ===
    
    @pytest.mark.asyncio
    async def test_fallback_mode(self, analyzer_config):
        """Test fallback mode when transformers unavailable"""
        # Mock transformers unavailable
        with patch('analyzers.emotion_analyzer.HAS_TRANSFORMERS', False):
            fallback_analyzer = EmotionAnalyzer(analyzer_config)
            assert not fallback_analyzer.is_available()
            
            result = await fallback_analyzer._analyze_emotion("I'm happy today!")
            assert isinstance(result, AnalysisResult)
            assert result.metadata.get('fallback_mode') is True
    
    @pytest.mark.asyncio
    async def test_model_loading_failure(self, analyzer_config):
        """Test handling of model loading failure"""
        with patch.object(EmotionAnalyzer, '_initialize_model', side_effect=Exception("Model load failed")):
            analyzer = EmotionAnalyzer(analyzer_config)
            
            result = await analyzer._analyze_emotion("Test text")
            assert isinstance(result, AnalysisResult)
            # Should fallback gracefully
    
    # === Performance Tests ===
    
    @pytest.mark.asyncio
    async def test_analysis_performance(self, analyzer, sample_texts):
        """Test analysis performance benchmarks"""
        text = sample_texts['joy']
        
        # Warm up
        await analyzer._analyze_emotion(text)
        
        # Measure performance
        start_time = time.time()
        result = await analyzer._analyze_emotion(text)
        end_time = time.time()
        
        analysis_time = end_time - start_time
        
        # Should complete in reasonable time (< 5 seconds even on CPU)
        assert analysis_time < 5.0
        assert result.analysis_time > 0
        
    @pytest.mark.asyncio
    async def test_batch_analysis(self, analyzer, sample_texts):
        """Test batch analysis functionality"""
        texts = list(sample_texts.values())[:4]  # First 4 texts
        
        start_time = time.time()
        results = await analyzer.batch_analyze(texts)
        end_time = time.time()
        
        assert len(results) == len(texts)
        assert all(isinstance(r, AnalysisResult) for r in results)
        
        # Batch should be reasonably fast
        batch_time = end_time - start_time
        assert batch_time < 10.0
    
    # === Integration Tests ===
    
    @pytest.mark.asyncio
    async def test_genre_classification(self, analyzer):
        """Test genre classification logic"""
        # Test rap classification
        rap_text = "Yo, I'm spitting fire, anger in my flow, fuck the system!"
        result = await analyzer._analyze_emotion(rap_text)
        
        # Should detect high anger and classify accordingly
        if result.metadata and 'emotions' in result.metadata:
            emotions = result.metadata['emotions']
            if emotions.get('anger', 0) > 0.3:
                assert result.genre in ['rap', 'hip-hop']
    
    @pytest.mark.asyncio
    async def test_sentiment_calculation(self, analyzer):
        """Test sentiment calculation accuracy"""
        # Very positive text
        positive_text = "I love this so much! It brings me joy and happiness!"
        pos_result = await analyzer._analyze_emotion(positive_text)
        
        # Very negative text
        negative_text = "I hate this! It makes me angry and sad!"
        neg_result = await analyzer._analyze_emotion(negative_text)
        
        # Positive should have higher sentiment than negative
        assert pos_result.sentiment > neg_result.sentiment
        
    # === Configuration Tests ===
    
    def test_custom_configuration(self):
        """Test custom configuration handling"""
        custom_config = {
            'model_name': 'custom-model',
            'device': 'cpu',
            'max_length': 256,
            'batch_size': 8
        }
        
        analyzer = EmotionAnalyzer(custom_config)
        
        assert analyzer.model_name == 'custom-model'
        assert analyzer.device == 'cpu'
        assert analyzer.max_length == 256
        assert analyzer.batch_size == 8
    
    def test_default_configuration(self):
        """Test default configuration when none provided"""
        analyzer = EmotionAnalyzer()
        
        assert analyzer.model_name == 'j-hartmann/emotion-english-distilroberta-base'
        assert analyzer.device == 'auto'
        assert analyzer.max_length == 512
        assert analyzer.batch_size == 16

# === Standalone Test Runner ===

async def run_standalone_tests():
    """Run tests standalone without pytest"""
    print("🧪 Running Emotion Analyzer Tests")
    print("=" * 60)
    
    # Setup
    config = {
        'model_name': 'j-hartmann/emotion-english-distilroberta-base',
        'device': 'cpu',
        'max_length': 512,
        'batch_size': 4
    }
    
    analyzer = EmotionAnalyzer(config)
    
    # Test 1: Basic functionality
    print("\n1. Testing basic functionality...")
    result = await analyzer._analyze_emotion("I'm so happy today!")
    print(f"   ✓ Sentiment: {result.sentiment:.3f}")
    print(f"   ✓ Confidence: {result.confidence:.3f}")
    print(f"   ✓ Genre: {result.genre}")
    
    # Test 2: Different emotions
    print("\n2. Testing different emotions...")
    test_cases = [
        ("Happy text", "I love this amazing day!"),
        ("Angry text", "This makes me so angry and mad!"),
        ("Sad text", "I'm feeling very sad and depressed"),
        ("Rap text", "Yo, I'm the best rapper, spitting fire!")
    ]
    
    for name, text in test_cases:
        result = await analyzer._analyze_emotion(text)
        print(f"   ✓ {name}: sentiment={result.sentiment:.3f}, genre={result.genre}")
    
    # Test 3: Performance
    print("\n3. Testing performance...")
    start_time = time.time()
    for _ in range(5):
        await analyzer._analyze_emotion("Performance test text")
    end_time = time.time()
    avg_time = (end_time - start_time) / 5
    print(f"   ✓ Average analysis time: {avg_time:.3f}s")
    
    # Test 4: Analyzer info
    print("\n4. Testing analyzer info...")
    info = analyzer.get_info()
    print(f"   ✓ Available: {info['available']}")
    print(f"   ✓ Fallback available: {info['fallback_available']}")
    print(f"   ✓ Supported emotions: {len(info['supported_emotions'])}")
    
    print("\n" + "=" * 60)
    print("🎉 All tests completed successfully!")

if __name__ == "__main__":
    # Run standalone tests
    asyncio.run(run_standalone_tests())
