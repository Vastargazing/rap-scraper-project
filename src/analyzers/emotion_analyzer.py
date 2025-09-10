#!/usr/bin/env python3
"""
🎯 Emotion Analyzer
Анализ эмоций в рэп-лирике с использованием Hugging Face transformers

НАЗНАЧЕНИЕ:
- Детекция множественных эмоций (радость, гнев, страх, грусть, удивление, любовь)
- Расширенный анализ настроения для рэп-текстов
- Интеграция с Hugging Face ecosystem

ИСПОЛЬЗОВАНИЕ:
python src/analyzers/emotion_analyzer.py --text "sample lyrics"
# Or via main.py interface

ЗАВИСИМОСТИ:
- Python 3.8+
- transformers >= 4.21.0
- torch >= 1.12.0
- src/interfaces/analyzer_interface.py

РЕЗУЛЬТАТ:
- AnalysisResult с детальными emotion scores
- Общий sentiment и confidence
- Интеграция с основным pipeline

АВТОР: Vastargazing | ДАТА: Сентябрь 2025
"""

import logging
import numpy as np
from typing import Dict, Any, Optional, List
from datetime import datetime
from dataclasses import dataclass

# Conditional imports для graceful degradation
try:
    import torch
    from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False

from interfaces.analyzer_interface import BaseAnalyzer, AnalysisResult, register_analyzer

# Внутренний класс для async результатов
@dataclass
class EmotionResult:
    analyzer_name: str
    sentiment: float
    confidence: float
    genre: str
    analysis_time: float
    metadata: Dict[str, Any]

logger = logging.getLogger(__name__)

@register_analyzer("emotion_analyzer")
class EmotionAnalyzer(BaseAnalyzer):
    """
    Анализатор эмоций на базе Hugging Face transformers
    
    Features:
    - Детекция 6 основных эмоций (joy, anger, fear, sadness, surprise, love)
    - Автоматический выбор устройства (GPU/CPU)
    - Batch processing для эффективности
    - Graceful fallback при отсутствии dependencies
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize emotion analyzer"""
        super().__init__(config)
        
        # Configuration
        self.model_name = self.config.get('model_name', 'j-hartmann/emotion-english-distilroberta-base')
        self.device = self.config.get('device', 'auto')
        self.max_length = self.config.get('max_length', 512)
        self.batch_size = self.config.get('batch_size', 16)
        
        # State
        self.model = None
        self.tokenizer = None
        self.classifier = None
        self._is_available = False
        
        # Initialize model
        self._initialize_model()
    
    def _initialize_model(self):
        """Initialize the Hugging Face model and tokenizer"""
        if not HAS_TRANSFORMERS:
            logger.warning("Transformers not available. Install with: pip install transformers torch")
            return
            
        try:
            logger.info(f"Loading emotion model: {self.model_name}")
            
            # Determine device
            if self.device == "auto":
                device = 0 if torch.cuda.is_available() else -1
            elif self.device == "cuda":
                device = 0 if torch.cuda.is_available() else -1
            else:
                device = -1  # CPU
            
            # Initialize pipeline
            self.classifier = pipeline(
                "text-classification",
                model=self.model_name,
                device=device,
                return_all_scores=True
            )
            
            self._is_available = True
            logger.info(f"Emotion analyzer initialized successfully on device: {'GPU' if device >= 0 else 'CPU'}")
            
        except Exception as e:
            logger.error(f"Failed to initialize emotion model: {e}")
            self._is_available = False
    
    def analyze_song(self, artist: str, title: str, lyrics: str) -> AnalysisResult:
        """
        Analyze a single song and return structured results.
        
        Args:
            artist: Artist name
            title: Song title  
            lyrics: Song lyrics text
            
        Returns:
            AnalysisResult with standardized output format
        """
        # Используем синхронную обертку для async метода
        import asyncio
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # Если уже в event loop, создаем новый task
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(asyncio.run, self._analyze_emotion(lyrics))
                    emotion_result = future.result()
            else:
                emotion_result = loop.run_until_complete(self._analyze_emotion(lyrics))
        except RuntimeError:
            # Если нет event loop, создаем новый
            emotion_result = asyncio.run(self._analyze_emotion(lyrics))
        
        # Адаптируем результат к старому формату AnalysisResult
        from datetime import datetime
        return AnalysisResult(
            artist=artist,
            title=title,
            analysis_type="emotion_analyzer",
            confidence=emotion_result.confidence,
            metadata={
                "analyzer_version": "1.0.0",
                "processing_date": datetime.now().isoformat(),
                "emotions": emotion_result.metadata.get('emotions', {}),
                "dominant_emotion": emotion_result.metadata.get('dominant_emotion', 'neutral'),
                "model_name": emotion_result.metadata.get('model_name', self.model_name),
                "fallback_mode": emotion_result.metadata.get('fallback_mode', False),
                "lyrics_length": len(lyrics),
                "sentiment_score": emotion_result.sentiment,
                "genre_prediction": emotion_result.genre
            },
            raw_output={
                "emotions": emotion_result.metadata.get('emotions', {}),
                "sentiment": emotion_result.sentiment,
                "genre": emotion_result.genre,
                "confidence": emotion_result.confidence,
                "analysis_time": emotion_result.analysis_time
            },
            processing_time=emotion_result.analysis_time,
            timestamp=datetime.now().isoformat()
        )

    async def _analyze_emotion(self, text: str, **kwargs) -> EmotionResult:
        """
        Analyze emotions in text
        
        Args:
            text: Input text to analyze
            **kwargs: Additional parameters
            
        Returns:
            AnalysisResult with emotion scores, sentiment, confidence
        """
        start_time = datetime.now()
        
        # Fallback if model not available
        if not self._is_available:
            return self._fallback_analysis(text, start_time)
        
        try:
            # Truncate text if too long
            if len(text) > self.max_length * 4:  # Rough token estimation
                text = text[:self.max_length * 4]
            
            # Get emotion predictions
            predictions = self.classifier(text)
            
            # Process results
            emotions = {}
            max_score = 0.0
            dominant_emotion = "neutral"
            
            for pred in predictions[0]:  # First (and only) text
                emotion_name = pred['label'].lower()
                score = pred['score']
                emotions[emotion_name] = score
                
                if score > max_score:
                    max_score = score
                    dominant_emotion = emotion_name
            
            # Calculate overall sentiment
            sentiment = self._calculate_sentiment(emotions)
            
            # Determine genre based on emotional patterns
            genre = self._determine_genre(emotions)
            
            analysis_time = (datetime.now() - start_time).total_seconds()
            
            return EmotionResult(
                analyzer_name=self.name,
                sentiment=sentiment,
                confidence=max_score,
                genre=genre,
                analysis_time=analysis_time,
                metadata={
                    'emotions': emotions,
                    'dominant_emotion': dominant_emotion,
                    'model_name': self.model_name,
                    'text_length': len(text),
                    'device': 'GPU' if torch.cuda.is_available() and self.device != 'cpu' else 'CPU'
                }
            )
            
        except Exception as e:
            logger.error(f"Emotion analysis failed: {e}")
            return self._fallback_analysis(text, start_time, error=str(e))
    
    def _calculate_sentiment(self, emotions: Dict[str, float]) -> float:
        """Calculate overall sentiment from emotion scores"""
        positive_emotions = ['joy', 'love', 'surprise']
        negative_emotions = ['anger', 'fear', 'sadness']
        
        positive_score = sum(emotions.get(emotion, 0.0) for emotion in positive_emotions)
        negative_score = sum(emotions.get(emotion, 0.0) for emotion in negative_emotions)
        
        # Normalize to 0-1 range
        total_score = positive_score + negative_score
        if total_score > 0:
            return positive_score / total_score
        else:
            return 0.5  # Neutral
    
    def _determine_genre(self, emotions: Dict[str, float]) -> str:
        """Determine genre based on emotional patterns"""
        anger_score = emotions.get('anger', 0.0)
        joy_score = emotions.get('joy', 0.0)
        love_score = emotions.get('love', 0.0)
        
        # Simple heuristics for genre classification
        if anger_score > 0.6:
            return "rap"  # Aggressive rap
        elif joy_score > 0.5 or love_score > 0.4:
            return "r&b"  # More melodic/positive
        elif anger_score > 0.3:
            return "hip-hop"  # General hip-hop
        else:
            return "other"
    
    def _fallback_analysis(self, text: str, start_time: datetime, error: str = None) -> EmotionResult:
        """Fallback analysis when model is unavailable"""
        # Simple keyword-based emotion detection
        emotion_keywords = {
            'joy': ['happy', 'joy', 'celebration', 'party', 'fun', 'good', 'great'],
            'anger': ['angry', 'mad', 'hate', 'damn', 'shit', 'fuck', 'rage'],
            'fear': ['scared', 'afraid', 'fear', 'worry', 'anxious'],
            'sadness': ['sad', 'cry', 'tear', 'pain', 'hurt', 'depression'],
            'love': ['love', 'heart', 'baby', 'girl', 'woman', 'kiss'],
            'surprise': ['wow', 'amazing', 'incredible', 'unbelievable']
        }
        
        text_lower = text.lower()
        emotions = {}
        
        for emotion, keywords in emotion_keywords.items():
            score = sum(1 for keyword in keywords if keyword in text_lower)
            emotions[emotion] = min(score / 10.0, 1.0)  # Normalize
        
        sentiment = self._calculate_sentiment(emotions)
        confidence = max(emotions.values()) if emotions else 0.0
        
        analysis_time = (datetime.now() - start_time).total_seconds()
        
        return EmotionResult(
            analyzer_name=self.name,
            sentiment=sentiment,
            confidence=confidence,
            genre=self._determine_genre(emotions),
            analysis_time=analysis_time,
            metadata={
                'emotions': emotions,
                'fallback_mode': True,
                'error': error,
                'text_length': len(text)
            }
        )
    
    async def batch_analyze(self, texts: List[str], **kwargs) -> List[EmotionResult]:
        """Batch analysis with optimized processing"""
        if not self._is_available:
            # Fallback to individual analysis
            results = []
            for text in texts:
                result = await self.analyze(text, **kwargs)
                results.append(result)
            return results
        
        # TODO: Implement true batch processing with transformers
        # For now, process individually but could be optimized
        results = []
        for text in texts:
            result = await self._analyze_emotion(text, **kwargs)
            results.append(result)
        
        return results
    
    def is_available(self) -> bool:
        """Check if analyzer is ready to use"""
        return HAS_TRANSFORMERS and self._is_available
    
    def get_info(self) -> Dict[str, Any]:
        """Get analyzer information"""
        return {
            'name': self.name,
            'version': '1.0.0',
            'description': 'Advanced emotion detection using Hugging Face transformers',
            'available': self.is_available(),
            'fallback_available': True,
            'model_name': self.model_name,
            'supported_emotions': ['joy', 'anger', 'fear', 'sadness', 'surprise', 'love'],
            'config': {
                'device': self.device,
                'max_length': self.max_length,
                'batch_size': self.batch_size
            },
            'requirements': [
                'transformers >= 4.21.0',
                'torch >= 1.12.0'
            ]
        }
    
    def get_analyzer_info(self) -> Dict[str, Any]:
        """Get analyzer metadata (required by BaseAnalyzer)"""
        return {
            "name": "EmotionAnalyzer",
            "version": "1.0.0", 
            "description": "Advanced emotion detection using Hugging Face transformers",
            "author": "Rap Scraper Project",
            "type": self.analyzer_type,
            "supported_features": self.supported_features,
            "model_name": self.model_name,
            "available": self.is_available(),
            "config_options": {
                "model_name": "Hugging Face model for emotion classification",
                "device": "Computation device (auto, cpu, cuda)",
                "max_length": "Maximum text length for processing",
                "batch_size": "Batch size for processing multiple texts"
            }
        }
    
    @property
    def analyzer_type(self) -> str:
        """Return analyzer type classification"""
        return "ai"
    
    @property  
    def supported_features(self) -> List[str]:
        """Return list of features this analyzer supports"""
        return [
            "emotion_detection",
            "sentiment_analysis",
            "genre_classification", 
            "confidence_scoring",
            "batch_processing",
            "fallback_mode"
        ]

# Example usage and testing
if __name__ == "__main__":
    import asyncio
    
    # Test configuration
    config = {
        'model_name': 'j-hartmann/emotion-english-distilroberta-base',
        'device': 'auto',
        'max_length': 512
    }
    
    analyzer = EmotionAnalyzer(config)
    
    # Test texts
    test_texts = [
        "I'm so happy today, life is beautiful and full of joy!",
        "I'm really angry about this situation, it makes me mad!",
        "I love you so much, you mean everything to me",
        "This makes me so sad, I could cry",
        "Yo, I'm the best rapper in the game, nobody can touch me!"
    ]
    
    async def test_analyzer():
        print(f"🎯 Testing Emotion Analyzer")
        print(f"Available: {analyzer.is_available()}")
        print(f"Info: {analyzer.get_info()}")
        print("-" * 60)
        
        for i, text in enumerate(test_texts, 1):
            print(f"\n{i}. Text: {text[:50]}...")
            result = await analyzer.analyze(text)
            print(f"   Sentiment: {result.sentiment:.3f}")
            print(f"   Confidence: {result.confidence:.3f}")
            print(f"   Genre: {result.genre}")
            if result.metadata and 'emotions' in result.metadata:
                emotions = result.metadata['emotions']
                top_emotions = sorted(emotions.items(), key=lambda x: x[1], reverse=True)[:3]
                print(f"   Top emotions: {top_emotions}")
    
    # Run test
    asyncio.run(test_analyzer())
