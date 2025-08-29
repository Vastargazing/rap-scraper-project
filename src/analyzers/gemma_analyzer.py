"""
Gemma AI Analyzer - основной анализатор на базе Google Gemma модели.

Реализует анализ текстов песен с использованием Gemma-2-27b-it модели
через Google Generative AI API.
"""

import json
import time
import logging
import os
from typing import Dict, Any, List, Optional
from datetime import datetime

from src.interfaces.analyzer_interface import BaseAnalyzer, AnalysisResult, register_analyzer

# Условный импорт для Google AI
try:
    import google.generativeai as genai
    HAS_GOOGLE_AI = True
except ImportError:
    HAS_GOOGLE_AI = False

logger = logging.getLogger(__name__)


@register_analyzer("gemma")
class GemmaAnalyzer(BaseAnalyzer):
    """
    Анализатор на базе Google Gemma модели.
    
    Основной AI анализатор для глубокого анализа текстов песен:
    - Жанровая классификация
    - Анализ настроения и эмоций
    - Оценка качества текстов
    - Определение тематики
    - Структурный анализ
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Инициализация Gemma анализатора"""
        super().__init__(config)
        
        # Настройки модели
        self.model_name = self.config.get('model_name', 'gemma-2-27b-it')
        self.temperature = self.config.get('temperature', 0.1)
        self.max_tokens = self.config.get('max_tokens', 1500)
        self.timeout = self.config.get('timeout', 30)
        
        # API ключ
        self.api_key = self.config.get('api_key') or os.getenv("GOOGLE_API_KEY")
        
        # Проверка доступности
        self.available = self._check_availability()
        
        if self.available:
            genai.configure(api_key=self.api_key)
            self.model = genai.GenerativeModel(self.model_name)
            logger.info(f"✅ Gemma анализатор инициализирован: {self.model_name}")
        else:
            logger.warning("⚠️ Gemma анализатор недоступен")
    
    def _check_availability(self) -> bool:
        """Проверка доступности Google AI API"""
        if not HAS_GOOGLE_AI:
            logger.error("❌ google-generativeai не установлен. Установите: pip install google-generativeai")
            return False
        
        if not self.api_key:
            logger.error("❌ GOOGLE_API_KEY не найден в конфигурации или переменных окружения")
            return False
        
        try:
            # Тестовое подключение
            genai.configure(api_key=self.api_key)
            # Простой тест API
            model = genai.GenerativeModel(self.model_name)
            response = model.generate_content(
                "Test",
                generation_config=genai.types.GenerationConfig(
                    temperature=0.1,
                    max_output_tokens=10,
                )
            )
            logger.info("✅ Google Gemma API успешно протестирован")
            return True
            
        except Exception as e:
            logger.error(f"❌ Ошибка проверки Gemma API: {e}")
            return False
    
    def analyze_song(self, artist: str, title: str, lyrics: str) -> AnalysisResult:
        """
        Анализ песни с использованием Gemma модели.
        
        Args:
            artist: Имя исполнителя
            title: Название песни
            lyrics: Текст песни
            
        Returns:
            AnalysisResult с результатами анализа
        """
        start_time = time.time()
        
        # Валидация входных данных
        if not self.validate_input(artist, title, lyrics):
            raise ValueError("Invalid input parameters")
        
        if not self.available:
            raise RuntimeError("Gemma analyzer is not available")
        
        # Предобработка текста
        processed_lyrics = self.preprocess_lyrics(lyrics)
        
        try:
            # Создание промпта
            prompt = self._create_analysis_prompt(artist, title, processed_lyrics)
            
            # Генерация ответа
            response = self.model.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=self.temperature,
                    max_output_tokens=self.max_tokens,
                )
            )
            
            if not response.text:
                raise RuntimeError("Empty response from Gemma model")
            
            # Парсинг результата
            analysis_data = self._parse_response(response.text)
            
            # Вычисление уверенности
            confidence = self._calculate_confidence(analysis_data)
            
            processing_time = time.time() - start_time
            
            return AnalysisResult(
                artist=artist,
                title=title,
                analysis_type="gemma",
                confidence=confidence,
                metadata={
                    "model_name": self.model_name,
                    "model_version": "gemma-2-27b-it",
                    "processing_date": datetime.now().isoformat(),
                    "lyrics_length": len(processed_lyrics),
                    "temperature": self.temperature,
                    "max_tokens": self.max_tokens
                },
                raw_output=analysis_data,
                processing_time=processing_time,
                timestamp=datetime.now().isoformat()
            )
            
        except Exception as e:
            logger.error(f"❌ Ошибка анализа Gemma для {artist} - {title}: {e}")
            raise RuntimeError(f"Gemma analysis failed: {e}") from e
    
    def _create_analysis_prompt(self, artist: str, title: str, lyrics: str) -> str:
        """Создание промпта для Gemma модели"""
        # Ограничиваем длину текста для API
        max_lyrics_length = 2000
        if len(lyrics) > max_lyrics_length:
            lyrics = lyrics[:max_lyrics_length] + "..."
        
        return f"""Analyze this rap song and return results in STRICT JSON format.

Artist: {artist}
Title: {title}
Lyrics: {lyrics}

Return ONLY valid JSON with these exact fields:
{{
    "genre_analysis": {{
        "primary_genre": "rap/trap/drill/old-school/gangsta/emo-rap/alternative-hip-hop",
        "subgenre": "string description",
        "confidence": 0.0-1.0
    }},
    "mood_analysis": {{
        "primary_mood": "aggressive/melancholic/energetic/confident/reflective/party/romantic",
        "emotional_intensity": "low/medium/high",
        "energy_level": "low/medium/high",
        "valence": "positive/negative/neutral"
    }},
    "content_analysis": {{
        "explicit_content": true/false,
        "explicit_level": "none/mild/moderate/high",
        "main_themes": ["money", "relationships", "street_life", "success", "struggle", "drugs", "violence", "family"],
        "narrative_style": "storytelling/boastful/confessional/abstract/conversational"
    }},
    "technical_analysis": {{
        "rhyme_scheme": "simple/complex/varied/internal",
        "flow_pattern": "steady/varied/aggressive/laid-back",
        "complexity_level": "beginner/intermediate/advanced/expert",
        "wordplay_quality": "basic/good/excellent/masterful",
        "metaphor_usage": "none/basic/moderate/heavy",
        "structure": "traditional/experimental/freestyle/mixed"
    }},
    "quality_metrics": {{
        "lyrical_creativity": 0.0-1.0,
        "technical_skill": 0.0-1.0,
        "authenticity": 0.0-1.0,
        "commercial_appeal": 0.0-1.0,
        "originality": 0.0-1.0,
        "overall_quality": 0.0-1.0,
        "ai_generated_likelihood": 0.0-1.0
    }},
    "cultural_context": {{
        "era_estimate": "1980s/1990s/2000s/2010s/2020s",
        "regional_style": "east_coast/west_coast/south/midwest/uk/international",
        "cultural_references": ["list", "of", "references"],
        "social_commentary": true/false
    }}
}}

Return ONLY the JSON object, no additional text or explanation!"""
    
    def _parse_response(self, response_text: str) -> Dict[str, Any]:
        """Парсинг ответа от Gemma модели"""
        try:
            # Поиск JSON в ответе
            json_start = response_text.find('{')
            json_end = response_text.rfind('}') + 1
            
            if json_start == -1 or json_end == 0:
                raise ValueError("No JSON found in response")
            
            json_str = response_text[json_start:json_end]
            
            # Парсинг JSON
            analysis_data = json.loads(json_str)
            
            # Валидация структуры
            self._validate_analysis_structure(analysis_data)
            
            return analysis_data
            
        except json.JSONDecodeError as e:
            logger.error(f"❌ Ошибка парсинга JSON: {e}")
            logger.error(f"Ответ модели: {response_text[:500]}...")
            raise ValueError(f"Invalid JSON response: {e}") from e
        
        except Exception as e:
            logger.error(f"❌ Ошибка обработки ответа: {e}")
            raise ValueError(f"Response parsing failed: {e}") from e
    
    def _validate_analysis_structure(self, data: Dict[str, Any]) -> None:
        """Валидация структуры результата анализа"""
        required_sections = [
            'genre_analysis', 'mood_analysis', 'content_analysis',
            'technical_analysis', 'quality_metrics', 'cultural_context'
        ]
        
        for section in required_sections:
            if section not in data:
                raise ValueError(f"Missing required section: {section}")
        
        # Проверка качественных метрик
        quality_metrics = data.get('quality_metrics', {})
        for metric in ['lyrical_creativity', 'technical_skill', 'authenticity', 
                      'commercial_appeal', 'originality', 'overall_quality']:
            if metric in quality_metrics:
                value = quality_metrics[metric]
                if not isinstance(value, (int, float)) or not 0 <= value <= 1:
                    logger.warning(f"Invalid metric value for {metric}: {value}")
    
    def _calculate_confidence(self, analysis_data: Dict[str, Any]) -> float:
        """Вычисление уверенности в результатах анализа"""
        confidence_factors = []
        
        # Проверка полноты анализа
        sections_completed = 0
        total_sections = 6
        
        for section_name in ['genre_analysis', 'mood_analysis', 'content_analysis',
                           'technical_analysis', 'quality_metrics', 'cultural_context']:
            if section_name in analysis_data and analysis_data[section_name]:
                sections_completed += 1
        
        completeness_score = sections_completed / total_sections
        confidence_factors.append(completeness_score)
        
        # Проверка уверенности в жанре
        genre_analysis = analysis_data.get('genre_analysis', {})
        if 'confidence' in genre_analysis:
            genre_confidence = genre_analysis['confidence']
            if isinstance(genre_confidence, (int, float)) and 0 <= genre_confidence <= 1:
                confidence_factors.append(genre_confidence)
        
        # Проверка качественных метрик
        quality_metrics = analysis_data.get('quality_metrics', {})
        if quality_metrics:
            # Средняя уверенность по метрикам
            valid_metrics = []
            for metric_value in quality_metrics.values():
                if isinstance(metric_value, (int, float)) and 0 <= metric_value <= 1:
                    valid_metrics.append(metric_value)
            
            if valid_metrics:
                avg_quality = sum(valid_metrics) / len(valid_metrics)
                confidence_factors.append(avg_quality)
        
        # Общая уверенность
        if confidence_factors:
            return sum(confidence_factors) / len(confidence_factors)
        else:
            return 0.5  # Средняя уверенность при отсутствии данных
    
    def get_analyzer_info(self) -> Dict[str, Any]:
        """Получение информации об анализаторе"""
        return {
            "name": "GemmaAnalyzer",
            "version": "2.0.0",
            "description": "AI-powered lyrics analysis using Google Gemma-2-27b-it model",
            "author": "Rap Scraper Project",
            "type": self.analyzer_type,
            "supported_features": self.supported_features,
            "model_info": {
                "model_name": self.model_name,
                "provider": "Google Generative AI",
                "temperature": self.temperature,
                "max_tokens": self.max_tokens
            },
            "requirements": ["google-generativeai", "GOOGLE_API_KEY"],
            "available": self.available,
            "config_options": {
                "model_name": "Gemma model to use (default: gemma-2-27b-it)",
                "temperature": "Generation temperature (default: 0.1)",
                "max_tokens": "Maximum output tokens (default: 1500)",
                "timeout": "Request timeout in seconds (default: 30)",
                "api_key": "Google API key (can use GOOGLE_API_KEY env var)"
            }
        }
    
    @property
    def analyzer_type(self) -> str:
        """Тип анализатора"""
        return "ai"
    
    @property
    def supported_features(self) -> List[str]:
        """Поддерживаемые функции анализа"""
        return [
            "genre_classification",
            "mood_analysis",
            "content_analysis", 
            "technical_analysis",
            "quality_assessment",
            "cultural_context",
            "authenticity_detection",
            "ai_generation_detection",
            "commercial_appeal",
            "lyrical_creativity"
        ]
    
    def preprocess_lyrics(self, lyrics: str) -> str:
        """Предобработка текста для Gemma модели"""
        # Базовая очистка
        lyrics = super().preprocess_lyrics(lyrics)
        
        # Удаление лишних символов, которые могут мешать анализу
        import re
        
        # Удаление повторяющихся символов
        lyrics = re.sub(r'(.)\1{3,}', r'\1\1\1', lyrics)
        
        # Нормализация пробелов
        lyrics = re.sub(r'\n{3,}', '\n\n', lyrics)
        
        # Удаление URL и специальных символов
        lyrics = re.sub(r'http[s]?://\S+', '', lyrics)
        lyrics = re.sub(r'[^\w\s\n.,!?\'"-]', '', lyrics)
        
        return lyrics.strip()
