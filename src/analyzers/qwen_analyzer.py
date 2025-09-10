"""
🤖 Qwen AI анализатор текстов песен (Novita AI)

НАЗНАЧЕНИЕ:
- Глубокий AI-анализ текстов песен с помощью Qwen-3-4B
- Классификация жанра, настроения, качества, тематики
- Использование облачного API Novita AI

ИСПОЛЬЗОВАНИЕ:
Используется через main.py, batch_processor, analyzer_cli

ЗАВИСИМОСТИ:
- Python 3.8+
- src/interfaces/analyzer_interface.py
- Novita AI/Qwen API ключи

РЕЗУЛЬТАТ:
- Метрики: жанр, настроение, качество, тематика, структура
- Глубокий анализ для production и исследований

АВТОР: AI Assistant
ДАТА: Сентябрь 2025
"""
import json
import time
import logging
import os
from typing import Dict, Any, List, Optional
from datetime import datetime

from interfaces.analyzer_interface import BaseAnalyzer, AnalysisResult, register_analyzer

# Условный импорт для OpenAI-совместимого клиента
try:
    import openai
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False

logger = logging.getLogger(__name__)


@register_analyzer("qwen")
class QwenAnalyzer(BaseAnalyzer):
    """
    Анализатор на базе Qwen модели через Novita AI API.
    
    Использует OpenAI-совместимый API для анализа текстов песен:
    - Жанровая классификация
    - Анализ настроения и эмоций
    - Оценка качества текстов
    - Определение тематики
    - Структурный анализ
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Инициализация Qwen анализатора"""
        super().__init__(config)
        
        # Настройки модели
        self.model_name = self.config.get('model_name', 'qwen/qwen3-4b-fp8')
        self.base_url = self.config.get('base_url', 'https://api.novita.ai/openai/v1')
        self.temperature = self.config.get('temperature', 0.1)
        self.max_tokens = self.config.get('max_tokens', 1500)
        self.timeout = self.config.get('timeout', 30)
        
        # API ключ
        self.api_key = self.config.get('api_key') or os.getenv("NOVITA_API_KEY")
        
        # Проверка доступности
        self.available = self._check_availability()
        
        if self.available:
            self.client = openai.OpenAI(
                api_key=self.api_key,
                base_url=self.base_url,
                timeout=self.timeout
            )
            logger.info(f"✅ Qwen анализатор инициализирован: {self.model_name}")
        else:
            logger.warning("⚠️ Qwen анализатор недоступен")
    
    def _check_availability(self) -> bool:
        """Проверка доступности Novita AI API"""
        if not HAS_OPENAI:
            logger.error("❌ openai не установлен. Установите: pip install openai")
            return False
        
        if not self.api_key:
            logger.error("❌ NOVITA_API_KEY не найден в конфигурации или переменных окружения")
            return False
        
        try:
            # Тестовое подключение
            client = openai.OpenAI(
                api_key=self.api_key,
                base_url=self.base_url
            )
            
            # Простой тест API
            response = client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": "Test"}
                ],
                max_tokens=10,
                temperature=0.1
            )
            
            if response.choices and response.choices[0].message:
                logger.info("✅ Novita AI Qwen API успешно протестирован")
                return True
            else:
                logger.error("❌ Получен пустой ответ от Qwen API")
                return False
            
        except Exception as e:
            logger.error(f"❌ Ошибка проверки Qwen API: {e}")
            return False
    
    def analyze_song(self, artist: str, title: str, lyrics: str) -> AnalysisResult:
        """
        Анализ песни с использованием Qwen модели.
        
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
            raise RuntimeError("Qwen analyzer is not available")
        
        # Предобработка текста
        processed_lyrics = self.preprocess_lyrics(lyrics)
        
        try:
            # Создание промпта
            system_prompt, user_prompt = self._create_analysis_prompts(artist, title, processed_lyrics)
            
            # Генерация ответа
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                max_tokens=self.max_tokens,
                temperature=self.temperature
            )
            
            if not response.choices or not response.choices[0].message.content:
                raise RuntimeError("Empty response from Qwen model")
            
            # Парсинг результата
            analysis_data = self._parse_response(response.choices[0].message.content)
            
            # Вычисление уверенности
            confidence = self._calculate_confidence(analysis_data)
            
            processing_time = time.time() - start_time
            
            return AnalysisResult(
                artist=artist,
                title=title,
                analysis_type="qwen",
                confidence=confidence,
                metadata={
                    "model_name": self.model_name,
                    "model_version": "qwen3-4b-fp8",
                    "processing_date": datetime.now().isoformat(),
                    "lyrics_length": len(processed_lyrics),
                    "temperature": self.temperature,
                    "max_tokens": self.max_tokens,
                    "provider": "Novita AI",
                    "usage": {
                        "prompt_tokens": response.usage.prompt_tokens if response.usage else 0,
                        "completion_tokens": response.usage.completion_tokens if response.usage else 0,
                        "total_tokens": response.usage.total_tokens if response.usage else 0
                    }
                },
                raw_output=analysis_data,
                processing_time=processing_time,
                timestamp=datetime.now().isoformat()
            )
            
        except Exception as e:
            logger.error(f"❌ Ошибка анализа Qwen для {artist} - {title}: {e}")
            raise RuntimeError(f"Qwen analysis failed: {e}") from e
    
    def _create_analysis_prompts(self, artist: str, title: str, lyrics: str) -> tuple[str, str]:
        """Создание системного и пользовательского промптов для Qwen модели"""
        # Ограничиваем длину текста для API
        max_lyrics_length = 2000
        if len(lyrics) > max_lyrics_length:
            lyrics = lyrics[:max_lyrics_length] + "..."
        
        system_prompt = """You are a rap lyrics analyzer. You MUST respond with ONLY a JSON object, no other text.

CRITICAL: Do not include ANY explanations, thoughts, or text outside the JSON. 
NO <think> tags, NO explanations, NO additional text.
Start your response with { and end with }.

Analyze rap songs and return JSON with this structure only."""
        
        user_prompt = f"""Artist: {artist}
Title: {title}
Lyrics: {lyrics}

Return ONLY this JSON structure (fill with actual analysis):
{{
    "genre_analysis": {{
        "primary_genre": "rap",
        "subgenre": "string",
        "confidence": 0.9
    }},
    "mood_analysis": {{
        "primary_mood": "confident",
        "emotional_intensity": "high",
        "energy_level": "high", 
        "valence": "positive"
    }},
    "content_analysis": {{
        "explicit_content": false,
        "explicit_level": "none",
        "main_themes": ["success"],
        "narrative_style": "boastful"
    }},
    "technical_analysis": {{
        "rhyme_scheme": "complex",
        "flow_pattern": "varied",
        "complexity_level": "advanced",
        "wordplay_quality": "excellent",
        "metaphor_usage": "moderate",
        "structure": "traditional"
    }},
    "quality_metrics": {{
        "lyrical_creativity": 0.8,
        "technical_skill": 0.9,
        "authenticity": 0.9,
        "commercial_appeal": 0.8,
        "originality": 0.8,
        "overall_quality": 0.8,
        "ai_generated_likelihood": 0.1
    }},
    "cultural_context": {{
        "era_estimate": "2020s",
        "regional_style": "mainstream",
        "cultural_references": [],
        "social_commentary": false
    }}
}}"""
        
        return system_prompt, user_prompt
    
    def _parse_response(self, response_text: str) -> Dict[str, Any]:
        """Парсинг ответа от Qwen модели"""
        try:
            # Очистка от лишнего текста
            response_text = response_text.strip()
            
            # Удаляем теги <think>...</think> 
            import re
            response_text = re.sub(r'<think>.*?</think>', '', response_text, flags=re.DOTALL)
            response_text = response_text.strip()
            
            # Логируем сырой ответ для отладки
            logger.debug(f"Cleaned response (first 500 chars): {response_text[:500]}")
            
            # Если ответ уже JSON, парсим напрямую
            if response_text.startswith('{') and response_text.endswith('}'):
                try:
                    analysis_data = json.loads(response_text)
                    logger.debug("✅ Direct JSON parsing successful")
                except json.JSONDecodeError as e:
                    logger.warning(f"Direct JSON parsing failed: {e}")
                    # Попробуем исправить и парсить снова
                    fixed_json = self._fix_common_json_issues(response_text)
                    try:
                        analysis_data = json.loads(fixed_json)
                        logger.debug("✅ JSON parsing successful after fixes")
                    except json.JSONDecodeError:
                        logger.error(f"JSON still invalid after fixes")
                        analysis_data = self._create_fallback_analysis()
            else:
                # Поиск JSON блока в тексте
                json_start = response_text.find('{')
                json_end = response_text.rfind('}') + 1
                
                if json_start == -1 or json_end == 0:
                    logger.error(f"No JSON found in response. Full response: {response_text}")
                    # Попробуем создать базовый ответ
                    analysis_data = self._create_fallback_analysis()
                else:
                    json_str = response_text[json_start:json_end]
                    try:
                        analysis_data = json.loads(json_str)
                    except json.JSONDecodeError:
                        logger.error(f"Invalid JSON in response: {json_str}")
                        analysis_data = self._create_fallback_analysis()
            
            # Валидация структуры
            self._validate_analysis_structure(analysis_data)
            
            return analysis_data
            
        except json.JSONDecodeError as e:
            logger.error(f"❌ Ошибка парсинга JSON: {e}")
            logger.error(f"Ответ модели: {response_text[:500]}...")
            return self._create_fallback_analysis()
        
        except Exception as e:
            logger.error(f"❌ Ошибка обработки ответа: {e}")
            return self._create_fallback_analysis()
    
    def _fix_common_json_issues(self, json_str: str) -> str:
        """Исправляет распространенные проблемы с JSON"""
        import re
        
        # Удаляем trailing commas
        json_str = re.sub(r',(\s*[}\]])', r'\1', json_str)
        
        # Исправляем одинарные кавычки на двойные (если есть)
        json_str = re.sub(r"'([^']*)':", r'"\1":', json_str)
        
        # Убираем лишние символы в конце
        json_str = json_str.strip()
        
        return json_str
    
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
            "name": "QwenAnalyzer",
            "version": "1.0.0",
            "description": "AI-powered lyrics analysis using Qwen-3-4B model via Novita AI",
            "author": "Rap Scraper Project",
            "type": self.analyzer_type,
            "supported_features": self.supported_features,
            "model_info": {
                "model_name": self.model_name,
                "provider": "Novita AI",
                "base_url": self.base_url,
                "temperature": self.temperature,
                "max_tokens": self.max_tokens
            },
            "requirements": ["openai", "NOVITA_API_KEY"],
            "available": self.available,
            "config_options": {
                "model_name": "Qwen model to use (default: qwen/qwen3-4b-fp8)",
                "base_url": "API base URL (default: https://api.novita.ai/openai/v1)",
                "temperature": "Generation temperature (default: 0.1)",
                "max_tokens": "Maximum output tokens (default: 1500)",
                "timeout": "Request timeout in seconds (default: 30)",
                "api_key": "Novita API key (can use NOVITA_API_KEY env var)"
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
        """Предобработка текста для Qwen модели"""
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
    
    def _create_fallback_analysis(self) -> Dict[str, Any]:
        """Создает базовый анализ в случае ошибки парсинга ответа"""
        return {
            "genre_analysis": {
                "primary_genre": "unknown",
                "subgenre": "unknown",
                "confidence": 0.0
            },
            "mood_analysis": {
                "primary_mood": "neutral",
                "emotional_intensity": "unknown",
                "energy_level": "unknown",
                "valence": "neutral"
            },
            "content_analysis": {
                "explicit_content": False,
                "explicit_level": "unknown",
                "main_themes": [],
                "narrative_style": "unknown"
            },
            "technical_analysis": {
                "rhyme_scheme": "unknown",
                "flow_pattern": "unknown",
                "complexity_level": "unknown",
                "wordplay_quality": "unknown",
                "metaphor_usage": "unknown",
                "structure": "unknown"
            },
            "quality_metrics": {
                "lyrical_creativity": 0.0,
                "technical_skill": 0.0,
                "authenticity": 0.0,
                "commercial_appeal": 0.0,
                "originality": 0.0,
                "overall_quality": 0.0,
                "ai_generated_likelihood": 0.0
            },
            "cultural_context": {
                "era_estimate": "unknown",
                "regional_style": "unknown",
                "cultural_references": [],
                "social_commentary": False
            }
        }


# Принудительная регистрация анализатора (если декоратор не сработал)
try:
    from interfaces.analyzer_interface import AnalyzerFactory
    AnalyzerFactory.register("qwen", QwenAnalyzer)
except ImportError:
    pass
