"""
🦙 Ollama AI анализатор текстов песен (локальные модели)

НАЗНАЧЕНИЕ:
- Локальный AI-анализ текстов песен через Ollama
- Эксперименты и обучение без затрат на облачные API
- Поддержка разных моделей и приватная обработка

ИСПОЛЬЗОВАНИЕ:
Используется через main.py, batch_processor, analyzer_cli

ЗАВИСИМОСТИ:
- Python 3.8+
- src/interfaces/analyzer_interface.py
- Ollama (локальный сервер)

РЕЗУЛЬТАТ:
- Метрики: настроение, качество, структура, жанр
- Быстрый локальный анализ для тестов и обучения

АВТОР: AI Assistant
ДАТА: Сентябрь 2025
"""

import json
import logging
import time
from datetime import datetime
from typing import Any

import requests

from interfaces.analyzer_interface import (
    AnalysisResult,
    BaseAnalyzer,
    register_analyzer,
)

logger = logging.getLogger(__name__)


@register_analyzer("ollama")
class OllamaAnalyzer(BaseAnalyzer):
    """
    Анализатор на базе локальных моделей Ollama.

    Локальный AI анализатор для экспериментов и обучения:
    - Бесплатное использование
    - Локальная обработка (приватность)
    - Поддержка различных моделей
    - Экспериментальные возможности
    """

    def __init__(self, config: dict[str, Any] | None = None):
        """Инициализация Ollama анализатора"""
        super().__init__(config)

        # Настройки Ollama
        self.model_name = self.config.get("model_name", "llama3.2:3b")
        self.base_url = self.config.get("base_url", "http://localhost:11434")
        self.temperature = self.config.get("temperature", 0.1)
        self.timeout = self.config.get("timeout", 60)

        # Проверка доступности
        self.available = self._check_availability()

        if self.available:
            logger.info(f"✅ Ollama анализатор инициализирован: {self.model_name}")
        else:
            logger.warning("⚠️ Ollama анализатор недоступен")

    def _check_availability(self) -> bool:
        """Проверка доступности Ollama сервера"""
        try:
            # Проверка статуса сервера
            response = requests.get(
                f"{self.base_url}/api/tags",
                timeout=5,
                proxies={"http": "", "https": ""},
            )

            if response.status_code == 200:
                models = response.json().get("models", [])
                available_models = [model["name"] for model in models]
                logger.info(f"🦙 Ollama доступен. Модели: {available_models}")

                # Проверяем наличие нужной модели
                if any(self.model_name in model for model in available_models):
                    logger.info(f"✅ Модель {self.model_name} найдена")
                    return True
                logger.warning(f"⚠️ Модель {self.model_name} не найдена")
                # Пытаемся загрузить модель
                return self._pull_model()

            return False

        except requests.exceptions.RequestException as e:
            logger.warning(f"❌ Ollama недоступен: {e}")
            logger.info("💡 Убедитесь что Ollama запущен: ollama serve")
            return False

    def _pull_model(self) -> bool:
        """Автоматическая загрузка модели если её нет"""
        try:
            logger.info(f"📥 Загружаем модель {self.model_name}...")

            response = requests.post(
                f"{self.base_url}/api/pull",
                json={"name": self.model_name},
                timeout=300,  # 5 минут на загрузку
                proxies={"http": "", "https": ""},
            )

            if response.status_code == 200:
                logger.info(f"✅ Модель {self.model_name} успешно загружена")
                return True
            logger.error(f"❌ Не удалось загрузить модель: {response.text}")
            return False

        except Exception as e:
            logger.error(f"❌ Ошибка загрузки модели: {e}")
            return False

    def analyze_song(self, artist: str, title: str, lyrics: str) -> AnalysisResult:
        """
        Анализ песни с использованием Ollama модели.

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
            raise RuntimeError(
                "Ollama analyzer is not available. Make sure Ollama is running."
            )

        # Предобработка текста
        processed_lyrics = self.preprocess_lyrics(lyrics)

        try:
            # Создание промпта
            prompt = self._create_analysis_prompt(artist, title, processed_lyrics)

            # Отправка запроса к Ollama
            response = requests.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": self.model_name,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": self.temperature,
                        "top_p": 0.9,
                        "num_ctx": 4096,  # Контекстное окно
                        "num_predict": 1500,  # Максимум токенов ответа
                    },
                },
                timeout=self.timeout,
                proxies={"http": "", "https": ""},
            )

            if response.status_code != 200:
                raise RuntimeError(
                    f"Ollama request failed: {response.status_code} - {response.text}"
                )

            result = response.json()
            analysis_text = result.get("response", "")

            if not analysis_text:
                raise RuntimeError("Empty response from Ollama model")

            # Парсинг результата
            analysis_data = self._parse_response(analysis_text)

            # Вычисление уверенности
            confidence = self._calculate_confidence(analysis_data)

            processing_time = time.time() - start_time

            return AnalysisResult(
                artist=artist,
                title=title,
                analysis_type="ollama",
                confidence=confidence,
                metadata={
                    "model_name": self.model_name,
                    "base_url": self.base_url,
                    "processing_date": datetime.now().isoformat(),
                    "lyrics_length": len(processed_lyrics),
                    "temperature": self.temperature,
                    "timeout": self.timeout,
                },
                raw_output=analysis_data,
                processing_time=processing_time,
                timestamp=datetime.now().isoformat(),
            )

        except requests.exceptions.RequestException as e:
            logger.error(f"❌ Ошибка подключения к Ollama: {e}")
            raise RuntimeError(f"Ollama connection failed: {e}") from e

        except Exception as e:
            logger.error(f"❌ Ошибка анализа Ollama для {artist} - {title}: {e}")
            raise RuntimeError(f"Ollama analysis failed: {e}") from e

    def _create_analysis_prompt(self, artist: str, title: str, lyrics: str) -> str:
        """Создание промпта для Ollama модели"""
        # Ограничиваем длину для локальных моделей
        max_lyrics_length = 1500
        if len(lyrics) > max_lyrics_length:
            lyrics = lyrics[:max_lyrics_length] + "..."

        return f"""Analyze this rap song and return ONLY a valid JSON response with the analysis.

Artist: {artist}
Title: {title}
Lyrics: {lyrics}

Return ONLY valid JSON with this structure:
{{
    "basic_analysis": {{
        "genre": "rap/trap/drill/old-school/gangsta/emo-rap",
        "mood": "aggressive/melancholic/energetic/confident/neutral",
        "energy": "low/medium/high",
        "explicit": true/false
    }},
    "content_themes": {{
        "main_topics": ["money", "relationships", "street_life", "success", "struggle"],
        "narrative_style": "storytelling/boastful/confessional/abstract",
        "emotional_tone": "positive/negative/neutral/mixed"
    }},
    "technical_aspects": {{
        "rhyme_complexity": "simple/moderate/complex",
        "flow_style": "steady/varied/aggressive/laid-back",
        "wordplay_level": "basic/good/excellent",
        "structure_type": "traditional/experimental/freestyle"
    }},
    "quality_assessment": {{
        "lyrical_skill": 0.0-1.0,
        "creativity": 0.0-1.0,
        "authenticity": 0.0-1.0,
        "overall_quality": 0.0-1.0
    }},
    "experimental_features": {{
        "cultural_era": "1990s/2000s/2010s/2020s",
        "regional_style": "east_coast/west_coast/south/midwest/international",
        "influences": ["list", "of", "influences"],
        "innovation_level": 0.0-1.0
    }}
}}

Respond with ONLY the JSON object, no additional text!"""

    def _parse_response(self, response_text: str) -> dict[str, Any]:
        """Парсинг ответа от Ollama модели"""
        try:
            # Поиск JSON блока в ответе
            json_start = response_text.find("{")
            json_end = response_text.rfind("}") + 1

            if json_start == -1 or json_end == 0:
                raise ValueError("No JSON found in response")

            json_str = response_text[json_start:json_end]

            # Очистка возможных проблем с форматированием
            json_str = json_str.replace("\\n", "\\\\n")
            json_str = json_str.replace("\n", " ")

            # Парсинг JSON
            analysis_data = json.loads(json_str)

            # Валидация основной структуры
            self._validate_analysis_structure(analysis_data)

            return analysis_data

        except json.JSONDecodeError as e:
            logger.error(f"❌ Ошибка парсинга JSON от Ollama: {e}")
            logger.error(f"Ответ модели: {response_text[:300]}...")

            # Попытка извлечь хотя бы базовую информацию
            return self._extract_basic_info(response_text)

        except Exception as e:
            logger.error(f"❌ Ошибка обработки ответа Ollama: {e}")
            raise ValueError(f"Ollama response parsing failed: {e}") from e

    def _extract_basic_info(self, response_text: str) -> dict[str, Any]:
        """Извлечение базовой информации при ошибке парсинга JSON"""
        logger.warning("⚠️ Извлекаем базовую информацию из нестандартного ответа")

        # Простой анализ по ключевым словам
        text_lower = response_text.lower()

        # Определение жанра
        genre = "rap"  # По умолчанию
        if "trap" in text_lower:
            genre = "trap"
        elif "drill" in text_lower:
            genre = "drill"
        elif "old school" in text_lower or "old-school" in text_lower:
            genre = "old-school"

        # Определение настроения
        mood = "neutral"
        if any(word in text_lower for word in ["aggressive", "angry", "hard"]):
            mood = "aggressive"
        elif any(word in text_lower for word in ["sad", "melancholic", "depressed"]):
            mood = "melancholic"
        elif any(word in text_lower for word in ["energetic", "upbeat", "hype"]):
            mood = "energetic"
        elif any(word in text_lower for word in ["confident", "boastful"]):
            mood = "confident"

        return {
            "basic_analysis": {
                "genre": genre,
                "mood": mood,
                "energy": "medium",
                "explicit": "explicit" in text_lower or "profanity" in text_lower,
            },
            "content_themes": {
                "main_topics": ["general"],
                "narrative_style": "abstract",
                "emotional_tone": "neutral",
            },
            "technical_aspects": {
                "rhyme_complexity": "moderate",
                "flow_style": "steady",
                "wordplay_level": "basic",
                "structure_type": "traditional",
            },
            "quality_assessment": {
                "lyrical_skill": 0.5,
                "creativity": 0.5,
                "authenticity": 0.5,
                "overall_quality": 0.5,
            },
            "experimental_features": {
                "cultural_era": "2020s",
                "regional_style": "international",
                "influences": ["modern_rap"],
                "innovation_level": 0.5,
            },
            "_parsing_note": "Extracted from non-JSON response",
        }

    def _validate_analysis_structure(self, data: dict[str, Any]) -> None:
        """Валидация структуры результата анализа"""
        required_sections = [
            "basic_analysis",
            "content_themes",
            "technical_aspects",
            "quality_assessment",
            "experimental_features",
        ]

        for section in required_sections:
            if section not in data:
                logger.warning(f"⚠️ Отсутствует секция: {section}")

        # Проверка качественных метрик
        quality_assessment = data.get("quality_assessment", {})
        for metric in [
            "lyrical_skill",
            "creativity",
            "authenticity",
            "overall_quality",
        ]:
            if metric in quality_assessment:
                value = quality_assessment[metric]
                if not isinstance(value, (int, float)) or not 0 <= value <= 1:
                    logger.warning(f"⚠️ Некорректное значение метрики {metric}: {value}")

    def _calculate_confidence(self, analysis_data: dict[str, Any]) -> float:
        """Вычисление уверенности в результатах анализа"""
        confidence_factors = []

        # Проверка полноты анализа
        expected_sections = 5
        completed_sections = 0

        for section_name in [
            "basic_analysis",
            "content_themes",
            "technical_aspects",
            "quality_assessment",
            "experimental_features",
        ]:
            if analysis_data.get(section_name):
                completed_sections += 1

        completeness_score = completed_sections / expected_sections
        confidence_factors.append(completeness_score)

        # Проверка качественных метрик
        quality_assessment = analysis_data.get("quality_assessment", {})
        if quality_assessment:
            valid_metrics = []
            for metric_value in quality_assessment.values():
                if isinstance(metric_value, (int, float)) and 0 <= metric_value <= 1:
                    valid_metrics.append(metric_value)

            if valid_metrics:
                # Средняя уверенность по метрикам
                avg_quality = sum(valid_metrics) / len(valid_metrics)
                confidence_factors.append(avg_quality)

        # Штраф за нестандартный парсинг
        if "_parsing_note" in analysis_data:
            confidence_factors.append(0.3)  # Низкая уверенность

        # Общая уверенность
        if confidence_factors:
            base_confidence = sum(confidence_factors) / len(confidence_factors)
            # Дополнительный штраф для локальных моделей (они менее точны)
            return base_confidence * 0.8
        return 0.4  # Низкая уверенность по умолчанию

    def get_analyzer_info(self) -> dict[str, Any]:
        """Получение информации об анализаторе"""
        return {
            "name": "OllamaAnalyzer",
            "version": "2.0.0",
            "description": "Local AI analysis using Ollama models for experimentation and learning",
            "author": "Rap Scraper Project",
            "type": self.analyzer_type,
            "supported_features": self.supported_features,
            "model_info": {
                "model_name": self.model_name,
                "base_url": self.base_url,
                "provider": "Ollama Local",
                "temperature": self.temperature,
                "cost": "Free (local)",
            },
            "requirements": ["Ollama server running", "Model downloaded"],
            "available": self.available,
            "config_options": {
                "model_name": "Ollama model to use (default: llama3.2:3b)",
                "base_url": "Ollama server URL (default: http://localhost:11434)",
                "temperature": "Generation temperature (default: 0.1)",
                "timeout": "Request timeout in seconds (default: 60)",
            },
            "setup_instructions": [
                "1. Install Ollama from https://ollama.ai",
                "2. Run: ollama serve",
                "3. Pull model: ollama pull llama3.2:3b",
                "4. Start analysis",
            ],
        }

    @property
    def analyzer_type(self) -> str:
        """Тип анализатора"""
        return "ai"

    @property
    def supported_features(self) -> list[str]:
        """Поддерживаемые функции анализа"""
        return [
            "basic_classification",
            "mood_analysis",
            "content_analysis",
            "technical_analysis",
            "quality_assessment",
            "experimental_features",
            "local_processing",
            "privacy_friendly",
            "cost_free",
        ]
