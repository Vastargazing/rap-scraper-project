# TODO(code-review): Remove emoji from module docstring (Google style guide)
# TODO(code-review): Add proper Google-style docstring with sections:
#   - Module description
#   - Typical usage example
#   - Attributes (if any module-level)
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

# TODO(code-review): Sort imports alphabetically within groups (stdlib, third-party, local)
# TODO(code-review): Add 'from typing import Optional' for better type safety
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

# TODO(code-review): Extract magic numbers to module-level constants with descriptive names
# Example: DEFAULT_MODEL_NAME, DEFAULT_BASE_URL, DEFAULT_TIMEOUT, etc.

# TODO(code-review): Add constants for retry logic, timeouts, and other configuration
# Example:
# _DEFAULT_MODEL = "llama3.2:3b"
# _DEFAULT_BASE_URL = "http://localhost:11434"
# _DEFAULT_TEMPERATURE = 0.1
# _DEFAULT_TIMEOUT = 60
# _MAX_LYRICS_LENGTH = 1500
# _MODEL_PULL_TIMEOUT = 300
# _AVAILABILITY_CHECK_TIMEOUT = 5
# _CONFIDENCE_PENALTY_FOR_LOCAL_MODELS = 0.8
# _LOW_CONFIDENCE_FALLBACK = 0.4

logger = logging.getLogger(__name__)


# TODO(code-review): Add type hint for logger (logging.Logger)
# logger: logging.Logger = logging.getLogger(__name__)

@register_analyzer("ollama")
class OllamaAnalyzer(BaseAnalyzer):
    # TODO(code-review): Improve class docstring to follow Google style guide format:
    #   - One-line summary
    #   - Detailed description (optional)
    #   - Attributes section listing all class attributes
    #   - Example usage section
    """
    Анализатор на базе локальных моделей Ollama.

    Локальный AI анализатор для экспериментов и обучения:
    - Бесплатное использование
    - Локальная обработка (приватность)
    - Поддержка различных моделей
    - Экспериментальные возможности
    """

    # TODO(code-review): Add type hints for all instance attributes at class level
    # model_name: str
    # base_url: str
    # temperature: float
    # timeout: int
    # available: bool

    def __init__(self, config: dict[str, Any] | None = None):
        # TODO(code-review): Expand docstring with Google style:
        #   Args:
        #       config: Configuration dictionary with optional keys...
        #   Raises:
        #       ConnectionError: If Ollama server is not accessible
        """Инициализация Ollama анализатора"""
        super().__init__(config)

        # TODO(code-review): Use module-level constants instead of magic values
        # TODO(code-review): Add type casting for config values to ensure type safety
        # TODO(code-review): Validate config values (e.g., temperature should be 0-2)
        # Настройки Ollama
        self.model_name = self.config.get("model_name", "llama3.2:3b")
        self.base_url = self.config.get("base_url", "http://localhost:11434")
        self.temperature = self.config.get("temperature", 0.1)
        self.timeout = self.config.get("timeout", 60)

        # Проверка доступности
        self.available = self._check_availability()

        # TODO(code-review): Remove emoji from log messages (not production-ready)
        # TODO(code-review): Use structured logging with extra fields instead of f-strings
        # Example: logger.info("Ollama analyzer initialized", extra={"model": self.model_name})
        if self.available:
            logger.info(f"✅ Ollama анализатор инициализирован: {self.model_name}")
        else:
            logger.warning("⚠️ Ollama анализатор недоступен")

    def _check_availability(self) -> bool:
        # TODO(code-review): Add proper Google-style docstring:
        #   Returns:
        #       bool: True if Ollama server is available and model is ready
        #   Raises:
        #       (Document any exceptions or note that they're caught internally)
        """Проверка доступности Ollama сервера"""
        try:
            # TODO(code-review): Extract "/api/tags" to a constant (e.g., _API_TAGS_ENDPOINT)
            # TODO(code-review): Extract timeout (5) to constant (_AVAILABILITY_CHECK_TIMEOUT)
            # TODO(code-review): Extract proxies dict to a constant or helper method
            # TODO(code-review): Add comment explaining why proxies are disabled
            # Проверка статуса сервера
            response = requests.get(
                f"{self.base_url}/api/tags",
                timeout=5,
                proxies={"http": "", "https": ""},
            )

            # TODO(code-review): Use response.raise_for_status() for better error handling
            # TODO(code-review): Add explicit check for response.ok or status code range
            if response.status_code == 200:
                models = response.json().get("models", [])
                # TODO(code-review): Add type hint and null check for models list
                # TODO(code-review): Handle case where "name" key might be missing
                available_models = [model["name"] for model in models]
                # TODO(code-review): Remove emoji from logs
                logger.info(f"🦙 Ollama доступен. Модели: {available_models}")

                # TODO(code-review): Use exact match instead of 'in' for model checking
                # Current logic: "llama" in "llama3.2" would match incorrectly
                # Проверяем наличие нужной модели
                if any(self.model_name in model for model in available_models):
                    logger.info(f"✅ Модель {self.model_name} найдена")
                    return True
                logger.warning(f"⚠️ Модель {self.model_name} не найдена")
                # Пытаемся загрузить модель
                return self._pull_model()

            # TODO(code-review): Log the actual status code for debugging
            return False

        # TODO(code-review): Catch more specific exceptions (ConnectionError, Timeout, etc.)
        # TODO(code-review): Consider retrying with exponential backoff
        except requests.exceptions.RequestException as e:
            # TODO(code-review): Remove emoji from logs
            logger.warning(f"❌ Ollama недоступен: {e}")
            logger.info("💡 Убедитесь что Ollama запущен: ollama serve")
            return False

    def _pull_model(self) -> bool:
        # TODO(code-review): Add proper Google-style docstring with Returns section
        """Автоматическая загрузка модели если её нет"""
        try:
            # TODO(code-review): Remove emoji from logs
            logger.info(f"📥 Загружаем модель {self.model_name}...")

            # TODO(code-review): Extract "/api/pull" to constant
            # TODO(code-review): Extract timeout (300) to constant (_MODEL_PULL_TIMEOUT)
            # TODO(code-review): Extract proxies to constant/helper
            # TODO(code-review): Add progress indication for long-running operation
            # TODO(code-review): Consider using streaming response to show download progress
            response = requests.post(
                f"{self.base_url}/api/pull",
                json={"name": self.model_name},
                timeout=300,  # 5 минут на загрузку
                proxies={"http": "", "https": ""},
            )

            # TODO(code-review): Use response.raise_for_status() or check response.ok
            if response.status_code == 200:
                # TODO(code-review): Remove emoji
                logger.info(f"✅ Модель {self.model_name} успешно загружена")
                return True
            # TODO(code-review): Don't log full response.text (might be large), log status code
            # TODO(code-review): Remove emoji
            logger.error(f"❌ Не удалось загрузить модель: {response.text}")
            return False

        # TODO(code-review): Never catch bare Exception - use specific exceptions
        # TODO(code-review): Should be: except requests.exceptions.RequestException
        except Exception as e:
            # TODO(code-review): Remove emoji, add more context to error message
            logger.error(f"❌ Ошибка загрузки модели: {e}")
            return False

    def analyze_song(self, artist: str, title: str, lyrics: str) -> AnalysisResult:
        # TODO(code-review): Add Raises section to docstring:
        #   Raises:
        #       ValueError: If input parameters are invalid
        #       RuntimeError: If Ollama analyzer is unavailable or request fails
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

        # TODO(code-review): Add input sanitization before validation
        # TODO(code-review): Validate individual parameters (check for empty strings, None, etc.)
        # Валидация входных данных
        if not self.validate_input(artist, title, lyrics):
            # TODO(code-review): Provide more specific error message about what's invalid
            raise ValueError("Invalid input parameters")

        # TODO(code-review): This check should be done in __init__ or raise specific exception
        if not self.available:
            # TODO(code-review): Consider custom exception class (OllamaUnavailableError)
            raise RuntimeError(
                "Ollama analyzer is not available. Make sure Ollama is running."
            )

        # Предобработка текста
        processed_lyrics = self.preprocess_lyrics(lyrics)

        try:
            # Создание промпта
            prompt = self._create_analysis_prompt(artist, title, processed_lyrics)

            # TODO(code-review): Extract "/api/generate" to constant
            # TODO(code-review): Extract all magic numbers to constants:
            #   - top_p (0.9) -> _DEFAULT_TOP_P
            #   - num_ctx (4096) -> _DEFAULT_CONTEXT_WINDOW
            #   - num_predict (1500) -> _MAX_TOKENS_RESPONSE
            # TODO(code-review): Extract proxies dict to constant/helper
            # TODO(code-review): Consider extracting request payload to separate method for testing
            # TODO(code-review): Add retry logic with exponential backoff for transient failures
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

            # TODO(code-review): Use response.raise_for_status() instead of manual check
            # TODO(code-review): Don't include full response.text in error (might be large/sensitive)
            if response.status_code != 200:
                raise RuntimeError(
                    f"Ollama request failed: {response.status_code} - {response.text}"
                )

            # TODO(code-review): Add error handling for invalid JSON response
            # TODO(code-review): Validate response structure before accessing fields
            result = response.json()
            analysis_text = result.get("response", "")

            # TODO(code-review): Check for whitespace-only responses, not just empty
            if not analysis_text:
                raise RuntimeError("Empty response from Ollama model")

            # Парсинг результата
            analysis_data = self._parse_response(analysis_text)

            # Вычисление уверенности
            confidence = self._calculate_confidence(analysis_data)

            processing_time = time.time() - start_time

            # TODO(code-review): Extract "ollama" string to constant or use class attribute
            # TODO(code-review): Don't call datetime.now() twice - reuse timestamp
            # TODO(code-review): Consider adding request_id for tracking/debugging
            # TODO(code-review): Validate confidence is in range [0, 1]
            # TODO(code-review): Don't include base_url in metadata (might be sensitive)
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

        # TODO(code-review): Separate different exception types and handle specifically
        # TODO(code-review): Add retry logic for ConnectionError, Timeout exceptions
        except requests.exceptions.RequestException as e:
            # TODO(code-review): Remove emoji from error logs
            # TODO(code-review): Add structured logging with extra fields
            logger.error(f"❌ Ошибка подключения к Ollama: {e}")
            raise RuntimeError(f"Ollama connection failed: {e}") from e

        # TODO(code-review): NEVER catch bare Exception - specify exact exception types
        # TODO(code-review): This catches ValueError, RuntimeError, etc. - handle each separately
        except Exception as e:
            # TODO(code-review): Remove emoji, don't log PII (artist/title might be sensitive)
            logger.error(f"❌ Ошибка анализа Ollama для {artist} - {title}: {e}")
            raise RuntimeError(f"Ollama analysis failed: {e}") from e

    def _create_analysis_prompt(self, artist: str, title: str, lyrics: str) -> str:
        # TODO(code-review): Add Google-style docstring with Args and Returns sections
        # TODO(code-review): Add sanitization for artist/title to prevent prompt injection
        """Создание промпта для Ollama модели"""
        # TODO(code-review): Extract 1500 to constant (_MAX_LYRICS_LENGTH)
        # TODO(code-review): Use textwrap.shorten() or smart truncation at word boundaries
        # TODO(code-review): Add logging when lyrics are truncated
        # Ограничиваем длину для локальных моделей
        max_lyrics_length = 1500
        if len(lyrics) > max_lyrics_length:
            lyrics = lyrics[:max_lyrics_length] + "..."

        # TODO(code-review): Extract prompt template to constant or separate file
        # TODO(code-review): Use a template engine (e.g., jinja2) for complex prompts
        # TODO(code-review): Validate that f-strings don't break JSON structure
        # TODO(code-review): Consider using triple-quoted strings with proper indentation
        # TODO(code-review): This method is too long (>50 lines) - consider splitting
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
        # TODO(code-review): Add validation that returned string is within model's context limit
        # TODO(code-review): Consider using Pydantic models for the expected JSON schema

    def _parse_response(self, response_text: str) -> dict[str, Any]:
        # TODO(code-review): Add Google-style docstring with Args, Returns, Raises sections
        # TODO(code-review): Add type hint: -> Dict[str, Any] or use TypedDict for structure
        """Парсинг ответа от Ollama модели"""
        try:
            # TODO(code-review): Use regex for more robust JSON extraction
            # TODO(code-review): Handle nested braces correctly (current logic might fail)
            # Поиск JSON блока в ответе
            json_start = response_text.find("{")
            json_end = response_text.rfind("}") + 1

            if json_start == -1 or json_end == 0:
                raise ValueError("No JSON found in response")

            json_str = response_text[json_start:json_end]

            # TODO(code-review): This string replacement is fragile and might break valid JSON
            # TODO(code-review): Use json.loads directly and catch errors instead of pre-processing
            # TODO(code-review): Document why these replacements are necessary
            # Очистка возможных проблем с форматированием
            json_str = json_str.replace("\\n", "\\\\n")
            json_str = json_str.replace("\n", " ")

            # Парсинг JSON
            analysis_data = json.loads(json_str)

            # Валидация основной структуры
            self._validate_analysis_structure(analysis_data)

            return analysis_data

        # TODO(code-review): Good - specific exception handling
        except json.JSONDecodeError as e:
            # TODO(code-review): Remove emoji from logs
            # TODO(code-review): Extract 300 to constant
            # TODO(code-review): Use logger.exception() to include traceback
            logger.error(f"❌ Ошибка парсинга JSON от Ollama: {e}")
            logger.error(f"Ответ модели: {response_text[:300]}...")

            # TODO(code-review): Add comment explaining why we fall back to basic extraction
            # Попытка извлечь хотя бы базовую информацию
            return self._extract_basic_info(response_text)

        # TODO(code-review): NEVER catch bare Exception - this is too broad
        # TODO(code-review): Specify exact exception types (ValueError, KeyError, etc.)
        except Exception as e:
            # TODO(code-review): Remove emoji
            logger.error(f"❌ Ошибка обработки ответа Ollama: {e}")
            raise ValueError(f"Ollama response parsing failed: {e}") from e

    def _extract_basic_info(self, response_text: str) -> dict[str, Any]:
        # TODO(code-review): Add Google-style docstring with Args, Returns sections
        # TODO(code-review): Add note that this is a fallback method with lower accuracy
        """Извлечение базовой информации при ошибке парсинга JSON"""
        # TODO(code-review): Remove emoji from logs
        logger.warning("⚠️ Извлекаем базовую информацию из нестандартного ответа")

        # TODO(code-review): This is a fragile keyword-based approach - document limitations
        # TODO(code-review): Consider using regex patterns instead of simple 'in' checks
        # Простой анализ по ключевым словам
        text_lower = response_text.lower()

        # TODO(code-review): Extract genre keywords to constants or configuration
        # TODO(code-review): Use elif properly to avoid redundant checks
        # Определение жанра
        genre = "rap"  # По умолчанию
        if "trap" in text_lower:
            genre = "trap"
        elif "drill" in text_lower:
            genre = "drill"
        elif "old school" in text_lower or "old-school" in text_lower:
            genre = "old-school"

        # TODO(code-review): Extract mood keywords to constants
        # TODO(code-review): This could match unrelated words - use word boundaries
        # TODO(code-review): Consider using a scoring system instead of first-match
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

        # TODO(code-review): Extract this default structure to a constant or factory method
        # TODO(code-review): Use dataclass or Pydantic model for type safety
        return {
            "basic_analysis": {
                "genre": genre,
                "mood": mood,
                # TODO(code-review): Extract "medium" to constant
                "energy": "medium",
                # TODO(code-review): This boolean logic might match "not explicit" as True
                "explicit": "explicit" in text_lower or "profanity" in text_lower,
            },
            "content_themes": {
                # TODO(code-review): Extract default values to constants
                "main_topics": ["general"],
                "narrative_style": "abstract",
                "emotional_tone": "neutral",
            },
            "technical_aspects": {
                # TODO(code-review): Extract default values to constants
                "rhyme_complexity": "moderate",
                "flow_style": "steady",
                "wordplay_level": "basic",
                "structure_type": "traditional",
            },
            "quality_assessment": {
                # TODO(code-review): Extract 0.5 fallback score to named constant
                # TODO(code-review): Consider using lower confidence scores for fallback
                "lyrical_skill": 0.5,
                "creativity": 0.5,
                "authenticity": 0.5,
                "overall_quality": 0.5,
            },
            "experimental_features": {
                # TODO(code-review): Extract default values to constants
                "cultural_era": "2020s",
                "regional_style": "international",
                "influences": ["modern_rap"],
                "innovation_level": 0.5,
            },
            # TODO(code-review): Use more descriptive key name (e.g., "fallback_parsing")
            "_parsing_note": "Extracted from non-JSON response",
        }

    def _validate_analysis_structure(self, data: dict[str, Any]) -> None:
        # TODO(code-review): Add Google-style docstring with Args and Raises sections
        # TODO(code-review): Consider raising ValidationError instead of just logging warnings
        """Валидация структуры результата анализа"""
        # TODO(code-review): Extract required sections list to class-level constant
        # TODO(code-review): Use a schema validation library (e.g., jsonschema, Pydantic)
        required_sections = [
            "basic_analysis",
            "content_themes",
            "technical_aspects",
            "quality_assessment",
            "experimental_features",
        ]

        # TODO(code-review): Collect all missing sections and log once
        # TODO(code-review): Consider raising exception if critical sections are missing
        for section in required_sections:
            if section not in data:
                # TODO(code-review): Remove emoji from logs
                logger.warning(f"⚠️ Отсутствует секция: {section}")

        # TODO(code-review): Extract expected metrics to constant
        # TODO(code-review): Validate nested structure (e.g., genre values, mood values)
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
                # TODO(code-review): Good validation, but should fix invalid values not just log
                # TODO(code-review): Consider clamping values to [0, 1] range instead of warning
                if not isinstance(value, (int, float)) or not 0 <= value <= 1:
                    # TODO(code-review): Remove emoji
                    logger.warning(f"⚠️ Некорректное значение метрики {metric}: {value}")

    def _calculate_confidence(self, analysis_data: dict[str, Any]) -> float:
        # TODO(code-review): Add Google-style docstring with Args and Returns sections
        # TODO(code-review): Document the confidence calculation algorithm
        """Вычисление уверенности в результатах анализа"""
        confidence_factors = []

        # TODO(code-review): Extract section names to constant (duplicated from validation)
        # TODO(code-review): Magic number 5 should be len() of the sections list
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
            # TODO(code-review): Check if section is non-empty dict, not just truthy
            if analysis_data.get(section_name):
                completed_sections += 1

        # TODO(code-review): Add check for division by zero (though unlikely here)
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

        # TODO(code-review): Extract 0.3 to named constant (_FALLBACK_PARSING_PENALTY)
        # TODO(code-review): Document why 0.3 was chosen
        # Штраф за нестандартный парсинг
        if "_parsing_note" in analysis_data:
            confidence_factors.append(0.3)  # Низкая уверенность

        # TODO(code-review): Extract 0.8 to constant (_LOCAL_MODEL_CONFIDENCE_PENALTY)
        # TODO(code-review): Extract 0.4 to constant (_DEFAULT_LOW_CONFIDENCE)
        # TODO(code-review): Document the reasoning behind these magic numbers
        # TODO(code-review): Ensure return value is always in range [0.0, 1.0]
        # Общая уверенность
        if confidence_factors:
            base_confidence = sum(confidence_factors) / len(confidence_factors)
            # Дополнительный штраф для локальных моделей (они менее точны)
            return base_confidence * 0.8
        return 0.4  # Низкая уверенность по умолчанию

    def get_analyzer_info(self) -> dict[str, Any]:
        # TODO(code-review): Add Google-style docstring with Returns section
        # TODO(code-review): Consider using dataclass or TypedDict for return type
        """Получение информации об анализаторе"""
        # TODO(code-review): Extract version to constant at module level
        # TODO(code-review): Extract all string literals to constants
        # TODO(code-review): Don't expose base_url (might be sensitive internal info)
        # TODO(code-review): Use semantic versioning strictly (current: 2.0.0)
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
                # TODO(code-review): Extract default values to module constants
                # TODO(code-review): Use the actual constants instead of hardcoded strings
                "model_name": "Ollama model to use (default: llama3.2:3b)",
                "base_url": "Ollama server URL (default: http://localhost:11434)",
                "temperature": "Generation temperature (default: 0.1)",
                "timeout": "Request timeout in seconds (default: 60)",
            },
            "setup_instructions": [
                # TODO(code-review): Load instructions from external file or config
                # TODO(code-review): Add links to documentation
                "1. Install Ollama from https://ollama.ai",
                "2. Run: ollama serve",
                "3. Pull model: ollama pull llama3.2:3b",
                "4. Start analysis",
            ],
        }

    @property
    def analyzer_type(self) -> str:
        # TODO(code-review): Add Google-style docstring with Returns section
        """Тип анализатора"""
        # TODO(code-review): Extract "ai" to class-level constant
        # TODO(code-review): Consider using Enum for analyzer types
        return "ai"

    @property
    def supported_features(self) -> list[str]:
        # TODO(code-review): Add Google-style docstring with Returns section
        # TODO(code-review): Use tuple instead of list for immutable return value
        # TODO(code-review): Consider using Enum or constants for feature names
        """Поддерживаемые функции анализа"""
        # TODO(code-review): Extract this list to class-level constant
        # TODO(code-review): Add documentation for what each feature means
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

# TODO(code-review): GENERAL FILE-LEVEL IMPROVEMENTS:
# 1. Add unit tests for all methods (especially _parse_response, _calculate_confidence)
# 2. Add integration tests with mock Ollama server
# 3. Consider dependency injection for requests library (easier mocking)
# 4. Add metrics/monitoring (e.g., track success rate, latency, model usage)
# 5. Consider circuit breaker pattern for Ollama connection failures
# 6. Add request/response logging for debugging (with privacy considerations)
# 7. Implement caching for repeated analyses
# 8. Add rate limiting to prevent overwhelming local Ollama instance
# 9. Consider async/await for non-blocking I/O operations
# 10. Add comprehensive error codes for different failure modes
# 11. Create custom exception hierarchy (OllamaError, ModelNotFoundError, etc.)
# 12. Add type checking with mypy in CI/CD pipeline
# 13. Run linters: pylint, flake8, black for formatting
# 14. Add performance profiling for slow methods
# 15. Document thread-safety considerations
# 16. Add examples in docstrings
# 17. Consider adding __repr__ and __str__ methods for debugging
# 18. Add logging of confidence scores distribution for monitoring
# 19. Consider feature flags for experimental functionality
# 20. Add deprecation warnings for any backwards-incompatible changes
