#!/usr/bin/env python3
"""
🤖 Объединенный массовый Qwen анализатор (PostgreSQL + встроенный Qwen)

НАЗНАЧЕНИЕ:
- Встроенный Qwen анализатор (из archive/qwen_analyzer.py)
- Массовый анализ PostgreSQL базы данных
- Оптимизированная обработка без внешних зависимостей
- Полная совместимость с системой

ИСПОЛЬЗОВАНИЕ:
python src/analyzers/mass_qwen_analysis.py --test      # Тестовый режим
python src/analyzers/mass_qwen_analysis.py --stats    # Статистика
python src/analyzers/mass_qwen_analysis.py --batch 100 # Кастомный батч
python src/analyzers/mass_qwen_analysis.py --resume   # Продолжить

АВТОР: AI Assistant
ДАТА: Сентябрь 2025
ВЕРСИЯ: 3.0 (Unified)
"""

import argparse
import asyncio
import json
import logging
import os
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

# Загружаем переменные окружения
load_dotenv()

# Добавляем корневую папку в path
project_root = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
sys.path.insert(0, project_root)

# Условный импорт для OpenAI-совместимого клиента
try:
    import openai

    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False

try:
    from src.core.app import create_app
    from src.database.postgres_adapter import PostgreSQLManager
except ImportError as e:
    print(f"❌ Ошибка импорта: {e}")
    print("💡 Убедитесь, что вы запускаете скрипт из корневой папки проекта")
    sys.exit(1)

logger = logging.getLogger(__name__)


# ============================================================================
# ВСТРОЕННЫЙ QWEN АНАЛИЗАТОР (из archive/qwen_analyzer.py)
# ============================================================================


@dataclass
class AnalysisResult:
    """Результат анализа для совместимости с массовым анализатором"""

    artist: str
    title: str
    analyzer_type: str
    confidence: float
    metadata: dict[str, Any]
    raw_output: dict[str, Any]
    processing_time: float
    timestamp: str


class EmbeddedQwenAnalyzer:
    """
    Встроенный Qwen анализатор (объединенная версия из archive/qwen_analyzer.py)
    """

    def __init__(self, config: dict[str, Any] | None = None):
        """Инициализация встроенного Qwen анализатора"""
        self.config = config or {}

        # Настройки модели
        self.model_name = self.config.get("model_name", "qwen/qwen3-4b-fp8")
        self.base_url = self.config.get("base_url", "https://api.novita.ai/openai/v1")
        self.temperature = self.config.get("temperature", 0.1)
        self.max_tokens = self.config.get("max_tokens", 1500)
        self.timeout = self.config.get("timeout", 30)

        # API ключ
        self.api_key = self.config.get("api_key") or os.getenv("NOVITA_API_KEY")

        # Проверка доступности
        self.available = self._check_availability()

        if self.available:
            self.client = openai.OpenAI(
                api_key=self.api_key, base_url=self.base_url, timeout=self.timeout
            )
            logger.info(
                f"✅ Встроенный Qwen анализатор инициализирован: {self.model_name}"
            )
        else:
            logger.warning("⚠️ Встроенный Qwen анализатор недоступен")

    def _check_availability(self) -> bool:
        """Проверка доступности Novita AI API"""
        if not HAS_OPENAI:
            logger.error("❌ openai не установлен. Установите: pip install openai")
            return False

        if not self.api_key:
            logger.error(
                "❌ NOVITA_API_KEY не найден в конфигурации или переменных окружения"
            )
            return False

        try:
            # Тестовое подключение
            client = openai.OpenAI(api_key=self.api_key, base_url=self.base_url)

            response = client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": "Test"},
                ],
                max_tokens=10,
                temperature=0.1,
            )

            if response.choices and response.choices[0].message:
                logger.info("✅ Novita AI Qwen API успешно протестирован")
                return True
            logger.error("❌ Получен пустой ответ от Qwen API")
            return False

        except Exception as e:
            logger.error(f"❌ Ошибка проверки Qwen API: {e}")
            return False

    def validate_input(self, artist: str, title: str, lyrics: str) -> bool:
        """Валидация входных данных"""
        if not all([artist, title, lyrics]):
            return False
        if len(lyrics.strip()) < 10:
            return False
        return True

    def preprocess_lyrics(self, lyrics: str) -> str:
        """Предобработка текста песни"""
        lyrics = lyrics.strip()

        # Удаление избыточных пробелов
        import re

        lyrics = re.sub(r"\s+", " ", lyrics)

        # Удаление повторяющихся символов
        lyrics = re.sub(r"(.)\1{3,}", r"\1\1\1", lyrics)

        # Нормализация пробелов
        lyrics = re.sub(r"\n{3,}", "\n\n", lyrics)

        # Удаление URL и специальных символов
        lyrics = re.sub(r"http[s]?://\S+", "", lyrics)
        lyrics = re.sub(r'[^\w\s\n.,!?\'"-]', "", lyrics)

        return lyrics.strip()

    def analyze_song(self, artist: str, title: str, lyrics: str) -> AnalysisResult:
        """
        Анализ песни с использованием Qwen модели
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
            system_prompt, user_prompt = self._create_analysis_prompts(
                artist, title, processed_lyrics
            )

            # Генерация ответа
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                max_tokens=self.max_tokens,
                temperature=self.temperature,
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
                analyzer_type="qwen",  # Правильное поле!
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
                        "prompt_tokens": response.usage.prompt_tokens
                        if response.usage
                        else 0,
                        "completion_tokens": response.usage.completion_tokens
                        if response.usage
                        else 0,
                        "total_tokens": response.usage.total_tokens
                        if response.usage
                        else 0,
                    },
                },
                raw_output=analysis_data,
                processing_time=processing_time,
                timestamp=datetime.now().isoformat(),
            )

        except Exception as e:
            logger.error(
                f"❌ Ошибка анализа встроенного Qwen для {artist} - {title}: {e}"
            )
            raise RuntimeError(f"Qwen analysis failed: {e}") from e

    def _create_analysis_prompts(
        self, artist: str, title: str, lyrics: str
    ) -> tuple[str, str]:
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

    def _parse_response(self, response_text: str) -> dict[str, Any]:
        """Парсинг ответа от Qwen модели"""
        try:
            # Очистка от лишнего текста
            response_text = response_text.strip()

            # Удаляем теги <think>...</think>
            import re

            response_text = re.sub(
                r"<think>.*?</think>", "", response_text, flags=re.DOTALL
            )
            response_text = response_text.strip()

            # Логируем сырой ответ для отладки
            logger.debug(f"Cleaned response (first 500 chars): {response_text[:500]}")

            # Если ответ уже JSON, парсим напрямую
            if response_text.startswith("{") and response_text.endswith("}"):
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
                        logger.error("JSON still invalid after fixes")
                        analysis_data = self._create_fallback_analysis()
            else:
                # Поиск JSON блока в тексте
                json_start = response_text.find("{")
                json_end = response_text.rfind("}") + 1

                if json_start == -1 or json_end == 0:
                    logger.error(
                        f"No JSON found in response. Full response: {response_text}"
                    )
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
        json_str = re.sub(r",(\s*[}\]])", r"\1", json_str)

        # Исправляем одинарные кавычки на двойные (если есть)
        json_str = re.sub(r"'([^']*)':", r'"\1":', json_str)

        # Убираем лишние символы в конце
        json_str = json_str.strip()

        return json_str

    def _validate_analysis_structure(self, data: dict[str, Any]) -> None:
        """Валидация структуры результата анализа"""
        required_sections = [
            "genre_analysis",
            "mood_analysis",
            "content_analysis",
            "technical_analysis",
            "quality_metrics",
            "cultural_context",
        ]

        for section in required_sections:
            if section not in data:
                logger.warning(f"Missing section: {section}")
                data[section] = {}

        # Проверка качественных метрик
        quality_metrics = data.get("quality_metrics", {})
        for metric in [
            "lyrical_creativity",
            "technical_skill",
            "authenticity",
            "commercial_appeal",
            "originality",
            "overall_quality",
        ]:
            if metric in quality_metrics:
                value = quality_metrics[metric]
                if not isinstance(value, (int, float)) or not 0 <= value <= 1:
                    logger.warning(f"Invalid metric value for {metric}: {value}")

    def _calculate_confidence(self, analysis_data: dict[str, Any]) -> float:
        """Вычисление уверенности в результатах анализа"""
        confidence_factors = []

        # Проверка полноты анализа
        sections_completed = 0
        total_sections = 6

        for section_name in [
            "genre_analysis",
            "mood_analysis",
            "content_analysis",
            "technical_analysis",
            "quality_metrics",
            "cultural_context",
        ]:
            if analysis_data.get(section_name):
                sections_completed += 1

        completeness_score = sections_completed / total_sections
        confidence_factors.append(completeness_score)

        # Проверка уверенности в жанре
        genre_analysis = analysis_data.get("genre_analysis", {})
        if "confidence" in genre_analysis:
            genre_confidence = genre_analysis["confidence"]
            if (
                isinstance(genre_confidence, (int, float))
                and 0 <= genre_confidence <= 1
            ):
                confidence_factors.append(genre_confidence)

        # Проверка качественных метрик
        quality_metrics = analysis_data.get("quality_metrics", {})
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
        return 0.5  # Средняя уверенность при отсутствии данных

    def _create_fallback_analysis(self) -> dict[str, Any]:
        """Создает базовый анализ в случае ошибки парсинга ответа"""
        return {
            "genre_analysis": {
                "primary_genre": "rap",
                "subgenre": "unknown",
                "confidence": 0.3,
            },
            "mood_analysis": {
                "primary_mood": "neutral",
                "emotional_intensity": "unknown",
                "energy_level": "unknown",
                "valence": "neutral",
            },
            "content_analysis": {
                "explicit_content": False,
                "explicit_level": "unknown",
                "main_themes": ["general"],
                "narrative_style": "unknown",
            },
            "technical_analysis": {
                "rhyme_scheme": "unknown",
                "flow_pattern": "unknown",
                "complexity_level": "unknown",
                "wordplay_quality": "unknown",
                "metaphor_usage": "unknown",
                "structure": "unknown",
            },
            "quality_metrics": {
                "lyrical_creativity": 0.3,
                "technical_skill": 0.3,
                "authenticity": 0.3,
                "commercial_appeal": 0.3,
                "originality": 0.3,
                "overall_quality": 0.3,
                "ai_generated_likelihood": 0.5,
            },
            "cultural_context": {
                "era_estimate": "unknown",
                "regional_style": "unknown",
                "cultural_references": [],
                "social_commentary": False,
            },
        }


# ============================================================================
# МАССОВЫЙ АНАЛИЗАТОР (из src/analyzers/mass_qwen_analysis.py)
# ============================================================================


@dataclass
class AnalysisStats:
    """Статистика анализа"""

    total_records: int = 0
    processed: int = 0
    errors: int = 0
    skipped: int = 0
    start_time: datetime | None = None
    current_batch: int = 0
    total_batches: int = 0

    @property
    def success_rate(self) -> float:
        """Процент успешности"""
        return (self.processed / max(self.total_records, 1)) * 100

    @property
    def processing_rate(self) -> float:
        """Скорость обработки (записей/минуту)"""
        if not self.start_time:
            return 0.0
        elapsed = (datetime.now() - self.start_time).total_seconds()
        return (self.processed / max(elapsed, 1)) * 60

    @property
    def estimated_remaining(self) -> timedelta:
        """Оценка оставшегося времени"""
        remaining = self.total_records - self.processed
        if remaining <= 0 or self.processing_rate == 0:
            return timedelta(0)
        minutes = remaining / self.processing_rate
        return timedelta(minutes=minutes)


class UnifiedQwenMassAnalyzer:
    """Объединенный массовый анализатор с встроенным Qwen"""

    def __init__(self):
        self.app = None
        self.analyzer = None
        self.db_manager = None
        self.stats = AnalysisStats()
        self.last_processed_id = 0
        # Сохраняем checkpoint в папку results
        self.checkpoint_file = Path("results") / "qwen_analysis_checkpoint.json"

    async def initialize(self) -> bool:
        """Инициализация компонентов"""
        try:
            print("🔧 Инициализация системы...")

            # Инициализация приложения
            self.app = create_app()

            # Получение встроенного анализатора
            self.analyzer = EmbeddedQwenAnalyzer()
            if not self.analyzer or not self.analyzer.available:
                print("❌ Встроенный Qwen анализатор недоступен!")
                print("💡 Проверьте NOVITA_API_KEY в .env файле")
                return False

            print(f"✅ Встроенный Qwen анализатор готов: {self.analyzer.model_name}")

            # Подключение к PostgreSQL
            self.db_manager = PostgreSQLManager()
            await self.db_manager.initialize()

            print("✅ PostgreSQL подключение установлено")
            return True

        except Exception as e:
            print(f"❌ Ошибка инициализации: {e}")
            return False

    async def get_database_stats(self) -> dict[str, Any]:
        """Получение статистики базы данных"""
        try:
            async with self.db_manager.connection_pool.acquire() as conn:
                stats_query = """
                SELECT 
                    COUNT(*) as total_tracks,
                    COUNT(CASE WHEN lyrics IS NOT NULL AND lyrics != '' THEN 1 END) as tracks_with_lyrics,
                    COUNT(DISTINCT ar.track_id) as qwen_analyzed,
                    COUNT(CASE WHEN ar.track_id IS NULL THEN 1 END) as unanalyzed
                FROM tracks t
                LEFT JOIN analysis_results ar ON t.id = ar.track_id 
                    AND ar.analyzer_type LIKE '%qwen%'
                WHERE t.lyrics IS NOT NULL AND t.lyrics != ''
                """

                result = await conn.fetchrow(stats_query)
                return dict(result) if result else {}

        except Exception as e:
            print(f"❌ Ошибка получения статистики: {e}")
            return {}

    async def load_checkpoint(self) -> bool:
        """Загрузка чекпоинта для продолжения работы"""
        try:
            if not self.checkpoint_file.exists():
                return False

            with open(self.checkpoint_file, encoding="utf-8") as f:
                data = json.load(f)

            self.last_processed_id = data.get("last_processed_id", 0)
            print(
                f"📍 Загружен чекпоинт: последняя обработанная запись ID {self.last_processed_id}"
            )
            return True

        except Exception as e:
            print(f"⚠️ Ошибка загрузки чекпоинта: {e}")
            return False

    async def save_checkpoint(self):
        """Сохранение чекпоинта"""
        try:
            # Создаем папку results если её нет
            self.checkpoint_file.parent.mkdir(exist_ok=True)

            data = {
                "last_processed_id": self.last_processed_id,
                "timestamp": datetime.now().isoformat(),
                "processed": self.stats.processed,
                "errors": self.stats.errors,
            }

            with open(self.checkpoint_file, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)

        except Exception as e:
            print(f"⚠️ Ошибка сохранения чекпоинта: {e}")

    async def get_unanalyzed_records(
        self, limit: int | None = None, resume: bool = False
    ) -> list[dict[str, Any]]:
        """Получение неанализированных записей"""
        try:
            async with self.db_manager.connection_pool.acquire() as conn:
                # Базовый запрос
                query = """
                SELECT t.id, t.artist, t.title, t.lyrics 
                FROM tracks t
                WHERE t.lyrics IS NOT NULL 
                AND t.lyrics != '' 
                AND t.id NOT IN (
                    SELECT DISTINCT track_id 
                    FROM analysis_results 
                    WHERE analyzer_type LIKE '%qwen%'
                )
                """

                # Условие для resume режима
                if resume and self.last_processed_id > 0:
                    query += f" AND t.id > {self.last_processed_id}"

                query += " ORDER BY t.id"

                # Лимит
                if limit:
                    query += f" LIMIT {limit}"

                records = await conn.fetch(query)
                return [dict(record) for record in records]

        except Exception as e:
            print(f"❌ Ошибка получения записей: {e}")
            return []

    async def analyze_single_record(self, record: dict[str, Any]) -> bool:
        """Анализ одной записи"""
        track_id = record["id"]
        artist = record["artist"]
        title = record["title"]
        lyrics = record["lyrics"]

        try:
            # Анализ текста (встроенный анализатор)
            result = self.analyzer.analyze_song(artist, title, lyrics)

            if result is None:
                return False

            # Подготовка данных для PostgreSQL
            analysis_data = {
                "track_id": track_id,
                "analyzer_type": "qwen-3-4b-fp8",
                "sentiment": self._extract_sentiment(result),
                "confidence": result.confidence or 0.5,
                "complexity_score": self._extract_complexity(result),
                "themes": self._extract_themes(result),
                "analysis_data": result.raw_output or {},
                "processing_time_ms": int((result.processing_time or 0) * 1000),
                "model_version": result.metadata.get("model_name", "qwen-3-4b-fp8"),
            }

            # Сохранение в базу
            success = await self._save_analysis_to_database(analysis_data)

            if success:
                self.last_processed_id = track_id
                return True
            return False

        except Exception as e:
            print(f"❌ Ошибка анализа записи {track_id}: {e}")
            return False

    async def _save_analysis_to_database(self, analysis_data: dict[str, Any]) -> bool:
        """Сохранение результата анализа в базу данных"""
        try:
            async with self.db_manager.connection_pool.acquire() as conn:
                # Проверяем, существует ли уже анализ для этого трека
                existing = await conn.fetchrow(
                    "SELECT id FROM analysis_results WHERE track_id = $1 AND analyzer_type = $2",
                    analysis_data["track_id"],
                    analysis_data["analyzer_type"],
                )

                if existing:
                    print(
                        f"  ⚠️ Анализ уже существует для трека {analysis_data['track_id']}"
                    )
                    return True

                # Вставляем новый анализ
                insert_query = """
                INSERT INTO analysis_results 
                (track_id, analyzer_type, analysis_data, confidence, sentiment, complexity_score, themes, processing_time_ms, model_version, created_at)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, NOW())
                RETURNING id
                """

                result = await conn.fetchrow(
                    insert_query,
                    analysis_data["track_id"],
                    analysis_data["analyzer_type"],
                    json.dumps(analysis_data["analysis_data"]),
                    analysis_data["confidence"],
                    analysis_data["sentiment"],
                    analysis_data["complexity_score"],
                    json.dumps(analysis_data["themes"]),
                    analysis_data["processing_time_ms"],
                    analysis_data["model_version"],
                )

                return result is not None

        except Exception as e:
            print(f"❌ Ошибка сохранения анализа: {e}")
            return False

    def _extract_sentiment(self, result) -> str:
        """Извлечение настроения из результата анализа"""
        try:
            if hasattr(result, "raw_output") and result.raw_output:
                mood_analysis = result.raw_output.get("mood_analysis", {})
                return mood_analysis.get("primary_mood", "neutral")
            return "neutral"
        except:
            return "neutral"

    def _extract_complexity(self, result) -> float:
        """Извлечение оценки сложности"""
        try:
            if hasattr(result, "raw_output") and result.raw_output:
                quality_metrics = result.raw_output.get("quality_metrics", {})
                return float(quality_metrics.get("overall_quality", 0.5)) * 5.0
            return 3.0
        except:
            return 3.0

    def _extract_themes(self, result) -> list[str]:
        """Извлечение тем из результата анализа"""
        try:
            if hasattr(result, "raw_output") and result.raw_output:
                content_analysis = result.raw_output.get("content_analysis", {})
                return content_analysis.get("main_themes", ["general"])
            return ["general"]
        except:
            return ["general"]

    async def process_batch(self, batch: list[dict[str, Any]]) -> tuple[int, int]:
        """Обработка батча записей"""
        processed = 0
        errors = 0

        for i, record in enumerate(batch, 1):
            track_id = record["id"]
            artist = record.get("artist", "Unknown")
            title = record.get("title", "Unknown")

            # Прогресс внутри батча
            print(f"  🎵 [{i}/{len(batch)}] {artist} - {title} (ID: {track_id})")

            try:
                if await self.analyze_single_record(record):
                    processed += 1
                    print("    ✅ Успешно проанализировано")
                else:
                    errors += 1
                    print("    ❌ Ошибка анализа")

                # Пауза между запросами для предотвращения rate limiting
                await asyncio.sleep(0.5)

            except Exception as e:
                errors += 1
                print(f"    ❌ Исключение: {e}")

        return processed, errors

    def print_progress(self):
        """Вывод прогресса"""
        print("\n📊 ПРОГРЕСС:")
        print(f"  📈 Обработано: {self.stats.processed}/{self.stats.total_records}")
        print(f"  ✅ Успешность: {self.stats.success_rate:.1f}%")
        print(f"  ⚡ Скорость: {self.stats.processing_rate:.1f} записей/мин")
        print(f"  ⏱️  Осталось: {self.stats.estimated_remaining}")
        print(f"  📦 Батч: {self.stats.current_batch}/{self.stats.total_batches}")

    async def run_analysis(
        self,
        batch_size: int = 100,
        max_records: int | None = None,
        resume: bool = False,
        test_mode: bool = False,
    ) -> dict[str, Any]:
        """Запуск массового анализа"""

        print("🎵 Объединенный Qwen массовый анализатор (PostgreSQL v3.0)")
        print("=" * 70)

        # Загрузка чекпоинта при необходимости
        if resume:
            await self.load_checkpoint()

        # Получение статистики базы
        db_stats = await self.get_database_stats()
        print("📊 База данных:")
        print(f"  📁 Всего треков: {db_stats.get('total_tracks', 0)}")
        print(f"  📝 С текстами: {db_stats.get('tracks_with_lyrics', 0)}")
        print(f"  🤖 Уже проанализировано Qwen: {db_stats.get('qwen_analyzed', 0)}")
        print(f"  ⏳ Ожидает анализа: {db_stats.get('unanalyzed', 0)}")

        # Определение лимита для тестового режима
        if test_mode:
            max_records = 10
            batch_size = 5
            print(f"\n🧪 ТЕСТОВЫЙ РЕЖИМ: анализируем только {max_records} записей")

        # Получение записей для анализа
        print("\n🔍 Загрузка записей для анализа...")
        records = await self.get_unanalyzed_records(limit=max_records, resume=resume)

        if not records:
            print("✅ Все записи уже проанализированы!")
            return {"status": "completed", "message": "No records to process"}

        # Инициализация статистики
        self.stats.total_records = len(records)
        self.stats.start_time = datetime.now()
        self.stats.total_batches = (len(records) + batch_size - 1) // batch_size

        print("\n🎯 План анализа:")
        print(f"  📊 Записей к обработке: {len(records)}")
        print(f"  📦 Размер батча: {batch_size}")
        print(f"  🔢 Количество батчей: {self.stats.total_batches}")
        print(f"  ⏱️  Примерное время: {(len(records) * 15) // 60} минут")
        print("  🆓 Бесплатная модель Qwen через Novita AI - без затрат!")

        if not test_mode:
            print("\n⏳ Начинаем через 3 секунды...")
            await asyncio.sleep(3)

        # Массовая обработка по батчам
        print("\n🚀 НАЧИНАЕМ МАССОВЫЙ АНАЛИЗ")
        print("=" * 50)

        for i in range(0, len(records), batch_size):
            self.stats.current_batch += 1
            batch = records[i : i + batch_size]
            batch_start = time.time()

            print(f"\n📦 Батч {self.stats.current_batch}/{self.stats.total_batches}")
            print(
                f"📊 Записи {i + 1}-{min(i + batch_size, len(records))} из {len(records)}"
            )

            # Обработка батча
            batch_processed, batch_errors = await self.process_batch(batch)

            # Обновление статистики
            self.stats.processed += batch_processed
            self.stats.errors += batch_errors

            # Сохранение чекпоинта
            await self.save_checkpoint()

            # Статистика батча
            batch_time = time.time() - batch_start
            print(f"  ⏱️  Батч завершен за {batch_time:.1f}с")
            print(f"  ✅ Успешно: {batch_processed}")
            print(f"  ❌ Ошибок: {batch_errors}")

            # Общий прогресс
            self.print_progress()

            # Пауза между батчами (кроме последнего)
            if i + batch_size < len(records):
                print("  ⏸️  Пауза между батчами...")
                await asyncio.sleep(2)

        # Финальная статистика
        total_time = (datetime.now() - self.stats.start_time).total_seconds()

        print("\n🏆 АНАЛИЗ ЗАВЕРШЕН!")
        print("=" * 50)
        print(f"✅ Успешно проанализировано: {self.stats.processed}")
        print(f"❌ Ошибок: {self.stats.errors}")
        print(f"📊 Всего записей: {self.stats.total_records}")
        print(f"🎯 Успешность: {self.stats.success_rate:.1f}%")
        print(f"⏱️  Общее время: {total_time // 60:.0f}м {total_time % 60:.0f}с")
        print(f"⚡ Средняя скорость: {self.stats.processing_rate:.1f} записей/мин")

        # Обновленная статистика базы
        final_db_stats = await self.get_database_stats()
        print("\n📈 Обновленная статистика базы:")
        print(f"  🤖 Qwen проанализировано: {final_db_stats.get('qwen_analyzed', 0)}")
        print(f"  ⏳ Осталось: {final_db_stats.get('unanalyzed', 0)}")

        # Удаление чекпоинта при успешном завершении
        if self.stats.errors == 0 and self.checkpoint_file.exists():
            self.checkpoint_file.unlink()
            print("🗑️  Чекпоинт удален (анализ завершен без ошибок)")

        return {
            "status": "completed",
            "processed": self.stats.processed,
            "errors": self.stats.errors,
            "success_rate": self.stats.success_rate,
            "total_time": total_time,
            "processing_rate": self.stats.processing_rate,
        }

    async def show_stats_only(self) -> dict[str, Any]:
        """Показать только статистику без анализа"""
        print("📊 Статистика объединенного Qwen анализа базы данных")
        print("=" * 60)

        db_stats = await self.get_database_stats()

        print(f"📁 Всего треков: {db_stats.get('total_tracks', 0)}")
        print(f"📝 С текстами: {db_stats.get('tracks_with_lyrics', 0)}")
        print(f"🤖 Проанализировано Qwen: {db_stats.get('qwen_analyzed', 0)}")
        print(f"⏳ Ожидает анализа: {db_stats.get('unanalyzed', 0)}")

        if db_stats.get("tracks_with_lyrics", 0) > 0:
            coverage = (
                db_stats.get("qwen_analyzed", 0) / db_stats.get("tracks_with_lyrics", 1)
            ) * 100
            print(f"📈 Покрытие анализом: {coverage:.1f}%")

        # Проверка чекпоинта
        if self.checkpoint_file.exists():
            try:
                with open(self.checkpoint_file) as f:
                    checkpoint = json.load(f)
                print("\n📍 Найден чекпоинт:")
                print(
                    f"  📄 Последняя обработанная запись: {checkpoint.get('last_processed_id', 0)}"
                )
                print(f"  📅 Дата: {checkpoint.get('timestamp', 'unknown')}")
                print(f"  ✅ Обработано в сессии: {checkpoint.get('processed', 0)}")
                print(f"  ❌ Ошибок в сессии: {checkpoint.get('errors', 0)}")
            except:
                print("\n⚠️ Найден поврежденный чекпоинт")

        return db_stats

    async def cleanup(self):
        """Очистка ресурсов"""
        if self.db_manager:
            await self.db_manager.close()
        print("🧹 Ресурсы освобождены")


async def main():
    """Главная функция с расширенным парсингом аргументов"""
    parser = argparse.ArgumentParser(
        description="Объединенный массовый анализ Qwen (PostgreSQL v3.0)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Примеры использования:
  python src/analyzers/mass_qwen_analysis.py                    # Полный анализ
  python src/analyzers/mass_qwen_analysis.py --batch 50         # Батч 50 записей
  python src/analyzers/mass_qwen_analysis.py --test             # Тестовый режим
  python src/analyzers/mass_qwen_analysis.py --max 1000         # Лимит 1000 записей
  python src/analyzers/mass_qwen_analysis.py --resume           # Продолжить с чекпоинта
  python src/analyzers/mass_qwen_analysis.py --stats            # Только статистика
        """,
    )

    parser.add_argument(
        "--batch",
        type=int,
        default=100,
        metavar="N",
        help="Размер батча (default: 100)",
    )
    parser.add_argument(
        "--max",
        type=int,
        metavar="N",
        help="Максимальное количество записей для анализа",
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Тестовый режим (только 10 записей с батчем 5)",
    )
    parser.add_argument(
        "--resume", action="store_true", help="Продолжить с последнего чекпоинта"
    )
    parser.add_argument(
        "--stats", action="store_true", help="Показать только статистику базы данных"
    )

    args = parser.parse_args()

    # Создание и инициализация анализатора
    analyzer = UnifiedQwenMassAnalyzer()

    try:
        # Режим только статистики
        if args.stats:
            if await analyzer.initialize():
                await analyzer.show_stats_only()
            return

        # Основной режим анализа
        if not await analyzer.initialize():
            print("❌ Не удалось инициализировать систему")
            return

        # Запуск анализа с параметрами
        result = await analyzer.run_analysis(
            batch_size=args.batch,
            max_records=args.max,
            resume=args.resume,
            test_mode=args.test,
        )

        print(f"\n🎯 Результат: {result['status']}")

    except KeyboardInterrupt:
        print("\n\n⏹️  Анализ прерван пользователем")
        print("💡 Используйте --resume для продолжения с последней позиции")
    except Exception as e:
        print(f"\n❌ Критическая ошибка: {e}")
    finally:
        await analyzer.cleanup()


if __name__ == "__main__":
    asyncio.run(main())
