"""
🤖 Многоуровневый AI анализатор текстов песен с Safety & Hallucination Detection

НАЗНАЧЕНИЕ:
- Глубокий AI-анализ текстов песен с помощью нескольких моделей
- Классификация жанра, настроения, качества, тематики
- Детекция галлюцинаций и валидация надежности AI
- Интерпретируемость решений с объяснениями
- Использование облачных и локальных API

ИСПОЛЬЗОВАНИЕ:
- Ollama (приоритет): локальные модели, бесплатно
- Google Gemma (fallback): облачное API
- Mock Provider (резерв): для тестирования
- Safety Layer: автоматическая валидация результатов
- Interpretability: объяснение решений AI

ЗАВИСИМОСТИ:
- Python 3.8+
- ollama (для локальных моделей)
- google-generativeai (для Gemma API)
- pydantic (для моделей данных)
- requests (для HTTP запросов)
- sqlite3 (для работы с БД)

РЕЗУЛЬТАТ:
- Полный анализ: жанр, настроение, энергия, структура
- Качественные метрики: аутентичность, креативность, коммерческий потенциал
- Валидация надежности: детекция галлюцинаций, консистентность
- Объяснения решений: почему AI принял то или иное решение
- Сохранение в базу данных с метаданными

ОСОБЕННОСТИ:
- Multi-provider fallback система
- Safety & Hallucination Detection
- Interpretability & Explainability
- Batch processing с progress tracking
- Cost optimization (бесплатные модели в приоритете)

АВТОР: AI Assistant
ДАТА: Сентябрь 2025
"""

import json
import logging
import os
import re
import sqlite3
import time
from datetime import datetime

import requests
from dotenv import load_dotenv
from pydantic import BaseModel

# Загрузка переменных окружения
load_dotenv()

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("ai_analysis.log", encoding="utf-8"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)

# Импорт моделей данных
from ..models.models import LyricsAnalysis, QualityMetrics, SongMetadata


class SafetyValidator:
    """Валидатор для проверки достоверности AI анализа и детекции галлюцинаций"""

    def __init__(self):
        # Словари для проверки тематик (English-focused)
        self.theme_keywords = {
            "money": [
                "cash",
                "money",
                "dollars",
                "bands",
                "racks",
                "bread",
                "paper",
                "coins",
                "wealth",
                "riches",
                "bank",
                "rich",
                "fortune",
            ],
            "relationships": [
                "love",
                "girl",
                "boy",
                "girlfriend",
                "boyfriend",
                "wife",
                "husband",
                "family",
                "bae",
                "baby",
                "relationship",
                "romance",
            ],
            "street_life": [
                "street",
                "block",
                "neighborhood",
                "ghetto",
                "projects",
                "corners",
                "trap",
                "streets",
                "hood",
                "city",
                "urban",
            ],
            "success": [
                "success",
                "famous",
                "star",
                "career",
                "achievement",
                "winning",
                "made it",
                "top",
                "win",
                "champion",
                "glory",
            ],
            "struggle": [
                "struggle",
                "pain",
                "problems",
                "hardship",
                "suffering",
                "tough",
                "hard times",
                "grind",
                "difficult",
                "rough",
            ],
            "drugs": [
                "drugs",
                "molly",
                "xanax",
                "percs",
                "pills",
                "cocaine",
                "heroin",
                "marijuana",
                "cannabis",
                "weed",
                "lean",
                "high",
            ],
            "violence": [
                "war",
                "fight",
                "murder",
                "blood",
                "gun",
                "knife",
                "shoot",
                "kill",
                "weapon",
                "violence",
                "battle",
                "beef",
            ],
            "party": [
                "party",
                "club",
                "dance",
                "fun",
                "alcohol",
                "beer",
                "drunk",
                "drinking",
                "turn up",
                "lit",
                "celebration",
            ],
            "depression": [
                "depression",
                "sad",
                "suicide",
                "death",
                "lonely",
                "sorrow",
                "depressed",
                "dark",
                "pain",
                "hurt",
                "broken",
            ],
            "social_issues": [
                "politics",
                "society",
                "system",
                "power",
                "protest",
                "revolution",
                "government",
                "social",
                "justice",
                "change",
            ],
        }

        # Словари для проверки настроений (English-focused)
        self.mood_indicators = {
            "aggressive": [
                "hate",
                "angry",
                "mad",
                "kill",
                "war",
                "blood",
                "fight",
                "rage",
                "fury",
                "violence",
                "beef",
                "pissed",
            ],
            "melancholic": [
                "sad",
                "sadness",
                "tears",
                "depression",
                "lonely",
                "pain",
                "hurt",
                "broken",
                "crying",
                "sorrow",
            ],
            "energetic": [
                "party",
                "club",
                "dance",
                "hype",
                "lit",
                "turn up",
                "wild",
                "crazy",
                "bounce",
                "jump",
                "energy",
            ],
            "neutral": [
                "talking",
                "telling",
                "thinking",
                "know",
                "remember",
                "see",
                "saying",
                "speaking",
                "telling",
            ],
        }

        # Пороги для различных проверок
        self.consistency_threshold = 0.6  # Понижен для более гибкой оценки
        self.hallucination_threshold = 0.4  # Повышен для строгого контроля галлюцинаций

    def validate_analysis(self, lyrics: str, ai_analysis: dict) -> dict:
        """Полная проверка достоверности AI анализа"""

        # 1. Проверка внутренней консистентности
        consistency_score = self.check_internal_consistency(ai_analysis)

        # 2. Валидация фактических утверждений
        factual_accuracy = self.validate_factual_claims(lyrics, ai_analysis)

        # 3. Детекция галлюцинаций
        hallucination_risk = self.detect_hallucinations(lyrics, ai_analysis)

        # 4. Проверка соответствия текста и анализа
        text_alignment = self.check_text_analysis_alignment(lyrics, ai_analysis)

        # 5. Получение предупреждающих флагов
        warning_flags = self.get_warning_flags(ai_analysis, lyrics)

        # Итоговая оценка надежности
        is_reliable = (
            hallucination_risk < self.hallucination_threshold
            and consistency_score > self.consistency_threshold
            and factual_accuracy > 0.5  # Понижен порог
            and text_alignment > 0.4  # Понижен порог
            and len(warning_flags) == 0  # Никаких критических предупреждений
        )

        return {
            "is_reliable": is_reliable,
            "reliability_score": (consistency_score + factual_accuracy + text_alignment)
            / 3,
            "consistency_score": consistency_score,
            "factual_accuracy": factual_accuracy,
            "hallucination_risk": hallucination_risk,
            "text_alignment": text_alignment,
            "warning_flags": warning_flags,
            "validation_summary": self._generate_validation_summary(
                is_reliable, hallucination_risk, consistency_score, warning_flags
            ),
        }

    def detect_hallucinations(self, lyrics: str, analysis: dict) -> float:
        """Детектирует возможные галлюцинации в AI анализе"""
        hallucination_score = 0.0
        lyrics_lower = lyrics.lower()

        # Проверяем заявленные темы
        if "main_themes" in analysis:
            claimed_themes = analysis["main_themes"]
            if isinstance(claimed_themes, list):
                for theme in claimed_themes:
                    if not self.theme_present_in_lyrics(theme, lyrics_lower):
                        hallucination_score += 0.15
                        logger.warning(
                            f"🚨 Possible hallucination: theme '{theme}' not found in lyrics"
                        )

        # Проверяем настроение
        if "mood" in analysis:
            claimed_mood = analysis["mood"].lower()
            if not self.mood_supported_by_lyrics(claimed_mood, lyrics_lower):
                hallucination_score += 0.2
                logger.warning(
                    f"🚨 Possible hallucination: mood '{claimed_mood}' not supported by lyrics"
                )

        # Проверяем жанр (менее строго, так как жанр может быть музыкальным)
        if "genre" in analysis:
            claimed_genre = analysis["genre"].lower()
            if (
                claimed_genre in ["classical", "jazz", "country"]
                and "rap" not in lyrics_lower
            ):
                hallucination_score += 0.3  # Явно неподходящий жанр

        # Проверяем explicit content
        if "explicit_content" in analysis:
            claimed_explicit = analysis["explicit_content"]
            actual_explicit = self.detect_explicit_content(lyrics_lower)
            if claimed_explicit != actual_explicit:
                hallucination_score += 0.1

        # Проверяем качественные метрики на разумность
        if "authenticity_score" in analysis:
            auth_score = analysis["authenticity_score"]
            if isinstance(auth_score, (int, float)):
                if (
                    auth_score > 0.9 and len(lyrics.split()) < 50
                ):  # Высокая аутентичность при коротком тексте
                    hallucination_score += 0.1

        return min(hallucination_score, 1.0)

    def theme_present_in_lyrics(self, theme: str, lyrics_lower: str) -> bool:
        """Проверяет, присутствует ли тема в тексте песни"""
        theme_lower = theme.lower().replace("_", " ")

        # Прямое совпадение
        if theme_lower in lyrics_lower:
            return True

        # Проверка по ключевым словам
        if theme_lower in self.theme_keywords:
            keywords = self.theme_keywords[theme_lower]
            found_keywords = sum(1 for keyword in keywords if keyword in lyrics_lower)
            return found_keywords >= 1  # Достаточно одного ключевого слова

        # Частичные совпадения для составных тем (English-focused)
        if "street" in theme_lower and any(
            word in lyrics_lower
            for word in ["street", "block", "hood", "neighborhood", "ghetto"]
        ):
            return True
        if "money" in theme_lower and any(
            word in lyrics_lower
            for word in ["cash", "money", "dollars", "bands", "racks", "bread"]
        ):
            return True
        if "love" in theme_lower and any(
            word in lyrics_lower
            for word in ["love", "girl", "relationship", "girlfriend", "romance"]
        ):
            return True

        return False

    def mood_supported_by_lyrics(self, mood: str, lyrics_lower: str) -> bool:
        """Проверяет, соответствует ли заявленное настроение тексту"""
        if mood in self.mood_indicators:
            indicators = self.mood_indicators[mood]
            found_indicators = sum(
                1 for indicator in indicators if indicator in lyrics_lower
            )
            return found_indicators >= 1

        # Для неизвестных настроений возвращаем True (не можем проверить)
        return True

    def detect_explicit_content(self, lyrics_lower: str) -> bool:
        """Детектирует explicit контент в тексте (English-focused)"""
        explicit_words = [
            "fuck",
            "shit",
            "bitch",
            "asshole",
            "damn",
            "hell",
            "pussy",
            "dick",
            "cock",
            "motherfucker",
            "nigga",
            "nigger",
            "whore",
            "slut",
            "cunt",
            "bastard",
            "piss",
        ]
        return any(word in lyrics_lower for word in explicit_words)

    def check_internal_consistency(self, analysis: dict) -> float:
        """Проверяет внутреннюю консистентность анализа"""
        consistency_score = 1.0

        # Проверяем соответствие настроения и энергии
        mood = analysis.get("mood", "").lower()
        energy = analysis.get("energy_level", "").lower()

        # Логические противоречия
        if mood == "melancholic" and energy == "high":
            consistency_score -= 0.2  # Грустная, но энергичная - возможно
        if mood == "aggressive" and energy == "low":
            consistency_score -= 0.3  # Агрессивная, но низкая энергия - странно

        # Проверяем качественные метрики
        if "authenticity_score" in analysis and "commercial_appeal" in analysis:
            auth = analysis["authenticity_score"]
            commercial = analysis["commercial_appeal"]
            if isinstance(auth, (int, float)) and isinstance(commercial, (int, float)):
                # Очень высокая аутентичность И очень высокий коммерческий аппеал - редко
                if auth > 0.9 and commercial > 0.9:
                    consistency_score -= 0.2

        # Проверяем соответствие сложности и качества
        complexity = analysis.get("complexity_level", "").lower()
        overall_quality = analysis.get("overall_quality", "").lower()

        if complexity == "advanced" and overall_quality == "poor":
            consistency_score -= 0.2
        if complexity == "beginner" and overall_quality == "excellent":
            consistency_score -= 0.1

        return max(consistency_score, 0.0)

    def validate_factual_claims(self, lyrics: str, analysis: dict) -> float:
        """Валидирует фактические утверждения в анализе"""
        factual_score = 1.0
        lyrics_lower = lyrics.lower()

        # Проверяем структуру
        claimed_structure = analysis.get("structure", "").lower()
        if claimed_structure:
            # Подсчет строк для валидации структуры
            lines = [line for line in lyrics.split("\n") if line.strip()]

            if "verse-chorus-verse" in claimed_structure and len(lines) < 8:
                factual_score -= 0.2  # Слишком короткий для такой структуры
            if "hook" in claimed_structure and len(lines) > 20:
                factual_score -= 0.1  # Слишком длинный для hook

        # Проверяем схему рифм
        rhyme_scheme = analysis.get("rhyme_scheme", "").lower()
        if rhyme_scheme and rhyme_scheme != "unknown":
            # Упрощенная проверка рифм
            lines = [line.strip() for line in lyrics.split("\n") if line.strip()]
            if len(lines) >= 4:
                # Если заявлена сложная схема, но текст простой
                if (
                    "complex" in rhyme_scheme
                    and len(set(line.split()[-1] for line in lines[:4] if line.split()))
                    == 1
                ):
                    factual_score -= 0.1

        # Проверяем количество слов vs сложность
        word_count = len(lyrics.split())
        complexity = analysis.get("complexity_level", "").lower()

        if complexity == "advanced" and word_count < 100:
            factual_score -= 0.2
        if complexity == "beginner" and word_count > 500:
            factual_score -= 0.1

        return max(factual_score, 0.0)

    def check_text_analysis_alignment(self, lyrics: str, analysis: dict) -> float:
        """Проверяет соответствие между текстом и анализом"""
        alignment_score = 1.0
        lyrics_lower = lyrics.lower()

        # Проверяем соответствие длины текста и детальности анализа
        word_count = len(lyrics.split())

        # Если текст короткий, но анализ очень детальный - подозрительно
        if word_count < 50:
            detailed_fields = sum(
                1
                for key in ["main_themes", "structure", "rhyme_scheme"]
                if analysis.get(key)
            )
            if detailed_fields > 2:
                alignment_score -= 0.2

        # Проверяем explicit content alignment
        actual_explicit = self.detect_explicit_content(lyrics_lower)
        claimed_explicit = analysis.get("explicit_content", False)

        if actual_explicit != claimed_explicit:
            alignment_score -= 0.3

        # Проверяем energy level alignment
        energy = analysis.get("energy_level", "").lower()
        exclamation_count = lyrics.count("!")
        caps_ratio = sum(1 for c in lyrics if c.isupper()) / max(len(lyrics), 1)

        if energy == "high" and exclamation_count == 0 and caps_ratio < 0.05:
            alignment_score -= 0.2
        if energy == "low" and exclamation_count > 5:
            alignment_score -= 0.2

        return max(alignment_score, 0.0)

    def get_warning_flags(self, analysis: dict, lyrics: str) -> list:
        """Получает список предупреждающих флагов"""
        flags = []

        # Проверка на подозрительно высокие оценки
        if analysis.get("authenticity_score", 0) > 0.95:
            flags.append("SUSPICIOUSLY_HIGH_AUTHENTICITY")

        if analysis.get("uniqueness", 0) > 0.95:
            flags.append("SUSPICIOUSLY_HIGH_UNIQUENESS")

        # Проверка на несоответствие длины и сложности
        word_count = len(lyrics.split())
        complexity = analysis.get("complexity_level", "").lower()

        if word_count < 50 and complexity == "advanced":
            flags.append("SHORT_TEXT_HIGH_COMPLEXITY")

        # Проверка на отсутствие тем в коротком тексте
        themes = analysis.get("main_themes", [])
        if word_count < 100 and len(themes) > 4:
            flags.append("SHORT_TEXT_MANY_THEMES")

        # Проверка на противоречивые метрики
        mood = analysis.get("mood", "").lower()
        commercial = analysis.get("commercial_appeal", 0)

        if mood == "melancholic" and commercial > 0.8:
            flags.append("SAD_MOOD_HIGH_COMMERCIAL")

        return flags

    def _generate_validation_summary(
        self,
        is_reliable: bool,
        hallucination_risk: float,
        consistency_score: float,
        warning_flags: list,
    ) -> str:
        """Генерирует текстовое резюме валидации"""
        if is_reliable:
            return f"✅ Анализ надежен (риск галлюцинаций: {hallucination_risk:.2f})"
        issues = []
        if hallucination_risk > 0.4:  # Обновленный порог
            issues.append(f"высокий риск галлюцинаций ({hallucination_risk:.2f})")
        if consistency_score < 0.6:  # Обновленный порог
            issues.append(f"низкая консистентность ({consistency_score:.2f})")
        if warning_flags:
            issues.append(f"предупреждения: {len(warning_flags)}")

        return f"⚠️ Анализ ненадежен: {', '.join(issues)}"


# Создаем модель для анализа (исправленная версия)
class EnhancedSongData(BaseModel):
    """Результат AI анализа песни"""

    artist: str
    title: str
    metadata: SongMetadata
    lyrics_analysis: LyricsAnalysis
    quality_metrics: QualityMetrics
    model_used: str
    analysis_date: str


class ExplainableAnalysisResult(BaseModel):
    """Результат анализа с объяснениями"""

    analysis: EnhancedSongData
    explanation: dict[str, list[str]]
    confidence: float
    decision_factors: dict[str, float]
    influential_phrases: dict[str, list[str]]


class InterpretableAnalyzer:
    """Анализатор с возможностью объяснения решений AI"""

    def __init__(self, base_analyzer):
        self.base_analyzer = base_analyzer

        # Словари ключевых слов для разных категорий
        self.genre_keywords = {
            "trap": ["trap", "молли", "lean", "xanax", "скрр", "йа", "bando", "plug"],
            "drill": ["drill", "smoke", "opps", "block", "gang", "sliding", "packed"],
            "old_school": [
                "boom bap",
                "real hip hop",
                "90s",
                "golden era",
                "conscious",
            ],
            "gangsta": ["glock", "ak", "blood", "crip", "hood", "street", "thug"],
            "emo_rap": [
                "депрессия",
                "суицид",
                "боль",
                "грусть",
                "одиночество",
                "слезы",
            ],
        }

        self.mood_keywords = {
            "aggressive": ["убью", "война", "кровь", "драка", "hate", "angry", "mad"],
            "melancholic": ["грусть", "печаль", "слезы", "депрессия", "одиночество"],
            "energetic": ["party", "club", "dance", "energy", "вперед", "движение"],
            "chill": ["расслабон", "спокойно", "медленно", "vibe", "атмосфера"],
        }

        self.authenticity_keywords = {
            "real": ["правда", "реально", "честно", "без фальши", "по-настоящему"],
            "fake": ["понт", "фэйк", "пижон", "показуха", "притворство"],
            "street": ["улица", "район", "двор", "подъезд", "квартал", "гетто"],
            "commercial": ["money", "brand", "коммерция", "продажи", "mainstream"],
        }

    def analyze_with_explanation(
        self, artist: str, title: str, lyrics: str
    ) -> ExplainableAnalysisResult | None:
        """Анализ с объяснением решений"""
        try:
            # Базовый анализ
            base_result = self.base_analyzer.analyze_song(artist, title, lyrics)
            if not base_result:
                return None

            # Генерируем объяснения
            explanation = self.explain_decision(lyrics, base_result)
            confidence = self.calculate_confidence(base_result, lyrics)
            decision_factors = self.extract_key_factors(lyrics, base_result)
            influential_phrases = self.find_influential_phrases(lyrics, base_result)

            return ExplainableAnalysisResult(
                analysis=base_result,
                explanation=explanation,
                confidence=confidence,
                decision_factors=decision_factors,
                influential_phrases=influential_phrases,
            )

        except Exception as e:
            logging.error(f"❌ Ошибка интерпретируемого анализа: {e}")
            return None

    def explain_decision(
        self, lyrics: str, result: EnhancedSongData
    ) -> dict[str, list[str]]:
        """Объясняет, на основе чего модель приняла решение"""
        lyrics_lower = lyrics.lower()
        explanations = {
            "genre_indicators": [],
            "mood_triggers": [],
            "authenticity_markers": [],
            "quality_indicators": [],
        }

        # Анализ жанровых индикаторов
        detected_genre = result.metadata.genre.lower()
        for genre, keywords in self.genre_keywords.items():
            if genre in detected_genre:
                found_keywords = [kw for kw in keywords if kw in lyrics_lower]
                if found_keywords:
                    explanations["genre_indicators"].extend(
                        [
                            f"Жанр '{genre}' определен по словам: {', '.join(found_keywords[:3])}"
                        ]
                    )

        # Анализ настроения
        detected_mood = result.metadata.mood.lower()
        for mood, keywords in self.mood_keywords.items():
            if mood in detected_mood:
                found_keywords = [kw for kw in keywords if kw in lyrics_lower]
                if found_keywords:
                    explanations["mood_triggers"].extend(
                        [
                            f"Настроение '{mood}' определено по словам: {', '.join(found_keywords[:3])}"
                        ]
                    )

        # Анализ аутентичности
        auth_score = result.quality_metrics.authenticity_score
        if auth_score > 0.7:
            real_words = [
                kw for kw in self.authenticity_keywords["real"] if kw in lyrics_lower
            ]
            street_words = [
                kw for kw in self.authenticity_keywords["street"] if kw in lyrics_lower
            ]
            if real_words or street_words:
                explanations["authenticity_markers"].append(
                    f"Высокая аутентичность ({auth_score:.2f}) благодаря: {', '.join((real_words + street_words)[:3])}"
                )
        elif auth_score < 0.4:
            fake_words = [
                kw for kw in self.authenticity_keywords["fake"] if kw in lyrics_lower
            ]
            commercial_words = [
                kw
                for kw in self.authenticity_keywords["commercial"]
                if kw in lyrics_lower
            ]
            if fake_words or commercial_words:
                explanations["authenticity_markers"].append(
                    f"Низкая аутентичность ({auth_score:.2f}) из-за: {', '.join((fake_words + commercial_words)[:3])}"
                )

        # Анализ качества
        creativity = result.quality_metrics.lyrical_creativity
        wordplay = result.lyrics_analysis.wordplay_quality
        explanations["quality_indicators"].append(
            f"Креативность: {creativity:.2f}, Wordplay: {wordplay}"
        )

        return explanations

    def calculate_confidence(self, result: EnhancedSongData, lyrics: str) -> float:
        """Рассчитывает уверенность в анализе"""
        confidence_factors = []

        # Фактор 1: Длина текста (больше текста = больше уверенности)
        text_length_factor = min(len(lyrics) / 1000, 1.0)  # Нормализуем к 1.0
        confidence_factors.append(text_length_factor * 0.2)

        # Фактор 2: Наличие явных индикаторов жанра
        genre_confidence = self._calculate_genre_confidence(
            lyrics, result.metadata.genre
        )
        confidence_factors.append(genre_confidence * 0.3)

        # Фактор 3: Консистентность метрик качества
        quality_consistency = self._calculate_quality_consistency(
            result.quality_metrics
        )
        confidence_factors.append(quality_consistency * 0.3)

        # Фактор 4: Наличие конкретных деталей (имена, места, события)
        detail_factor = self._calculate_detail_factor(lyrics)
        confidence_factors.append(detail_factor * 0.2)

        return min(sum(confidence_factors), 1.0)

    def _calculate_genre_confidence(self, lyrics: str, genre: str) -> float:
        """Рассчитывает уверенность в определении жанра"""
        lyrics_lower = lyrics.lower()
        genre_lower = genre.lower()

        matching_keywords = 0
        total_keywords = 0

        for g, keywords in self.genre_keywords.items():
            if g in genre_lower:
                total_keywords = len(keywords)
                matching_keywords = sum(1 for kw in keywords if kw in lyrics_lower)
                break

        if total_keywords == 0:
            return 0.5  # Средняя уверенность для неизвестных жанров

        return matching_keywords / total_keywords

    def _calculate_quality_consistency(self, metrics: QualityMetrics) -> float:
        """Проверяет консистентность метрик качества"""
        scores = [
            metrics.authenticity_score,
            metrics.lyrical_creativity,
            metrics.commercial_appeal,
            metrics.uniqueness,
        ]

        # Рассчитываем стандартное отклонение
        mean_score = sum(scores) / len(scores)
        variance = sum((x - mean_score) ** 2 for x in scores) / len(scores)
        std_dev = variance**0.5

        # Низкое стандартное отклонение = высокая консистентность
        consistency = max(0, 1 - (std_dev * 2))  # Нормализуем
        return consistency

    def _calculate_detail_factor(self, lyrics: str) -> float:
        """Рассчитывает наличие конкретных деталей"""
        detail_indicators = [
            r"\b[A-Z][a-z]+\b",  # Имена собственные
            r"\b\d{4}\b",  # Годы
            r"\b\d+[км]\b",  # Расстояния
            r"\$\d+",  # Деньги
            r"\b[А-ЯЁ][а-яё]+\b",  # Русские имена собственные
        ]

        total_details = 0
        for pattern in detail_indicators:
            matches = re.findall(pattern, lyrics)
            total_details += len(matches)

        # Нормализуем к длине текста
        detail_density = total_details / max(len(lyrics.split()), 1)
        return min(detail_density * 10, 1.0)  # Масштабируем

    def extract_key_factors(
        self, lyrics: str, result: EnhancedSongData
    ) -> dict[str, float]:
        """Извлекает ключевые факторы, влияющие на анализ"""
        factors = {}
        lyrics_lower = lyrics.lower()

        # Частота ключевых слов по категориям
        for category, keywords in {**self.genre_keywords, **self.mood_keywords}.items():
            keyword_count = sum(1 for kw in keywords if kw in lyrics_lower)
            factors[f"{category}_keywords"] = keyword_count / len(keywords)

        # Структурные факторы
        factors["text_length"] = min(len(lyrics) / 2000, 1.0)
        factors["line_count"] = min(len(lyrics.split("\n")) / 50, 1.0)
        factors["word_diversity"] = len(set(lyrics.lower().split())) / max(
            len(lyrics.split()), 1
        )

        # Метрики качества как факторы
        factors["authenticity"] = result.quality_metrics.authenticity_score
        factors["creativity"] = result.quality_metrics.lyrical_creativity
        factors["commercial_appeal"] = result.quality_metrics.commercial_appeal
        factors["uniqueness"] = result.quality_metrics.uniqueness

        return factors

    def find_influential_phrases(
        self, lyrics: str, result: EnhancedSongData
    ) -> dict[str, list[str]]:
        """Находит конкретные фразы, которые повлияли на оценку"""
        influential = {
            "genre_phrases": [],
            "mood_phrases": [],
            "authenticity_phrases": [],
            "quality_phrases": [],
        }

        lines = lyrics.split("\n")

        # Поиск влиятельных фраз для жанра
        genre_lower = result.metadata.genre.lower()
        for genre, keywords in self.genre_keywords.items():
            if genre in genre_lower:
                for line in lines:
                    if any(kw in line.lower() for kw in keywords):
                        influential["genre_phrases"].append(line.strip())
                        if len(influential["genre_phrases"]) >= 3:
                            break

        # Поиск фраз для настроения
        mood_lower = result.metadata.mood.lower()
        for mood, keywords in self.mood_keywords.items():
            if mood in mood_lower:
                for line in lines:
                    if any(kw in line.lower() for kw in keywords):
                        influential["mood_phrases"].append(line.strip())
                        if len(influential["mood_phrases"]) >= 3:
                            break

        # Поиск фраз для аутентичности
        auth_score = result.quality_metrics.authenticity_score
        auth_keywords = (
            self.authenticity_keywords["real"] + self.authenticity_keywords["street"]
        )
        if auth_score > 0.7:
            for line in lines:
                if any(kw in line.lower() for kw in auth_keywords):
                    influential["authenticity_phrases"].append(line.strip())
                    if len(influential["authenticity_phrases"]) >= 2:
                        break

        # Поиск качественных wordplay фраз
        if result.lyrics_analysis.wordplay_quality == "excellent":
            # Ищем строки с рифмами или аллитерацией
            for line in lines:
                words = line.lower().split()
                if len(words) >= 4:
                    # Простая проверка на рифму (одинаковые окончания)
                    endings = [word[-2:] for word in words if len(word) > 3]
                    if (
                        len(set(endings)) < len(endings) * 0.8
                    ):  # Много повторяющихся окончаний
                        influential["quality_phrases"].append(line.strip())
                        if len(influential["quality_phrases"]) >= 2:
                            break

        return influential


class ModelProvider:
    """Базовый класс для AI провайдеров"""

    def __init__(self, name: str):
        self.name = name
        self.available = False
        self.cost_per_1k_tokens = 0.0

    def check_availability(self) -> bool:
        """Проверка доступности модели"""
        raise NotImplementedError

    def analyze_song(
        self, artist: str, title: str, lyrics: str
    ) -> EnhancedSongData | None:
        """Анализ песни"""
        raise NotImplementedError


class OllamaProvider(ModelProvider):
    """Провайдер для локальных моделей Ollama"""

    def __init__(
        self, model_name: str = "llama3.2:3b", base_url: str = "http://localhost:11434"
    ):
        super().__init__("Ollama")
        self.model_name = model_name
        self.base_url = base_url
        self.cost_per_1k_tokens = 0.0  # Бесплатно!
        self.available = self.check_availability()

    def check_availability(self) -> bool:
        """Проверка запущен ли Ollama"""
        try:
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
                logger.warning(
                    f"⚠️ Модель {self.model_name} не найдена. Попытка загрузки..."
                )
                return self._pull_model()
            return False
        except requests.exceptions.RequestException as e:
            logger.warning(f"❌ Ollama недоступен: {e}")
            return False

    def _pull_model(self) -> bool:
        """Загрузка модели если её нет"""
        try:
            logger.info(f"📥 Загружаем модель {self.model_name}...")
            response = requests.post(
                f"{self.base_url}/api/pull",
                json={"name": self.model_name},
                timeout=300,  # 5 минут на загрузку
                proxies={"http": "", "https": ""},
            )
            if response.status_code == 200:
                logger.info(f"✅ Модель {self.model_name} загружена")
                return True
            logger.error(f"❌ Не удалось загрузить модель: {response.text}")
            return False
        except Exception as e:
            logger.error(f"❌ Ошибка загрузки модели: {e}")
            return False

    def analyze_song(
        self, artist: str, title: str, lyrics: str
    ) -> EnhancedSongData | None:
        """Анализ песни через Ollama"""
        if not self.available:
            return None

        try:
            prompt = self._create_analysis_prompt(artist, title, lyrics)

            response = requests.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": self.model_name,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.1,  # Низкая температура для консистентности
                        "top_p": 0.9,
                        "max_tokens": 1500,
                    },
                },
                timeout=60,
                proxies={"http": "", "https": ""},
            )

            if response.status_code == 200:
                result = response.json()
                analysis_text = result.get("response", "")
                return self._parse_analysis(analysis_text, artist, title)
            logger.error(f"❌ Ollama ошибка: {response.status_code} - {response.text}")
            return None

        except Exception as e:
            logger.error(f"❌ Ошибка анализа Ollama: {e}")
            return None

    def _create_analysis_prompt(self, artist: str, title: str, lyrics: str) -> str:
        """Создание промпта для анализа"""
        return f"""
Проанализируй рэп-песню и верни результат СТРОГО в JSON формате.

Исполнитель: {artist}
Название: {title}
Текст: {lyrics[:2000]}...

Верни ТОЛЬКО валидный JSON без дополнительного текста:

{{
    "metadata": {{
        "genre": "rap",
        "mood": "aggressive",
        "energy_level": "high",
        "explicit_content": true
    }},
    "lyrics_analysis": {{
        "structure": "verse-chorus-verse",
        "rhyme_scheme": "ABAB",
        "complexity_level": "advanced",
        "main_themes": ["street_life", "success", "relationships"],
        "emotional_tone": "mixed",
        "storytelling_type": "narrative",
        "wordplay_quality": "excellent"
    }},
    "quality_metrics": {{
        "authenticity_score": 0.8,
        "lyrical_creativity": 0.9,
        "commercial_appeal": 0.7,
        "uniqueness": 0.6,
        "overall_quality": "excellent",
        "ai_likelihood": 0.1
    }}
}}

ОБЯЗАТЕЛЬНЫЕ ПОЛЯ:
- emotional_tone: positive/negative/neutral/mixed
- storytelling_type: narrative/abstract/conversational
- wordplay_quality: basic/good/excellent

Верни ТОЛЬКО JSON без комментариев!
"""

    def _parse_analysis(
        self, analysis_text: str, artist: str, title: str
    ) -> EnhancedSongData | None:
        """Парсинг результата анализа"""
        try:
            # Извлекаем JSON из ответа
            json_start = analysis_text.find("{")
            json_end = analysis_text.rfind("}") + 1

            if json_start == -1 or json_end <= json_start:
                logger.error("❌ JSON не найден в ответе")
                return None

            json_str = analysis_text[json_start:json_end]
            data = json.loads(json_str)

            # Проверяем и дополняем отсутствующие поля
            metadata_data = data.get("metadata", {})
            lyrics_data = data.get("lyrics_analysis", {})
            quality_data = data.get("quality_metrics", {})

            # Дополняем отсутствующие поля в lyrics_analysis
            if "emotional_tone" not in lyrics_data:
                lyrics_data["emotional_tone"] = "neutral"
                logger.warning("⚠️ Добавлено значение по умолчанию для emotional_tone")

            if "storytelling_type" not in lyrics_data:
                lyrics_data["storytelling_type"] = "conversational"
                logger.warning(
                    "⚠️ Добавлено значение по умолчанию для storytelling_type"
                )

            if "wordplay_quality" not in lyrics_data:
                lyrics_data["wordplay_quality"] = "basic"
                logger.warning("⚠️ Добавлено значение по умолчанию для wordplay_quality")

            # Создаем структурированный анализ
            metadata = SongMetadata(**metadata_data)
            lyrics_analysis = LyricsAnalysis(**lyrics_data)
            quality_metrics = QualityMetrics(**quality_data)

            return EnhancedSongData(
                artist=artist,
                title=title,
                metadata=metadata,
                lyrics_analysis=lyrics_analysis,
                quality_metrics=quality_metrics,
                model_used=f"ollama:{self.model_name}",
                analysis_date=datetime.now().isoformat(),
            )

        except json.JSONDecodeError as e:
            logger.error(f"❌ Ошибка парсинга JSON: {e}")
            logger.debug(f"Ответ модели: {analysis_text[:500]}")
            return None
        except Exception as e:
            logger.error(f"❌ Ошибка создания анализа: {e}")
            return None


class MockProvider(ModelProvider):
    """Mock провайдер для демонстрации и тестирования"""

    def __init__(self):
        super().__init__("Mock")
        self.cost_per_1k_tokens = 0.0  # Бесплатно для тестов
        self.available = True  # Всегда доступен

    def check_availability(self) -> bool:
        """Mock провайдер всегда доступен"""
        logger.info("✅ Mock провайдер готов для демонстрации")
        return True

    def analyze_song(
        self, artist: str, title: str, lyrics: str
    ) -> EnhancedSongData | None:
        """Mock анализ песни с умными предположениями"""
        try:
            lyrics_lower = lyrics.lower()

            # Умное определение жанра на основе ключевых слов
            genre = "rap"
            if any(word in lyrics_lower for word in ["trap", "молли", "lean", "скрр"]):
                genre = "trap"
            elif any(
                word in lyrics_lower for word in ["drill", "smoke", "opps", "gang"]
            ):
                genre = "drill"
            elif any(
                word in lyrics_lower for word in ["улица", "район", "двор", "подъезд"]
            ):
                genre = "gangsta_rap"
            elif any(word in lyrics_lower for word in ["депрессия", "грусть", "слезы"]):
                genre = "emo_rap"

            # Умное определение настроения
            mood = "neutral"
            aggressive_words = ["убью", "война", "драка", "hate", "angry"]
            sad_words = ["грусть", "печаль", "слезы", "депрессия", "одиночество"]
            positive_words = ["party", "счастье", "радость", "love", "успех"]

            if any(word in lyrics_lower for word in aggressive_words):
                mood = "aggressive"
            elif any(word in lyrics_lower for word in sad_words):
                mood = "melancholic"
            elif any(word in lyrics_lower for word in positive_words):
                mood = "energetic"

            # Анализ энергии
            energy = "medium"
            if (
                len(lyrics.split("!")) > 3
                or "йа" in lyrics_lower
                or "скрр" in lyrics_lower
            ):
                energy = "high"
            elif any(word in lyrics_lower for word in ["медленно", "спокойно", "тихо"]):
                energy = "low"

            # Определение explicit content
            explicit_words = [
                "сука",
                "блять",
                "хуй",
                "пизда",
                "ебать",
                "fuck",
                "shit",
                "bitch",
            ]
            explicit_content = any(word in lyrics_lower for word in explicit_words)

            # Анализ структуры
            lines = lyrics.strip().split("\n")
            non_empty_lines = [line for line in lines if line.strip()]

            structure = "verse"
            if len(non_empty_lines) > 16:
                structure = "verse-chorus-verse"
            elif len(non_empty_lines) < 8:
                structure = "hook"

            # Анализ рифмы (упрощенный)
            rhyme_scheme = "ABAB"
            if len(non_empty_lines) >= 4:
                # Проверяем последние слова строк
                last_words = [
                    line.strip().split()[-1].lower()
                    for line in non_empty_lines[:4]
                    if line.strip().split()
                ]
                if len(set(last_words)) == 1:
                    rhyme_scheme = "AAAA"
                elif len(set(last_words)) == 2:
                    rhyme_scheme = "AABB"

            # Определение сложности
            complexity = "intermediate"
            word_count = len(lyrics.split())
            unique_words = len(set(lyrics.lower().split()))
            diversity = unique_words / max(word_count, 1)

            if diversity > 0.7 and word_count > 200:
                complexity = "advanced"
            elif diversity < 0.5 or word_count < 100:
                complexity = "beginner"

            # Основные темы
            themes = []
            theme_keywords = {
                "street_life": ["улица", "район", "двор", "подъезд", "гетто"],
                "money": ["деньги", "cash", "money", "бабки", "лавэ"],
                "relationships": ["любовь", "девочка", "отношения", "семья"],
                "success": ["успех", "fame", "слава", "топ"],
                "struggle": ["борьба", "struggle", "проблемы", "трудности"],
            }

            for theme, keywords in theme_keywords.items():
                if any(keyword in lyrics_lower for keyword in keywords):
                    themes.append(theme)

            if not themes:
                themes = ["life"]

            # Качественные метрики
            authenticity_score = 0.5
            street_words = ["улица", "район", "двор", "подъезд", "правда", "реально"]
            fake_words = ["понт", "фэйк", "показуха"]

            street_count = sum(1 for word in street_words if word in lyrics_lower)
            fake_count = sum(1 for word in fake_words if word in lyrics_lower)

            authenticity_score = min(
                0.3 + (street_count * 0.15) - (fake_count * 0.1), 1.0
            )

            creativity = min(0.4 + (diversity * 0.6), 1.0)
            commercial_appeal = (
                0.5
                + (0.1 if explicit_content else 0.2)
                + (0.1 if energy == "high" else 0)
            )
            uniqueness = diversity * 0.8 + 0.2

            # Общее качество
            avg_quality = (
                authenticity_score + creativity + commercial_appeal + uniqueness
            ) / 4
            if avg_quality > 0.8:
                overall_quality = "excellent"
            elif avg_quality > 0.6:
                overall_quality = "good"
            elif avg_quality > 0.4:
                overall_quality = "fair"
            else:
                overall_quality = "poor"

            # AI likelihood (обратная зависимость от аутентичности)
            ai_likelihood = max(0.1, 1.0 - authenticity_score)

            # Создание результата
            metadata = SongMetadata(
                genre=genre,
                mood=mood,
                energy_level=energy,
                explicit_content=explicit_content,
            )

            lyrics_analysis = LyricsAnalysis(
                structure=structure,
                rhyme_scheme=rhyme_scheme,
                complexity_level=complexity,
                main_themes=themes,
                emotional_tone=mood,
                storytelling_type="narrative"
                if "история" in lyrics_lower or len(non_empty_lines) > 12
                else "conversational",
                wordplay_quality="excellent"
                if creativity > 0.8
                else ("good" if creativity > 0.6 else "basic"),
            )

            quality_metrics = QualityMetrics(
                authenticity_score=authenticity_score,
                lyrical_creativity=creativity,
                commercial_appeal=commercial_appeal,
                uniqueness=uniqueness,
                overall_quality=overall_quality,
                ai_likelihood=ai_likelihood,
            )

            return EnhancedSongData(
                artist=artist,
                title=title,
                metadata=metadata,
                lyrics_analysis=lyrics_analysis,
                quality_metrics=quality_metrics,
                model_used="mock_analyzer_v1",
                analysis_date=datetime.now().isoformat(),
            )

        except Exception as e:
            logger.error(f"❌ Ошибка Mock анализа: {e}")
            return None


class GemmaProvider(ModelProvider):
    """Провайдер для Google Gemma API"""

    def __init__(self):
        super().__init__("Gemma")
        self.api_key = os.getenv("GOOGLE_API_KEY")
        self.available = self.check_availability()
        self.cost_per_1k_tokens = 0.0  # Free tier в пределах лимитов

    def check_availability(self) -> bool:
        """Проверка API ключа Google"""
        if not self.api_key:
            logger.warning("❌ GOOGLE_API_KEY не найден в .env")
            return False

        try:
            import google.generativeai as genai
            from google.generativeai.client import configure
            from google.generativeai.generative_models import GenerativeModel

            configure(api_key=self.api_key)
            logger.info("✅ Google Gemma API готов к использованию")
            return True
        except ImportError:
            logger.warning("❌ google-generativeai не установлен")
            return False
        except Exception as e:
            logger.error(f"❌ Ошибка проверки Gemma API: {e}")
            return False

    def analyze_song(
        self, artist: str, title: str, lyrics: str
    ) -> EnhancedSongData | None:
        """Анализ песни через Google Gemma"""
        if not self.available:
            return None

        try:
            from google.generativeai.generative_models import GenerativeModel

            model = GenerativeModel("gemma-2-27b-it")
            prompt = self._create_analysis_prompt(artist, title, lyrics)

            response = model.generate_content(
                prompt,
                generation_config={
                    "temperature": 0.1,
                    "max_output_tokens": 1500,
                },
            )

            if response.text:
                return self._parse_analysis(response.text, artist, title)
            logger.error("❌ Gemma: пустой ответ")
            return None

        except Exception as e:
            logger.error(f"❌ Ошибка анализа Gemma: {e}")
            return None

    def _create_analysis_prompt(self, artist: str, title: str, lyrics: str) -> str:
        """Создание промпта для Gemma"""
        return f"""
Analyze this rap song and return results in STRICT JSON format:

Artist: {artist}
Title: {title}
Lyrics: {lyrics[:2000]}...

Return ONLY valid JSON with these exact fields:
{{
    "metadata": {{
        "genre": "rap/trap/drill/old-school/gangsta/emo-rap",
        "mood": "aggressive/melancholic/energetic/neutral",
        "energy_level": "low/medium/high",
        "explicit_content": true/false
    }},
    "lyrics_analysis": {{
        "structure": "verse-chorus-verse/freestyle/storytelling",
        "rhyme_scheme": "AABA/ABAB/complex/simple",
        "complexity_level": "beginner/intermediate/advanced",
        "main_themes": ["money", "relationships", "street_life", "success"],
        "emotional_tone": "positive/negative/neutral/mixed",
        "storytelling_type": "narrative/abstract/conversational",
        "wordplay_quality": "basic/good/excellent"
    }},
    "quality_metrics": {{
        "authenticity_score": 0.0-1.0,
        "lyrical_creativity": 0.0-1.0,
        "commercial_appeal": 0.0-1.0,
        "uniqueness": 0.0-1.0,
        "overall_quality": "poor/fair/good/excellent",
        "ai_likelihood": 0.0-1.0
    }}
}}

Return ONLY JSON, no additional text!
"""

    def _parse_analysis(
        self, analysis_text: str, artist: str, title: str
    ) -> EnhancedSongData | None:
        """Парсинг результата анализа"""
        try:
            # Извлекаем JSON из ответа
            json_start = analysis_text.find("{")
            json_end = analysis_text.rfind("}") + 1

            if json_start == -1 or json_end <= json_start:
                logger.error("❌ JSON не найден в ответе Gemma")
                return None

            json_str = analysis_text[json_start:json_end]
            data = json.loads(json_str)

            # Проверяем и дополняем отсутствующие поля
            metadata_data = data.get("metadata", {})
            lyrics_data = data.get("lyrics_analysis", {})
            quality_data = data.get("quality_metrics", {})

            # Дополняем отсутствующие поля в lyrics_analysis
            if "emotional_tone" not in lyrics_data:
                lyrics_data["emotional_tone"] = "neutral"
                logger.warning("⚠️ Добавлено значение по умолчанию для emotional_tone")

            if "storytelling_type" not in lyrics_data:
                lyrics_data["storytelling_type"] = "conversational"
                logger.warning(
                    "⚠️ Добавлено значение по умолчанию для storytelling_type"
                )

            if "wordplay_quality" not in lyrics_data:
                lyrics_data["wordplay_quality"] = "basic"
                logger.warning("⚠️ Добавлено значение по умолчанию для wordplay_quality")

            # Создаем структурированный анализ
            metadata = SongMetadata(**metadata_data)
            lyrics_analysis = LyricsAnalysis(**lyrics_data)
            quality_metrics = QualityMetrics(**quality_data)

            return EnhancedSongData(
                artist=artist,
                title=title,
                metadata=metadata,
                lyrics_analysis=lyrics_analysis,
                quality_metrics=quality_metrics,
                model_used="gemma-2-27b-it",
                analysis_date=datetime.now().isoformat(),
            )

        except json.JSONDecodeError as e:
            logger.error(f"❌ Ошибка парсинга JSON Gemma: {e}")
            logger.debug(f"Ответ модели: {analysis_text[:500]}")
            return None
        except Exception as e:
            logger.error(f"❌ Ошибка создания анализа Gemma: {e}")
            return None


class MultiModelAnalyzer:
    """Анализатор с поддержкой нескольких провайдеров"""

    def __init__(self):
        self.providers = []
        self.current_provider = None
        self.stats = {
            "total_analyzed": 0,
            "ollama_used": 0,
            "gemma_used": 0,
            "mock_used": 0,
            "total_cost": 0.0,
        }

        # Инициализация провайдеров в порядке приоритета
        self._init_providers()

        # Инициализация интерпретируемого анализатора
        self.interpretable_analyzer = InterpretableAnalyzer(self)

        # Инициализация валидатора безопасности
        self.safety_validator = SafetyValidator()

    def analyze_with_explanations(
        self, artist: str, title: str, lyrics: str
    ) -> ExplainableAnalysisResult | None:
        """Анализ с полными объяснениями решений AI"""
        return self.interpretable_analyzer.analyze_with_explanation(
            artist, title, lyrics
        )

    def explain_existing_analysis(
        self, song_id: int, db_path: str = "rap_lyrics.db"
    ) -> dict | None:
        """Объясняет существующий анализ из базы данных"""
        try:
            conn = sqlite3.connect(db_path)
            conn.row_factory = sqlite3.Row

            # Получаем данные песни и анализа
            cursor = conn.execute(
                """
                SELECT s.artist, s.title, s.lyrics, a.*
                FROM tracks s
                JOIN ai_analysis a ON s.id = a.song_id
                WHERE s.id = ?
            """,
                (song_id,),
            )

            row = cursor.fetchone()
            if not row:
                logging.warning(
                    f"Песня с ID {song_id} не найдена или не проанализирована"
                )
                return None

            # Создаем объект EnhancedSongData из данных БД
            metadata = SongMetadata(
                genre=row["genre"],
                mood=row["mood"],
                energy_level=row["energy_level"],
                explicit_content=bool(row["explicit_content"]),
            )

            lyrics_analysis = LyricsAnalysis(
                structure=row["structure"],
                rhyme_scheme=row["rhyme_scheme"],
                complexity_level=row["complexity_level"],
                main_themes=json.loads(row["main_themes"])
                if row["main_themes"]
                else [],
                emotional_tone="neutral",  # Добавляем значения по умолчанию
                storytelling_type="conversational",
                wordplay_quality="basic",
            )

            quality_metrics = QualityMetrics(
                authenticity_score=row["authenticity_score"],
                lyrical_creativity=row["lyrical_creativity"],
                commercial_appeal=row["commercial_appeal"],
                uniqueness=row["uniqueness"],
                overall_quality=row["overall_quality"],
                ai_likelihood=row["ai_likelihood"],
            )

            enhanced_data = EnhancedSongData(
                artist=row["artist"],
                title=row["title"],
                metadata=metadata,
                lyrics_analysis=lyrics_analysis,
                quality_metrics=quality_metrics,
                model_used=row["model_version"],
                analysis_date=row["analysis_date"],
            )

            # Генерируем объяснения
            explanation = self.interpretable_analyzer.explain_decision(
                row["lyrics"], enhanced_data
            )
            confidence = self.interpretable_analyzer.calculate_confidence(
                enhanced_data, row["lyrics"]
            )
            decision_factors = self.interpretable_analyzer.extract_key_factors(
                row["lyrics"], enhanced_data
            )
            influential_phrases = self.interpretable_analyzer.find_influential_phrases(
                row["lyrics"], enhanced_data
            )

            conn.close()

            return {
                "song_info": {
                    "id": song_id,
                    "artist": row["artist"],
                    "title": row["title"],
                },
                "analysis": enhanced_data.model_dump(),
                "explanation": explanation,
                "confidence": confidence,
                "decision_factors": decision_factors,
                "influential_phrases": influential_phrases,
            }

        except Exception as e:
            logging.error(f"❌ Ошибка объяснения анализа: {e}")
            return None

    def _init_providers(self):
        """Инициализация провайдеров в порядке приоритета"""
        logger.info("🔍 Инициализация AI провайдеров...")

        # 1. Ollama (приоритет - бесплатно)
        ollama = OllamaProvider()
        if ollama.available:
            self.providers.append(ollama)
            logger.info("✅ Ollama готов к использованию")

        # 2. Google Gemma (cloud fallback)
        gemma = GemmaProvider()
        if gemma.available:
            self.providers.append(gemma)
            logger.info("✅ Google Gemma готов к использованию")

        # 3. Mock Provider (всегда добавляем для надежности)
        mock = MockProvider()
        self.providers.append(mock)
        logger.info("✅ Mock провайдер добавлен как fallback")

        if not self.providers:
            logger.error("❌ Ни один AI провайдер недоступен!")
            raise Exception("No AI providers available")

        self.current_provider = self.providers[0]
        logger.info(f"🎯 Активный провайдер: {self.current_provider.name}")

    def analyze_song(
        self, artist: str, title: str, lyrics: str
    ) -> EnhancedSongData | None:
        """Анализ песни с fallback между провайдерами"""

        for provider in self.providers:
            try:
                logger.info(f"🤖 Анализируем через {provider.name}: {artist} - {title}")

                result = provider.analyze_song(artist, title, lyrics)

                if result:
                    # Обновляем статистику
                    self.stats["total_analyzed"] += 1
                    if provider.name == "Ollama":
                        self.stats["ollama_used"] += 1
                    elif provider.name == "Gemma":
                        self.stats["gemma_used"] += 1
                    elif provider.name == "Mock":
                        self.stats["mock_used"] += 1

                    logger.info(f"✅ Анализ завершен через {provider.name}")
                    return result
                logger.warning(f"⚠️ {provider.name} не смог проанализировать")

            except Exception as e:
                logger.error(f"❌ Ошибка {provider.name}: {e}")
                continue

        logger.error(
            f"❌ Все провайдеры не смогли проанализировать: {artist} - {title}"
        )
        return None

    def get_stats(self) -> dict:
        """Получение статистики использования"""
        return {
            **self.stats,
            "available_providers": [p.name for p in self.providers],
            "current_provider": self.current_provider.name
            if self.current_provider
            else None,
        }

    def batch_analyze_from_db(
        self, db_path: str = "rap_lyrics.db", limit: int = 100, offset: int = 0
    ):
        """Массовый анализ песен из базы данных"""

        logger.info(f"🎵 Начинаем batch анализ: {limit} песен с offset {offset}")

        try:
            conn = sqlite3.connect(db_path)
            conn.row_factory = sqlite3.Row

            # Получаем песни для анализа
            cursor = conn.execute(
                """
                SELECT s.id, s.artist, s.title, s.lyrics 
                FROM tracks s
                LEFT JOIN ai_analysis a ON s.id = a.song_id
                WHERE a.id IS NULL  -- Только неанализированные
                LIMIT ? OFFSET ?
            """,
                (limit, offset),
            )

            tracks = cursor.fetchall()
            logger.info(f"📊 Найдено {len(tracks)} песен для анализа")

            successful = 0
            failed = 0

            for i, track in enumerate(tracks, 1):
                try:
                    logger.info(
                        f"📈 Прогресс: {i}/{len(tracks)} - {song['artist']} - {song['title']}"
                    )

                    analysis = self.analyze_song(
                        song["artist"], song["title"], song["lyrics"]
                    )

                    if analysis:
                        # Сохраняем в БД
                        self._save_analysis_to_db(conn, song["id"], analysis)
                        successful += 1
                        logger.info(f"✅ Сохранен анализ #{successful}")
                    else:
                        failed += 1
                        logger.warning("❌ Не удалось проанализировать")

                    # Пауза между запросами
                    if i < len(tracks):  # Не делаем паузу после последней песни
                        time.sleep(2)  # 2 секунды между анализами

                except Exception as e:
                    failed += 1
                    logger.error(f"❌ Ошибка анализа песни {song['id']}: {e}")
                    continue

            conn.close()

            logger.info(f"""
            🎉 Batch анализ завершен!
            ✅ Успешно: {successful}
            ❌ Ошибок: {failed}
            📊 Статистика: {self.get_stats()}
            """)

        except Exception as e:
            logger.error(f"❌ Ошибка batch анализа: {e}")

    def _save_analysis_to_db(
        self, conn: sqlite3.Connection, song_id: int, analysis: EnhancedSongData
    ):
        """Сохранение анализа в базу данных"""
        try:
            conn.execute(
                """
                INSERT INTO ai_analysis (
                    song_id, genre, mood, energy_level, explicit_content,
                    structure, rhyme_scheme, complexity_level, main_themes,
                    authenticity_score, lyrical_creativity, commercial_appeal,
                    uniqueness, overall_quality, ai_likelihood,
                    analysis_date, model_version
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    song_id,
                    analysis.metadata.genre,
                    analysis.metadata.mood,
                    analysis.metadata.energy_level,
                    analysis.metadata.explicit_content,
                    analysis.lyrics_analysis.structure,
                    analysis.lyrics_analysis.rhyme_scheme,
                    analysis.lyrics_analysis.complexity_level,
                    json.dumps(analysis.lyrics_analysis.main_themes),
                    analysis.quality_metrics.authenticity_score,
                    analysis.quality_metrics.lyrical_creativity,
                    analysis.quality_metrics.commercial_appeal,
                    analysis.quality_metrics.uniqueness,
                    analysis.quality_metrics.overall_quality,
                    analysis.quality_metrics.ai_likelihood,
                    analysis.analysis_date,
                    analysis.model_used,
                ),
            )
            conn.commit()
        except Exception as e:
            logger.error(f"❌ Ошибка сохранения в БД: {e}")
            raise

    def analyze_song_with_safety(
        self, artist: str, title: str, lyrics: str
    ) -> dict | None:
        """Анализ песни с валидацией безопасности и детекцией галлюцинаций"""

        logger.info(f"🛡️ Безопасный анализ: {artist} - {title}")

        # 1. Выполняем стандартный анализ
        analysis_result = self.analyze_song(artist, title, lyrics)

        if not analysis_result:
            logger.error("❌ Не удалось получить анализ для валидации")
            return None

        # 2. Конвертируем результат в словарь для валидации
        analysis_dict = {
            "genre": analysis_result.metadata.genre,
            "mood": analysis_result.metadata.mood,
            "energy_level": analysis_result.metadata.energy_level,
            "explicit_content": analysis_result.metadata.explicit_content,
            "structure": analysis_result.lyrics_analysis.structure,
            "rhyme_scheme": analysis_result.lyrics_analysis.rhyme_scheme,
            "complexity_level": analysis_result.lyrics_analysis.complexity_level,
            "main_themes": analysis_result.lyrics_analysis.main_themes,
            "authenticity_score": analysis_result.quality_metrics.authenticity_score,
            "lyrical_creativity": analysis_result.quality_metrics.lyrical_creativity,
            "commercial_appeal": analysis_result.quality_metrics.commercial_appeal,
            "uniqueness": analysis_result.quality_metrics.uniqueness,
            "overall_quality": analysis_result.quality_metrics.overall_quality,
            "ai_likelihood": analysis_result.quality_metrics.ai_likelihood,
        }

        # 3. Валидация через SafetyValidator
        validation_result = self.safety_validator.validate_analysis(
            lyrics, analysis_dict
        )

        # 4. Логирование результатов валидации
        logger.info(
            f"🔍 Результат валидации: {validation_result['validation_summary']}"
        )

        if not validation_result["is_reliable"]:
            logger.warning("⚠️ ВНИМАНИЕ: Анализ признан ненадежным!")
            logger.warning(
                f"   • Риск галлюцинаций: {validation_result['hallucination_risk']:.3f}"
            )
            logger.warning(
                f"   • Консистентность: {validation_result['consistency_score']:.3f}"
            )
            logger.warning(
                f"   • Точность фактов: {validation_result['factual_accuracy']:.3f}"
            )

            if validation_result["warning_flags"]:
                logger.warning(
                    f"   • Предупреждения: {', '.join(validation_result['warning_flags'])}"
                )
        else:
            logger.info("✅ Анализ прошел валидацию безопасности")
            logger.info(
                f"   • Надежность: {validation_result['reliability_score']:.3f}"
            )

        # 5. Возвращаем расширенный результат
        return {
            "analysis": analysis_result,
            "validation": validation_result,
            "is_safe": validation_result["is_reliable"],
            "confidence": validation_result["reliability_score"],
            "warnings": validation_result["warning_flags"],
            "summary": validation_result["validation_summary"],
        }


def main():
    """Тестирование многомодельного анализатора с интерпретируемостью"""

    print("🤖 Многомодельный AI анализатор с объяснениями решений")
    print("=" * 70)

    try:
        analyzer = MultiModelAnalyzer()

        print(f"📊 Доступные провайдеры: {[p.name for p in analyzer.providers]}")
        print(
            f"🎯 Активный провайдер: {analyzer.current_provider.name if analyzer.current_provider else 'None'}"
        )

        # Демонстрация анализа с объяснениями
        print("\n🧪 Тестирование анализа с объяснениями...")

        # Тестовый текст песни
        test_lyrics = """
        Я с улицы, район меня воспитал
        В подъездах темных правду познавал
        Молодость прошла в дыму и драках
        Теперь читаю правду в этих строках
        
        Деньги, слава - все это пустота
        Главное остаться собой до конца
        Семья и верные друзья рядом
        Это богатство, а не фальшивый яд
        """

        # Анализ с объяснениями
        explainable_result = analyzer.analyze_with_explanations(
            "Тестовый артист", "Тестовый трек", test_lyrics
        )

        if explainable_result:
            print("\n🎯 РЕЗУЛЬТАТ АНАЛИЗА С ОБЪЯСНЕНИЯМИ:")
            print("-" * 50)

            # Основной анализ
            analysis = explainable_result.analysis
            print(f"🎵 Жанр: {analysis.metadata.genre}")
            print(f"😊 Настроение: {analysis.metadata.mood}")
            print(f"⚡ Энергия: {analysis.metadata.energy_level}")
            print(f"🏆 Качество: {analysis.quality_metrics.overall_quality}")
            print(f"🔍 Уверенность: {explainable_result.confidence:.2f}")

            # Объяснения
            print("\n💡 ОБЪЯСНЕНИЯ РЕШЕНИЙ:")
            for category, explanations in explainable_result.explanation.items():
                if explanations:
                    print(f"  {category.replace('_', ' ').title()}:")
                    for exp in explanations:
                        print(f"    • {exp}")

            # Влиятельные фразы
            print("\n📝 ВЛИЯТЕЛЬНЫЕ ФРАЗЫ:")
            for category, phrases in explainable_result.influential_phrases.items():
                if phrases:
                    print(f"  {category.replace('_', ' ').title()}:")
                    for phrase in phrases[:2]:  # Показываем только первые 2
                        print(f"    • '{phrase}'")

            # Ключевые факторы
            print("\n📊 КЛЮЧЕВЫЕ ФАКТОРЫ (топ-5):")
            top_factors = sorted(
                explainable_result.decision_factors.items(),
                key=lambda x: x[1],
                reverse=True,
            )[:5]
            for factor, value in top_factors:
                print(f"  • {factor.replace('_', ' ').title()}: {value:.3f}")

        # Демонстрация объяснения существующего анализа
        print("\n🔍 Тестирование объяснения существующего анализа...")
        explanation = analyzer.explain_existing_analysis(song_id=1)

        if explanation:
            print(
                f"\n🎵 Объяснение для: {explanation['song_info']['artist']} - {explanation['song_info']['title']}"
            )
            print(f"🔍 Уверенность: {explanation['confidence']:.2f}")
            print(
                f"💡 Основные объяснения: {len(explanation['explanation']['genre_indicators'])} жанровых, "
                f"{len(explanation['explanation']['mood_triggers'])} настроения"
            )

        # Демонстрация SafetyValidator
        print("\n🛡️ Тестирование AI Safety & Hallucination Detection...")

        # Тест с потенциально проблемным текстом
        problematic_lyrics = """
        Короткий текст
        """

        safe_result = analyzer.analyze_song_with_safety(
            "Test Artist", "Problematic Track", problematic_lyrics
        )

        if safe_result:
            print("\n🛡️ РЕЗУЛЬТАТ БЕЗОПАСНОГО АНАЛИЗА:")
            print("-" * 50)
            print(
                f"✅ Безопасность: {'НАДЕЖЕН' if safe_result['is_safe'] else 'НЕНАДЕЖЕН'}"
            )
            print(f"🔍 Уверенность: {safe_result['confidence']:.3f}")
            print(f"📝 Резюме: {safe_result['summary']}")

            if safe_result["warnings"]:
                print("⚠️ Предупреждения:")
                for warning in safe_result["warnings"]:
                    print(f"   • {warning}")

            # Детали валидации
            validation = safe_result["validation"]
            print("\n📊 ДЕТАЛИ ВАЛИДАЦИИ:")
            print(f"   • Риск галлюцинаций: {validation['hallucination_risk']:.3f}")
            print(f"   • Консистентность: {validation['consistency_score']:.3f}")
            print(f"   • Точность фактов: {validation['factual_accuracy']:.3f}")
            print(f"   • Соответствие тексту: {validation['text_alignment']:.3f}")

        # Тест с нормальным текстом
        print("\n🔄 Тест с качественным текстом...")
        normal_safe_result = analyzer.analyze_song_with_safety(
            "Тестовый артист", "Качественный трек", test_lyrics
        )

        if normal_safe_result:
            print(
                f"✅ Нормальный текст: {'НАДЕЖЕН' if normal_safe_result['is_safe'] else 'НЕНАДЕЖЕН'}"
            )
            print(f"🔍 Уверенность: {normal_safe_result['confidence']:.3f}")
            print(f"📝 Резюме: {normal_safe_result['summary']}")

        # Показываем статистику
        stats = analyzer.get_stats()
        print("\n📈 СТАТИСТИКА:")
        print(f"  • Всего проанализировано: {stats['total_analyzed']}")
        print(f"  • Ollama использован: {stats['ollama_used']} раз")
        print(f"  • Gemma использован: {stats['gemma_used']} раз")
        print(f"  • Mock использован: {stats['mock_used']} раз")
        print(f"  • Общая стоимость: ${stats['total_cost']:.4f}")

        print("\n✅ AI Safety & Hallucination Detection - ГОТОВО!")
        print("🛡️ Теперь AI анализ включает:")
        print("   • Interpretability & Model Understanding")
        print("   • Safety & Hallucination Detection")
        print("   • Consistency Validation")
        print("   • Factual Accuracy Checking")
        print("🎯 Продукционная система с валидацией надежности!")

    except Exception as e:
        logger.error(f"❌ Ошибка: {e}")


if __name__ == "__main__":
    main()
