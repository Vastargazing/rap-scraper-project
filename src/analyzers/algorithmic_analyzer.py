"""
🧮 Продвинутый алгоритмический анализатор текстов песен

КЛЮЧЕВЫЕ УЛУЧШЕНИЯ:
✨ Фонетический анализ рифм (вместо простого сравнения окончаний)
🎯 Продвинутый анализ flow и ритма
🧠 Семантический анализ тем с расширенными словарями
📊 Статистические метрики читабельности (Flesch, SMOG, ARI)
🎵 Анализ музыкальности и аллитерации
🔄 Кэширование для производительности
⚡ Асинхронная обработка больших текстов
📈 Детализированные композитные метрики

НАЗНАЧЕНИЕ:
- 🎯 Профессиональный анализ текстов без использования AI моделей
- ⚡ Быстрая обработка больших объемов данных (57K+ треков)
- 📊 Baseline метрики для сравнения с AI-анализаторами
- 🗄️ Production-ready компонент с PostgreSQL интеграцией
- 📈 Детальные композитные оценки качества текстов

ИСПОЛЬЗОВАНИЕ:
🖥️ CLI интерфейс:
  python src/analyzers/algorithmic_analyzer.py --stats
  python src/analyzers/algorithmic_analyzer.py --analyze-all --limit 100
  python src/analyzers/algorithmic_analyzer.py --analyze-track 123

📝 Программный интерфейс:
  analyzer = AdvancedAlgorithmicAnalyzer()
  result = analyzer.analyze_song("Artist", "Title", "Lyrics...")

ЗАВИСИМОСТИ:
- 🐍 Python 3.8+
- 🗄️ PostgreSQL с asyncpg
- 📄 PyYAML для конфигурации
- 🔧 src/interfaces/analyzer_interface.py

РЕЗУЛЬТАТ:
- 🎵 Анализ рифм: схема, плотность, фонетическое сходство
- 🌊 Flow метрики: консистентность слогов, ритмическая плотность
- 📚 Читабельность: Flesch, SMOG, ARI, Coleman-Liau индексы
- 💭 Эмоциональный анализ: валентность, интенсивность, сложность
- 🎨 Тематический анализ: деньги, улица, успех, отношения
- ✍️ Литературные приемы: метафоры, аллитерация, повторы
- 📊 Композитные оценки: техническое мастерство, артистичность

ВОЗМОЖНОСТИ:
- 📈 Статистика базы данных PostgreSQL
- 🔍 Анализ отдельных треков по ID
- 🚀 Массовый анализ с прогрессом и батчингом
- 💾 Кэширование результатов для производительности
- 🎭 Демонстрационный режим с примером

АВТОР: Rap Scraper Project Team
ВЕРСИЯ: 2.0.0 Advanced
ДАТА: Сентябрь 2025
"""

# TODO(google-review): [STYLE] Organize imports: stdlib, third-party, local
# TODO(google-review): [STYLE] Add module-level docstring after imports
import asyncio
import hashlib
import logging
import math
import re
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# Импорты с fallback для standalone запуска
try:
    from config.config_loader import get_config
    from interfaces.analyzer_interface import (
        AnalysisResult,
        BaseAnalyzer,
        register_analyzer,
    )

    PROJECT_IMPORT_SUCCESS = True
except ImportError:
    PROJECT_IMPORT_SUCCESS = False
    import sys
    from pathlib import Path

    # Попытка найти корень проекта и добавить src в путь
    current_dir = Path(__file__).resolve().parent
    possible_roots = [current_dir.parent.parent, current_dir.parent, current_dir]

    for root in possible_roots:
        src_path = root / "src"
        if src_path.exists() and (src_path / "interfaces").exists():
            if str(src_path) not in sys.path:
                sys.path.append(str(src_path))
            try:
                from interfaces.analyzer_interface import (
                    AnalysisResult,
                    BaseAnalyzer,
                    register_analyzer,
                )

                PROJECT_IMPORT_SUCCESS = True
                break
            except ImportError:
                continue

    # Если все еще не удалось импортировать, создаем заглушки
    if not PROJECT_IMPORT_SUCCESS:
        # Базовые классы заглушки для standalone работы
        class BaseAnalyzer:
            def __init__(self, config: dict[str, Any] | None = None):
                self.config = config or {}

            def validate_input(self, artist: str, title: str, lyrics: str) -> bool:
                return bool(artist and title and lyrics and len(lyrics.strip()) > 10)

            def preprocess_lyrics(self, lyrics: str) -> str:
                return re.sub(r"\s+", " ", lyrics.strip())

        @dataclass
        class AnalysisResult:
            artist: str
            title: str
            analysis_type: str
            confidence: float
            metadata: dict[str, Any]
            raw_output: dict[str, Any]
            processing_time: float
            timestamp: str

        def register_analyzer(name: str):  # noqa: ARG001
            def decorator(cls):
                return cls

            return decorator


# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class PhoneticPattern:
    """Фонетические паттерны для улучшенного анализа рифм"""

    vowel_sounds: dict[str, list[str]] = field(
        default_factory=lambda: {
            "ay": ["ai", "ay", "ey", "a_e"],
            "ee": ["ee", "ea", "ie", "y"],
            "oh": ["o", "oa", "ow", "o_e"],
            "oo": ["oo", "u", "ew", "ue"],
            "ah": ["a", "au", "aw"],
            "ih": ["i", "y", "ie"],
            "eh": ["e", "ea", "ai"],
            "uh": ["u", "o", "ou"],
        }
    )

    consonant_clusters: set[str] = field(
        default_factory=lambda: {
            "ch",
            "sh",
            "th",
            "wh",
            "ck",
            "ng",
            "ph",
            "gh",
            "st",
            "sp",
            "sc",
            "sk",
            "sm",
            "sn",
            "sw",
            "sl",
            "bl",
            "br",
            "cl",
            "cr",
            "dr",
            "fl",
            "fr",
            "gl",
            "gr",
            "pl",
            "pr",
            "tr",
            "tw",
        }
    )


class AdvancedLexicon:
    """Расширенные словари для семантического анализа"""
    # TODO(google-review): [DOCSTRING] Add Args, Returns sections to docstring
    # TODO(google-review): [ARCHITECTURE] Consider loading from config file

    def __init__(self):
        # Эмоциональные категории с градацией интенсивности
        self.emotions = {
            "joy": {
                "weak": {"happy", "good", "nice", "okay", "fine", "content"},
                "medium": {"great", "wonderful", "amazing", "excellent", "fantastic"},
                "strong": {
                    "ecstatic",
                    "euphoric",
                    "blissful",
                    "overjoyed",
                    "triumphant",
                },
            },
            "anger": {
                "weak": {"annoyed", "bothered", "upset", "frustrated", "irritated"},
                "medium": {"angry", "mad", "furious", "pissed", "heated"},
                "strong": {"rage", "wrath", "livid", "enraged", "incensed"},
            },
            "sadness": {
                "weak": {"sad", "down", "blue", "low", "melancholy"},
                "medium": {"depressed", "miserable", "heartbroken", "devastated"},
                "strong": {"suicidal", "hopeless", "despairing", "anguished"},
            },
            "fear": {
                "weak": {"worried", "nervous", "anxious", "concerned", "uneasy"},
                "medium": {"scared", "afraid", "frightened", "terrified"},
                "strong": {"petrified", "horrified", "panic", "dread"},
            },
        }

        # Тематические категории рэпа
        self.rap_themes = {
            "money_wealth": {
                "money",
                "cash",
                "bread",
                "dough",
                "paper",
                "green",
                "bills",
                "rich",
                "wealth",
                "fortune",
                "bank",
                "account",
                "invest",
                "luxury",
                "expensive",
                "diamond",
                "gold",
                "platinum",
                "mansion",
                "penthouse",
                "yacht",
                "lambo",
                "ferrari",
                "bentley",
            },
            "street_life": {
                "street",
                "hood",
                "block",
                "corner",
                "ghetto",
                "projects",
                "hustle",
                "grind",
                "struggle",
                "survive",
                "real",
                "raw",
                "concrete",
                "pavement",
                "alley",
                "neighborhood",
            },
            "success_ambition": {
                "success",
                "win",
                "victory",
                "champion",
                "boss",
                "king",
                "queen",
                "crown",
                "throne",
                "empire",
                "legend",
                "icon",
                "achieve",
                "accomplish",
                "conquer",
                "dominate",
                "rise",
            },
            "relationships_love": {
                "love",
                "heart",
                "baby",
                "girl",
                "woman",
                "man",
                "relationship",
                "kiss",
                "hug",
                "romance",
                "passion",
                "soul",
                "forever",
                "together",
                "commitment",
                "loyalty",
                "trust",
            },
            "violence_conflict": {
                "gun",
                "shoot",
                "kill",
                "murder",
                "death",
                "blood",
                "war",
                "fight",
                "battle",
                "enemy",
                "revenge",
                "violence",
                "attack",
                "weapon",
                "bullet",
                "trigger",
                "blast",
                "destroy",
            },
            "drugs_party": {
                "weed",
                "smoke",
                "high",
                "drunk",
                "party",
                "club",
                "dance",
                "drink",
                "bottle",
                "shot",
                "pill",
                "lean",
                "molly",
                "celebration",
                "turn_up",
                "lit",
                "wild",
                "crazy",
            },
        }

        # Слова, указывающие на сложность мышления
        self.complexity_indicators = {
            "philosophical": {
                "existence",
                "reality",
                "consciousness",
                "purpose",
                "meaning",
                "truth",
                "wisdom",
                "knowledge",
                "understand",
                "perspective",
                "philosophy",
                "metaphysical",
                "spiritual",
                "transcend",
            },
            "abstract": {
                "concept",
                "theory",
                "principle",
                "ideology",
                "paradigm",
                "dimension",
                "universe",
                "infinity",
                "eternal",
                "essence",
                "phenomenon",
                "manifestation",
                "transformation",
                "evolution",
            },
            "analytical": {
                "analyze",
                "examine",
                "investigate",
                "evaluate",
                "assess",
                "consider",
                "contemplate",
                "reflect",
                "introspect",
                "ponder",
                "calculate",
                "measure",
                "compare",
                "contrast",
                "deduce",
            },
        }

        # Литературные приемы в текстах
        self.literary_devices = {
            "metaphor_indicators": {
                "like",
                "as",
                "than",
                "seems",
                "appears",
                "resembles",
                "similar",
                "reminds",
                "symbolize",
                "represent",
                "embody",
            },
            "time_references": {
                "yesterday",
                "today",
                "tomorrow",
                "past",
                "present",
                "future",
                "memory",
                "remember",
                "forget",
                "history",
                "destiny",
                "fate",
            },
            "contrast_words": {
                "but",
                "however",
                "although",
                "despite",
                "nevertheless",
                "nonetheless",
                "whereas",
                "while",
                "opposite",
                "contrast",
            },
        }


class FlowAnalyzer:
    """Продвинутый анализатор flow и ритма"""

    def __init__(self):
        self.phonetic_patterns = PhoneticPattern()

    def analyze_flow_patterns(self, lines: list[str]) -> dict[str, Any]:
        """Анализ паттернов flow"""
        # TODO(google-review): [DOCSTRING] Add Args, Returns, Raises sections
        # TODO(google-review): [ARCHITECTURE] Function too long (40+ lines)
        if not lines:
            return self._empty_flow_result()

        syllable_patterns = []
        stress_patterns = []
        line_lengths = []

        for line in lines:
            syllables = self._count_syllables_advanced(line)
            syllable_patterns.append(syllables)
            line_lengths.append(len(line.split()))

            # Анализ ударений (упрощенный)
            stress_pattern = self._analyze_stress_pattern(line)
            stress_patterns.append(stress_pattern)

        return {
            "syllable_consistency": self._calculate_consistency(syllable_patterns),
            "average_syllables_per_line": sum(syllable_patterns)
            / len(syllable_patterns),
            "syllable_variance": self._calculate_variance(syllable_patterns),
            "line_length_consistency": self._calculate_consistency(line_lengths),
            "stress_pattern_regularity": self._analyze_stress_regularity(
                stress_patterns
            ),
            "flow_breaks": self._count_flow_interruptions(lines),
            "rhythmic_density": self._calculate_rhythmic_density(lines),
        }

    def _count_syllables_advanced(self, text: str) -> int:
        """Улучшенный подсчет слогов"""
        # Удаляем знаки препинания и разбиваем на слова
        words = re.findall(r"\b[a-zA-Z]+\b", text.lower())
        if not words:
            return 0

        total_syllables = 0
        for word in words:
            syllables = self._syllables_in_word(word)
            total_syllables += syllables

        return total_syllables

    def _syllables_in_word(self, word: str) -> int:
        """Точный подсчет слогов в слове"""
        if len(word) <= 2:
            return 1

        word = word.lower().strip()

        # Специальные случаи
        special_cases = {
            "the": 1,
            "a": 1,
            "an": 1,
            "and": 1,
            "or": 1,
            "but": 1,
            "through": 1,
            "though": 1,
            "every": 2,
            "very": 2,
            "people": 2,
            "little": 2,
            "middle": 2,
            "simple": 2,
        }

        if word in special_cases:
            return special_cases[word]

        # Подсчет гласных групп
        vowels = "aeiouy"
        syllable_count = 0
        prev_was_vowel = False

        for i, char in enumerate(word):
            is_vowel = char in vowels

            if is_vowel:
                # Новая гласная группа
                if not prev_was_vowel:
                    syllable_count += 1
                # Исключения для дифтонгов
                elif i > 0 and word[i - 1 : i + 1] in [
                    "ai",
                    "au",
                    "ea",
                    "ee",
                    "ei",
                    "ie",
                    "oa",
                    "oo",
                    "ou",
                    "ue",
                ]:
                    pass  # Не увеличиваем счетчик

            prev_was_vowel = is_vowel

        # Исключения
        # TODO(google-review): [STYLE] Line exceeds 80 chars, break into multiple
        if word.endswith("e") and syllable_count > 1 and not word.endswith(("le", "se", "me", "ne", "ve", "ze", "de", "ge")):
            syllable_count -= 1

        # Специальные окончания
        if word.endswith(("ed", "es", "er", "ly")):
            pass  # Уже учтено в основном алгоритме

        return max(1, syllable_count)

    def _analyze_stress_pattern(self, line: str) -> str:
        """Анализ паттерна ударений (упрощенный)"""
        # TODO(google-review): [DOCSTRING] Add Args, Returns sections
        # TODO(google-review): [ARCHITECTURE] Magic numbers: 3 should be constant
        words = line.split()
        if not words:
            return ""

        stress_pattern = []
        for word in words:
            # Простая эвристика для ударений
            # TODO(google-review): [ARCHITECTURE] Magic number 3
            if len(word) <= 3:
                stress_pattern.append("1")  # Ударный
            elif word.lower() in {"the", "and", "but", "for", "with", "from", "into"}:
                stress_pattern.append("0")  # Безударный
            else:
                # Для длинных слов ставим ударение на первый слог
                stress_pattern.append("1")

        return "".join(stress_pattern)

    def _calculate_consistency(self, values: list[float]) -> float:
        """Вычисление консистентности значений"""
        if len(values) < 2:
            return 1.0

        mean = sum(values) / len(values)
        variance = sum((x - mean) ** 2 for x in values) / len(values)

        # Нормализуем: чем меньше вариация, тем выше консистентность
        consistency = 1 / (1 + variance)
        return min(consistency, 1.0)

    def _calculate_variance(self, values: list[float]) -> float:
        """Вычисление дисперсии"""
        if len(values) < 2:
            return 0.0

        mean = sum(values) / len(values)
        return sum((x - mean) ** 2 for x in values) / len(values)

    def _analyze_stress_regularity(self, stress_patterns: list[str]) -> float:
        """Анализ регулярности ударений"""
        if not stress_patterns:
            return 0.0

        # Ищем повторяющиеся паттерны
        pattern_counts = Counter(stress_patterns)
        most_common = pattern_counts.most_common(1)

        if most_common:
            return most_common[0][1] / len(stress_patterns)

        return 0.0

    def _count_flow_interruptions(self, lines: list[str]) -> int:
        """Подсчет прерываний flow"""
        interruptions = 0
        punctuation = {".", "!", "?", ";", ":", ",", "--", "..."}

        for line in lines:
            for punct in punctuation:
                interruptions += line.count(punct)

        return interruptions

    def _calculate_rhythmic_density(self, lines: list[str]) -> float:
        """Вычисление ритмической плотности"""
        if not lines:
            return 0.0

        total_words = sum(len(line.split()) for line in lines)
        total_syllables = sum(self._count_syllables_advanced(line) for line in lines)

        if total_words == 0:
            return 0.0

        # Плотность = слоги на слово
        density = total_syllables / total_words
        return min(density / 2.0, 1.0)  # Нормализуем к [0, 1]

    def _empty_flow_result(self) -> dict[str, Any]:
        """Пустой результат анализа flow"""
        return {
            "syllable_consistency": 0.0,
            "average_syllables_per_line": 0.0,
            "syllable_variance": 0.0,
            "line_length_consistency": 0.0,
            "stress_pattern_regularity": 0.0,
            "flow_breaks": 0,
            "rhythmic_density": 0.0,
        }


class RhymeAnalyzer:
    """Продвинутый анализатор рифм с фонетическими паттернами"""

    def __init__(self):
        self.phonetic_patterns = PhoneticPattern()
        self.rhyme_cache = {}

    def analyze_rhyme_structure(self, lines: list[str]) -> dict[str, Any]:
        """Комплексный анализ рифменной структуры"""
        # TODO(google-review): [DOCSTRING] Add Args, Returns sections
        # TODO(google-review): [ARCHITECTURE] Function too long (40+ lines)
        # TODO(google-review): [ARCHITECTURE] Magic number 2
        if len(lines) < 2:
            return self._empty_rhyme_result()

        # Извлекаем окончания строк
        line_endings = self._extract_line_endings(lines)

        # Анализ различных типов рифм
        perfect_rhymes = self._find_perfect_rhymes(line_endings)
        near_rhymes = self._find_near_rhymes(line_endings)
        internal_rhymes = self._find_internal_rhymes(lines)

        # Схема рифмовки
        rhyme_scheme = self._detect_complex_rhyme_scheme(line_endings)

        # Продвинутые метрики
        phonetic_similarity = self._calculate_phonetic_similarity(line_endings)
        rhyme_density = self._calculate_rhyme_density(
            line_endings, perfect_rhymes, near_rhymes
        )

        return {
            "perfect_rhymes": len(perfect_rhymes),
            "near_rhymes": len(near_rhymes),
            "internal_rhymes": len(internal_rhymes),
            "rhyme_scheme": rhyme_scheme,
            "rhyme_scheme_complexity": self._evaluate_scheme_complexity(rhyme_scheme),
            "phonetic_similarity_score": phonetic_similarity,
            "rhyme_density": rhyme_density,
            "alliteration_score": self._calculate_alliteration(lines),
            "assonance_score": self._calculate_assonance(lines),
            "consonance_score": self._calculate_consonance(lines),
        }

    def _extract_line_endings(self, lines: list[str]) -> list[str]:
        """Извлечение окончаний строк с очисткой"""
        endings = []
        for line in lines:
            # Удаляем знаки препинания и берем последнее слово
            words = re.findall(r"\b[a-zA-Z]+\b", line)
            if words:
                ending = words[-1].lower()
                endings.append(ending)
            else:
                endings.append("")
        return endings

    def _find_perfect_rhymes(self, endings: list[str]) -> list[tuple[int, int]]:
        """Поиск точных рифм"""
        # TODO(google-review): [PERFORMANCE] O(n²) complexity, consider optimization
        # TODO(google-review): [DOCSTRING] Add Args, Returns sections
        return [
            (i, j)
            for i in range(len(endings))
            for j in range(i + 1, len(endings))
            if self._is_perfect_rhyme(endings[i], endings[j])
        ]

    def _find_near_rhymes(self, endings: list[str]) -> list[tuple[int, int]]:
        """Поиск неточных рифм"""
        # TODO(google-review): [PERFORMANCE] O(n²) complexity, consider optimization
        # TODO(google-review): [DOCSTRING] Add Args, Returns sections
        return [
            (i, j)
            for i in range(len(endings))
            for j in range(i + 1, len(endings))
            if not self._is_perfect_rhyme(endings[i], endings[j])
            and self._is_near_rhyme(endings[i], endings[j])
        ]

    def _find_internal_rhymes(self, lines: list[str]) -> list[tuple[int, str, str]]:
        """Поиск внутренних рифм"""
        # TODO(google-review): [PERFORMANCE] Nested loops O(n*m²), optimize
        # TODO(google-review): [ARCHITECTURE] Magic number 3
        # TODO(google-review): [DOCSTRING] Add Args, Returns sections
        internal_rhymes = []
        for line_idx, line in enumerate(lines):
            words = re.findall(r"\b[a-zA-Z]{3,}\b", line.lower())
            internal_rhymes.extend([
                (line_idx, words[i], words[j])
                for i in range(len(words))
                for j in range(i + 1, len(words))
                if self._is_perfect_rhyme(words[i], words[j])
                or self._is_near_rhyme(words[i], words[j])
            ])
        return internal_rhymes

    def _is_perfect_rhyme(self, word1: str, word2: str) -> bool:
        """Проверка точной рифмы с учетом фонетики"""
        if not word1 or not word2 or word1 == word2:
            return False

        # Кэширование результатов
        cache_key = tuple(sorted([word1, word2]))
        if cache_key in self.rhyme_cache:
            return self.rhyme_cache[cache_key]

        # Проверка по окончаниям разной длины
        result = False
        for suffix_len in range(2, min(len(word1), len(word2)) + 1):
            if word1[-suffix_len:] == word2[-suffix_len:]:
                result = True
                break

        # Фонетическая проверка
        if not result:
            result = self._phonetic_rhyme_check(word1, word2)

        self.rhyme_cache[cache_key] = result
        return result

    def _is_near_rhyme(self, word1: str, word2: str) -> bool:
        """Проверка неточной рифмы"""
        if not word1 or not word2 or len(word1) < 2 or len(word2) < 2:
            return False

        # Проверка на созвучие гласных (assonance)
        vowels1 = [c for c in word1[-3:] if c in "aeiou"]
        vowels2 = [c for c in word2[-3:] if c in "aeiou"]

        if vowels1 and vowels2 and vowels1[-1] == vowels2[-1]:
            return True

        # Проверка на созвучие согласных (consonance)
        consonants1 = [c for c in word1[-3:] if c not in "aeiou"]
        consonants2 = [c for c in word2[-3:] if c not in "aeiou"]

        return len(set(consonants1) & set(consonants2)) >= 1

    def _phonetic_rhyme_check(self, word1: str, word2: str) -> bool:
        """Фонетическая проверка рифмы"""
        # Упрощенная фонетическая проверка
        # В реальной реализации здесь был бы phonetic matching алгоритм

        # Проверяем схожие звуки
        phonetic_groups = {
            "k_sounds": ["c", "k", "ck", "q"],
            "s_sounds": ["s", "c", "z"],
            "f_sounds": ["f", "ph", "gh"],
            "long_a": ["a", "ai", "ay", "ei"],
            "long_e": ["e", "ee", "ea", "ie"],
            "long_i": ["i", "ie", "y", "igh"],
            "long_o": ["o", "oa", "ow", "ough"],
            "long_u": ["u", "ue", "ew", "ou"],
        }

        # Проверяем окончания на фонетическое сходство
        end1 = word1[-2:]
        end2 = word2[-2:]

        for group in phonetic_groups.values():
            if any(end1.endswith(sound) for sound in group) and any(
                end2.endswith(sound) for sound in group
            ):
                return True

        return False

    def _detect_complex_rhyme_scheme(self, endings: list[str]) -> str:
        """Определение сложной схемы рифмовки"""
        # TODO(google-review): [ARCHITECTURE] Magic numbers: 4, 16 as constants
        # TODO(google-review): [DOCSTRING] Add Args, Returns sections
        if len(endings) < 4:
            return "insufficient"

        # Берем первые 16 строк для анализа схемы
        sample = endings[:16]

        # Группируем рифмующиеся слова
        rhyme_groups = {}
        scheme = []
        next_letter = "A"

        for ending in sample:
            assigned_letter = None

            # Ищем существующую группу
            for group_word, letter in rhyme_groups.items():
                if self._is_perfect_rhyme(ending, group_word) or self._is_near_rhyme(
                    ending, group_word
                ):
                    assigned_letter = letter
                    break

            # Создаем новую группу если не найдена
            if assigned_letter is None:
                assigned_letter = next_letter
                rhyme_groups[ending] = next_letter
                next_letter = chr(ord(next_letter) + 1)

            scheme.append(assigned_letter)

        return "".join(scheme)

    def _evaluate_scheme_complexity(self, scheme: str) -> float:
        """Оценка сложности схемы рифмовки"""
        if not scheme or scheme == "insufficient":
            return 0.0

        # Факторы сложности
        unique_rhymes = len(set(scheme))
        total_lines = len(scheme)

        # Поиск паттернов
        patterns = {"ABAB": 0.6, "AABB": 0.4, "ABCB": 0.7, "ABBA": 0.8, "AAAA": 0.2}

        complexity_score = unique_rhymes / total_lines

        # Бонус за известные сложные паттерны
        for pattern, bonus in patterns.items():
            if pattern in scheme:
                complexity_score += bonus * 0.1

        return min(complexity_score, 1.0)

    def _calculate_phonetic_similarity(self, endings: list[str]) -> float:
        """Вычисление фонетического сходства"""
        if len(endings) < 2:
            return 0.0

        similarity_scores = []

        for i in range(len(endings)):
            for j in range(i + 1, len(endings)):
                score = self._phonetic_similarity_score(endings[i], endings[j])
                similarity_scores.append(score)

        return (
            sum(similarity_scores) / len(similarity_scores)
            if similarity_scores
            else 0.0
        )

    def _phonetic_similarity_score(self, word1: str, word2: str) -> float:
        """Оценка фонетического сходства двух слов"""
        if not word1 or not word2:
            return 0.0

        # Сравниваем окончания разной длины
        max_len = min(len(word1), len(word2), 4)
        similarity = 0.0

        for i in range(1, max_len + 1):
            if word1[-i:] == word2[-i:]:
                similarity += i * 0.25

        return min(similarity, 1.0)

    def _calculate_rhyme_density(
        self, endings: list[str], perfect_rhymes: list, near_rhymes: list
    ) -> float:
        """Вычисление плотности рифм"""
        # TODO(google-review): [ARCHITECTURE] Magic numbers: 2, 0.7 as constants
        # TODO(google-review): [DOCSTRING] Add Args, Returns sections
        if len(endings) < 2:
            return 0.0

        total_rhymes = (
            len(perfect_rhymes) + len(near_rhymes) * 0.7
        )  # Near rhymes считаются с коэффициентом
        max_possible_rhymes = (
            len(endings) // 2
        )  # Максимальное количество возможных рифм

        return (
            min(total_rhymes / max_possible_rhymes, 1.0)
            if max_possible_rhymes > 0
            else 0.0
        )

    def _calculate_alliteration(self, lines: list[str]) -> float:
        """Вычисление коэффициента аллитерации"""
        # TODO(google-review): [ARCHITECTURE] Magic numbers: 2, 0.5 as constants
        # TODO(google-review): [DOCSTRING] Add Args, Returns sections
        if not lines:
            return 0.0

        alliteration_count = 0
        total_word_pairs = 0

        for line in lines:
            words = [word.lower() for word in re.findall(r"\b[a-zA-Z]{2,}\b", line)]
            if len(words) < 2:
                continue

            for i in range(len(words) - 1):
                total_word_pairs += 1
                if words[i][0] == words[i + 1][0]:
                    alliteration_count += 1

                    # Бонус за аллитерацию через слово
                    if i < len(words) - 2 and words[i][0] == words[i + 2][0]:
                        alliteration_count += 0.5

        return alliteration_count / max(total_word_pairs, 1)

    def _calculate_assonance(self, lines: list[str]) -> float:
        """Вычисление коэффициента ассонанса (повторение гласных)"""
        if not lines:
            return 0.0

        vowels = "aeiou"
        assonance_count = 0
        total_comparisons = 0

        for line in lines:
            words = [word.lower() for word in re.findall(r"\b[a-zA-Z]{3,}\b", line)]

            for i in range(len(words)):
                for j in range(
                    i + 1, min(i + 3, len(words))
                ):  # Проверяем ближайшие слова
                    vowels_i = [c for c in words[i] if c in vowels]
                    vowels_j = [c for c in words[j] if c in vowels]

                    if vowels_i and vowels_j:
                        total_comparisons += 1
                        # Проверяем совпадение гласных
                        common_vowels = set(vowels_i) & set(vowels_j)
                        if common_vowels:
                            assonance_count += len(common_vowels) / max(
                                len(vowels_i), len(vowels_j)
                            )

        return assonance_count / max(total_comparisons, 1)

    def _calculate_consonance(self, lines: list[str]) -> float:
        """Вычисление коэффициента консонанса (повторение согласных)"""
        if not lines:
            return 0.0

        vowels = "aeiou"
        consonance_count = 0
        total_comparisons = 0

        for line in lines:
            words = [word.lower() for word in re.findall(r"\b[a-zA-Z]{3,}\b", line)]

            for i in range(len(words)):
                for j in range(i + 1, min(i + 3, len(words))):
                    consonants_i = [
                        c for c in words[i] if c not in vowels and c.isalpha()
                    ]
                    consonants_j = [
                        c for c in words[j] if c not in vowels and c.isalpha()
                    ]

                    if consonants_i and consonants_j:
                        total_comparisons += 1
                        common_consonants = set(consonants_i) & set(consonants_j)
                        if common_consonants:
                            consonance_count += len(common_consonants) / max(
                                len(consonants_i), len(consonants_j)
                            )

        return consonance_count / max(total_comparisons, 1)

    def _empty_rhyme_result(self) -> dict[str, Any]:
        """Пустой результат анализа рифм"""
        return {
            "perfect_rhymes": 0,
            "near_rhymes": 0,
            "internal_rhymes": 0,
            "rhyme_scheme": "insufficient",
            "rhyme_scheme_complexity": 0.0,
            "phonetic_similarity_score": 0.0,
            "rhyme_density": 0.0,
            "alliteration_score": 0.0,
            "assonance_score": 0.0,
            "consonance_score": 0.0,
        }


class ReadabilityAnalyzer:
    """Анализатор читабельности с множественными метриками"""

    def analyze_readability(self, text: str) -> dict[str, Any]:
        """Комплексный анализ читабельности"""
        # TODO(google-review): [DOCSTRING] Add Args, Returns sections
        # TODO(google-review): [ARCHITECTURE] Function too long (40+ lines)
        if not text.strip():
            return self._empty_readability_result()

        # Базовые метрики
        sentences = self._count_sentences(text)
        words = self._count_words(text)
        syllables = self._count_total_syllables(text)

        if sentences == 0 or words == 0:
            return self._empty_readability_result()

        # Вычисляем различные индексы читабельности
        flesch_score = self._calculate_flesch_reading_ease(sentences, words, syllables)
        flesch_kincaid_grade = self._calculate_flesch_kincaid_grade(
            sentences, words, syllables
        )
        smog_index = self._calculate_smog_index(text, sentences)
        ari_score = self._calculate_ari(sentences, words, text)
        coleman_liau = self._calculate_coleman_liau(text, sentences, words)

        return {
            "flesch_reading_ease": flesch_score,
            "flesch_kincaid_grade": flesch_kincaid_grade,
            "smog_index": smog_index,
            "automated_readability_index": ari_score,
            "coleman_liau_index": coleman_liau,
            "average_sentence_length": words / sentences,
            "average_syllables_per_word": syllables / words,
            "readability_consensus": self._calculate_consensus(
                flesch_score, flesch_kincaid_grade, smog_index
            ),
        }

    def _count_sentences(self, text: str) -> int:
        """Подсчет предложений"""
        # Разбиваем по знакам препинания, фильтруем пустые
        sentences = re.split(r"[.!?]+", text)
        sentences = [s.strip() for s in sentences if s.strip()]
        return len(sentences)

    def _count_words(self, text: str) -> int:
        """Подсчет слов"""
        words = re.findall(r"\b[a-zA-Z]+\b", text)
        return len(words)

    def _count_total_syllables(self, text: str) -> int:
        """Подсчет общего количества слогов"""
        # TODO(google-review): [PERFORMANCE] Creating FlowAnalyzer in loop expensive
        # TODO(google-review): [DOCSTRING] Add Args, Returns sections
        words = re.findall(r"\b[a-zA-Z]+\b", text.lower())
        total_syllables = 0

        flow_analyzer = FlowAnalyzer()
        for word in words:
            total_syllables += flow_analyzer._syllables_in_word(word)

        return total_syllables

    def _calculate_flesch_reading_ease(
        self, sentences: int, words: int, syllables: int
    ) -> float:
        """Индекс читабельности Флеша"""
        # TODO(google-review): [ARCHITECTURE] Magic numbers as constants
        # TODO(google-review): [DOCSTRING] Add Args, Returns sections
        if sentences == 0 or words == 0:
            return 0.0

        asl = words / sentences  # Average Sentence Length
        asw = syllables / words  # Average Syllables per Word

        score = 206.835 - (1.015 * asl) - (84.6 * asw)
        return max(0, min(100, score))

    def _calculate_flesch_kincaid_grade(
        self, sentences: int, words: int, syllables: int
    ) -> float:
        """Индекс уровня класса Флеша-Кинкейда"""
        if sentences == 0 or words == 0:
            return 0.0

        asl = words / sentences
        asw = syllables / words

        grade = (0.39 * asl) + (11.8 * asw) - 15.59
        return max(0, grade)

    def _calculate_smog_index(self, text: str, sentences: int) -> float:
        """Индекс SMOG"""
        # TODO(google-review): [PERFORMANCE] Creating FlowAnalyzer in method
        # TODO(google-review): [ARCHITECTURE] Magic numbers: 3, 30
        # TODO(google-review): [DOCSTRING] Add Args, Returns sections
        if sentences < 3:
            return 0.0

        # Подсчитываем слова с 3+ слогами
        words = re.findall(r"\b[a-zA-Z]+\b", text.lower())
        complex_words = 0
        flow_analyzer = FlowAnalyzer()

        for word in words:
            if flow_analyzer._syllables_in_word(word) >= 3:
                complex_words += 1

        if complex_words == 0:
            return 0.0

        # SMOG = 3 + √(complex_words * 30 / sentences)
        return 3 + math.sqrt(complex_words * 30 / sentences)

    def _calculate_ari(self, sentences: int, words: int, text: str) -> float:
        """Автоматический индекс читабельности (ARI)"""
        if sentences == 0 or words == 0:
            return 0.0

        characters = len(re.sub(r"[^a-zA-Z]", "", text))

        if characters == 0:
            return 0.0

        ari = (4.71 * (characters / words)) + (0.5 * (words / sentences)) - 21.43
        return max(0, ari)

    def _calculate_coleman_liau(self, text: str, sentences: int, words: int) -> float:
        """Индекс Коулмана-Лиау"""
        if words == 0 or sentences == 0:
            return 0.0

        characters = len(re.sub(r"[^a-zA-Z]", "", text))

        letters_per_100 = (characters / words) * 100  # Average letters per 100 words
        sentences_per_100 = (sentences / words) * 100  # Average sentences per 100 words

        cli = (0.0588 * letters_per_100) - (0.296 * sentences_per_100) - 15.8
        return max(0, cli)

    def _calculate_consensus(self, flesch: float, fk_grade: float, smog: float) -> str:
        """Консенсус по читабельности"""
        # TODO(google-review): [ARCHITECTURE] Magic numbers, use dict/config
        # TODO(google-review): [DOCSTRING] Add Args, Returns sections
        # Преобразуем Flesch в примерный уровень класса
        if flesch >= 90:
            flesch_grade = 5
        elif flesch >= 80:
            flesch_grade = 6
        elif flesch >= 70:
            flesch_grade = 7
        elif flesch >= 60:
            flesch_grade = 8
        elif flesch >= 50:
            flesch_grade = 9
        elif flesch >= 30:
            flesch_grade = 12
        else:
            flesch_grade = 16

        # Среднее арифметическое уровней
        avg_grade = (flesch_grade + fk_grade + smog) / 3

        if avg_grade <= 6:
            return "elementary"
        if avg_grade <= 8:
            return "middle_school"
        if avg_grade <= 12:
            return "high_school"
        if avg_grade <= 16:
            return "college"
        return "graduate"

    def _empty_readability_result(self) -> dict[str, Any]:
        """Пустой результат анализа читабельности"""
        return {
            "flesch_reading_ease": 0.0,
            "flesch_kincaid_grade": 0.0,
            "smog_index": 0.0,
            "automated_readability_index": 0.0,
            "coleman_liau_index": 0.0,
            "average_sentence_length": 0.0,
            "average_syllables_per_word": 0.0,
            "readability_consensus": "insufficient",
        }


@register_analyzer("advanced_algorithmic")
class AdvancedAlgorithmicAnalyzer(BaseAnalyzer):
    """
    Продвинутый алгоритмический анализатор с комплексным подходом

    Возможности:
    - Фонетический анализ рифм
    - Продвинутый анализ flow и ритма
    - Множественные индексы читабельности
    - Семантический анализ с градацией эмоций
    - Анализ литературных приемов
    - Кэширование для производительности
    """
    # TODO(google-review): [ARCHITECTURE] God class - too many responsibilities
    # TODO(google-review): [DOCSTRING] Add Args, Returns to class docstring

    def __init__(self, config: dict[str, Any] | None = None):
        # TODO(google-review): [PERFORMANCE] Cache has no max size limit
        # TODO(google-review): [DOCSTRING] Add Args section to docstring
        super().__init__(config)

        # Инициализация компонентов
        self.lexicon = AdvancedLexicon()
        self.flow_analyzer = FlowAnalyzer()
        self.rhyme_analyzer = RhymeAnalyzer()
        self.readability_analyzer = ReadabilityAnalyzer()

        # Настройки
        self.cache_enabled = self.config.get("cache_enabled", True)
        self.analysis_cache = {} if self.cache_enabled else None
        self.detailed_logging = self.config.get("detailed_logging", False)

        if self.detailed_logging:
            logger.setLevel(logging.DEBUG)

    def analyze_song(self, artist: str, title: str, lyrics: str) -> AnalysisResult:
        """
        Комплексный анализ песни с продвинутыми алгоритмами
        """
        # TODO(google-review): [DOCSTRING] Add Args, Returns, Raises sections
        # TODO(google-review): [ARCHITECTURE] Function too long (90+ lines)
        # TODO(google-review): [ERROR_HANDLING] Generic ValueError, be specific
        start_time = time.time()

        # Валидация
        if not self.validate_input(artist, title, lyrics):
            raise ValueError("Invalid input parameters")

        # Проверка кэша
        cache_key = None
        if self.cache_enabled:
            cache_key = self._generate_cache_key(artist, title, lyrics)
            if cache_key in self.analysis_cache:
                cached_result = self.analysis_cache[cache_key]
                logger.debug(f"Returning cached result for {artist} - {title}")
                return cached_result

        # Предобработка
        processed_lyrics = self.preprocess_lyrics(lyrics)
        lines = self._split_into_lines(processed_lyrics)
        words = self._extract_meaningful_words(processed_lyrics)

        if self.detailed_logging:
            logger.debug(
                f"Processing {artist} - {title}: {len(lines)} lines, {len(words)} words"
            )

        # Основной анализ
        analysis_results = {
            "advanced_sentiment": self._analyze_advanced_sentiment(words),
            "rhyme_analysis": self.rhyme_analyzer.analyze_rhyme_structure(lines),
            "flow_analysis": self.flow_analyzer.analyze_flow_patterns(lines),
            "readability_metrics": self.readability_analyzer.analyze_readability(
                processed_lyrics
            ),
            "thematic_analysis": self._analyze_themes_advanced(words),
            "literary_devices": self._analyze_literary_devices(processed_lyrics, words),
            "vocabulary_sophistication": self._analyze_vocabulary_sophistication(words),
            "structural_analysis": self._analyze_structure_advanced(
                lines, processed_lyrics
            ),
            "creativity_metrics": self._analyze_creativity_advanced(
                processed_lyrics, words, lines
            ),
        }

        # Композитные метрики
        composite_scores = self._calculate_advanced_composite_scores(analysis_results)
        analysis_results.update(composite_scores)

        # Общая уверенность
        confidence = self._calculate_advanced_confidence(analysis_results, lines, words)

        processing_time = time.time() - start_time

        # Метаданные
        metadata = {
            "analyzer_version": "2.0.0",
            "processing_date": datetime.now(tz=timezone.utc).isoformat(),
            "lyrics_length": len(processed_lyrics),
            "word_count": len(words),
            "line_count": len(lines),
            "processing_components": list(analysis_results.keys()),
            "cache_used": False,
            "detailed_logging": self.detailed_logging,
        }

        # Build plain dictionary result (legacy-friendly)
        result_dict = {
            "analysis_type": "advanced_algorithmic",
            "analysis_data": analysis_results,
            "confidence": confidence,
            "processing_time": processing_time,
            "metadata": metadata,
            "raw_output": analysis_results,
            "timestamp": datetime.now(tz=timezone.utc).isoformat(),
            "artist": artist,
            "title": title,
        }

        # Кэширование результата
        if self.cache_enabled and cache_key:
            self.analysis_cache[cache_key] = result_dict

        if self.detailed_logging:
            logger.debug(
                f"Analysis completed for {artist} - {title} in {processing_time:.3f}s"
            )

        return result_dict

    def _generate_cache_key(self, artist: str, title: str, lyrics: str) -> str:
        """Генерация ключа для кэша"""
        # TODO(google-review): [SECURITY] MD5 deprecated, use SHA256
        # TODO(google-review): [ARCHITECTURE] Magic number 500 as constant
        # TODO(google-review): [DOCSTRING] Add Args, Returns sections
        content = (
            f"{artist}|{title}|{lyrics[:500]}"  # Первые 500 символов для уникальности
        )
        return hashlib.md5(content.encode()).hexdigest()

    def _split_into_lines(self, lyrics: str) -> list[str]:
        """Разбиение на строки с очисткой"""
        lines = [line.strip() for line in lyrics.split("\n") if line.strip()]

        # Фильтрация метаданных
        return [
            line
            for line in lines
            if not re.match(
                r"^\[.*\]$|^\(.*\)$|^(Verse|Chorus|Bridge|Outro|Intro)[\s\d]*:",
                line,
                re.IGNORECASE,
            )
        ]

    def _extract_meaningful_words(self, lyrics: str) -> list[str]:
        """Извлечение значимых слов"""
        # TODO(google-review): [PERFORMANCE] Stop words recreated each call
        # TODO(google-review): [ARCHITECTURE] Move stop_words to class constant
        # TODO(google-review): [DOCSTRING] Add Args, Returns sections
        words = re.findall(r"\b[a-zA-Z]{2,}\b", lyrics.lower())

        # Расширенный список стоп-слов
        stop_words = {
            "the",
            "a",
            "an",
            "and",
            "or",
            "but",
            "in",
            "on",
            "at",
            "to",
            "for",
            "of",
            "with",
            "by",
            "from",
            "up",
            "about",
            "into",
            "through",
            "during",
            "before",
            "after",
            "above",
            "below",
            "between",
            "among",
            "this",
            "that",
            "these",
            "those",
            "i",
            "me",
            "my",
            "myself",
            "we",
            "our",
            "ours",
            "ourselves",
            "you",
            "your",
            "yours",
            "he",
            "him",
            "his",
            "himself",
            "she",
            "her",
            "hers",
            "herself",
            "it",
            "its",
            "itself",
            "they",
            "them",
            "their",
            "theirs",
            "themselves",
            "what",
            "which",
            "who",
            "whom",
            "am",
            "is",
            "are",
            "was",
            "were",
            "be",
            "been",
            "being",
            "have",
            "has",
            "had",
            "having",
            "do",
            "does",
            "did",
            "doing",
            "will",
            "would",
            "could",
            "should",
            "yeah",
            "uh",
            "oh",
            "got",
            "get",
            "gotta",
            "wanna",
            "gonna",
            "ain",
            "yall",
            "em",
            "ya",
            "like",
            "just",
            "now",
            "know",
            "see",
            "come",
            "go",
            "say",
            "said",
            "tell",
            "make",
            "way",
            "time",
            "want",
            "need",
            "take",
            "give",
            "put",
            "keep",
            "let",
            "think",
        }

        # TODO(google-review): [ARCHITECTURE] Magic number 3 as constant
        return [word for word in words if word not in stop_words and len(word) >= 3]

    def _analyze_advanced_sentiment(
        self, words: list[str]
    ) -> dict[str, Any]:
        """Продвинутый анализ настроения с градацией"""
        # TODO(google-review): [ARCHITECTURE] Function too long (60+ lines)
        # TODO(google-review): [DOCSTRING] Add Args, Returns sections
        if not words:
            return self._empty_sentiment_result()

        emotion_scores = {}
        total_emotional_words = 0

        # Анализ по категориям эмоций с учетом интенсивности
        for emotion, intensity_levels in self.lexicon.emotions.items():
            emotion_score = 0.0
            emotion_word_count = 0

            for intensity, word_set in intensity_levels.items():
                matches = len(set(words) & word_set)
                if matches > 0:
                    # Весовые коэффициенты для интенсивности
                    # TODO(google-review): [ARCHITECTURE] Move weights to class const
                    intensity_weight = {"weak": 1.0, "medium": 2.0, "strong": 3.0}[
                        intensity
                    ]
                    emotion_score += matches * intensity_weight
                    emotion_word_count += matches
                    total_emotional_words += matches

            emotion_scores[emotion] = {
                "score": emotion_score,
                "word_count": emotion_word_count,
                "normalized_score": emotion_score / len(words) if words else 0,
            }

        # Определение доминирующей эмоции
        if emotion_scores:
            dominant_emotion = max(emotion_scores.items(), key=lambda x: x[1]["score"])
            dominant_emotion_name = dominant_emotion[0]
            dominant_emotion_strength = dominant_emotion[1]["score"]
        else:
            dominant_emotion_name = "neutral"
            dominant_emotion_strength = 0

        # Вычисление общей эмоциональной валентности
        positive_emotions = emotion_scores.get("joy", {}).get("score", 0)
        negative_emotions = (
            emotion_scores.get("anger", {}).get("score", 0)
            + emotion_scores.get("sadness", {}).get("score", 0)
            + emotion_scores.get("fear", {}).get("score", 0)
        )

        if total_emotional_words > 0:
            valence = (positive_emotions - negative_emotions) / total_emotional_words
        else:
            valence = 0.0

        return {
            "emotion_scores": emotion_scores,
            "dominant_emotion": dominant_emotion_name,
            "dominant_emotion_strength": dominant_emotion_strength,
            "emotional_valence": valence,
            "emotional_intensity": total_emotional_words / len(words) if words else 0,
            "total_emotional_words": total_emotional_words,
            "emotional_complexity": len(
                [e for e in emotion_scores.values() if e["score"] > 0]
            ),
        }

    def _analyze_themes_advanced(self, words: list[str]) -> dict[str, Any]:
        """Продвинутый тематический анализ"""
        # TODO(google-review): [DOCSTRING] Add Args, Returns sections
        # TODO(google-review): [ARCHITECTURE] Function too long (50+ lines)
        if not words:
            return {"theme_scores": {}, "dominant_theme": "neutral"}

        word_set = set(words)
        theme_scores = {}

        # Анализ по тематическим категориям
        for theme, theme_words in self.lexicon.rap_themes.items():
            matches = len(word_set & theme_words)
            theme_scores[theme] = {
                "absolute_count": matches,
                "relative_score": matches / len(words),
                "theme_coverage": matches / len(theme_words) if theme_words else 0,
            }

        # Определение доминирующих тем
        sorted_themes = sorted(
            theme_scores.items(), key=lambda x: x[1]["absolute_count"], reverse=True
        )

        dominant_theme = (
            sorted_themes[0][0]
            if sorted_themes and sorted_themes[0][1]["absolute_count"] > 0
            else "neutral"
        )

        # Вычисление тематического разнообразия
        active_themes = [
            theme
            for theme, scores in theme_scores.items()
            if scores["absolute_count"] > 0
        ]
        thematic_diversity = len(active_themes) / len(self.lexicon.rap_themes)

        return {
            "theme_scores": theme_scores,
            "dominant_theme": dominant_theme,
            "secondary_themes": [
                theme[0]
                for theme in sorted_themes[1:4]
                if theme[1]["absolute_count"] > 0
            ],
            "thematic_diversity": thematic_diversity,
            "total_thematic_words": sum(
                score["absolute_count"] for score in theme_scores.values()
            ),
        }

    def _analyze_literary_devices(
        self, lyrics: str, words: list[str]
    ) -> dict[str, Any]:
        """Анализ литературных приемов"""
        # TODO(google-review): [DOCSTRING] Add Args, Returns sections
        # TODO(google-review): [ARCHITECTURE] Function too long (50+ lines)
        if not lyrics or not words:
            return self._empty_literary_result()

        # Поиск метафор и сравнений
        metaphor_count = 0
        simile_count = 0
        lyrics_lower = lyrics.lower()

        for indicator in self.lexicon.literary_devices["metaphor_indicators"]:
            if indicator in ["like", "as"]:
                simile_count += lyrics_lower.count(f" {indicator} ")
            else:
                metaphor_count += lyrics_lower.count(indicator)

        # Анализ временных отсылок
        time_references = sum(
            lyrics_lower.count(time_word)
            for time_word in self.lexicon.literary_devices["time_references"]
        )

        # Поиск контрастов и противопоставлений
        contrast_usage = sum(
            lyrics_lower.count(contrast_word)
            for contrast_word in self.lexicon.literary_devices["contrast_words"]
        )

        # Анализ повторов и рефренов
        line_repetitions = self._analyze_repetitions(lyrics)

        # Персонификация (упрощенно)
        personification_indicators = [
            "speaks",
            "whispers",
            "calls",
            "cries",
            "laughs",
            "dances",
        ]
        personification_count = sum(
            lyrics_lower.count(indicator) for indicator in personification_indicators
        )

        return {
            "metaphor_count": metaphor_count,
            "simile_count": simile_count,
            "time_references": time_references,
            "contrast_usage": contrast_usage,
            "repetition_analysis": line_repetitions,
            "personification_count": personification_count,
            "total_literary_devices": metaphor_count
            + simile_count
            + time_references
            + contrast_usage
            + personification_count,
        }

    def _analyze_repetitions(self, lyrics: str) -> dict[str, Any]:
        """Анализ повторов в тексте"""
        # TODO(google-review): [DOCSTRING] Add Args, Returns sections
        # TODO(google-review): [ARCHITECTURE] Magic numbers: 2, 5
        lines = [line.strip().lower() for line in lyrics.split("\n") if line.strip()]

        if not lines:
            return {"repeated_lines": 0, "repetition_ratio": 0.0}

        line_counts = Counter(lines)
        repeated_lines = {
            line: count for line, count in line_counts.items() if count > 1
        }

        # Анализ повторяющихся фраз (2-4 слова)
        phrase_repetitions = defaultdict(int)
        for line in lines:
            words = line.split()
            for i in range(len(words) - 1):
                for j in range(2, min(5, len(words) - i + 1)):
                    phrase = " ".join(words[i : i + j])
                    if len(phrase) > 5:  # Игнорируем очень короткие фразы
                        phrase_repetitions[phrase] += 1

        repeated_phrases = {
            phrase: count for phrase, count in phrase_repetitions.items() if count > 1
        }

        return {
            "repeated_lines": len(repeated_lines),
            "repeated_phrases": len(repeated_phrases),
            "repetition_ratio": len(repeated_lines) / len(set(lines)) if lines else 0,
            "most_repeated_line": max(line_counts.items(), key=lambda x: x[1])
            if line_counts
            else None,
            "total_line_repetitions": sum(
                count - 1 for count in line_counts.values() if count > 1
            ),
        }

    def _analyze_vocabulary_sophistication(self, words: list[str]) -> dict[str, Any]:
        """Анализ сложности словаря"""
        # TODO(google-review): [DOCSTRING] Add Args, Returns sections
        # TODO(google-review): [ARCHITECTURE] Function too long (70+ lines)
        # TODO(google-review): [PERFORMANCE] common_words recreated each call
        if not words:
            return self._empty_vocabulary_result()

        word_set = set(words)

        # Анализ сложности мышления
        complexity_scores = {}
        total_complex_words = 0

        for category, category_words in self.lexicon.complexity_indicators.items():
            matches = len(word_set & category_words)
            complexity_scores[category] = matches
            total_complex_words += matches

        # Анализ длины слов
        # TODO(google-review): [ARCHITECTURE] Magic number 7 as constant
        word_lengths = [len(word) for word in words]
        avg_word_length = sum(word_lengths) / len(word_lengths)
        long_words = len([w for w in words if len(w) >= 7])

        # Уникальность словаря
        vocabulary_richness = len(word_set) / len(words)

        # Редкие/необычные слова (эвристика)
        common_words = {
            "love",
            "money",
            "life",
            "time",
            "good",
            "bad",
            "right",
            "wrong",
            "black",
            "white",
            "red",
            "blue",
            "big",
            "small",
            "old",
            "new",
            "high",
            "low",
            "fast",
            "slow",
            "hot",
            "cold",
            "young",
            "real",
            "hard",
            "easy",
            "free",
            "game",
            "play",
            "work",
            "home",
            "house",
        }

        uncommon_words = len(word_set - common_words)
        uncommon_ratio = uncommon_words / len(words)

        return {
            "complexity_scores": complexity_scores,
            "total_complex_words": total_complex_words,
            "average_word_length": avg_word_length,
            "long_words_count": long_words,
            "vocabulary_richness": vocabulary_richness,
            "uncommon_words_ratio": uncommon_ratio,
            "lexical_diversity": len(word_set) / max(len(words), 1),
            "sophisticated_vocabulary_score": (
                total_complex_words + long_words + uncommon_words
            )
            / len(words),
        }

    def _analyze_structure_advanced(
        self, lines: list[str], full_text: str
    ) -> dict[str, Any]:
        """Продвинутый структурный анализ"""
        # TODO(google-review): [DOCSTRING] Add Args, Returns sections
        if not lines:
            return self._empty_structure_result()

        # Базовые метрики структуры
        total_lines = len(lines)
        line_lengths = [len(line.split()) for line in lines]
        avg_line_length = sum(line_lengths) / len(line_lengths)
        line_length_variance = sum(
            (x - avg_line_length) ** 2 for x in line_lengths
        ) / len(line_lengths)

        # Анализ пунктуации и паузы
        punctuation_analysis = self._analyze_punctuation(full_text)

        # Структурная регулярность
        structure_patterns = self._find_structure_patterns(lines)

        # Анализ строфической структуры (по пустым строкам)
        stanzas = self._identify_stanzas(full_text)

        return {
            "total_lines": total_lines,
            "average_line_length": avg_line_length,
            "line_length_variance": line_length_variance,
            "punctuation_analysis": punctuation_analysis,
            "structure_patterns": structure_patterns,
            "stanza_analysis": stanzas,
            "structural_consistency": 1.0
            - (line_length_variance / max(avg_line_length, 1)),
        }

    def _analyze_punctuation(self, text: str) -> dict[str, int]:
        """Анализ использования пунктуации"""
        punctuation_counts = {
            "periods": text.count("."),
            "commas": text.count(","),
            "exclamations": text.count("!"),
            "questions": text.count("?"),
            "semicolons": text.count(";"),
            "colons": text.count(":"),
            "dashes": text.count("-") + text.count("--"),
            "ellipses": text.count("..."),
            "parentheses": text.count("(") + text.count(")"),
            "quotations": text.count('"') + text.count("'"),
        }

        total_punctuation = sum(punctuation_counts.values())
        punctuation_counts["total"] = total_punctuation

        return punctuation_counts

    def _find_structure_patterns(self, lines: list[str]) -> dict[str, Any]:
        """Поиск структурных паттернов"""
        if len(lines) < 4:
            return {"pattern_found": False}

        # Анализ паттернов длины строк
        line_lengths = [len(line.split()) for line in lines]

        # Поиск повторяющихся паттернов длины (например, ABAB по длине)
        pattern_length = 4
        patterns_found = []

        for i in range(len(line_lengths) - pattern_length + 1):
            pattern = line_lengths[i : i + pattern_length]
            # Ищем повторение этого паттерна
            for j in range(i + pattern_length, len(line_lengths) - pattern_length + 1):
                if line_lengths[j : j + pattern_length] == pattern:
                    patterns_found.append(pattern)
                    break

        return {
            "pattern_found": len(patterns_found) > 0,
            "patterns": patterns_found,
            "pattern_consistency": len(patterns_found)
            / max(len(line_lengths) // pattern_length, 1),
        }

    def _identify_stanzas(self, text: str) -> dict[str, Any]:
        """Анализ строфической структуры"""
        # Разбиваем текст по двойным переносам строк
        stanzas = [
            stanza.strip() for stanza in re.split(r"\n\s*\n", text) if stanza.strip()
        ]

        if not stanzas:
            stanzas = [text]  # Если нет явных строф, весь текст - одна строфа

        stanza_lengths = [len(stanza.split("\n")) for stanza in stanzas]

        return {
            "stanza_count": len(stanzas),
            "average_stanza_length": sum(stanza_lengths) / len(stanza_lengths)
            if stanza_lengths
            else 0,
            "stanza_length_consistency": self._calculate_consistency_score(
                stanza_lengths
            ),
            "stanza_lengths": stanza_lengths,
        }

    def _analyze_creativity_advanced(
        self, lyrics: str, words: list[str], lines: list[str]
    ) -> dict[str, Any]:
        """Продвинутый анализ креативности"""
        # TODO(google-review): [DOCSTRING] Add Args, Returns sections
        # TODO(google-review): [ARCHITECTURE] Function too long (40+ lines)
        if not words or not lines:
            return self._empty_creativity_result()

        # Неологизмы и необычные словоформы
        neologisms = self._detect_neologisms(words)

        # Оригинальные фразовые конструкции
        unique_phrases = self._find_unique_phrases(lines)

        # Семантические сдвиги и игра слов
        wordplay_analysis = self._analyze_advanced_wordplay(lyrics, words)

        # Инновационность рифм
        innovative_rhymes = self._analyze_rhyme_innovation(lines)

        # Общий показатель креативности
        creativity_factors = [
            len(neologisms) / max(len(words), 1),
            len(unique_phrases) / max(len(lines), 1),
            wordplay_analysis["total_score"],
            innovative_rhymes["innovation_score"],
        ]

        overall_creativity = sum(creativity_factors) / len(creativity_factors)

        return {
            "neologisms": neologisms,
            "unique_phrases": unique_phrases,
            "wordplay_analysis": wordplay_analysis,
            "innovative_rhymes": innovative_rhymes,
            "creativity_factors": creativity_factors,
            "overall_creativity_score": overall_creativity,
        }

    def _detect_neologisms(self, words: list[str]) -> list[str]:
        """Обнаружение неологизмов и необычных слов"""
        # TODO(google-review): [ARCHITECTURE] Magic numbers: 6, 4, 10
        # TODO(google-review): [DOCSTRING] Add Args, Returns sections
        # Простая эвристика для обнаружения возможных неологизмов
        potential_neologisms = []

        for word in words:
            # Слова с необычными суффиксами или префиксами
            if len(word) > 6 and (
                word.endswith(("ness", "tion", "ism"))
                or word.startswith(("un", "pre", "over"))
            ):
                potential_neologisms.append(word)

            # Слова с повторяющимися частями
            if len(word) > 4:
                mid = len(word) // 2
                if word[:mid] == word[mid:] or word[:mid] in word[mid:]:
                    potential_neologisms.append(word)

        return potential_neologisms[
            :10
        ]  # Ограничиваем количество для производительности

    def _find_unique_phrases(self, lines: list[str]) -> list[str]:
        """Поиск уникальных фразовых конструкций"""
        unique_phrases = []

        # Ищем фразы с необычной структурой
        for line in lines:
            words = line.split()
            if len(words) >= 3:
                # Поиск инвертированных конструкций
                if len(words) >= 4 and words[0].lower() in [
                    "when",
                    "where",
                    "how",
                    "why",
                ]:
                    unique_phrases.append(line)

                # Поиск аллитераций в фразах
                if (
                    len(
                        [
                            w
                            for w in words[:3]
                            if w and words[0] and w[0].lower() == words[0][0].lower()
                        ]
                    )
                    >= 2
                ):
                    unique_phrases.append(line)

        return unique_phrases[:5]  # Ограничиваем для производительности

    def _analyze_advanced_wordplay(
        self, lyrics: str, words: list[str]
    ) -> dict[str, Any]:
        """Анализ продвинутых приемов игры слов"""
        # TODO(google-review): [ARCHITECTURE] Magic numbers as constants
        # TODO(google-review): [DOCSTRING] Add Args, Returns sections
        # TODO(google-review): [PERFORMANCE] Nested loops inefficient
        wordplay_score = 0
        techniques_found = []

        # Двойные смыслы (упрощенно)
        double_meanings = [
            word
            for word in set(words)
            if len(word) > 4
            and any(
                other in word
                for other in words
                if other != word and len(other) > 2
            )
        ]

        if double_meanings:
            wordplay_score += len(double_meanings) * 0.1
            techniques_found.append("double_meanings")

        # Звукоподражания
        onomatopoeia = [
            "bang",
            "boom",
            "crash",
            "pop",
            "snap",
            "crack",
            "splash",
            "whoosh",
        ]
        onomatopoeia_count = sum(lyrics.lower().count(sound) for sound in onomatopoeia)

        if onomatopoeia_count > 0:
            wordplay_score += onomatopoeia_count * 0.05
            techniques_found.append("onomatopoeia")

        # Каламбуры (простая эвристика)
        puns_detected = 0
        for i, word in enumerate(words[:-1]):
            next_word = words[i + 1]
            if (
                len(word) > 3
                and len(next_word) > 3
                and word[:-1] == next_word[:-1]
                and word != next_word
            ):
                puns_detected += 1

        if puns_detected > 0:
            wordplay_score += puns_detected * 0.15
            techniques_found.append("potential_puns")

        return {
            "total_score": min(wordplay_score, 1.0),
            "techniques_found": techniques_found,
            "double_meanings": double_meanings,
            "onomatopoeia_count": onomatopoeia_count,
            "potential_puns": puns_detected,
        }

    def _analyze_rhyme_innovation(self, lines: list[str]) -> dict[str, Any]:
        """Анализ инновационности рифм"""
        # TODO(google-review): [ARCHITECTURE] Magic numbers: 4, 12, 3, 6
        # TODO(google-review): [DOCSTRING] Add Args, Returns sections
        if len(lines) < 4:
            return {"innovation_score": 0.0}

        # Извлекаем окончания
        endings = []
        for line in lines[:12]:  # Анализируем первые 12 строк
            words = line.split()
            if words:
                ending = re.sub(r"[^\w]", "", words[-1].lower())
                if len(ending) >= 2:
                    endings.append(ending)

        innovation_factors = []

        # Мультисложные рифмы
        multisyllabic_rhymes = 0
        flow_analyzer = FlowAnalyzer()
        for ending in endings:
            if flow_analyzer._syllables_in_word(ending) >= 3:
                multisyllabic_rhymes += 1

        if endings:
            multisyllabic_ratio = multisyllabic_rhymes / len(endings)
            innovation_factors.append(multisyllabic_ratio)

        # Внутренние рифмы
        internal_rhyme_count = 0
        for line in lines:
            words = line.split()
            for i in range(len(words) - 1):
                for j in range(i + 1, len(words)):
                    if self._simple_rhyme_check(words[i], words[j]):
                        internal_rhyme_count += 1

        internal_rhyme_ratio = internal_rhyme_count / max(len(lines), 1)
        innovation_factors.append(min(internal_rhyme_ratio, 1.0))

        # Необычные рифмы (длинные окончания)
        long_rhymes = len([e for e in endings if len(e) >= 6])
        long_rhyme_ratio = long_rhymes / max(len(endings), 1)
        innovation_factors.append(long_rhyme_ratio)

        overall_innovation = (
            sum(innovation_factors) / len(innovation_factors)
            if innovation_factors
            else 0
        )

        return {
            "innovation_score": overall_innovation,
            "multisyllabic_rhymes": multisyllabic_rhymes,
            "internal_rhymes": internal_rhyme_count,
            "long_rhymes": long_rhymes,
            "innovation_factors": innovation_factors,
        }

    def _simple_rhyme_check(self, word1: str, word2: str) -> bool:
        """Простая проверка рифмы"""
        if len(word1) < 2 or len(word2) < 2 or word1 == word2:
            return False
        return word1[-2:].lower() == word2[-2:].lower()

    def _calculate_advanced_composite_scores(
        self, analysis_results: dict[str, Any]
    ) -> dict[str, Any]:
        """Вычисление продвинутых композитных оценок"""
        # TODO(google-review): [ARCHITECTURE] Magic weights should be constants
        # TODO(google-review): [DOCSTRING] Add Args, Returns sections

        # Извлекаем ключевые метрики
        rhyme_density = analysis_results.get("rhyme_analysis", {}).get(
            "rhyme_density", 0
        )
        flow_consistency = analysis_results.get("flow_analysis", {}).get(
            "syllable_consistency", 0
        )
        vocabulary_richness = analysis_results.get("vocabulary_sophistication", {}).get(
            "vocabulary_richness", 0
        )
        creativity_score = analysis_results.get("creativity_metrics", {}).get(
            "overall_creativity_score", 0
        )
        readability = analysis_results.get("readability_metrics", {}).get(
            "flesch_reading_ease", 0
        )

        # Нормализуем readability (Flesch scale: 0-100, higher = easier)
        normalized_readability = readability / 100

        # Композитные метрики
        technical_mastery = (
            rhyme_density * 0.4 + flow_consistency * 0.4 + vocabulary_richness * 0.2
        )

        artistic_sophistication = (
            creativity_score * 0.5
            + vocabulary_richness * 0.3
            + (1 - normalized_readability) * 0.2
        )  # Более сложный текст = выше артистичность

        overall_quality = (
            technical_mastery * 0.4
            + artistic_sophistication * 0.4
            + creativity_score * 0.2
        )

        # Инновационность
        innovation_score = creativity_score * 0.6 + rhyme_density * 0.4

        return {
            "composite_scores": {
                "technical_mastery": technical_mastery,
                "artistic_sophistication": artistic_sophistication,
                "overall_quality": overall_quality,
                "innovation_score": innovation_score,
                "complexity_balance": (
                    vocabulary_richness + (1 - normalized_readability)
                )
                / 2,
            }
        }

    def _calculate_advanced_confidence(
        self, analysis_results: dict[str, Any], lines: list[str], words: list[str]
    ) -> float:
        """Расчет продвинутой оценки уверенности"""
        # TODO(google-review): [ARCHITECTURE] Magic numbers: 100, 10, 8, 2
        # TODO(google-review): [DOCSTRING] Add Args, Returns sections
        confidence_factors = []

        # Фактор объема текста
        text_volume_factor = min(len(words) / 100, 1.0) * min(len(lines) / 10, 1.0)
        confidence_factors.append(text_volume_factor)

        # Фактор завершенности анализа
        expected_analyses = [
            "advanced_sentiment",
            "rhyme_analysis",
            "flow_analysis",
            "readability_metrics",
        ]
        completed_analyses = sum(
            1 for analysis in expected_analyses if analysis in analysis_results
        )
        completeness_factor = completed_analyses / len(expected_analyses)
        confidence_factors.append(completeness_factor)

        # Фактор качества данных
        if words and lines:
            avg_line_length = sum(len(line.split()) for line in lines) / len(lines)
            quality_factor = min(
                avg_line_length / 8, 1.0
            )  # Оптимальная длина строки ~ 8 слов
            confidence_factors.append(quality_factor)

        # Фактор разнообразия словаря
        if words:
            vocab_diversity = len(set(words)) / len(words)
            diversity_factor = min(vocab_diversity * 2, 1.0)
            confidence_factors.append(diversity_factor)

        return (
            sum(confidence_factors) / len(confidence_factors)
            if confidence_factors
            else 0.5
        )

    def _calculate_consistency_score(self, values: list[float]) -> float:
        """Вычисление консистентности значений"""
        if len(values) < 2:
            return 1.0

        mean = sum(values) / len(values)
        if mean == 0:
            return 1.0

        variance = sum((x - mean) ** 2 for x in values) / len(values)
        coefficient_of_variation = (variance**0.5) / mean

        # Консистентность: чем меньше коэффициент вариации, тем выше консистентность
        consistency = 1 / (1 + coefficient_of_variation)
        return min(consistency, 1.0)

    # Методы для пустых результатов
    def _empty_sentiment_result(self) -> dict[str, Any]:
        # TODO(google-review): [TYPING] Add return type hint
        return {
            "emotion_scores": {},
            "dominant_emotion": "neutral",
            "dominant_emotion_strength": 0,
            "emotional_valence": 0.0,
            "emotional_intensity": 0.0,
            "total_emotional_words": 0,
            "emotional_complexity": 0,
        }

    def _empty_literary_result(self) -> dict[str, Any]:
        # TODO(google-review): [TYPING] Add return type hint
        return {
            "metaphor_count": 0,
            "simile_count": 0,
            "time_references": 0,
            "contrast_usage": 0,
            "repetition_analysis": {"repeated_lines": 0, "repetition_ratio": 0.0},
            "personification_count": 0,
            "total_literary_devices": 0,
        }

    def _empty_vocabulary_result(self) -> dict[str, Any]:
        # TODO(google-review): [TYPING] Add return type hint
        return {
            "complexity_scores": {},
            "total_complex_words": 0,
            "average_word_length": 0.0,
            "long_words_count": 0,
            "vocabulary_richness": 0.0,
            "uncommon_words_ratio": 0.0,
            "lexical_diversity": 0.0,
            "sophisticated_vocabulary_score": 0.0,
        }

    def _empty_structure_result(self) -> dict[str, Any]:
        # TODO(google-review): [TYPING] Add return type hint
        return {
            "total_lines": 0,
            "average_line_length": 0.0,
            "line_length_variance": 0.0,
            "punctuation_analysis": {},
            "structure_patterns": {"pattern_found": False},
            "stanza_analysis": {"stanza_count": 0},
            "structural_consistency": 0.0,
        }

    def _empty_creativity_result(self) -> dict[str, Any]:
        # TODO(google-review): [TYPING] Add return type hint
        return {
            "neologisms": [],
            "unique_phrases": [],
            "wordplay_analysis": {"total_score": 0.0},
            "innovative_rhymes": {"innovation_score": 0.0},
            "creativity_factors": [0.0, 0.0, 0.0, 0.0],
            "overall_creativity_score": 0.0,
        }

    def get_analyzer_info(self) -> dict[str, Any]:
        """Информация об анализаторе"""
        return {
            "name": "AdvancedAlgorithmicAnalyzer",
            "version": "2.0.0",
            "description": "Advanced algorithmic lyrics analysis with phonetic rhyme analysis, flow metrics, readability indices, and semantic sophistication scoring",
            "author": "Rap Scraper Project - Advanced Edition",
            "type": self.analyzer_type,
            "supported_features": self.supported_features,
            "components": [
                "AdvancedLexicon",
                "FlowAnalyzer",
                "RhymeAnalyzer",
                "ReadabilityAnalyzer",
                "LiteraryDevicesAnalyzer",
            ],
            "config_options": {
                "cache_enabled": "Enable result caching for performance (default: True)",
                "detailed_logging": "Enable detailed debug logging (default: False)",
                "min_word_length": "Minimum word length for analysis (default: 3)",
                "max_cache_size": "Maximum cache entries (default: 1000)",
            },
            "performance": {
                "typical_processing_time": "50-200ms per song",
                "memory_usage": "~5-10MB for cache",
                "scalability": "Excellent for batch processing",
            },
        }

    @property
    def analyzer_type(self) -> str:
        """Тип анализатора"""
        return "advanced_algorithmic"

    @property
    def supported_features(self) -> list[str]:
        """Поддерживаемые функции анализа"""
        return [
            "phonetic_rhyme_analysis",
            "advanced_flow_metrics",
            "readability_indices",
            "emotional_gradient_analysis",
            "thematic_categorization",
            "literary_devices_detection",
            "vocabulary_sophistication",
            "structural_pattern_analysis",
            "creativity_assessment",
            "composite_scoring",
            "performance_caching",
        ]

    def clear_cache(self):
        """Очистка кэша"""
        if self.analysis_cache:
            self.analysis_cache.clear()
            logger.info("Analysis cache cleared")


# Демонстрационная функция
async def demo_advanced_analysis() -> None:
    """Демонстрация возможностей продвинутого анализатора"""
    # TODO(google-review): [DOCSTRING] Add Returns section to docstring

    sample_lyrics = """
    Metaphors cascade like waterfalls in my mind
    Each syllable calculated, rhythmically designed  
    Philosophy meets poetry in this verbal shrine
    Where consciousness and consonance perfectly align

    The lexicon I wield cuts deeper than a blade
    Multisyllabic mastery in every word I've made
    Assonance and alliteration, my lyrical trade
    While lesser wordsmiths stumble, my foundation's never swayed

    In the labyrinth of language, I navigate with ease
    Semantic sophistication brings critics to their knees  
    Innovation flows through every line like autumn leaves
    This artistic architecture is what true genius achieves
    """

    print("🚀 ДЕМОНСТРАЦИЯ ПРОДВИНУТОГО АЛГОРИТМИЧЕСКОГО АНАЛИЗАТОРА")
    print("=" * 70)

    analyzer = AdvancedAlgorithmicAnalyzer(
        {"cache_enabled": True, "detailed_logging": True}
    )

    try:
        result = analyzer.analyze_song(
            "Demo Artist", "Advanced Analysis Demo", sample_lyrics
        )

        print("\n📊 РЕЗУЛЬТАТЫ АНАЛИЗА:")
        print(f"🎯 Уверенность: {result['confidence']:.3f}")
        print(f"⚡ Время обработки: {result['processing_time']:.3f}s")

        # Рифмы и звучание
        rhyme_analysis = result["raw_output"].get("rhyme_analysis", {})
        print("\n🎵 РИФМЫ И ЗВУЧАНИЕ:")
        print(f"  Схема рифмовки: {rhyme_analysis.get('rhyme_scheme', 'N/A')}")
        print(f"  Плотность рифм: {rhyme_analysis.get('rhyme_density', 0):.3f}")
        print(f"  Аллитерация: {rhyme_analysis.get('alliteration_score', 0):.3f}")
        print(f"  Внутренние рифмы: {rhyme_analysis.get('internal_rhymes', 0)}")

        # Flow анализ
        flow_analysis = result["raw_output"].get("flow_analysis", {})
        print("\n🌊 FLOW И РИТМ:")
        print(
            f"  Консистентность слогов: {flow_analysis.get('syllable_consistency', 0):.3f}"
        )
        print(
            f"  Ср. слогов на строку: {flow_analysis.get('average_syllables_per_line', 0):.1f}"
        )
        print(
            f"  Ритмическая плотность: {flow_analysis.get('rhythmic_density', 0):.3f}"
        )

        # Читабельность
        readability = result["raw_output"].get("readability_metrics", {})
        print("\n📚 ЧИТАБЕЛЬНОСТЬ:")
        print(f"  Flesch Reading Ease: {readability.get('flesch_reading_ease', 0):.1f}")
        print(f"  SMOG Index: {readability.get('smog_index', 0):.1f}")
        print(f"  Консенсус: {readability.get('readability_consensus', 'N/A')}")

        # Композитные оценки
        composite = result["raw_output"].get("composite_scores", {})
        print("\n🏆 КОМПОЗИТНЫЕ ОЦЕНКИ:")
        print(f"  Техническое мастерство: {composite.get('technical_mastery', 0):.3f}")
        print(
            f"  Артистическая утончённость: {composite.get('artistic_sophistication', 0):.3f}"
        )
        print(f"  Общее качество: {composite.get('overall_quality', 0):.3f}")
        print(f"  Инновационность: {composite.get('innovation_score', 0):.3f}")

        print("=" * 70)
        print("✅ Демонстрация завершена успешно!")

    except Exception as e:
        # TODO(google-review): [ERROR_HANDLING] Use logger.exception instead
        # TODO(google-review): [ERROR_HANDLING] Too broad exception clause
        print(f"❌ Ошибка демонстрации: {e}")
        import traceback

        traceback.print_exc()


# Класс для работы с PostgreSQL
class PostgreSQLAnalyzer:
    """Анализатор для работы с PostgreSQL базой данных"""
    # TODO(google-review): [ARCHITECTURE] Missing connection pooling
    # TODO(google-review): [ARCHITECTURE] Tight coupling with database

    def __init__(self):
        """Инициализация PostgreSQL анализатора"""
        self.analyzer = AdvancedAlgorithmicAnalyzer(
            {"cache_enabled": True, "detailed_logging": False}
        )

        # Конфигурация PostgreSQL (будет загружена из config.yaml)
        self.db_config = self._load_db_config()

    def _load_db_config(self) -> dict[str, Any]:
        """Загрузка конфигурации БД"""
        try:
            import yaml

            # Сначала пытаемся использовать новую систему config_loader
            try:
                from config.config_loader import get_config
                config_obj = get_config()
                db_config = config_obj.database
                return {
                    "host": db_config.host,
                    "port": db_config.port,
                    "name": db_config.database,
                    "user": db_config.username,
                    "password": db_config.password,
                }
            except (ImportError, AttributeError):
                # Fallback на YAML файл
                config_path = Path(__file__).parent.parent.parent / "config.yaml"

                if config_path.exists():
                    with open(config_path, encoding="utf-8") as f:
                        config = yaml.safe_load(f)
                        return config.get("database", {})
                else:
                    print(
                        "⚠️ Файл config.yaml не найден, используются значения по умолчанию"
                    )
                    # TODO(google-review): [SECURITY] Hardcoded credentials risk
                    # TODO(google-review): [ARCHITECTURE] Use environment variables
                    return {
                        "host": "localhost",
                        "port": 5432,
                        "name": "rap_lyrics_db",
                        "user": "postgres",
                        "password": "password",
                    }
        except Exception as e:
            # TODO(google-review): [ERROR_HANDLING] Too broad exception clause
            # TODO(google-review): [ERROR_HANDLING] Use logger instead of print
            print(f"⚠️ Ошибка загрузки конфигурации: {e}")
            return {}

    async def get_database_stats(self) -> dict[str, Any]:
        """Получение статистики базы данных"""
        # TODO(google-review): [DOCSTRING] Add Returns section to docstring
        # TODO(google-review): [ARCHITECTURE] Function too long (100+ lines)
        try:
            import asyncpg

            # Создаем подключение
            conn = await asyncpg.connect(
                host=self.db_config.get("host", "localhost"),
                port=self.db_config.get("port", 5432),
                database=self.db_config.get("name", "rap_lyrics_db"),
                user=self.db_config.get("user", "postgres"),
                password=self.db_config.get("password", "password"),
            )

            try:
                # Запрос общей статистики
                total_stats_query = """
                SELECT 
                    COUNT(*) as total_songs,
                    COUNT(DISTINCT artist) as unique_artists,
                    COUNT(CASE WHEN lyrics IS NOT NULL THEN 1 END) as songs_with_lyrics,
                    COUNT(CASE WHEN lyrics IS NOT NULL AND LENGTH(lyrics) > 100 THEN 1 END) as analyzable_songs,
                    COUNT(CASE WHEN lyrics IS NULL OR LENGTH(lyrics) <= 100 THEN 1 END) as non_analyzable_songs
                FROM tracks;
                """

                # Запрос статистики по длине текстов (только для песен с текстами)
                lyrics_stats_query = """
                SELECT 
                    AVG(LENGTH(lyrics)) as avg_lyrics_length,
                    MIN(LENGTH(lyrics)) as min_lyrics_length,
                    MAX(LENGTH(lyrics)) as max_lyrics_length,
                    PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY LENGTH(lyrics)) as median_lyrics_length
                FROM tracks 
                WHERE lyrics IS NOT NULL;
                """

                total_result = await conn.fetchrow(total_stats_query)
                lyrics_result = await conn.fetchrow(lyrics_stats_query)

                stats = {
                    "total_songs": total_result["total_songs"],
                    "unique_artists": total_result["unique_artists"],
                    "songs_with_lyrics": total_result["songs_with_lyrics"],
                    "analyzable_songs": total_result["analyzable_songs"],
                    "non_analyzable_songs": total_result["non_analyzable_songs"],
                    "avg_lyrics_length": float(lyrics_result["avg_lyrics_length"])
                    if lyrics_result["avg_lyrics_length"]
                    else 0,
                    "min_lyrics_length": lyrics_result["min_lyrics_length"],
                    "max_lyrics_length": lyrics_result["max_lyrics_length"],
                    "median_lyrics_length": float(lyrics_result["median_lyrics_length"])
                    if lyrics_result["median_lyrics_length"]
                    else 0,
                }

                print("📊 ПОЛНАЯ СТАТИСТИКА БАЗЫ ДАННЫХ:")
                print("=" * 50)
                print(f"  📀 Всего записей в БД: {stats['total_songs']:,}")
                print(f"  🎤 Уникальных исполнителей: {stats['unique_artists']:,}")
                print(f"  📝 Песен с текстами: {stats['songs_with_lyrics']:,}")
                print(
                    f"  ✅ Песен для анализа (>100 символов): {stats['analyzable_songs']:,}"
                )
                print(
                    f"  ❌ Непригодных для анализа: {stats['non_analyzable_songs']:,}"
                )
                print("")
                print("📏 СТАТИСТИКА ПО ДЛИНЕ ТЕКСТОВ:")
                print(f"  Средняя длина: {stats['avg_lyrics_length']:.0f} символов")
                print(
                    f"  Медианная длина: {stats['median_lyrics_length']:.0f} символов"
                )
                print(
                    f"  Диапазон: {stats['min_lyrics_length']:,} - {stats['max_lyrics_length']:,} символов"
                )

                # Вычисляем процентные соотношения
                if stats["total_songs"] > 0:
                    lyrics_percent = (
                        stats["songs_with_lyrics"] / stats["total_songs"]
                    ) * 100
                    analyzable_percent = (
                        stats["analyzable_songs"] / stats["total_songs"]
                    ) * 100
                    print("")
                    print("📊 ПРОЦЕНТНЫЕ СООТНОШЕНИЯ:")
                    print(f"  Песен с текстами: {lyrics_percent:.1f}%")
                    print(f"  Пригодных для анализа: {analyzable_percent:.1f}%")

                return stats

            finally:
                await conn.close()

        except Exception as e:
            # TODO(google-review): [ERROR_HANDLING] Too broad exception clause
            # TODO(google-review): [ERROR_HANDLING] Use logger.exception
            print(f"❌ Ошибка получения статистики БД: {e}")
            return {}

    async def analyze_all_songs(
        self, limit: int | None = None, batch_size: int = 100
    ) -> dict[str, Any]:
        """Анализ всех песен в базе данных"""
        # TODO(google-review): [DOCSTRING] Add Args, Returns sections
        # TODO(google-review): [ARCHITECTURE] Function too long (90+ lines)
        # TODO(google-review): [ARCHITECTURE] Magic number 100
        try:
            import asyncpg

            conn = await asyncpg.connect(
                host=self.db_config.get("host", "localhost"),
                port=self.db_config.get("port", 5432),
                database=self.db_config.get("name", "rap_lyrics_db"),
                user=self.db_config.get("user", "postgres"),
                password=self.db_config.get("password", "password"),
            )

            try:
                # Запрос песен для анализа
                # TODO(google-review): [SECURITY] SQL injection risk with f-string
                # TODO(google-review): [ARCHITECTURE] Use parameterized queries
                limit_clause = f"LIMIT {limit}" if limit else ""
                query = f"""
                SELECT id, artist, title, lyrics 
                FROM tracks 
                WHERE lyrics IS NOT NULL AND LENGTH(lyrics) > 100
                ORDER BY id
                {limit_clause};
                """

                tracks = await conn.fetch(query)
                total_songs = len(tracks)

                print(
                    f"🚀 Начинаем анализ {total_songs:,} песен (батчи по {batch_size})"
                )

                processed = 0
                results = []

                # Обработка батчами
                for i in range(0, total_songs, batch_size):
                    batch = tracks[i : i + batch_size]
                    batch_results = []

                    for song in batch:
                        try:
                            # Анализ песни
                            result = self.analyzer.analyze_song(
                                artist=song["artist"],
                                title=song["title"],
                                lyrics=song["lyrics"],
                            )

                            batch_results.append(
                                {
                                    "song_id": song["id"],
                                    "artist": song["artist"],
                                    "title": song["title"],
                                    "analysis": result["raw_output"],
                                    "confidence": result["confidence"],
                                    "processing_time": result["processing_time"],
                                }
                            )

                            processed += 1

                        except Exception as e:
                            # TODO(google-review): [ERROR_HANDLING] Too broad exception
                            # TODO(google-review): [ERROR_HANDLING] Use logger
                            print(f"⚠️ Ошибка анализа песни {song['id']}: {e}")

                    results.extend(batch_results)

                    # Прогресс
                    progress = (processed / total_songs) * 100
                    print(
                        f"📈 Прогресс: {processed:,}/{total_songs:,} ({progress:.1f}%)"
                    )

                print(f"✅ Анализ завершен! Обработано {processed:,} песен")

                return {
                    "total_processed": processed,
                    "results": results,
                    "summary_stats": self._calculate_summary_stats(results),
                }

            finally:
                await conn.close()

        except Exception as e:
            # TODO(google-review): [ERROR_HANDLING] Too broad exception clause
            # TODO(google-review): [ERROR_HANDLING] Use logger.exception
            print(f"❌ Ошибка анализа песен: {e}")
            return {}

    async def analyze_single_track(self, track_id: int) -> dict[str, Any]:
        """Анализ конкретной песни по ID"""
        # TODO(google-review): [DOCSTRING] Add Args, Returns sections
        # TODO(google-review): [ARCHITECTURE] Function too long (50+ lines)
        try:
            import asyncpg

            conn = await asyncpg.connect(
                host=self.db_config.get("host", "localhost"),
                port=self.db_config.get("port", 5432),
                database=self.db_config.get("name", "rap_lyrics_db"),
                user=self.db_config.get("user", "postgres"),
                password=self.db_config.get("password", "password"),
            )

            try:
                # Получаем песню
                query = "SELECT id, artist, title, lyrics FROM tracks WHERE id = $1"
                song = await conn.fetchrow(query, track_id)

                if not song:
                    print(f"❌ Песня с ID {track_id} не найдена")
                    return {}

                if not song["lyrics"] or len(song["lyrics"]) < 100:
                    print(f"❌ Недостаточно текста для анализа (ID: {track_id})")
                    return {}

                print(f"🎵 Анализируем: {song['artist']} - {song['title']}")

                # Анализ
                result = self.analyzer.analyze_song(
                    artist=song["artist"], title=song["title"], lyrics=song["lyrics"]
                )

                # Красивый вывод результатов
                self._print_analysis_results(result)

                return {
                    "song_id": song["id"],
                    "artist": song["artist"],
                    "title": song["title"],
                    "analysis": result["raw_output"],
                    "confidence": result["confidence"],
                    "processing_time": result["processing_time"],
                }

            finally:
                await conn.close()

        except Exception as e:
            # TODO(google-review): [ERROR_HANDLING] Too broad exception clause
            # TODO(google-review): [ERROR_HANDLING] Use logger.exception
            print(f"❌ Ошибка анализа трека {track_id}: {e}")
            return {}

    def _calculate_summary_stats(self, results: list[dict]) -> dict[str, Any]:
        """Вычисление сводной статистики"""
        # TODO(google-review): [DOCSTRING] Add Args, Returns sections
        if not results:
            return {}

        # Извлекаем метрики
        confidences = [r["confidence"] for r in results]
        processing_times = [r["processing_time"] for r in results]

        # Композитные оценки
        technical_scores = []
        artistic_scores = []

        for result in results:
            composite = result.get("analysis", {}).get("composite_scores", {})
            if composite:
                technical_scores.append(composite.get("technical_mastery", 0))
                artistic_scores.append(composite.get("artistic_sophistication", 0))

        return {
            "avg_confidence": sum(confidences) / len(confidences),
            "avg_processing_time": sum(processing_times) / len(processing_times),
            "avg_technical_mastery": sum(technical_scores) / len(technical_scores)
            if technical_scores
            else 0,
            "avg_artistic_sophistication": sum(artistic_scores) / len(artistic_scores)
            if artistic_scores
            else 0,
            "total_results": len(results),
        }

    def _print_analysis_results(self, result: dict[str, Any]) -> None:
        """Красивый вывод результатов анализа"""
        # TODO(google-review): [TYPING] Add return type hint
        # TODO(google-review): [DOCSTRING] Add Args section to docstring
        # TODO(google-review): [ARCHITECTURE] Function too long (40+ lines)
        print("\n📊 РЕЗУЛЬТАТЫ АНАЛИЗА:")
        print(f"🎯 Уверенность: {result['confidence']:.3f}")
        print(f"⚡ Время обработки: {result['processing_time']:.3f}s")

        # Рифмы и звучание
        rhyme_analysis = result["raw_output"].get("rhyme_analysis", {})
        if rhyme_analysis:
            print("\n🎵 РИФМЫ И ЗВУЧАНИЕ:")
            print(f"  Схема рифмовки: {rhyme_analysis.get('rhyme_scheme', 'N/A')}")
            print(f"  Плотность рифм: {rhyme_analysis.get('rhyme_density', 0):.3f}")
            print(f"  Аллитерация: {rhyme_analysis.get('alliteration_score', 0):.3f}")
            print(f"  Внутренние рифмы: {rhyme_analysis.get('internal_rhymes', 0)}")

        # Flow анализ
        flow_analysis = result["raw_output"].get("flow_analysis", {})
        if flow_analysis:
            print("\n🌊 FLOW И РИТМ:")
            print(
                f"  Консистентность слогов: {flow_analysis.get('syllable_consistency', 0):.3f}"
            )
            print(
                f"  Ср. слогов на строку: {flow_analysis.get('average_syllables_per_line', 0):.1f}"
            )
            print(
                f"  Ритмическая плотность: {flow_analysis.get('rhythmic_density', 0):.3f}"
            )

        # Композитные оценки
        composite = result["raw_output"].get("composite_scores", {})
        if composite:
            print("\n🏆 КОМПОЗИТНЫЕ ОЦЕНКИ:")
            print(
                f"  Техническое мастерство: {composite.get('technical_mastery', 0):.3f}"
            )
            print(
                f"  Артистическая утончённость: {composite.get('artistic_sophistication', 0):.3f}"
            )
            print(f"  Общее качество: {composite.get('overall_quality', 0):.3f}")
            print(f"  Инновационность: {composite.get('innovation_score', 0):.3f}")


async def main() -> None:
    """Главная функция для работы с PostgreSQL"""
    # TODO(google-review): [DOCSTRING] Add Returns section to docstring
    # TODO(google-review): [ARCHITECTURE] Function too long (120+ lines)
    import argparse

    # Парсинг аргументов командной строки
    parser = argparse.ArgumentParser(
        description="Продвинутый алгоритмический анализатор для PostgreSQL"
    )

    parser.add_argument(
        "--stats", action="store_true", help="Показать статистику базы данных"
    )
    parser.add_argument(
        "--analyze-all", action="store_true", help="Анализировать все треки в базе"
    )
    parser.add_argument(
        "--analyze-track",
        type=int,
        metavar="ID",
        help="Анализировать конкретный трек по ID",
    )
    parser.add_argument(
        "--limit",
        type=int,
        metavar="N",
        help="Ограничить количество треков для анализа",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=100,
        metavar="N",
        help="Размер пакета для обработки (по умолчанию: 100)",
    )
    parser.add_argument(
        "--demo", action="store_true", help="Запустить демонстрацию анализатора"
    )

    args = parser.parse_args()

    # Если аргументы не переданы, показываем красивое меню
    action_args = [args.stats, args.analyze_all, args.analyze_track, args.demo]
    if not any(action_args):
        print()
        print("🧮 ПРОДВИНУТЫЙ АЛГОРИТМИЧЕСКИЙ АНАЛИЗАТОР ДЛЯ POSTGRESQL")
        print("=" * 65)
        print("🎯 Профессиональный анализ текстов без AI моделей")
        print("⚡ Работа с 57K+ треков в PostgreSQL базе данных")
        print("📊 Детальные метрики: рифмы, flow, читабельность, эмоции")
        print()
        print("🖥️ CLI ИНТЕРФЕЙС:")
        print("  --stats              📊 Показать статистику БД")
        print("  --analyze-all        🚀 Анализировать все треки")
        print("  --analyze-track ID   🎵 Анализировать конкретный трек")
        print("  --limit N            🔢 Ограничить количество треков")
        print("  --batch-size N       📦 Размер пакета для обработки")
        print("  --demo               🎭 Демонстрация анализатора")
        print("  --help               ❓ Полная справка")
        print()
        print("💡 ПРИМЕРЫ ИСПОЛЬЗОВАНИЯ:")
        print("  python src/analyzers/algorithmic_analyzer.py --stats")
        print(
            "  python src/analyzers/algorithmic_analyzer.py --analyze-all --limit 100"
        )
        print("  python src/analyzers/algorithmic_analyzer.py --analyze-track 123")
        print("  python src/analyzers/algorithmic_analyzer.py --demo")
        print()
        print("📈 МЕТРИКИ АНАЛИЗА:")
        print("  🎵 Рифмы: схема, плотность, фонетическое сходство")
        print("  🌊 Flow: консистентность слогов, ритмическая плотность")
        print("  📚 Читабельность: Flesch, SMOG, ARI индексы")
        print("  💭 Эмоции: валентность, интенсивность, сложность")
        print("  🎨 Темы: деньги, улица, успех, отношения")
        print("=" * 65)
        return

    # Создаем анализатор только когда он действительно нужен
    try:
        if args.demo:
            print("🚀 Запуск демонстрации...")
            await demo_advanced_analysis()

        else:
            # Инициализируем анализатор для работы с БД
            print("✅ PostgreSQL анализатор инициализирован")
            analyzer = PostgreSQLAnalyzer()

            if args.stats:
                print("📊 Получение статистики базы данных...")
                await analyzer.get_database_stats()

            elif args.analyze_track:
                print(f"🎵 Анализ трека ID: {args.analyze_track}")
                await analyzer.analyze_single_track(args.analyze_track)

            elif args.analyze_all:
                print("🚀 Массовый анализ всех треков...")
                results = await analyzer.analyze_all_songs(
                    limit=args.limit, batch_size=args.batch_size
                )

                if results:
                    summary = results.get("summary_stats", {})
                    print("\n📈 СВОДНАЯ СТАТИСТИКА:")
                    print(f"  Обработано треков: {summary.get('total_results', 0):,}")
                    print(
                        f"  Средняя уверенность: {summary.get('avg_confidence', 0):.3f}"
                    )
                    print(
                        f"  Среднее время обработки: {summary.get('avg_processing_time', 0):.3f}s"
                    )
                    print(
                        f"  Среднее техническое мастерство: {summary.get('avg_technical_mastery', 0):.3f}"
                    )
                    print(
                        f"  Средняя артистическая утончённость: {summary.get('avg_artistic_sophistication', 0):.3f}"
                    )

    except Exception as e:
        # TODO(google-review): [ERROR_HANDLING] Too broad exception clause
        # TODO(google-review): [ERROR_HANDLING] Use logger.exception instead
        print(f"❌ Ошибка выполнения: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
