"""
🧮 Базовый алгоритмический анализатор текстов песен

НАЗНАЧЕНИЕ:
- Быстрый анализ текста без использования AI моделей
- Оценка настроения, сложности, рифм, повторяемости
- Демонстрация интерфейса BaseAnalyzer

ИСПОЛЬЗОВАНИЕ:
Используется через main.py, batch_processor, analyzer_cli

ЗАВИСИМОСТИ:
- Python 3.8+
- src/interfaces/analyzer_interface.py

РЕЗУЛЬТАТ:
- Метрики по тексту: настроение, сложность, рифмы, повторяемость
- Быстрый baseline для сравнения с AI-анализаторами

АВТОР: AI Assistant
ДАТА: Сентябрь 2025
"""
import re
import time
from datetime import datetime
from typing import Dict, Any, List
from collections import Counter

from interfaces.analyzer_interface import BaseAnalyzer, AnalysisResult, register_analyzer


@register_analyzer("algorithmic_basic")
class AlgorithmicAnalyzer(BaseAnalyzer):
    """
    Базовый алгоритмический анализатор текста песен.
    
    Реализует простые метрики без использования AI:
    - Анализ настроения по ключевым словам
    - Сложность текста (readability)
    - Рифмовая схема
    - Повторяемость слов
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """Инициализация анализатора с настройками"""
        super().__init__(config)
        
        # Настройки по умолчанию
        self.min_word_length = self.config.get('min_word_length', 3)
        self.sentiment_threshold = self.config.get('sentiment_threshold', 0.1)
        
        # Словари для анализа настроения
        self._load_sentiment_dictionaries()
    
    def _load_sentiment_dictionaries(self):
        """Загрузка словарей для анализа настроения"""
        # Позитивные слова
        self.positive_words = {
            'love', 'happy', 'joy', 'good', 'great', 'amazing', 'beautiful',
            'win', 'success', 'best', 'awesome', 'perfect', 'wonderful',
            'smile', 'laugh', 'peace', 'hope', 'dream', 'shine', 'bright'
        }
        
        # Негативные слова
        self.negative_words = {
            'hate', 'sad', 'angry', 'bad', 'terrible', 'awful', 'ugly',
            'lose', 'fail', 'worst', 'horrible', 'pain', 'hurt', 'cry',
            'dark', 'death', 'kill', 'fight', 'war', 'broke', 'poor'
        }
        
        # Слова агрессии/насилия
        self.aggressive_words = {
            'kill', 'murder', 'gun', 'shoot', 'fight', 'blood', 'violence',
            'punch', 'hit', 'attack', 'destroy', 'revenge', 'enemy'
        }
        
        # Слова о деньгах/материальном
        self.money_words = {
            'money', 'cash', 'rich', 'wealth', 'gold', 'diamond', 'expensive',
            'luxury', 'mansion', 'car', 'chain', 'brand', 'designer'
        }
    
    def analyze_song(self, artist: str, title: str, lyrics: str) -> AnalysisResult:
        """
        Анализ песни с использованием алгоритмических методов.
        
        Args:
            artist: Имя исполнителя
            title: Название песни  
            lyrics: Текст песни
            
        Returns:
            AnalysisResult со структурированными результатами
        """
        start_time = time.time()
        
        # Валидация входных данных
        if not self.validate_input(artist, title, lyrics):
            raise ValueError("Invalid input parameters")
        
        # Предобработка текста
        processed_lyrics = self.preprocess_lyrics(lyrics)
        
        # Выполнение анализа
        analysis_results = self._perform_analysis(processed_lyrics)
        
        # Вычисление общей уверенности
        confidence = self._calculate_confidence(analysis_results)
        
        processing_time = time.time() - start_time
        
        return AnalysisResult(
            artist=artist,
            title=title,
            analysis_type="algorithmic_basic",
            confidence=confidence,
            metadata={
                "analyzer_version": "1.0.0",
                "processing_date": datetime.now().isoformat(),
                "lyrics_length": len(processed_lyrics),
                "word_count": len(processed_lyrics.split())
            },
            raw_output=analysis_results,
            processing_time=processing_time,
            timestamp=datetime.now().isoformat()
        )
    
    def _perform_analysis(self, lyrics: str) -> Dict[str, Any]:
        """Выполнение всех видов анализа"""
        words = self._extract_words(lyrics)
        
        return {
            "sentiment_analysis": self._analyze_sentiment(words),
            "complexity_analysis": self._analyze_complexity(lyrics, words),
            "themes_analysis": self._analyze_themes(words),
            "structure_analysis": self._analyze_structure(lyrics),
            "vocabulary_analysis": self._analyze_vocabulary(words)
        }
    
    def _extract_words(self, lyrics: str) -> List[str]:
        """Извлечение слов из текста"""
        # Удаление знаков препинания и приведение к нижнему регистру
        words = re.findall(r'\b[a-zA-Z]{%d,}\b' % self.min_word_length, lyrics.lower())
        return words
    
    def _analyze_sentiment(self, words: List[str]) -> Dict[str, Any]:
        """Анализ настроения по ключевым словам"""
        word_set = set(words)
        
        positive_count = len(word_set.intersection(self.positive_words))
        negative_count = len(word_set.intersection(self.negative_words))
        aggressive_count = len(word_set.intersection(self.aggressive_words))
        
        total_sentiment_words = positive_count + negative_count + aggressive_count
        
        if total_sentiment_words == 0:
            sentiment_score = 0
            sentiment_label = "neutral"
        else:
            sentiment_score = (positive_count - negative_count - aggressive_count) / total_sentiment_words
            
            if sentiment_score > self.sentiment_threshold:
                sentiment_label = "positive"
            elif sentiment_score < -self.sentiment_threshold:
                sentiment_label = "negative"
            else:
                sentiment_label = "neutral"
        
        return {
            "sentiment_score": round(sentiment_score, 3),
            "sentiment_label": sentiment_label,
            "positive_words_count": positive_count,
            "negative_words_count": negative_count,
            "aggressive_words_count": aggressive_count,
            "total_sentiment_words": total_sentiment_words
        }
    
    def _analyze_complexity(self, lyrics: str, words: List[str]) -> Dict[str, Any]:
        """Анализ сложности текста"""
        sentences = re.split(r'[.!?]+', lyrics)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        if not sentences:
            return {"error": "No sentences found"}
        
        # Средняя длина предложения
        avg_sentence_length = sum(len(s.split()) for s in sentences) / len(sentences)
        
        # Уникальность словаря
        unique_words = len(set(words))
        total_words = len(words)
        vocabulary_richness = unique_words / total_words if total_words > 0 else 0
        
        # Сложность на основе длины слов
        avg_word_length = sum(len(word) for word in words) / len(words) if words else 0
        
        # Простая метрика читабельности (Flesch-подобная)
        readability_score = 206.835 - (1.015 * avg_sentence_length) - (84.6 * (avg_word_length / 4.7))
        readability_score = max(0, min(100, readability_score))  # Ограничение 0-100
        
        return {
            "avg_sentence_length": round(avg_sentence_length, 2),
            "vocabulary_richness": round(vocabulary_richness, 3),
            "avg_word_length": round(avg_word_length, 2),
            "readability_score": round(readability_score, 1),
            "unique_words": unique_words,
            "total_words": total_words,
            "sentences_count": len(sentences)
        }
    
    def _analyze_themes(self, words: List[str]) -> Dict[str, Any]:
        """Анализ тематики песни"""
        word_set = set(words)
        
        themes = {
            "money_materialism": len(word_set.intersection(self.money_words)),
            "violence_aggression": len(word_set.intersection(self.aggressive_words)),
            "love_relationships": len(word_set.intersection(self.positive_words)),
            "negative_emotions": len(word_set.intersection(self.negative_words))
        }
        
        # Определение доминирующей темы
        dominant_theme = max(themes.items(), key=lambda x: x[1])
        
        return {
            "themes_scores": themes,
            "dominant_theme": dominant_theme[0] if dominant_theme[1] > 0 else "neutral",
            "dominant_theme_score": dominant_theme[1]
        }
    
    def _analyze_structure(self, lyrics: str) -> Dict[str, Any]:
        """Анализ структуры текста"""
        lines = [line.strip() for line in lyrics.split('\n') if line.strip()]
        
        # Поиск повторяющихся строк (припевы)
        line_counts = Counter(lines)
        repeated_lines = {line: count for line, count in line_counts.items() if count > 1}
        
        # Анализ рифм (упрощенный)
        rhyme_analysis = self._simple_rhyme_analysis(lines)
        
        return {
            "total_lines": len(lines),
            "unique_lines": len(set(lines)),
            "repeated_lines_count": len(repeated_lines),
            "repetition_ratio": len(repeated_lines) / len(set(lines)) if lines else 0,
            "rhyme_density": rhyme_analysis["rhyme_density"],
            "avg_line_length": sum(len(line.split()) for line in lines) / len(lines) if lines else 0
        }
    
    def _simple_rhyme_analysis(self, lines: List[str]) -> Dict[str, Any]:
        """Упрощенный анализ рифм"""
        if len(lines) < 2:
            return {"rhyme_density": 0, "rhyming_pairs": 0}
        
        # Извлечение последних слов из строк
        last_words = []
        for line in lines:
            words = line.split()
            if words:
                last_word = re.sub(r'[^a-zA-Z]', '', words[-1].lower())
                if len(last_word) >= 2:
                    last_words.append(last_word)
        
        # Поиск возможных рифм (упрощенно - по окончаниям)
        rhyming_pairs = 0
        for i in range(len(last_words)):
            for j in range(i + 1, len(last_words)):
                if self._words_rhyme(last_words[i], last_words[j]):
                    rhyming_pairs += 1
        
        rhyme_density = rhyming_pairs / len(last_words) if last_words else 0
        
        return {
            "rhyme_density": round(rhyme_density, 3),
            "rhyming_pairs": rhyming_pairs,
            "total_line_endings": len(last_words)
        }
    
    def _words_rhyme(self, word1: str, word2: str) -> bool:
        """Простая проверка рифмы по окончанию"""
        if len(word1) < 2 or len(word2) < 2:
            return False
        
        # Проверка совпадения последних 2-3 символов
        return word1[-2:] == word2[-2:] or word1[-3:] == word2[-3:]
    
    def _analyze_vocabulary(self, words: List[str]) -> Dict[str, Any]:
        """Анализ словарного запаса"""
        if not words:
            return {"error": "No words to analyze"}
        
        word_freq = Counter(words)
        
        # Самые частые слова
        most_common = word_freq.most_common(10)
        
        # Распределение частот
        freq_distribution = {}
        for word, freq in word_freq.items():
            freq_key = f"freq_{freq}"
            freq_distribution[freq_key] = freq_distribution.get(freq_key, 0) + 1
        
        return {
            "most_common_words": most_common,
            "frequency_distribution": freq_distribution,
            "hapax_legomena": sum(1 for freq in word_freq.values() if freq == 1),  # Слова, встречающиеся 1 раз
            "vocabulary_size": len(word_freq)
        }
    
    def _calculate_confidence(self, analysis_results: Dict[str, Any]) -> float:
        """Вычисление общей уверенности в результатах"""
        # Простая метрика уверенности на основе полноты анализа
        confidence_factors = []
        
        # Проверка наличия результатов анализа
        for analysis_type, results in analysis_results.items():
            if isinstance(results, dict) and "error" not in results:
                confidence_factors.append(1.0)
            else:
                confidence_factors.append(0.0)
        
        # Дополнительные факторы
        if "vocabulary_analysis" in analysis_results:
            vocab_results = analysis_results["vocabulary_analysis"]
            if "vocabulary_size" in vocab_results:
                # Больше словарь = выше уверенность
                vocab_size = vocab_results["vocabulary_size"]
                vocab_confidence = min(1.0, vocab_size / 50)  # Нормализация
                confidence_factors.append(vocab_confidence)
        
        return sum(confidence_factors) / len(confidence_factors) if confidence_factors else 0.0
    
    def get_analyzer_info(self) -> Dict[str, Any]:
        """Получение метаинформации об анализаторе"""
        return {
            "name": "AlgorithmicAnalyzer",
            "version": "1.0.0",
            "description": "Basic algorithmic text analysis without AI models",
            "author": "Rap Scraper Project",
            "type": self.analyzer_type,
            "supported_features": self.supported_features,
            "config_options": {
                "min_word_length": "Minimum word length for analysis (default: 3)",
                "sentiment_threshold": "Threshold for sentiment classification (default: 0.1)"
            }
        }
    
    @property
    def analyzer_type(self) -> str:
        """Тип анализатора"""
        return "algorithmic"
    
    @property
    def supported_features(self) -> List[str]:
        """Поддерживаемые функции анализа"""
        return [
            "sentiment_analysis",
            "complexity_analysis", 
            "themes_analysis",
            "structure_analysis",
            "vocabulary_analysis",
            "rhyme_analysis"
        ]
